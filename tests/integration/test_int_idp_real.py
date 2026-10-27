"""TokenValidator / JWKSCache against a real HTTP IdP (OIDC discovery, rotation, outage, hostile JWKS)."""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ai_gateway.auth import TokenValidator
from ai_gateway.config import IdPConfig
from ai_gateway.errors import AuthenticationError, AuthFailure
from ai_gateway.observability.telemetry import NullTelemetry
from integration.servers import ThreadedServer
from support import FakeClock, KeyPair


class IdpState:
    def __init__(self) -> None:
        self.keys: list[dict[str, Any]] = []
        self.issuer_override: str | None = None
        self.jwks_status = 200
        self.jwks_body: bytes | None = None
        self.discovery_status = 200
        self.jwks_uri_override: str | None = None
        self.hits: dict[str, int] = {"jwks": 0, "disco": 0}


def build_idp(state: IdpState, base: dict[str, str]) -> Starlette:
    async def disco(request: Request) -> Response:
        state.hits["disco"] += 1
        if state.discovery_status != 200:
            return Response(status_code=state.discovery_status)
        return JSONResponse(
            {
                "issuer": state.issuer_override or base["url"],
                "jwks_uri": state.jwks_uri_override or f"{base['url']}/jwks",
            }
        )

    async def jwks(request: Request) -> Response:
        state.hits["jwks"] += 1
        if state.jwks_body is not None:
            return Response(state.jwks_body, status_code=state.jwks_status, media_type="application/json")
        return JSONResponse({"keys": state.keys}, status_code=state.jwks_status)

    return Starlette(routes=[Route("/.well-known/openid-configuration", disco), Route("/jwks", jwks)])


@pytest.fixture
def idp_env() -> Any:
    state, base = IdpState(), {"url": ""}
    srv = ThreadedServer(build_idp(state, base)).__enter__()
    base["url"] = srv.url
    try:
        yield state, srv
    finally:
        srv.stop()


def make_validator(srv: ThreadedServer, http: httpx.AsyncClient, **kw: Any) -> TokenValidator:
    """Validator on a fake clock (JWKS min refresh interval is >= 1 s; tests advance `v.clock`)."""
    idp = IdPConfig(
        name="real",
        issuer=srv.url,
        audiences=["ai-gateway"],
        jwks_min_refresh_interval_s=1,
        jwks_allow_insecure_local=True,
        **kw,
    )
    clock = FakeClock(time.time())
    v = TokenValidator([idp], http, NullTelemetry(), clock=clock, jwks_clock=clock)
    v.clock = clock  # type: ignore[attr-defined]
    return v


def token(kp: KeyPair, srv: ThreadedServer, **kw: Any) -> str:
    return kp.mint(issuer=srv.url, **kw)


async def fails(v: TokenValidator, tok: str) -> AuthFailure:
    with pytest.raises(AuthenticationError) as ei:
        await v.validate(tok)
    return ei.value.failure


async def test_oidc_discovery_rotation_and_hostile_keys(idp_env: Any) -> None:
    state, srv = idp_env
    k1, k2 = KeyPair("k1"), KeyPair("k2")
    oct_key = {"kty": "oct", "kid": "sym", "k": "c2VjcmV0"}
    private = {**k1.jwk(), "kid": "priv", "d": "AQAB"}
    state.keys = [
        oct_key,
        private,
        {"kty": "RSA"},
        "junk",
        {**k1.jwk(), "use": "enc", "kid": "enc"},
        k1.jwk(),
    ]
    async with httpx.AsyncClient(timeout=5, trust_env=False) as http:
        v = make_validator(srv, http)
        p = await v.validate(token(k1, srv, scopes="a b", client_id="c1"))
        assert p.client_id == "c1" and p.scopes == frozenset({"a", "b"})
        assert state.hits["disco"] == 1 and v.ready
        # hostile / unusable key ids resolve to nothing
        for kid in ("sym", "priv", "enc"):
            fake = KeyPair(kid)
            assert await fails(v, token(fake, srv)) is AuthFailure.UNKNOWN_KID
        # key rotation: new kid appears at the IdP, gateway refetches on the unknown kid
        state.keys = [k1.jwk(), k2.jwk()]
        v.clock.advance(2)  # type: ignore[attr-defined]  # past the min refresh interval
        assert (await v.validate(token(k2, srv, client_id="c2"))).client_id == "c2"
        # known kid but wrong signature
        forged = KeyPair("k2")
        assert await fails(v, token(forged, srv)) is AuthFailure.BAD_SIGNATURE


async def test_discovery_failures_and_issuer_mismatch_fail_closed(idp_env: Any) -> None:
    state, srv = idp_env
    kp = KeyPair("k1")
    state.keys = [kp.jwk()]
    async with httpx.AsyncClient(timeout=5, trust_env=False) as http:

        def evil_issuer() -> None:
            state.issuer_override = "http://evil.example"

        def discovery_500() -> None:
            state.issuer_override, state.discovery_status = None, 500

        def evil_jwks_uri() -> None:
            state.discovery_status, state.jwks_uri_override = 200, "http://evil.example/jwks"

        for mutate in (evil_issuer, discovery_500, evil_jwks_uri):
            mutate()
            v = make_validator(srv, http)
            # no usable keys can be obtained: never accept the token
            assert await fails(v, token(kp, srv)) in (AuthFailure.JWKS_UNAVAILABLE, AuthFailure.UNKNOWN_KID)
            assert not v.ready


async def test_jwks_endpoint_garbage_and_outage_keep_last_good_keys(idp_env: Any) -> None:
    state, srv = idp_env
    kp = KeyPair("k1")
    state.keys = [kp.jwk()]
    clock = FakeClock(time.time())
    async with httpx.AsyncClient(timeout=5, trust_env=False) as http:
        idp = IdPConfig(
            name="real", issuer=srv.url, audiences=["ai-gateway"], jwks_min_refresh_interval_s=1,
            jwks_cache_ttl_s=10, jwks_stale_if_error_s=3600, jwks_allow_insecure_local=True,
        )  # fmt: skip
        v = TokenValidator([idp], http, NullTelemetry(), clock=clock, jwks_clock=clock)
        await v.validate(token(kp, srv))
        hits = state.hits["jwks"]
        for body, status in (
            (b"not json", 200),
            (b'["list"]', 200),
            (b"{}", 200),
            (b'{"keys": []}', 200),
            (b"{}", 503),
        ):
            state.jwks_body, state.jwks_status = body, status
            clock.advance(11)  # cache expired -> refresh attempted, fails, stale keys still serve
            assert (await v.validate(token(kp, srv, client_id="still-ok"))).client_id == "still-ok"
            assert state.hits["jwks"] > hits
            hits = state.hits["jwks"]
        state.jwks_body, state.jwks_status = None, 200
        clock.advance(11)
        await v.validate(token(kp, srv))  # recovered


async def test_unusable_algorithms_in_jwks_are_ignored(idp_env: Any) -> None:
    state, srv = idp_env
    kp = KeyPair("k1")
    # a key that claims an algorithm outside the IdP allow-list / mismatched kty is dropped
    state.keys = [
        {**kp.jwk(), "alg": "HS256", "kid": "k-hs"},
        {**kp.jwk(), "kid": "k-es", "alg": "ES256"},
        kp.jwk(),
    ]
    async with httpx.AsyncClient(timeout=5, trust_env=False) as http:
        v = make_validator(srv, http)
        await v.validate(token(kp, srv))
        other = KeyPair("k-es")
        assert await fails(v, token(other, srv)) is AuthFailure.UNKNOWN_KID
