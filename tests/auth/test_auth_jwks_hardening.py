"""JWKS key selection, bounded streaming fetch and endpoint (SSRF) policy."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import jwt
import pytest
import respx
from cryptography.hazmat.primitives.asymmetric import ec
from helpers import AUDIENCE, ISSUER, JWKS_URL, FakeClock, RecTelemetry, make_idp
from jwt.algorithms import ECAlgorithm

from ai_gateway.auth import JWKSCache, TokenValidator
from ai_gateway.auth import jwks as jwks_mod
from ai_gateway.errors import AuthenticationError, AuthFailure
from support import KeyPair

DISCOVERY = f"{ISSUER}/.well-known/openid-configuration"


class ECKeyPair:
    def __init__(self, kid: str) -> None:
        self.kid = kid
        self.private = ec.generate_private_key(ec.SECP256R1())

    def jwk(self) -> dict[str, Any]:
        d: dict[str, Any] = json.loads(ECAlgorithm.to_jwk(self.private.public_key()))
        d.update(kid=self.kid, use="sig", alg="ES256")
        return d

    def mint(self, *, kid: str | None = "unset") -> str:
        now = int(time.time())
        claims = {"iss": ISSUER, "aud": AUDIENCE, "sub": "c", "iat": now, "exp": now + 300}
        headers = {} if kid is None else {"kid": self.kid if kid == "unset" else kid}
        return jwt.encode(claims, self.private, algorithm="ES256", headers=headers)


def _cache(resolver: Any = None, **idp_kw: Any) -> tuple[JWKSCache, FakeClock, httpx.AsyncClient]:
    clock = FakeClock(0.0)
    client = httpx.AsyncClient()
    kw: dict[str, Any] = {"resolver": resolver} if resolver is not None else {}
    return JWKSCache(make_idp(**idp_kw), client, RecTelemetry(), clock=clock, **kw), clock, client


# --------------------------------------------------------------------------- key selection


async def test_duplicate_kid_across_rsa_and_ec_verifies_with_compatible_key() -> None:
    rsa_kp, ec_kp = KeyPair("dup"), ECKeyPair("dup")
    for keys in ([rsa_kp.jwk(), ec_kp.jwk()], [ec_kp.jwk(), rsa_kp.jwk()]):
        clock = FakeClock(time.time())
        async with httpx.AsyncClient() as client:
            v = TokenValidator([make_idp()], client, RecTelemetry(), jwks_clock=clock)
            with respx.mock:
                respx.get(JWKS_URL).respond(json={"keys": keys})
                assert (await v.validate(ec_kp.mint())).client_id == "c"
                assert (await v.validate(rsa_kp.mint(client_id="c"))).client_id == "c"


async def test_kid_present_but_only_incompatible_keys_is_alg_not_allowed_without_refresh() -> None:
    ec_kp = ECKeyPair("k")
    cache, clock, client = _cache()
    with respx.mock:
        route = respx.get(JWKS_URL).respond(json={"keys": [ec_kp.jwk()]})
        await cache.get_key("k", "ES256")
        clock.advance(100)
        with pytest.raises(AuthenticationError) as ei:
            await cache.get_key("k", "RS256")
        assert ei.value.failure is AuthFailure.ALG_NOT_ALLOWED
        assert route.call_count == 1
    await client.aclose()


async def test_no_kid_with_multiple_compatible_keys_fails_fast_without_refresh() -> None:
    a, b = KeyPair("a"), KeyPair("b")
    cache, clock, client = _cache(jwks_min_refresh_interval_s=1)
    with respx.mock:
        route = respx.get(JWKS_URL).respond(json={"keys": [a.jwk(), b.jwk()]})
        await cache.get_key("a", "RS256")
        assert route.call_count == 1
        clock.advance(100)  # cooldown long over: a refresh would be allowed, but must not happen
        for _ in range(5):
            with pytest.raises(AuthenticationError) as ei:
                await cache.get_key(None, "RS256")
            assert ei.value.failure is AuthFailure.UNKNOWN_KID
        assert route.call_count == 1
    await client.aclose()


async def test_no_kid_resolves_when_exactly_one_key_fits_the_alg() -> None:
    rsa_kp, ec_kp = KeyPair("r"), ECKeyPair("e")
    cache, _, client = _cache()
    with respx.mock:
        respx.get(JWKS_URL).respond(json={"keys": [rsa_kp.jwk(), ec_kp.jwk()]})
        assert await cache.get_key(None, "ES256") is not None
        assert await cache.get_key(None, "RS256") is not None
    await client.aclose()


# --------------------------------------------------------------------------- streaming size cap


def _chunks(n: int, size: int, counter: list[int]) -> Any:
    async def gen() -> AsyncIterator[bytes]:
        for _ in range(n):
            counter[0] += 1
            yield b" " * size

    return gen()


async def test_chunked_oversize_body_aborts_once_cap_is_exceeded() -> None:
    cache, _, client = _cache()
    sent = [0]
    with respx.mock:
        respx.get(JWKS_URL).mock(return_value=httpx.Response(200, content=_chunks(100, 100_000, sent)))
        with pytest.raises(AuthenticationError) as ei:
            await cache.get_key("k1", "RS256")
        assert ei.value.failure is AuthFailure.JWKS_UNAVAILABLE
    assert sent[0] <= jwks_mod.MAX_JWKS_BYTES // 100_000 + 2  # stopped reading, did not drain 10 MB
    await client.aclose()


async def test_large_content_length_rejected_before_reading_body() -> None:
    cache, _, client = _cache()
    sent = [0]
    with respx.mock:
        respx.get(JWKS_URL).mock(
            return_value=httpx.Response(
                200,
                headers={"content-length": str(jwks_mod.MAX_JWKS_BYTES + 1)},
                content=_chunks(1, 10, sent),
            )
        )
        with pytest.raises(AuthenticationError):
            await cache.get_key("k1", "RS256")
    assert sent[0] == 0
    await client.aclose()


async def test_oversize_does_not_evict_good_cache_and_normal_size_still_works() -> None:
    kp = KeyPair("k1")
    cache, clock, client = _cache(jwks_cache_ttl_s=10, jwks_min_refresh_interval_s=1)
    with respx.mock:
        route = respx.get(JWKS_URL).respond(json=kp.jwks())
        await cache.get_key("k1", "RS256")
        clock.advance(11)
        route.mock(return_value=httpx.Response(200, content=_chunks(100, 100_000, [0])))
        assert await cache.get_key("k1", "RS256") is not None  # stale cache kept
    await client.aclose()


# --------------------------------------------------------------------------- endpoint policy


async def _fails_unavailable(cache: JWKSCache) -> None:
    with pytest.raises(AuthenticationError) as ei:
        await cache.get_key("k1", "RS256")
    assert ei.value.failure is AuthFailure.JWKS_UNAVAILABLE


@pytest.mark.parametrize(
    "uri",
    [
        "https://evil.test/keys",  # other host
        "http://idp.test/keys",  # not https
        "https://user:pw@idp.test/keys",  # userinfo
        "https://169.254.169.254/latest/meta-data",  # metadata, not allow-listed host
    ],
)
async def test_discovered_jwks_uri_must_be_https_on_issuer_host(uri: str) -> None:
    cache, _, client = _cache(jwks_uri=None)
    with respx.mock(assert_all_called=False) as mock:
        mock.get(DISCOVERY).respond(json={"issuer": ISSUER, "jwks_uri": uri})
        mock.route().respond(json=KeyPair("k1").jwks())
        await _fails_unavailable(cache)
        assert [str(c.request.url) for c in mock.calls] == [DISCOVERY]  # nothing else was fetched
    await client.aclose()


async def test_discovered_jwks_uri_on_allow_listed_host_is_accepted() -> None:
    kp = KeyPair("k1")
    cache, _, client = _cache(jwks_uri=None, jwks_allowed_hosts=["Keys.Example.com"])
    with respx.mock:
        respx.get(DISCOVERY).respond(json={"issuer": ISSUER, "jwks_uri": "https://keys.example.com/jwks"})
        respx.get("https://keys.example.com/jwks").respond(json=kp.jwks())
        assert await cache.get_key("k1", "RS256") is not None
    await client.aclose()


async def test_allow_listed_host_still_refused_if_it_is_a_metadata_address() -> None:
    cache, _, client = _cache(jwks_uri=None, jwks_allowed_hosts=["169.254.169.254"])
    with respx.mock(assert_all_called=False) as mock:
        mock.get(DISCOVERY).respond(json={"issuer": ISSUER, "jwks_uri": "https://169.254.169.254/jwks"})
        meta = mock.get("https://169.254.169.254/jwks").respond(json=KeyPair("k1").jwks())
        await _fails_unavailable(cache)
        assert not meta.called
    await client.aclose()


@pytest.mark.parametrize(
    "uri",
    [
        "https://169.254.169.254/keys",
        "https://[fe80::1]/keys",
        "https://[fd00:ec2::254]/keys",
        "https://127.0.0.1/keys",
        "https://[::1]/keys",
        "https://[::ffff:169.254.169.254]/keys",
        "https://0.0.0.0/keys",
        "https://localhost/keys",
        "https://metadata.localhost/keys",
        "http://localhost:9000/keys",  # insecure-local needs the explicit opt-in
    ],
)
async def test_configured_endpoint_that_is_loopback_or_metadata_is_refused(uri: str) -> None:
    cache, _, client = _cache(jwks_uri=uri)
    with respx.mock(assert_all_called=False) as mock:
        hit = mock.route().respond(json=KeyPair("k1").jwks())
        await _fails_unavailable(cache)
        assert not hit.called
    await client.aclose()


async def test_hostname_resolving_to_forbidden_address_is_refused() -> None:
    async def rebinds(host: str) -> list[str]:
        return ["93.184.216.34", "169.254.169.254"]

    cache, _, client = _cache(resolver=rebinds, jwks_uri="https://keys.idp.test/jwks")
    with respx.mock(assert_all_called=False) as mock:
        hit = mock.route().respond(json=KeyPair("k1").jwks())
        await _fails_unavailable(cache)
        assert not hit.called
    await client.aclose()


async def test_hostname_resolving_to_public_or_private_address_is_fetched() -> None:
    async def resolves(host: str) -> list[str]:
        return ["10.1.2.3"]  # internal IdPs on RFC1918 space are normal

    cache, _, client = _cache(resolver=resolves, jwks_uri="https://keys.idp.test/jwks")
    with respx.mock:
        respx.get("https://keys.idp.test/jwks").respond(json=KeyPair("k1").jwks())
        assert await cache.get_key("k1", "RS256") is not None
    await client.aclose()


async def test_insecure_local_opt_in_allows_http_loopback_idp() -> None:
    cache, _, client = _cache(jwks_uri="http://127.0.0.1:9000/jwks", jwks_allow_insecure_local=True)
    with respx.mock:
        respx.get("http://127.0.0.1:9000/jwks").respond(json=KeyPair("k1").jwks())
        assert await cache.get_key("k1", "RS256") is not None
    # http to a non-loopback host stays refused even with the opt-in
    cache2, _, client2 = _cache(jwks_uri="http://idp.test/jwks", jwks_allow_insecure_local=True)
    with respx.mock(assert_all_called=False) as mock:
        hit = mock.route().respond(json=KeyPair("k1").jwks())
        await _fails_unavailable(cache2)
        assert not hit.called
    await client.aclose()
    await client2.aclose()


def test_config_defaults_for_new_idp_fields() -> None:
    idp = make_idp()
    assert idp.jwks_allowed_hosts == [] and idp.jwks_allow_insecure_local is False
