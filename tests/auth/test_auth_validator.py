from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, cast

import httpx
import jwt
import pytest
import respx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from helpers import AUDIENCE, ISSUER, JWKS_URL, FakeClock, RecTelemetry, make_idp

from ai_gateway.auth import TokenValidator
from ai_gateway.errors import AuthenticationError, AuthFailure
from support import KeyPair

KP = KeyPair("k1")


def b64(d: bytes) -> str:
    return base64.urlsafe_b64encode(d).rstrip(b"=").decode()


def raw_token(header: dict[str, Any], payload: dict[str, Any], key: bytes | None = None) -> str:
    h = b64(json.dumps(header).encode())
    p = b64(json.dumps(payload).encode())
    sig = b""
    if key is not None:
        sig = hmac.new(key, f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{b64(sig)}"


def claims(**over: Any) -> dict[str, Any]:
    now = int(time.time())
    c: dict[str, Any] = {"iss": ISSUER, "aud": AUDIENCE, "sub": "client-a", "exp": now + 300, "nbf": now}
    c.update(over)
    return {k: v for k, v in c.items() if v is not None}


@pytest.fixture
def tel() -> RecTelemetry:
    return RecTelemetry()


@pytest.fixture
async def http():  # type: ignore[no-untyped-def]
    async with httpx.AsyncClient() as c:
        yield c


@pytest.fixture
def jwks_route():  # type: ignore[no-untyped-def]
    with respx.mock:
        yield respx.get(JWKS_URL).respond(json=KP.jwks())


def validator(http: httpx.AsyncClient, tel: RecTelemetry, **idp_kw: Any) -> TokenValidator:
    return TokenValidator([make_idp(**idp_kw)], http, tel)


async def fails(v: TokenValidator, token: str) -> AuthFailure:
    with pytest.raises(AuthenticationError) as ei:
        await v.validate(token)
    return ei.value.failure


async def test_happy_path(http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any) -> None:
    v = validator(http, tel)
    tok = KP.mint(scopes="read write", jti="abc", email="x@y.z")
    p = await v.validate(tok)
    assert p.client_id == "client-a" and p.subject == "client-a"
    assert p.scopes == {"read", "write"}
    assert p.issuer == ISSUER and p.idp == "main" and p.audience == (AUDIENCE,)
    assert p.token_id == "abc" and p.is_client_credentials
    assert "email" not in p.claims and tok not in repr(p)
    assert tel.failures == []


async def test_expired(http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any) -> None:
    v = validator(http, tel)
    assert await fails(v, KP.mint(ttl=-3600)) is AuthFailure.EXPIRED
    assert tel.failures == [("expired", ISSUER)]


async def test_leeway(http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any) -> None:
    v = validator(http, tel, leeway_s=30)
    await v.validate(KP.mint(ttl=-10))  # within leeway
    assert await fails(v, KP.mint(ttl=-31)) is AuthFailure.EXPIRED
    nbf = int(time.time()) + 20
    await v.validate(KP.mint(nbf=nbf))
    assert await fails(v, KP.mint(nbf=int(time.time()) + 60)) is AuthFailure.NOT_YET_VALID
    assert await fails(v, KP.mint(iat=int(time.time()) + 60)) is AuthFailure.NOT_YET_VALID


async def test_injected_clock(http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any) -> None:
    clock = FakeClock(time.time())
    v = TokenValidator([make_idp()], http, tel, clock=clock)
    tok = KP.mint(ttl=300)
    await v.validate(tok)
    clock.advance(1000)
    assert await fails(v, tok) is AuthFailure.EXPIRED


async def test_wrong_audience_and_audience_list(
    http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any
) -> None:
    v = validator(http, tel, audiences=[AUDIENCE, "other"])
    assert await fails(v, KP.mint(audience="nope")) is AuthFailure.BAD_AUDIENCE
    p = await v.validate(KP.mint(audience=cast(Any, ["x", "other"])))
    assert p.audience == ("x", "other")


async def test_wrong_issuer(http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any) -> None:
    v = validator(http, tel)
    assert await fails(v, KP.mint(issuer="https://evil.test")) is AuthFailure.UNKNOWN_ISSUER
    assert await fails(v, KP.mint(issuer=ISSUER + "/")) is AuthFailure.UNKNOWN_ISSUER
    # unknown (attacker-controlled) issuer is never used as a metric attribute
    assert tel.failures[0] == ("unknown_issuer", None)


async def test_missing_claims(http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any) -> None:
    v = validator(http, tel)

    def tok(**kw: Any) -> str:
        return jwt.encode(claims(**kw), KP.private, algorithm="RS256", headers={"kid": "k1"})

    assert await fails(v, tok(exp=None)) is AuthFailure.MISSING_CLAIM
    assert await fails(v, tok(aud=None)) is AuthFailure.MISSING_CLAIM
    assert await fails(v, tok(iss=None)) is AuthFailure.MISSING_CLAIM
    assert await fails(v, tok(exp="soon")) is AuthFailure.MALFORMED
    await v.validate(tok(nbf=None))  # design: nbf verified only if present


async def test_alg_none(http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any) -> None:
    v = validator(http, tel)
    tok = jwt.encode(claims(), cast(Any, None), algorithm="none", headers={"kid": "k1"})
    assert await fails(v, tok) is AuthFailure.ALG_NOT_ALLOWED
    for alg in ("None", "NONE", "nOnE"):
        t = raw_token({"alg": alg, "kid": "k1"}, claims())
        assert await fails(v, t) is AuthFailure.ALG_NOT_ALLOWED
    assert not jwks_route.called  # rejected before any key lookup


async def test_hs256_public_key_confusion(
    http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any
) -> None:
    v = validator(http, tel)
    pem = KP.private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    for secret in (pem, json.dumps(KP.jwk()).encode()):
        t = raw_token({"alg": "HS256", "kid": "k1", "typ": "JWT"}, claims(), secret)
        assert await fails(v, t) is AuthFailure.ALG_NOT_ALLOWED
    assert not jwks_route.called


async def test_key_type_must_match_alg(http: httpx.AsyncClient, tel: RecTelemetry) -> None:
    """RSA key published without `alg`; token claims ES256 with the same kid."""
    jwk = KP.jwk()
    del jwk["alg"]
    ec_key = ec.generate_private_key(ec.SECP256R1())
    t = jwt.encode(claims(), ec_key, algorithm="ES256", headers={"kid": "k1"})
    with respx.mock:
        respx.get(JWKS_URL).respond(json={"keys": [jwk]})
        v = validator(http, tel, algorithms=["RS256", "ES256"])
        assert await fails(v, t) is AuthFailure.ALG_NOT_ALLOWED


async def test_es256_ok(http: httpx.AsyncClient, tel: RecTelemetry) -> None:
    from jwt.algorithms import ECAlgorithm

    ec_key = ec.generate_private_key(ec.SECP256R1())
    jwk = json.loads(ECAlgorithm.to_jwk(ec_key.public_key())) | {"kid": "e1", "use": "sig"}
    t = jwt.encode(claims(client_id="svc"), ec_key, algorithm="ES256", headers={"kid": "e1"})
    with respx.mock:
        respx.get(JWKS_URL).respond(json={"keys": [jwk]})
        p = await validator(http, tel).validate(t)
    assert p.client_id == "svc"


async def test_alg_not_in_idp_allow_list(http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any) -> None:
    v = validator(http, tel, algorithms=["ES256"])
    assert await fails(v, KP.mint()) is AuthFailure.ALG_NOT_ALLOWED
    assert not jwks_route.called


async def test_bad_signature(http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any) -> None:
    other = KeyPair("k1")  # same kid, different key
    v = validator(http, tel)
    assert await fails(v, other.mint()) is AuthFailure.BAD_SIGNATURE
    good = KP.mint()
    h, _p, s = good.split(".")
    tampered = raw_token(json.loads(base64.urlsafe_b64decode(h + "==")), claims(sub="admin"))
    assert await fails(v, tampered.rsplit(".", 1)[0] + "." + s) is AuthFailure.BAD_SIGNATURE


@pytest.mark.parametrize(
    "token",
    ["", "garbage", "a.b", "a.b.c", "a.b.c.d", "eyJ.eyJ.sig", "é.é.é", "x" * 9000],
)
async def test_malformed_tokens(
    http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any, token: str
) -> None:
    assert await fails(validator(http, tel), token) is AuthFailure.MALFORMED


async def test_max_token_size(http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any) -> None:
    v = validator(http, tel)
    big = KP.mint(blob="a" * 10_000)
    assert await fails(v, big) is AuthFailure.MALFORMED
    assert not jwks_route.called


async def test_crit_header_rejected(http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any) -> None:
    t = jwt.encode(claims(), KP.private, algorithm="RS256", headers={"kid": "k1", "crit": ["exp"]})
    assert await fails(validator(http, tel), t) is AuthFailure.MALFORMED


async def test_jku_x5u_jwk_headers_ignored(http: httpx.AsyncClient, tel: RecTelemetry) -> None:
    attacker = KeyPair("k1")
    t = jwt.encode(
        claims(),
        attacker.private,
        algorithm="RS256",
        headers={"kid": "k1", "jku": "https://evil.test/jwks", "jwk": attacker.jwk()},
    )
    with respx.mock:
        respx.get(JWKS_URL).respond(json=KP.jwks())
        evil = respx.get("https://evil.test/jwks").respond(json=attacker.jwks())
        assert await fails(validator(http, tel), t) is AuthFailure.BAD_SIGNATURE
        assert not evil.called


async def test_bad_kid_header(http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any) -> None:
    t = raw_token({"alg": "RS256", "kid": 5}, claims())
    assert await fails(validator(http, tel), t) is AuthFailure.MALFORMED
    t = raw_token({"alg": "RS256", "kid": "k" * 300}, claims())
    assert await fails(validator(http, tel), t) is AuthFailure.MALFORMED


async def test_unknown_kid_rotation_end_to_end(http: httpx.AsyncClient, tel: RecTelemetry) -> None:
    new = KeyPair("k2")
    jclock = FakeClock(0.0)
    with respx.mock:
        route = respx.get(JWKS_URL).mock(
            side_effect=[
                httpx.Response(200, json=KP.jwks()),
                httpx.Response(200, json={"keys": [KP.jwk(), new.jwk()]}),
            ]
        )
        v = TokenValidator([make_idp(jwks_min_refresh_interval_s=30)], http, tel, jwks_clock=jclock)
        await v.validate(KP.mint())
        jclock.advance(31)
        p = await v.validate(new.mint())  # rotated key picked up without restart
        assert p.client_id == "client-a"
        assert route.call_count == 2
        bogus = KeyPair("zzz")
        for _ in range(10):
            assert await fails(v, bogus.mint()) is AuthFailure.UNKNOWN_KID
        assert route.call_count == 2  # no refetch storm


async def test_jwks_outage_stale_if_error(http: httpx.AsyncClient, tel: RecTelemetry) -> None:
    jclock = FakeClock(0.0)
    with respx.mock:
        route = respx.get(JWKS_URL).respond(json=KP.jwks())
        v = TokenValidator(
            [make_idp(jwks_cache_ttl_s=60, jwks_stale_if_error_s=600)], http, tel, jwks_clock=jclock
        )
        await v.validate(KP.mint())
        route.respond(500)
        jclock.advance(120)
        await v.validate(KP.mint())  # served from stale cache
        assert v.ready
        jclock.advance(1000)
        assert await fails(v, KP.mint()) is AuthFailure.JWKS_UNAVAILABLE
        assert not v.ready
        assert tel.failures[-1] == ("jwks_unavailable", ISSUER)


async def test_jwks_outage_cold_start(http: httpx.AsyncClient, tel: RecTelemetry) -> None:
    with respx.mock:
        respx.get(JWKS_URL).mock(side_effect=httpx.ConnectTimeout("t"))
        v = validator(http, tel)
        with pytest.raises(AuthenticationError) as ei:
            await v.validate(KP.mint())
        assert ei.value.status_code == 503 and ei.value.code == "idp_unavailable"


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"scope": "a b  c"}, {"a", "b", "c"}),
        ({"scope": ""}, set()),
        ({"scp": ["x", "y"]}, {"x", "y"}),
        ({"scope": "a b", "scp": ["c"]}, {"a", "b", "c"}),
        ({"scp": "solo"}, {"solo"}),
    ],
)
async def test_scope_parsing(
    http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any, extra: dict[str, Any], expected: set[str]
) -> None:
    c = claims(**extra)
    t = jwt.encode(c, KP.private, algorithm="RS256", headers={"kid": "k1"})
    p = await validator(http, tel).validate(t)
    assert p.scopes == frozenset(expected)


async def test_scope_wrong_type(http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any) -> None:
    t = jwt.encode(claims(scope={"a": 1}), KP.private, algorithm="RS256", headers={"kid": "k1"})
    assert await fails(validator(http, tel), t) is AuthFailure.MALFORMED


async def test_client_id_claim_order(http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any) -> None:
    def enc(**kw: Any) -> str:
        return jwt.encode(claims(**kw), KP.private, algorithm="RS256", headers={"kid": "k1"})

    v = validator(http, tel)
    p = await v.validate(enc(client_id="cid", azp="az", sub="user-1"))
    assert (p.client_id, p.subject, p.is_client_credentials) == ("cid", "user-1", False)
    p = await v.validate(enc(azp="az", sub="user-1"))
    assert p.client_id == "az"
    p = await v.validate(enc(sub="user-1"))
    assert p.client_id == "user-1" and p.is_client_credentials
    p = await validator(http, tel, client_id_claims=["azp"]).validate(enc(azp="app", sub="u"))
    assert p.client_id == "app"
    assert (
        await fails(validator(http, tel, client_id_claims=["azp"]), enc(sub="u")) is AuthFailure.MISSING_CLAIM
    )
    p = await v.validate(enc(client_id="cid", sub=None))
    assert p.subject == "cid" and p.is_client_credentials


async def test_max_token_lifetime(http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any) -> None:
    v = validator(http, tel, max_token_lifetime_s=600)
    await v.validate(KP.mint(ttl=500))
    assert await fails(v, KP.mint(ttl=3600)) is AuthFailure.MALFORMED
    # No iat: fall back to capping exp - now (not MISSING_CLAIM)
    t = jwt.encode(claims(), KP.private, algorithm="RS256", headers={"kid": "k1"})
    await v.validate(t)
    far = jwt.encode(
        claims(exp=int(time.time()) + 3600), KP.private, algorithm="RS256", headers={"kid": "k1"}
    )
    assert await fails(v, far) is AuthFailure.MALFORMED


async def test_default_lifetime_cap_is_one_day_and_zero_or_null_opt_out(
    http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any
) -> None:
    assert make_idp().max_token_lifetime_s == 86_400
    assert make_idp(max_token_lifetime_s=0).max_token_lifetime_s is None
    assert make_idp(max_token_lifetime_s=None).max_token_lifetime_s is None
    week = 7 * 86_400
    assert await fails(validator(http, tel), KP.mint(ttl=week)) is AuthFailure.MALFORMED
    for opt_out in (0, None):
        await validator(http, tel, max_token_lifetime_s=opt_out).validate(KP.mint(ttl=week))


async def test_multiple_idps_selected_by_issuer(http: httpx.AsyncClient, tel: RecTelemetry) -> None:
    other = KeyPair("o1")
    idps = [
        make_idp(),
        make_idp(
            name="second", issuer="https://idp2.test", jwks_uri="https://idp2.test/jwks", audiences=["aud2"]
        ),
    ]
    with respx.mock:
        respx.get(JWKS_URL).respond(json=KP.jwks())
        r2 = respx.get("https://idp2.test/jwks").respond(json=other.jwks())
        v = TokenValidator(idps, http, tel)
        p1 = await v.validate(KP.mint())
        p2 = await v.validate(other.mint(issuer="https://idp2.test", audience="aud2"))
        assert (p1.idp, p2.idp) == ("main", "second")
        # IdP 2's key must not validate an IdP 1 token and vice versa
        assert await fails(v, other.mint(kid_dummy=1)) is AuthFailure.UNKNOWN_KID
        assert r2.called
        assert await fails(v, other.mint(issuer="https://idp2.test")) is AuthFailure.BAD_AUDIENCE


async def test_warm_and_ready(http: httpx.AsyncClient, tel: RecTelemetry) -> None:
    idps = [make_idp(), make_idp(name="b", issuer="https://b.test", jwks_uri="https://b.test/jwks")]
    with respx.mock:
        respx.get(JWKS_URL).respond(json=KP.jwks())
        respx.get("https://b.test/jwks").respond(503)
        v = TokenValidator(idps, http, tel)
        assert not v.ready
        await v.warm()  # must not raise
        assert not v.ready  # b is down
        respx.get("https://b.test/jwks").respond(json=KeyPair("b").jwks())
        await v.warm()
        assert v.ready


def test_duplicate_issuer_rejected(tel: RecTelemetry) -> None:
    with pytest.raises(ValueError):
        TokenValidator([make_idp(), make_idp(name="dup")], httpx.AsyncClient(), tel)


async def test_no_token_in_errors_or_logs(
    http: httpx.AsyncClient, tel: RecTelemetry, jwks_route: Any, caplog: pytest.LogCaptureFixture
) -> None:
    tok = KP.mint(ttl=-3600)
    with caplog.at_level("DEBUG"), pytest.raises(AuthenticationError) as ei:
        await validator(http, tel).validate(tok)
    sig = tok.split(".")[2]
    assert tok not in str(ei.value) and sig not in str(ei.value)
    assert sig not in caplog.text and tok not in caplog.text
    assert ei.value.__cause__ is None
