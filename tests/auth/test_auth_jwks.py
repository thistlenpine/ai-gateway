from __future__ import annotations

import httpx
import pytest
import respx
from helpers import ISSUER, JWKS_URL, FakeClock, RecTelemetry, make_idp

from ai_gateway.auth import JWKSCache
from ai_gateway.errors import AuthenticationError, AuthFailure
from support import KeyPair


def _setup(**idp_kw: object) -> tuple[JWKSCache, FakeClock, RecTelemetry, httpx.AsyncClient]:
    clock = FakeClock(0.0)
    tel = RecTelemetry()
    client = httpx.AsyncClient()
    return JWKSCache(make_idp(**idp_kw), client, tel, clock=clock), clock, tel, client


async def test_fetch_then_ttl_cache() -> None:
    kp = KeyPair("k1")
    cache, clock, tel, client = _setup(jwks_cache_ttl_s=100)
    with respx.mock:
        route = respx.get(JWKS_URL).respond(json=kp.jwks())
        assert not cache.ready
        assert await cache.get_key("k1", "RS256") is not None
        await cache.get_key("k1", "RS256")
        assert route.call_count == 1
        assert cache.ready
        clock.advance(101)
        await cache.get_key("k1", "RS256")
        assert route.call_count == 2
    assert tel.jwks[0] == "refreshed" and "hit" in tel.jwks
    assert "gateway.authn.jwks_refresh" in tel.spans
    await client.aclose()


async def test_unknown_kid_rotation_refetch_once_and_rate_limited() -> None:
    old, new = KeyPair("old"), KeyPair("new")
    cache, clock, _, client = _setup(jwks_min_refresh_interval_s=30)
    with respx.mock:
        route = respx.get(JWKS_URL).mock(
            side_effect=[
                httpx.Response(200, json=old.jwks()),
                httpx.Response(200, json={"keys": [old.jwk(), new.jwk()]}),
            ]
        )
        await cache.get_key("old", "RS256")
        clock.advance(31)
        assert await cache.get_key("new", "RS256") is not None  # rotation picked up
        assert route.call_count == 2
        # flood of bogus kids: no refetch inside the cooldown
        for i in range(20):
            with pytest.raises(AuthenticationError) as ei:
                await cache.get_key(f"bogus{i}", "RS256")
            assert ei.value.failure is AuthFailure.UNKNOWN_KID
        assert route.call_count == 2
        # after cooldown exactly one more fetch is allowed
        clock.advance(31)
        respx.get(JWKS_URL).respond(json={"keys": [old.jwk(), new.jwk()]})
        with pytest.raises(AuthenticationError):
            await cache.get_key("bogus", "RS256")
    await client.aclose()


async def test_unknown_kid_right_after_fetch_does_not_double_fetch() -> None:
    kp = KeyPair("k1")
    cache, _, _, client = _setup()
    with respx.mock:
        route = respx.get(JWKS_URL).respond(json=kp.jwks())
        with pytest.raises(AuthenticationError) as ei:
            await cache.get_key("nope", "RS256")
        assert ei.value.failure is AuthFailure.UNKNOWN_KID
        assert route.call_count == 1
    await client.aclose()


async def test_single_flight_concurrent() -> None:
    import asyncio

    kp = KeyPair("k1")
    cache, _, _, client = _setup()
    with respx.mock:
        route = respx.get(JWKS_URL).respond(json=kp.jwks())
        await asyncio.gather(*(cache.get_key("k1", "RS256") for _ in range(25)))
        assert route.call_count == 1
    await client.aclose()


async def test_stale_if_error_and_expiry_of_stale() -> None:
    kp = KeyPair("k1")
    cache, clock, tel, client = _setup(
        jwks_cache_ttl_s=100, jwks_stale_if_error_s=1000, jwks_min_refresh_interval_s=30
    )
    with respx.mock:
        route = respx.get(JWKS_URL).respond(json=kp.jwks())
        await cache.get_key("k1", "RS256")
        route.respond(503)
        clock.advance(200)  # TTL expired, IdP down
        assert await cache.get_key("k1", "RS256") is not None  # stale served
        assert "failed" in tel.jwks and "stale" in tel.jwks
        assert cache.ready
        calls = route.call_count
        for _ in range(10):  # outage must not cause a refetch storm
            await cache.get_key("k1", "RS256")
        assert route.call_count == calls
        clock.advance(2000)  # beyond stale window
        assert not cache.ready
        with pytest.raises(AuthenticationError) as ei:
            await cache.get_key("k1", "RS256")
        assert ei.value.failure is AuthFailure.JWKS_UNAVAILABLE
        assert ei.value.status_code == 503
    await client.aclose()


async def test_outage_without_cache_is_unavailable_and_rate_limited() -> None:
    cache, clock, _, client = _setup(jwks_min_refresh_interval_s=30)
    with respx.mock:
        route = respx.get(JWKS_URL).mock(side_effect=httpx.ConnectError("boom"))
        for _ in range(5):
            with pytest.raises(AuthenticationError) as ei:
                await cache.get_key("k1", "RS256")
            assert ei.value.failure is AuthFailure.JWKS_UNAVAILABLE
        assert route.call_count == 1
        clock.advance(31)
        route.mock(side_effect=None, return_value=httpx.Response(200, json=KeyPair("k1").jwks()))
        assert await cache.get_key("k1", "RS256") is not None
    await client.aclose()


async def test_garbage_jwks_does_not_replace_good_cache() -> None:
    kp = KeyPair("k1")
    cache, clock, _, client = _setup(jwks_cache_ttl_s=10, jwks_min_refresh_interval_s=1)
    with respx.mock:
        route = respx.get(JWKS_URL).respond(json=kp.jwks())
        await cache.get_key("k1", "RS256")
        clock.advance(11)
        route.respond(200, json={"keys": []})
        assert await cache.get_key("k1", "RS256") is not None
        route.respond(200, content=b"not json")
        clock.advance(2)
        assert await cache.get_key("k1", "RS256") is not None
    await client.aclose()


async def test_key_filtering() -> None:
    good = KeyPair("good")
    enc = good.jwk() | {"kid": "enc", "use": "enc"}
    wrong_alg = good.jwk() | {"kid": "wrong", "alg": "RS512"}
    priv = good.jwk() | {"kid": "priv", "d": "AQAB"}
    sym = {"kty": "oct", "kid": "sym", "k": "c2VjcmV0"}
    cache, _, _, client = _setup()
    with respx.mock:
        respx.get(JWKS_URL).respond(json={"keys": [good.jwk(), enc, wrong_alg, priv, sym, "junk"]})
        await cache.get_key("good", "RS256")
        for kid in ("enc", "wrong", "priv", "sym"):
            with pytest.raises(AuthenticationError) as ei:
                await cache.get_key(kid, "RS256")
            assert ei.value.failure is AuthFailure.UNKNOWN_KID
            cache._last_attempt = None  # bypass cooldown
    await client.aclose()


async def test_kid_none_only_with_single_candidate() -> None:
    a, b = KeyPair("a"), KeyPair("b")
    cache, _, _, client = _setup()
    with respx.mock:
        respx.get(JWKS_URL).respond(json=a.jwks())
        assert await cache.get_key(None, "RS256") is not None
    cache2, _, _, client2 = _setup()
    with respx.mock:
        respx.get(JWKS_URL).respond(json={"keys": [a.jwk(), b.jwk()]})
        with pytest.raises(AuthenticationError) as ei:
            await cache2.get_key(None, "RS256")
        assert ei.value.failure is AuthFailure.UNKNOWN_KID
    await client.aclose()
    await client2.aclose()


async def test_alg_not_in_allow_list_and_key_type_mismatch() -> None:
    kp = KeyPair("k1")
    nokey_alg = kp.jwk()
    del nokey_alg["alg"]
    cache, _, _, client = _setup(algorithms=["RS256", "ES256"])
    with respx.mock:
        respx.get(JWKS_URL).respond(json={"keys": [nokey_alg]})
        with pytest.raises(AuthenticationError) as ei:
            await cache.get_key("k1", "PS256")
        assert ei.value.failure is AuthFailure.ALG_NOT_ALLOWED
        with pytest.raises(AuthenticationError) as ei:
            await cache.get_key("k1", "ES256")  # RSA key presented for an EC alg
        assert ei.value.failure is AuthFailure.ALG_NOT_ALLOWED
        assert await cache.get_key("k1", "RS256") is not None
    await client.aclose()


async def test_oidc_discovery() -> None:
    kp = KeyPair("k1")
    cache, _, _, client = _setup(jwks_uri=None)
    with respx.mock:
        disc = respx.get(f"{ISSUER}/.well-known/openid-configuration").respond(
            json={"issuer": ISSUER, "jwks_uri": "https://idp.test/keys"}
        )
        keys = respx.get("https://idp.test/keys").respond(json=kp.jwks())
        await cache.get_key("k1", "RS256")
        assert disc.call_count == 1 and keys.call_count == 1
    await client.aclose()


async def test_discovery_issuer_mismatch_rejected() -> None:
    cache, _, _, client = _setup(jwks_uri=None)
    with respx.mock:
        respx.get(f"{ISSUER}/.well-known/openid-configuration").respond(
            json={"issuer": "https://evil.test", "jwks_uri": "https://evil.test/keys"}
        )
        evil = respx.get("https://evil.test/keys").respond(json=KeyPair("k1").jwks())
        with pytest.raises(AuthenticationError) as ei:
            await cache.get_key("k1", "RS256")
        assert ei.value.failure is AuthFailure.JWKS_UNAVAILABLE
        assert not evil.called
    await client.aclose()


async def test_insecure_jwks_uri_rejected_but_localhost_allowed() -> None:
    kp = KeyPair("k1")
    cache, _, _, client = _setup(jwks_uri="http://idp.test/jwks")
    with respx.mock:
        r = respx.get("http://idp.test/jwks").respond(json=kp.jwks())
        with pytest.raises(AuthenticationError):
            await cache.get_key("k1", "RS256")
        assert not r.called
    cache2, _, _, client2 = _setup(jwks_uri="http://localhost:9000/jwks", jwks_allow_insecure_local=True)
    with respx.mock:
        respx.get("http://localhost:9000/jwks").respond(json=kp.jwks())
        assert await cache2.get_key("k1", "RS256") is not None
    await client.aclose()
    await client2.aclose()


async def test_redirects_not_followed() -> None:
    cache, _, _, client = _setup()
    with respx.mock:
        respx.get(JWKS_URL).respond(302, headers={"location": "https://evil.test/k"})
        evil = respx.get("https://evil.test/k").respond(json=KeyPair("k1").jwks())
        with pytest.raises(AuthenticationError):
            await cache.get_key("k1", "RS256")
        assert not evil.called
    await client.aclose()


async def test_warm_swallows_errors_and_sets_ready() -> None:
    cache, _, _, client = _setup()
    with respx.mock:
        respx.get(JWKS_URL).mock(side_effect=httpx.ConnectError("down"))
        await cache.warm()
        assert not cache.ready
    with respx.mock:
        respx.get(JWKS_URL).respond(json=KeyPair("k1").jwks())
        await cache.warm()
        assert cache.ready
    await client.aclose()
