"""JWT / IdP attacks against TokenValidator and JWKSCache. Passing = gateway is secure."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import jwt
import pytest
import respx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from sec_helpers import (
    ISSUER,
    ISSUER_B,
    JWKS_B,
    b64,
    claims,
    forge,
    forge_raw,
    only_failure,
    rsa_key,
    validator,
)

from ai_gateway.config import IdPConfig
from ai_gateway.errors import AuthFailure
from support import FakeClock, KeyPair

JWKS_A = "https://idp.test/jwks.json"
KP = KeyPair("k1")


def idp(**kw: Any) -> IdPConfig:
    base: dict[str, Any] = {"name": "a", "issuer": ISSUER, "jwks_uri": JWKS_A, "audiences": ["ai-gateway"]}
    base.update(kw)
    return IdPConfig(**base)


@pytest.fixture
def router() -> Any:
    with respx.mock(assert_all_called=False) as r:
        r.jwks_a = r.get(JWKS_A).respond(json=KP.jwks())  # type: ignore[attr-defined]
        yield r


@pytest.fixture
async def http(router: Any) -> Any:
    async with httpx.AsyncClient() as c:
        yield c


def signed(payload: dict[str, Any] | None = None, kp: KeyPair = KP, **headers: Any) -> str:
    return jwt.encode(payload or claims(), kp.private, algorithm="RS256", headers={"kid": kp.kid, **headers})


# ------------------------------------------------------------------ alg confusion


@pytest.mark.parametrize("alg", ["none", "None", "NONE", "nOnE"])
async def test_alg_none_variants_rejected(http: Any, alg: str) -> None:
    v = validator([idp()], http)
    tok = forge({"alg": alg, "typ": "JWT", "kid": "k1"}, claims())
    with only_failure(AuthFailure.ALG_NOT_ALLOWED):
        await v.validate(tok)


@pytest.mark.parametrize("alg", ["HS256", "HS384", "HS512", "hs256"])
async def test_hs_with_public_key_as_secret_rejected(http: Any, alg: str) -> None:
    """Classic RS->HS confusion: sign HMAC with the public key bytes (PEM and raw JWK 'n')."""
    v = validator([idp()], http)
    pem = KP.private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    for secret in (pem, json.dumps(KP.jwk()).encode(), KP.jwk()["n"].encode()):
        tok = forge({"alg": alg, "kid": "k1"}, claims(), secret=secret)
        with only_failure():
            await v.validate(tok)


async def test_jwk_oct_key_in_jwks_ignored_so_hs256_cannot_verify(router: Any, http: Any) -> None:
    secret = b"attacker-chosen-secret-0123456789"
    jwks = {"keys": [{"kty": "oct", "kid": "k1", "k": b64(secret), "alg": "HS256", "use": "sig"}]}
    router.jwks_a.respond(json=jwks)
    v = validator([idp()], http)
    tok = forge({"alg": "HS256", "kid": "k1"}, claims(), secret=secret)
    with only_failure():
        await v.validate(tok)


async def test_alg_not_in_idp_allowlist_rejected(http: Any) -> None:
    v = validator([idp(algorithms=["RS256"])], http)
    ps = jwt.encode(claims(), KP.private, algorithm="PS256", headers={"kid": "k1"})
    with only_failure(AuthFailure.ALG_NOT_ALLOWED):
        await v.validate(ps)


async def test_es256_header_with_rsa_key_rejected(http: Any) -> None:
    """Header says ES256 (allowed) but the key under that kid is RSA: key family must match."""
    v = validator([idp(algorithms=["RS256", "ES256"])], http)
    ec_key = ec.generate_private_key(ec.SECP256R1())
    tok = jwt.encode(claims(), ec_key, algorithm="ES256", headers={"kid": "k1"})
    with only_failure():
        await v.validate(tok)


async def test_ec_curve_confusion_rejected(router: Any, http: Any) -> None:
    """ES256 header with a P-384 key published under the kid must not verify."""
    k384 = ec.generate_private_key(ec.SECP384R1())
    from jwt.algorithms import ECAlgorithm

    jwk = json.loads(ECAlgorithm.to_jwk(k384.public_key()))
    jwk.update(kid="k1", use="sig")
    router.jwks_a.respond(json={"keys": [jwk]})
    v = validator([idp(algorithms=["ES256"])], http)
    tok = jwt.encode(claims(), k384, algorithm="ES384", headers={"kid": "k1"})
    with only_failure():
        await v.validate(tok)


async def test_small_rsa_key_in_jwks_rejected(router: Any, http: Any) -> None:
    from cryptography.hazmat.primitives.asymmetric import rsa as _rsa
    from jwt.algorithms import RSAAlgorithm

    weak = _rsa.generate_private_key(public_exponent=65537, key_size=1024)
    jwk = json.loads(RSAAlgorithm.to_jwk(weak.public_key()))
    jwk.update(kid="k1", use="sig", alg="RS256")
    router.jwks_a.respond(json={"keys": [jwk]})
    v = validator([idp()], http)
    with only_failure():
        await v.validate(jwt.encode(claims(), weak, algorithm="RS256", headers={"kid": "k1"}))


# ------------------------------------------------------------------ header tricks


@pytest.mark.parametrize(
    "kid",
    ["../../../etc/passwd", "k1' OR '1'='1", "k1\x00", "a" * 5000, "", 7, ["k1"], {"x": 1}, True],
)
async def test_kid_injection_is_just_unknown(http: Any, kid: Any) -> None:
    v = validator([idp()], http)
    tok = forge({"alg": "RS256", "kid": kid}, claims(), sig=b"x" * 256)
    with only_failure():
        await v.validate(tok)


async def test_jku_x5u_jwk_x5c_headers_never_followed(router: Any, http: Any) -> None:
    evil = router.get("https://evil.test/jwks.json").respond(json={"keys": []})
    evil_x5u = router.get("https://evil.test/cert.pem").respond(text="x")
    attacker = rsa_key()
    from jwt.algorithms import RSAAlgorithm

    attacker_jwk = json.loads(RSAAlgorithm.to_jwk(attacker.public_key()))
    attacker_jwk.update(kid="evil", alg="RS256")
    tok = jwt.encode(
        claims(),
        attacker,
        algorithm="RS256",
        headers={
            "kid": "evil",
            "jku": "https://evil.test/jwks.json",
            "x5u": "https://evil.test/cert.pem",
            "jwk": attacker_jwk,
        },
    )
    v = validator([idp()], http)
    with only_failure():
        await v.validate(tok)
    assert evil.call_count == 0 and evil_x5u.call_count == 0


async def test_attacker_key_with_legit_kid_rejected(http: Any) -> None:
    v = validator([idp()], http)
    forged = KeyPair("k1")  # same kid, attacker private key
    with only_failure(AuthFailure.BAD_SIGNATURE):
        await v.validate(signed(kp=forged))


async def test_crit_header_rejected(http: Any) -> None:
    v = validator([idp()], http)
    with only_failure():
        await v.validate(signed(crit=["exp"], exp=1))


@pytest.mark.parametrize("suffix", [".", ".extra", ""])
async def test_structurally_wrong_tokens(http: Any, suffix: str) -> None:
    v = validator([idp()], http)
    tok = signed() + suffix if suffix else signed().rsplit(".", 1)[0] + "."
    with only_failure():
        await v.validate(tok)


async def test_empty_signature_rejected(http: Any) -> None:
    v = validator([idp()], http)
    h, p, _ = signed().split(".")
    with only_failure():
        await v.validate(f"{h}.{p}.")


async def test_payload_tamper_rejected(http: Any) -> None:
    v = validator([idp()], http)
    h, _, s = signed().split(".")
    evil = b64(json.dumps(claims(scope="admin:everything")).encode())
    with only_failure(AuthFailure.BAD_SIGNATURE):
        await v.validate(f"{h}.{evil}.{s}")


@pytest.mark.parametrize("size", [8193, 20_000, 1_000_000])
async def test_oversized_token_rejected_cheaply(router: Any, http: Any, size: int) -> None:
    v = validator([idp()], http)
    before = router.calls.call_count
    with only_failure(AuthFailure.MALFORMED):
        await v.validate("a." + "b" * size + ".c")
    assert router.calls.call_count == before  # rejected before any JWKS traffic


async def test_non_ascii_token_rejected(http: Any) -> None:
    v = validator([idp()], http)
    with only_failure():
        await v.validate("é.é.é")


async def test_deeply_nested_header_json_is_a_clean_401(http: Any) -> None:
    """json.loads of '[[[[...' raises RecursionError (not a PyJWTError): must not become a 500."""
    v = validator([idp()], http)
    nested = "[" * 3000 + "]" * 3000
    tok = forge_raw(nested, "{}")
    with only_failure():
        await v.validate(tok)


async def test_deeply_nested_payload_json_is_a_clean_401(http: Any) -> None:
    v = validator([idp()], http)
    h = b64(json.dumps({"alg": "RS256", "kid": "k1"}).encode())
    tok = f"{h}.{b64(b'[' * 3000)}.{b64(b'sig')}"
    assert len(tok) < 8192
    with only_failure():
        await v.validate(tok)


# ------------------------------------------------------------------ claims


def _signed_raw_payload(payload_json: str) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    h = b64(json.dumps({"alg": "RS256", "kid": "k1"}).encode())
    body = b64(payload_json.encode())
    sig = KP.private.sign(f"{h}.{body}".encode(), padding.PKCS1v15(), hashes.SHA256())
    return f"{h}.{body}.{b64(sig)}"


@pytest.mark.parametrize("exp", ["NaN", "Infinity"])
async def test_non_finite_exp_does_not_mean_immortal_token(http: Any, exp: str) -> None:
    """json.loads accepts NaN/Infinity; `exp <= now` is False for NaN, so the token never expires.
    Needs a validly signed token, so this is defence in depth against a sloppy/compromised IdP."""
    v = validator([idp()], http)
    payload = json.dumps(claims(exp=0)).replace('"exp": 0', f'"exp": {exp}')
    with only_failure():
        await v.validate(_signed_raw_payload(payload))


async def test_negative_infinity_exp_rejected(http: Any) -> None:
    v = validator([idp()], http)
    payload = json.dumps(claims(exp=0)).replace('"exp": 0', '"exp": -Infinity')
    with only_failure():
        await v.validate(_signed_raw_payload(payload))


@pytest.mark.parametrize("claim", ["nbf", "iat"])
async def test_non_finite_nbf_iat_rejected(http: Any, claim: str) -> None:
    v = validator([idp()], http)
    payload = json.dumps(claims(**{claim: 0})).replace(f'"{claim}": 0', f'"{claim}": Infinity')
    with only_failure():
        await v.validate(_signed_raw_payload(payload))


@pytest.mark.parametrize("exp", [True, "9999999999", [1], {"a": 1}])
async def test_exp_wrong_type_rejected(http: Any, exp: Any) -> None:
    v = validator([idp()], http)
    with only_failure():
        await v.validate(signed(claims(exp=exp)))


async def test_leeway_boundaries_with_injected_clock(http: Any) -> None:
    clock = FakeClock(2_000_000.0)
    v = validator([idp(leeway_s=30)], http, clock=clock)
    ok = signed(claims(exp=2_000_000 - 29, iat=1_999_000, nbf=1_999_000))
    assert (await v.validate(ok)).client_id == "client-a"
    for exp in (2_000_000 - 30, 2_000_000 - 31):  # boundary is exclusive
        with only_failure(AuthFailure.EXPIRED):
            await v.validate(signed(claims(exp=exp, iat=1_999_000, nbf=1_999_000)))
    with only_failure(AuthFailure.NOT_YET_VALID):
        await v.validate(signed(claims(exp=2_000_900, nbf=2_000_031, iat=2_000_000)))


async def test_leeway_cannot_exceed_five_minutes() -> None:
    with pytest.raises(ValueError):
        idp(leeway_s=301)


async def test_max_token_lifetime_enforced_and_iat_required(http: Any) -> None:
    v = validator([idp(max_token_lifetime_s=600)], http)
    now = int(time.time())
    with only_failure():
        await v.validate(signed(claims(iat=now, exp=now + 86_400)))
    await v.validate(signed(claims(iat=None)))  # no iat: exp - now is capped instead
    with only_failure(AuthFailure.MALFORMED):
        await v.validate(signed(claims(iat=None, exp=now + 86_400)))


async def test_default_lifetime_bound_without_revocation(http: Any) -> None:
    v = validator([idp()], http)  # default cap: 24 h
    now = int(time.time())
    await v.validate(signed(claims(iat=now, exp=now + 80_000)))
    with only_failure(AuthFailure.MALFORMED):
        await v.validate(signed(claims(iat=now, exp=now + 30 * 86_400)))
    with only_failure(AuthFailure.MALFORMED):
        await v.validate(signed(claims(iat=None, exp=now + 30 * 86_400)))


async def test_empty_or_missing_client_id_rejected(http: Any) -> None:
    v = validator([idp()], http)
    with only_failure(AuthFailure.MISSING_CLAIM):
        await v.validate(signed(claims(client_id=None, sub=None)))
    with only_failure(AuthFailure.MISSING_CLAIM):
        await v.validate(signed(claims(client_id="", sub="")))


async def test_scope_claim_wrong_type_rejected(http: Any) -> None:
    v = validator([idp()], http)
    with only_failure(AuthFailure.MALFORMED):
        await v.validate(signed(claims(scope={"billing:read": True})))
    with only_failure(AuthFailure.MALFORMED):
        await v.validate(signed(claims(scope=["ok", 5])))


async def test_scopes_do_not_gain_wildcards_from_claims(http: Any) -> None:
    v = validator([idp()], http)
    p = await v.validate(signed(claims(scope="billing:* *")))
    assert p.scopes == frozenset({"billing:*", "*"})  # literal strings only; policy must not glob scopes


async def test_principal_claims_are_allow_listed(http: Any) -> None:
    v = validator([idp()], http)
    p = await v.validate(signed(claims(email="a@b.c", password="hunter2", jti="j1")))
    assert set(p.claims) <= {"azp", "client_id", "jti", "iat", "nbf", "exp"}


# ------------------------------------------------------------------ multi-IdP confusion


@pytest.fixture
def two_idps(router: Any) -> tuple[IdPConfig, IdPConfig, KeyPair]:
    kp_b = KeyPair("k1")  # deliberately the same kid as IdP A
    router.get(JWKS_B).respond(json=kp_b.jwks())
    a = idp(name="a", audiences=["aud-a"])
    b = idp(name="b", issuer=ISSUER_B, jwks_uri=JWKS_B, audiences=["aud-b"])
    return a, b, kp_b


async def test_token_signed_by_b_claiming_issuer_a_rejected(http: Any, two_idps: Any) -> None:
    a, b, kp_b = two_idps
    v = validator([a, b], http)
    tok = jwt.encode(claims(aud="aud-a"), kp_b.private, algorithm="RS256", headers={"kid": "k1"})
    with only_failure(AuthFailure.BAD_SIGNATURE):
        await v.validate(tok)


async def test_audience_of_a_not_valid_at_b_and_vice_versa(http: Any, two_idps: Any) -> None:
    a, b, kp_b = two_idps
    v = validator([a, b], http)
    tok_b_with_a_aud = jwt.encode(
        claims(iss=ISSUER_B, aud="aud-a"), kp_b.private, algorithm="RS256", headers={"kid": "k1"}
    )
    with only_failure(AuthFailure.BAD_AUDIENCE):
        await v.validate(tok_b_with_a_aud)
    ok = jwt.encode(claims(iss=ISSUER_B, aud="aud-b"), kp_b.private, algorithm="RS256", headers={"kid": "k1"})
    p = await v.validate(ok)
    assert p.idp == "b" and p.issuer == ISSUER_B


async def test_issuer_matching_is_exact(http: Any, two_idps: Any) -> None:
    a, b, _ = two_idps
    v = validator([a, b], http)
    for iss in (
        ISSUER + "/",
        ISSUER.upper(),
        ISSUER + "#x",
        " " + ISSUER,
        ISSUER + "\x00",
        "https://idp.test:443",
    ):
        with only_failure(AuthFailure.UNKNOWN_ISSUER):
            await v.validate(
                forge({"alg": "RS256", "kid": "k1"}, claims(iss=iss, aud="aud-a"), sig=b"x" * 256)
            )


async def test_unknown_issuer_never_triggers_fetch_or_metric_label(router: Any, http: Any) -> None:
    probe = router.get("https://attacker.test/.well-known/openid-configuration").respond(json={})
    v = validator([idp()], http)
    with only_failure(AuthFailure.UNKNOWN_ISSUER):
        await v.validate(forge({"alg": "RS256", "kid": "k1"}, claims(iss="https://attacker.test")))
    assert probe.call_count == 0
    assert v._telemetry.failures[-1] == ("unknown_issuer", None)  # type: ignore[attr-defined]


# ------------------------------------------------------------------ JWKS behaviour


async def test_unknown_kid_flood_costs_at_most_one_fetch_per_cooldown(router: Any, http: Any) -> None:
    clock = FakeClock(1000.0)
    v = validator([idp(jwks_min_refresh_interval_s=30)], http, jwks_clock=clock)
    await v.validate(signed())  # warm
    route = router.jwks_a
    start = route.call_count
    for i in range(200):
        tok = forge({"alg": "RS256", "kid": f"bogus-{i}"}, claims(), sig=b"x" * 256)
        with only_failure(AuthFailure.UNKNOWN_KID):
            await v.validate(tok)
    assert route.call_count - start <= 1
    clock.advance(31)
    with only_failure(AuthFailure.UNKNOWN_KID):
        await v.validate(forge({"alg": "RS256", "kid": "again"}, claims(), sig=b"x" * 256))
    assert route.call_count - start <= 2


async def test_rotated_out_key_stops_working_after_refresh(router: Any, http: Any) -> None:
    clock = FakeClock(1000.0)
    v = validator([idp(jwks_cache_ttl_s=60, jwks_min_refresh_interval_s=1)], http, jwks_clock=clock)
    old_tok = signed()
    await v.validate(old_tok)
    new = KeyPair("k2")
    router.jwks_a.respond(json=new.jwks())
    clock.advance(61)
    with only_failure():
        await v.validate(old_tok)


async def test_stale_keys_not_served_beyond_stale_window(router: Any, http: Any) -> None:
    clock = FakeClock(1000.0)
    v = validator(
        [idp(jwks_cache_ttl_s=10, jwks_stale_if_error_s=100, jwks_min_refresh_interval_s=1)],
        http,
        jwks_clock=clock,
    )
    tok = signed()
    await v.validate(tok)
    router.jwks_a.respond(status_code=500)
    clock.advance(50)
    assert (await v.validate(tok)).client_id == "client-a"  # stale-if-error window
    clock.advance(100)
    with only_failure(AuthFailure.JWKS_UNAVAILABLE):
        await v.validate(tok)


async def test_poisoned_jwks_cannot_replace_good_cache_with_nothing(router: Any, http: Any) -> None:
    clock = FakeClock(1000.0)
    v = validator([idp(jwks_cache_ttl_s=5, jwks_min_refresh_interval_s=1)], http, jwks_clock=clock)
    tok = signed()
    await v.validate(tok)
    router.jwks_a.respond(json={"keys": []})
    clock.advance(10)
    assert (await v.validate(tok)).client_id == "client-a"


async def test_jwks_with_private_material_or_wrong_use_ignored(router: Any, http: Any) -> None:
    priv = json.loads(__import__("jwt").algorithms.RSAAlgorithm.to_jwk(KP.private))  # includes d,p,q
    priv.update(kid="k1", use="sig", alg="RS256")
    router.jwks_a.respond(json={"keys": [priv]})
    v = validator([idp()], http)
    with only_failure():
        await v.validate(signed())
    enc = {**KP.jwk(), "use": "enc"}
    router.jwks_a.respond(json={"keys": [enc]})
    v2 = validator([idp()], http)
    with only_failure():
        await v2.validate(signed())


async def test_oversized_jwks_ignored(router: Any, http: Any) -> None:
    router.jwks_a.respond(content=b'{"keys": [], "pad": "' + b"A" * 1_100_000 + b'"}')
    v = validator([idp()], http)
    with only_failure(AuthFailure.JWKS_UNAVAILABLE):
        await v.validate(signed())


async def test_jwks_redirect_is_not_followed(router: Any, http: Any) -> None:
    hop = router.get("http://169.254.169.254/latest/meta-data").respond(json=KP.jwks())
    router.jwks_a.respond(status_code=302, headers={"Location": "http://169.254.169.254/latest/meta-data"})
    v = validator([idp()], http)
    with only_failure(AuthFailure.JWKS_UNAVAILABLE):
        await v.validate(signed())
    assert hop.call_count == 0


async def test_oidc_discovery_rejects_issuer_mismatch_and_plain_http_jwks(router: Any, http: Any) -> None:
    disc = ISSUER + "/.well-known/openid-configuration"
    d = idp(jwks_uri=None)
    router.get(disc).respond(json={"issuer": "https://evil.test", "jwks_uri": JWKS_A})
    with only_failure(AuthFailure.JWKS_UNAVAILABLE):
        await validator([d], http).validate(signed())
    router.get(disc).respond(json={"issuer": ISSUER, "jwks_uri": "http://169.254.169.254/keys"})
    probe = router.get("http://169.254.169.254/keys").respond(json=KP.jwks())
    with only_failure(AuthFailure.JWKS_UNAVAILABLE):
        await validator([d], http).validate(signed())
    assert probe.call_count == 0


async def test_discovery_over_plain_http_issuer_refused(router: Any, http: Any) -> None:
    d = idp(issuer="http://idp.test", jwks_uri=None)
    probe = router.get("http://idp.test/.well-known/openid-configuration").respond(
        json={"issuer": "http://idp.test", "jwks_uri": JWKS_A}
    )
    with only_failure():
        await validator([d], http).validate(signed(claims(iss="http://idp.test")))
    assert probe.call_count == 0


async def test_single_flight_refresh_under_concurrent_cold_start(router: Any, http: Any) -> None:
    import asyncio

    v = validator([idp()], http)
    route = router.jwks_a
    start = route.call_count
    toks = [signed(claims(sub=f"c{i}", client_id=f"c{i}")) for i in range(20)]
    await asyncio.gather(*(v.validate(t) for t in toks))
    assert route.call_count - start == 1


@pytest.mark.parametrize("alg", ["None", "NONE", "nOnE"])
def test_config_rejects_none_alg_case_variants(alg: str) -> None:
    with pytest.raises(ValueError):
        idp(algorithms=[alg])


@pytest.mark.parametrize("alg", ["HS256", "hs512", "none", "Hs384"])
def test_config_rejects_weak_algorithms(alg: str) -> None:
    with pytest.raises(ValueError):
        idp(algorithms=[alg])
