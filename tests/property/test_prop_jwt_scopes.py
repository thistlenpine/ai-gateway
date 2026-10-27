"""Property tests for JWT scope parsing (TokenValidator end to end with real RS256 tokens)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
import respx
from hypothesis import given, settings
from hypothesis import strategies as st

from ai_gateway.auth import TokenValidator
from ai_gateway.config import IdPConfig
from ai_gateway.errors import AuthenticationError, AuthFailure
from ai_gateway.observability.telemetry import NullTelemetry
from support import KeyPair

KP = KeyPair("k1")
ISSUER = "https://idp.test"
JWKS_URL = "https://idp.test/jwks.json"
IDP: IdPConfig = IdPConfig(name="main", issuer=ISSUER, jwks_uri=JWKS_URL, audiences=["ai-gateway"])

WS = st.sampled_from([" ", "  ", "\t", "\n", " ", " "])
SCOPE = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Zs", "Zl", "Zp", "Cc"), blacklist_characters=" "),
    min_size=1,
    max_size=12,
)
scope_lists = st.lists(SCOPE, max_size=8)


def scopes_of(claims: dict[str, Any]) -> frozenset[str]:
    return TokenValidator._scopes(claims, IDP)


@given(scope_lists, st.lists(WS, min_size=1, max_size=3))
def test_space_string_roundtrip(scopes: list[str], seps: list[str]) -> None:
    text = "".join(s + seps[i % len(seps)] for i, s in enumerate(scopes))
    assert scopes_of({"scope": text}) == frozenset(scopes)


@given(scope_lists)
def test_list_claim_equals_string_claim(scopes: list[str]) -> None:
    assert scopes_of({"scp": scopes}) == scopes_of({"scope": " ".join(scopes)}) == frozenset(scopes)


@given(scope_lists, scope_lists)
def test_scope_and_scp_union(a: list[str], b: list[str]) -> None:
    assert scopes_of({"scope": " ".join(a), "scp": b}) == frozenset(a) | frozenset(b)


@given(st.text(max_size=60))
def test_never_empty_or_whitespace_members(text: str) -> None:
    got = scopes_of({"scope": text})
    assert all(s and not any(c.isspace() for c in s) for s in got)


@given(
    st.one_of(
        st.integers(),
        st.floats(allow_nan=False),
        st.booleans(),
        st.dictionaries(st.text(max_size=3), st.text(max_size=3), max_size=2),
        st.lists(
            st.one_of(st.integers(), st.none(), st.booleans(), st.lists(st.text(max_size=2))),
            min_size=1,
            max_size=3,
        ),
    )
)
def test_wrong_types_are_rejected_not_coerced(value: Any) -> None:
    with pytest.raises(AuthenticationError) as ei:
        scopes_of({"scope": value})
    assert ei.value.failure is AuthFailure.MALFORMED


def test_absent_claims_mean_no_scopes() -> None:
    assert scopes_of({}) == frozenset()
    assert scopes_of({"scope": None, "scp": None}) == frozenset()


@settings(max_examples=40)
@given(scope_lists)
def test_validate_populates_principal_scopes_from_signed_token(scopes: list[str]) -> None:
    async def go() -> frozenset[str]:
        with respx.mock:
            respx.get(JWKS_URL).respond(json=KP.jwks())
            async with httpx.AsyncClient() as http:
                v = TokenValidator([IDP], http, NullTelemetry())
                tok = KP.mint(scopes=" ".join(scopes), issuer=ISSUER, client_id="c1")
                return (await v.validate(tok)).scopes

    assert asyncio.run(go()) == frozenset(scopes)


def test_json_roundtrip_does_not_change_scopes() -> None:
    s = {"scope": "a b", "scp": ["c d"]}
    assert scopes_of(json.loads(json.dumps(s))) == frozenset("abcd")
