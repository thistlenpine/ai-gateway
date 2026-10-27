"""Property tests for path normalization and prefix routing (policy.matchers / PolicyEngine)."""

from __future__ import annotations

import re

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from ai_gateway.config import PolicyConfig
from ai_gateway.policy import PolicyEngine
from ai_gateway.policy.matchers import normalize_path, path_match

# Alphabet deliberately dense in the dangerous tokens.
SEGMENT = st.sampled_from(
    ["", ".", "..", "%2e", "%2E", "%2e%2e", ".%2e", "a", "b", "billing", "billingx", "mcp", "tools", "x y",
     "%61", "..%2f", "%2f", "%5c", "%00", "...", " .. ", "é", "․", "．", "%", "%zz", "%252e"]
)  # fmt: skip
RAW_PATH = st.lists(SEGMENT, max_size=8).map(lambda segs: "/" + "/".join(segs))
ARBITRARY = st.text(alphabet=st.characters(blacklist_categories=["Cs"]), max_size=40).map(lambda s: "/" + s)


@given(st.one_of(RAW_PATH, ARBITRARY))
@example("/billing/../../etc/passwd")
@example("/billing/%2e%2e/%2e%2e/x")
@example("//billing//x")
def test_normalized_output_is_canonical_or_rejected(raw: str) -> None:
    try:
        out = normalize_path(raw)
    except ValueError:
        return  # fail-closed is always acceptable
    assert out.startswith("/")
    assert "//" not in out
    assert "\x00" not in out and "\\" not in out
    segs = [] if out == "/" else out[1:].split("/")
    assert all(s not in ("", ".", "..") and s.strip() not in (".", "..") for s in segs)
    assert not out.endswith("/") or out == "/"
    assert not re.search(r"%2f|%5c", out, re.IGNORECASE)
    assert normalize_path(out) == out or "%" in out  # idempotent unless a literal '%' survived decoding


@given(st.lists(st.sampled_from(["a", "b", "billing", "mcp"]), max_size=6))
def test_idempotent_on_clean_paths(segs: list[str]) -> None:
    p = "/" + "/".join(segs)
    assert normalize_path(p) == p
    assert normalize_path(normalize_path(p)) == p


def _reference(raw: str) -> str | None:
    """Independent model for the restricted alphabet below: decode %2e, resolve with a stack.
    None means "must be rejected"."""
    out: list[str] = []
    for seg in raw[1:].split("/"):
        decoded = re.sub(r"%2e", ".", seg, flags=re.IGNORECASE)
        if decoded != seg and decoded in (".", ".."):
            return None  # encoded dot segments are rejected outright (stricter than RFC 3986)
        seg = decoded
        if seg in ("", "."):
            continue
        if seg == "..":
            if not out:
                return None
            out.pop()
            continue
        out.append(seg)
    return "/" + "/".join(out)


SIMPLE = st.lists(
    st.sampled_from(["", ".", "..", "%2e", "%2E", "%2e%2e", "a", "b", "billing", "mcp"]), max_size=8
).map(lambda segs: "/" + "/".join(segs))


@given(SIMPLE)
def test_matches_reference_model(raw: str) -> None:
    expected = _reference(raw)
    if expected is None:
        with pytest.raises(ValueError):
            normalize_path(raw)
    else:
        assert normalize_path(raw) == expected


def _engine() -> PolicyEngine:
    return PolicyEngine(
        PolicyConfig.model_validate(
            {
                "upstreams": {
                    "billing": {"kind": "http", "base_url": "http://b", "prefix": "/billing"},
                    "billing-admin": {"kind": "http", "base_url": "http://ba", "prefix": "/billing/admin"},
                    "tools": {"kind": "mcp", "base_url": "http://t", "prefix": "/mcp/tools"},
                },
                "rules": [],
            }
        )
    )


ENGINE = _engine()
PREFIXES = {"billing": "/billing", "billing-admin": "/billing/admin", "tools": "/mcp/tools"}


def _literal_route(path: str) -> tuple[str, str] | None:
    """Literal-prefix semantics: longest configured prefix that is a whole-segment prefix."""
    best: tuple[str, str] | None = None
    for name, prefix in PREFIXES.items():
        if path == prefix or path.startswith(prefix + "/"):
            if best is None or len(prefix) > len(PREFIXES[best[0]]):
                rel = path[len(prefix) :] or "/"
                best = (name, rel)
    return best


@given(st.one_of(RAW_PATH, SIMPLE))
@example("/billing/../mcp/tools/x")
@example("/mcp/tools/../../billing/admin/x")
@example("/billingx/admin")
@example("/billing/admin/../../mcp/tools")
def test_routing_equals_literal_prefix_semantics_on_normalized_path(raw: str) -> None:
    try:
        n = normalize_path(raw)
    except ValueError:
        return
    m = ENGINE.resolve_route(n)
    ref = _literal_route(n)
    if ref is None:
        assert m is None
    else:
        assert m is not None and (m.upstream, m.relative_path) == ref
        # the proxy forwards prefix + relative_path: they must reassemble the normalized path
        assert (m.prefix + ("" if m.relative_path == "/" else m.relative_path)) == n
        assert ".." not in m.relative_path.split("/")


@given(st.text(max_size=30))
def test_unnormalized_paths_never_route(raw: str) -> None:
    """resolve_route fails closed on anything that is not already clean."""
    if (
        "//" in raw
        or "/./" in raw
        or "/../" in raw
        or raw.endswith(("/.", "/..", "/"))
        or not raw.startswith("/")
    ):
        if raw != "/":
            assert ENGINE.resolve_route(raw) is None


@given(
    st.lists(st.sampled_from(["a", "b", "x1"]), max_size=5),
    st.lists(st.sampled_from(["a", "b", "*", "**", "x1"]), max_size=5),
)
def test_glob_star_never_crosses_segments(segs: list[str], pat: list[str]) -> None:
    path = "/" + "/".join(segs)
    pattern = "/" + "/".join(pat)
    if "**" not in pat:
        # without '**' a glob can only match paths with the same number of segments
        if path_match(pattern, path):
            assert len(pat) == len(segs)
    if path_match(pattern, path):
        # every literal pattern segment equals the corresponding path segment when no '**'
        if "**" not in pat:
            assert all(p in ("*", s) for p, s in zip(pat, segs, strict=True))
