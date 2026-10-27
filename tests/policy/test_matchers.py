from __future__ import annotations

import pytest

from ai_gateway.policy.matchers import name_match, normalize_path, path_match, scopes_satisfied


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("/", "/"),
        ("/a/b", "/a/b"),
        ("//a///b//", "/a/b"),
        ("/a/./b", "/a/b"),
        ("/a/b/../c", "/a/c"),
        ("/a/b/..", "/a"),
        ("/a/..", "/"),
        ("/a%20b", "/a b"),
        ("/%41bc", "/Abc"),
        ("/caf%C3%A9", "/café"),
        ("/Billing/X", "/Billing/X"),  # case preserved
        ("/a/...", "/a/..."),
        ("/a/b/", "/a/b"),
    ],
)
def test_normalize_ok(raw: str, expected: str) -> None:
    assert normalize_path(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "a/b",
        "http://evil/x",
        "/a\x00b",
        "/a%00b",
        "/a%2fb",
        "/a%2Fb",
        "/a%5cb",
        "/a%5Cb",
        "/a\\b",
        "/..",
        "/a/../..",
        "/%2e%2e/etc",
        "/%2E%2E/etc",
        "/a/%2e/b",
        "/.%2e/x",
        "/a/.. /b",
        "/a/%2e%2e%20/b",
        "/%252e%252e/x",  # double encoding
        "/%252fadmin",
        "/%41%2541",
        "/a%",
        "/a%zz",
        "/a%c0%afb",  # overlong slash
        "/a%ffb",
        "/a\nb",
        "/a%0ab",
        "/a?x=1",
        "/a%3fx",
        "/a%23b",
        "/a;x",
        "/\uff0e\uff0e/x",  # fullwidth dots
        "/a\uff0fb",  # fullwidth slash
    ],
)
def test_normalize_rejects(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_path(raw)


@pytest.mark.parametrize("p", ["/a//b/../c/./d/", "/x%20y/../z"])
def test_normalize_idempotent(p: str) -> None:
    once = normalize_path(p)
    assert normalize_path(once) == once


@pytest.mark.parametrize(
    ("pattern", "path", "ok"),
    [
        ("/", "/", True),
        ("/", "/a", False),
        ("/invoices", "/invoices", True),
        ("/invoices", "/invoices/1", False),
        ("/invoices", "/invoicesx", False),
        ("/invoices/*", "/invoices/42", True),
        ("/invoices/*", "/invoices", False),
        ("/invoices/*", "/invoices/42/lines", False),
        ("/invoices/inv-*", "/invoices/inv-7", True),
        ("/invoices/inv-*", "/invoices/INV-7", False),
        ("/invoices/inv-*", "/invoices/inv-7/x", False),
        ("/customers/**", "/customers", True),
        ("/customers/**", "/customers/1/a/b", True),
        ("/customers/**", "/customersx", False),
        ("/customers/**", "/customers-x/1", False),
        ("/**", "/", True),
        ("/**", "/a/b/c", True),
        ("/**/edit", "/a/b/edit", True),
        ("/**/edit", "/edit", True),
        ("/**/edit", "/a/edit/x", False),
        ("/a/**/b/**/c", "/a/x/b/y/z/c", True),
        ("/a/**/b/**/c", "/a/b/c", True),
        ("/a/**/b/**/c", "/a/x/c", False),
        ("/a/?", "/a/b", True),
        ("/a/?", "/a/bc", False),
        ("/a/*", "/a/b.c", True),
        ("/Users", "/users", False),
        ("a", "/a", False),
        ("/a.b", "/axb", False),  # '.' literal
        ("/a+(b)", "/a+(b)", True),  # regex metachars literal
        ("/a/[bc]", "/a/b", False),  # no char classes
        ("/a/", "/a", False),
    ],
)
def test_path_match(pattern: str, path: str, ok: bool) -> None:
    assert path_match(pattern, path) is ok


def test_path_match_many_globstars_is_fast() -> None:
    assert path_match("/**" * 20 + "/z", "/" + "/".join(["a"] * 60)) is False


@pytest.mark.parametrize(
    ("pattern", "name", "ok"),
    [
        ("*", "anything", True),
        ("*", "", True),
        ("search_*", "search_docs", True),
        ("search_*", "xsearch_docs", False),
        ("get_?", "get_a", True),
        ("get_?", "get_ab", False),
        ("claude-opus-*", "claude-opus-4", True),
        ("claude-opus-*", "Claude-Opus-4", False),
        ("claude-haiku-*", "claude-opus-4", False),
        ("gpt-*-mini", "gpt-4o-mini", True),
        ("gpt-*-mini", "gpt-4o-mini-x", False),
        ("a.b", "axb", False),
        ("[ab]", "a", False),
        ("tools/*", "tools/list", True),
        ("tools/*", "tools/call", True),
        ("notifications/*", "notifications/initialized", True),
        ("ping", "ping2", False),
    ],
)
def test_name_match(pattern: str, name: str, ok: bool) -> None:
    assert name_match(pattern, name) is ok


def test_name_match_multiline_name() -> None:
    assert name_match("a*", "a\nb")
    assert not name_match("a", "a\n")


@pytest.mark.parametrize(
    ("granted", "any_of", "all_of", "ok"),
    [
        ({"a"}, ["a", "b"], [], True),
        ({"c"}, ["a", "b"], [], False),
        ({"a", "b"}, [], ["a", "b"], True),
        ({"a"}, [], ["a", "b"], False),
        ({"a", "x"}, ["a", "b"], ["x"], True),
        ({"a"}, ["a", "b"], ["x"], False),
        ({"x"}, ["a", "b"], ["x"], False),
        ({"a", "b"}, [], [], False),
        (set(), [], [], False),
        ({"billing:*"}, ["billing:read"], [], False),  # no wildcard scopes
        ({"billing"}, ["billing:read"], [], False),
        ({"*"}, ["billing:read"], [], False),
    ],
)
def test_scopes_satisfied(granted: set[str], any_of: list[str], all_of: list[str], ok: bool) -> None:
    assert scopes_satisfied(granted, any_of=any_of, all_of=all_of) is ok
