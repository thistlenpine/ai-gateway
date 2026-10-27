from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from ai_gateway.config import PolicyConfig
from ai_gateway.models import DecisionReason, Principal, Resource, UpstreamKind
from ai_gateway.policy import PolicyEngine

DEFAULT_YAML = Path(__file__).parents[2] / "config" / "examples" / "policies" / "default.yaml"
HTTP, MCP, LLM = UpstreamKind.HTTP, UpstreamKind.MCP, UpstreamKind.LLM


def P(*scopes: str) -> Principal:
    return Principal(client_id="c", subject="c", issuer="https://idp.test", scopes=frozenset(scopes))


def http(method: str, path: str, upstream: str = "billing") -> Resource:
    return Resource(HTTP, upstream, method, path)


def tool(name: str | None, upstream: str = "tools") -> Resource:
    return Resource(MCP, upstream, "POST", "/", mcp_method="tools/call", tool=name)


def rpc(method: str, upstream: str = "tools") -> Resource:
    return Resource(MCP, upstream, "POST", "/", mcp_method=method)


def llm(model: str | None, upstream: str = "anthropic") -> Resource:
    return Resource(LLM, upstream, "POST", "/v1/messages", model=model)


@pytest.fixture(scope="module")
def engine() -> PolicyEngine:
    return PolicyEngine(PolicyConfig.model_validate(yaml.safe_load(DEFAULT_YAML.read_text())))


def make(rules: list[dict[str, Any]], **extra: Any) -> PolicyEngine:
    upstreams = {
        "billing": {"kind": "http", "base_url": "http://b", "prefix": "/billing"},
        "tools": {"kind": "mcp", "base_url": "http://t", "prefix": "/mcp/tools"},
        "anthropic": {"kind": "llm", "base_url": "http://a", "prefix": "/llm/a", "provider": "anthropic"},
    }
    upstreams.update(extra)
    return PolicyEngine(PolicyConfig.model_validate({"upstreams": upstreams, "rules": rules}))


# ---------------------------------------------------------------- routing


def test_route_basic(engine: PolicyEngine) -> None:
    m = engine.resolve_route("/billing/invoices/42")
    assert m is not None
    assert (m.upstream, m.kind, m.prefix, m.relative_path) == ("billing", HTTP, "/billing", "/invoices/42")


def test_route_exact_prefix_gives_root(engine: PolicyEngine) -> None:
    m = engine.resolve_route("/billing")
    assert m is not None and m.relative_path == "/"


def test_route_multi_segment_prefix(engine: PolicyEngine) -> None:
    m = engine.resolve_route("/llm/anthropic/v1/messages")
    assert m is not None and m.upstream == "anthropic" and m.relative_path == "/v1/messages"


@pytest.mark.parametrize(
    "path",
    [
        "/billingx",
        "/billingx/invoices",
        "/billing-admin",
        "/Billing/invoices",
        "/",
        "/unknown",
        "/mcp",
        "/mcp/toolsx",
        "/llm/anthropicx/v1",
        "billing/x",
        "",
    ],
)
def test_route_no_match(engine: PolicyEngine, path: str) -> None:
    assert engine.resolve_route(path) is None


@pytest.mark.parametrize(
    "path", ["/billing//x", "/billing/../x", "/billing/./x", "/billing/x/", "/billing\x00/x"]
)
def test_route_refuses_unnormalized(engine: PolicyEngine, path: str) -> None:
    assert engine.resolve_route(path) is None


def test_route_longest_prefix_wins() -> None:
    e = make(
        [],
        api={"kind": "http", "base_url": "http://x", "prefix": "/api"},
        apiary={"kind": "http", "base_url": "http://y", "prefix": "/apiary"},
        v2={"kind": "http", "base_url": "http://z", "prefix": "/api/v2"},
    )
    assert e.resolve_route("/api/x").upstream == "api"  # type: ignore[union-attr]
    assert e.resolve_route("/apiary/x").upstream == "apiary"  # type: ignore[union-attr]
    assert e.resolve_route("/api/v2/x").upstream == "v2"  # type: ignore[union-attr]
    assert e.resolve_route("/api/v2").upstream == "v2"  # type: ignore[union-attr]
    assert e.resolve_route("/api/v20/x").upstream == "api"  # type: ignore[union-attr]
    assert e.resolve_route("/api/v20/x").relative_path == "/v20/x"  # type: ignore[union-attr]
    assert e.resolve_route("/apix") is None


def test_route_root_prefix_is_fallback() -> None:
    e = make([], root={"kind": "http", "base_url": "http://r", "prefix": "/"})
    m = e.resolve_route("/anything/else")
    assert m is not None and m.upstream == "root" and m.relative_path == "/anything/else"
    assert e.resolve_route("/billing/x").upstream == "billing"  # type: ignore[union-attr]


# ---------------------------------------------------------------- HTTP


@pytest.mark.parametrize(
    ("method", "path", "ok"),
    [
        ("GET", "/invoices", True),
        ("HEAD", "/invoices", True),
        ("get", "/invoices", True),
        ("GET", "/invoices/42", True),
        ("GET", "/invoices/42/lines", False),
        ("GET", "/customers", True),
        ("GET", "/customers/1/x/y", True),
        ("GET", "/admin", False),
        ("GET", "/invoicesx", False),
        ("GET", "/Invoices", False),
        ("POST", "/invoices", False),  # write needs billing:write
        ("OPTIONS", "/invoices", False),
        ("TRACE", "/invoices", False),
    ],
)
def test_http_read_scope(engine: PolicyEngine, method: str, path: str, ok: bool) -> None:
    d = engine.authorize(P("billing:read"), http(method, path))
    assert d.allowed is ok
    if ok:
        assert d.reason is DecisionReason.ALLOWED and d.rule_id == "billing-read"
        assert d.matched_scopes == {"billing:read"}
        assert d.resource is not None
    else:
        assert d.reason in (DecisionReason.NO_MATCHING_RULE, DecisionReason.INSUFFICIENT_SCOPE)


def test_http_any_of(engine: PolicyEngine) -> None:
    assert engine.authorize(P("billing:admin"), http("GET", "/invoices")).allowed
    assert engine.authorize(P("billing:admin", "billing:read"), http("GET", "/invoices")).matched_scopes == {
        "billing:admin",
        "billing:read",
    }


def test_http_all_of_requires_every_scope(engine: PolicyEngine) -> None:
    d = engine.authorize(P("billing:write"), http("POST", "/invoices/1"))
    assert d.allowed and d.rule_id == "billing-write"
    assert engine.authorize(P("billing:read"), http("POST", "/invoices/1")).allowed is False


def test_method_mismatch_with_right_scope(engine: PolicyEngine) -> None:
    # billing:write grants POST/PUT/PATCH only; GET is not covered by it
    d = engine.authorize(P("billing:write"), http("GET", "/invoices/1"))
    assert not d.allowed and d.reason is DecisionReason.INSUFFICIENT_SCOPE  # read rule matches, scope lacking
    d = engine.authorize(P("billing:write"), http("PUT", "/customers/1"))  # right scope, path outside grant
    assert not d.allowed and d.reason is DecisionReason.NO_MATCHING_RULE
    d = engine.authorize(P("billing:write"), http("DELETE", "/invoices/1"))
    assert d.reason is DecisionReason.EXPLICIT_DENY


def test_insufficient_scope_reports_hint(engine: PolicyEngine) -> None:
    d = engine.authorize(P("other"), http("GET", "/invoices"))
    assert not d.allowed and d.reason is DecisionReason.INSUFFICIENT_SCOPE
    assert d.missing_scopes_hint == ("billing:admin", "billing:read")
    d = engine.authorize(P(), http("POST", "/invoices/1"))
    assert d.missing_scopes_hint == ("billing:write",)


def test_no_scopes_denied(engine: PolicyEngine) -> None:
    assert not engine.authorize(P(), http("GET", "/invoices")).allowed


def test_explicit_deny_wins_over_any_scope(engine: PolicyEngine) -> None:
    d = engine.authorize(P("billing:admin", "billing:write", "billing:read"), http("DELETE", "/invoices/1"))
    assert not d.allowed and d.reason is DecisionReason.EXPLICIT_DENY and d.rule_id == "deny-billing-delete"
    assert not engine.authorize(P(), http("DELETE", "/")).allowed
    assert engine.authorize(P(), http("DELETE", "/")).reason is DecisionReason.EXPLICIT_DENY


def test_deny_precedence_regardless_of_rule_order() -> None:
    allow = {
        "id": "a",
        "scopes_any": ["s"],
        "http": [{"upstream": "billing", "methods": ["GET"], "paths": ["/**"]}],
    }
    deny = {
        "id": "d",
        "effect": "deny",
        "http": [{"upstream": "billing", "methods": ["GET"], "paths": ["/secret"]}],
    }
    for rules in ([allow, deny], [deny, allow]):
        e = make(rules)
        assert e.authorize(P("s"), http("GET", "/secret")).reason is DecisionReason.EXPLICIT_DENY
        assert e.authorize(P("s"), http("GET", "/public")).allowed


def test_scoped_deny_only_hits_holders() -> None:
    e = make(
        [
            {"id": "a", "scopes_any": ["s"], "http": [{"upstream": "billing", "paths": ["/**"]}]},
            {
                "id": "d",
                "effect": "deny",
                "scopes_any": ["banned"],
                "http": [{"upstream": "billing", "paths": ["/**"]}],
            },
        ]
    )
    assert e.authorize(P("s"), http("GET", "/x")).allowed
    assert e.authorize(P("s", "banned"), http("GET", "/x")).reason is DecisionReason.EXPLICIT_DENY
    assert not e.authorize(P("banned"), http("GET", "/x")).allowed


def test_scoped_deny_all_of() -> None:
    e = make(
        [
            {"id": "a", "scopes_any": ["s"], "http": [{"upstream": "billing", "paths": ["/**"]}]},
            {
                "id": "d",
                "effect": "deny",
                "scopes_all": ["x", "y"],
                "http": [{"upstream": "billing", "paths": ["/**"]}],
            },
        ]
    )
    assert e.authorize(P("s", "x"), http("GET", "/p")).allowed
    assert not e.authorize(P("s", "x", "y"), http("GET", "/p")).allowed


def test_wildcard_scopes_do_not_over_match() -> None:
    e = make(
        [{"id": "a", "scopes_any": ["billing:read"], "http": [{"upstream": "billing", "paths": ["/**"]}]}]
    )
    for s in (
        "billing:*",
        "billing",
        "*",
        "billing:read:extra",
        "Billing:Read",
        "billing:rea",
        "billing:read ",
    ):
        assert not e.authorize(P(s), http("GET", "/x")).allowed, s


def test_http_grant_on_other_upstream_does_not_leak() -> None:
    e = make(
        [{"id": "a", "scopes_any": ["s"], "http": [{"upstream": "billing", "paths": ["/**"]}]}],
        other={"kind": "http", "base_url": "http://o", "prefix": "/other"},
    )
    assert not e.authorize(P("s"), http("GET", "/x", "other")).allowed


def test_unknown_upstream_is_no_route(engine: PolicyEngine) -> None:
    d = engine.authorize(P("billing:read"), http("GET", "/invoices", "ghost"))
    assert not d.allowed and d.reason is DecisionReason.NO_ROUTE


def test_kind_mismatch_denied(engine: PolicyEngine) -> None:
    d = engine.authorize(
        P("billing:read", "mcp:tools:admin"), Resource(MCP, "billing", "POST", "/", mcp_method="ping")
    )
    assert not d.allowed


@pytest.mark.parametrize(
    "path", ["/invoices/../admin", "/invoices//1", "/invoices/./1", "invoices", "/invoices/1/"]
)
def test_authorize_rejects_unnormalized_resource_path(engine: PolicyEngine, path: str) -> None:
    d = engine.authorize(P("billing:read"), http("GET", path))
    assert not d.allowed and d.reason is DecisionReason.MALFORMED_REQUEST


def test_allow_rule_without_scopes_never_matches() -> None:
    cfg = PolicyConfig.model_construct(
        upstreams=PolicyConfig.model_validate(
            {"upstreams": {"billing": {"kind": "http", "base_url": "http://b", "prefix": "/billing"}}}
        ).upstreams,
        rules=[],
    )
    from ai_gateway.config import HttpGrant, Rule

    bad = Rule.model_construct(
        id="x",
        effect="allow",
        scopes_any=[],
        scopes_all=[],
        http=[HttpGrant(upstream="billing", methods=["GET"], paths=["/**"])],
        mcp=[],
        llm=[],
    )
    cfg.rules.append(bad)
    assert not PolicyEngine(cfg).authorize(P(), http("GET", "/x")).allowed
    assert not PolicyEngine(cfg).authorize(P("anything"), http("GET", "/x")).allowed


def test_engine_snapshot_is_immutable() -> None:
    cfg = PolicyConfig.model_validate(yaml.safe_load(DEFAULT_YAML.read_text()))
    e = PolicyEngine(cfg)
    cfg.rules.clear()
    cfg.upstreams.clear()
    assert e.authorize(P("billing:read"), http("GET", "/invoices")).allowed
    assert e.resolve_route("/billing/x") is not None


# ---------------------------------------------------------------- MCP


def test_mcp_tool_globs(engine: PolicyEngine) -> None:
    p = P("mcp:tools:read")
    for name in ("search_docs", "get_user", "list_x"):
        assert engine.authorize(p, tool(name)).allowed, name
    for name in ("delete_user", "xsearch_docs", "SEARCH_docs", "search", "run_get_x", ""):
        assert not engine.authorize(p, tool(name)).allowed, name


def test_mcp_admin_all_tools(engine: PolicyEngine) -> None:
    d = engine.authorize(P("mcp:tools:admin"), tool("delete_everything"))
    assert d.allowed and d.rule_id == "mcp-tools-admin"


def test_mcp_tool_without_scope(engine: PolicyEngine) -> None:
    d = engine.authorize(P("billing:read"), tool("search_x"))
    assert not d.allowed and d.reason is DecisionReason.INSUFFICIENT_SCOPE
    assert "mcp:tools:read" in d.missing_scopes_hint


@pytest.mark.parametrize(
    ("method", "ok_read", "ok_admin"),
    [
        ("initialize", True, True),
        ("ping", True, True),
        ("tools/list", True, True),
        ("notifications/initialized", True, True),
        ("notifications/cancelled", True, True),
        ("resources/list", False, True),
        ("resources/read", False, True),
        ("prompts/list", False, False),
        ("sampling/createMessage", False, False),
        ("completion/complete", False, False),
        ("logging/setLevel", False, False),
    ],
)
def test_mcp_methods(engine: PolicyEngine, method: str, ok_read: bool, ok_admin: bool) -> None:
    assert engine.authorize(P("mcp:tools:read"), rpc(method)).allowed is ok_read
    assert engine.authorize(P("mcp:tools:admin"), rpc(method)).allowed is ok_admin


def test_mcp_tools_call_not_authorized_via_methods_glob() -> None:
    e = make(
        [
            {
                "id": "a",
                "scopes_any": ["s"],
                "mcp": [{"upstream": "tools", "tools": [], "methods": ["tools/*"]}],
            }
        ]
    )
    assert e.authorize(P("s"), rpc("tools/list")).allowed
    assert not e.authorize(P("s"), tool("anything")).allowed
    # tools/call with the tool name missing cannot be classified
    d = e.authorize(P("s"), rpc("tools/call"))
    assert not d.allowed and d.reason is DecisionReason.MALFORMED_REQUEST


def test_mcp_malformed_resources(engine: PolicyEngine) -> None:
    p = P("mcp:tools:admin")
    bad = [
        Resource(MCP, "tools", "POST", "/", mcp_method="tools/list", tool="x"),  # tool on non-call
        Resource(MCP, "tools", "POST", "/", mcp_method=None, tool="x"),
        Resource(MCP, "tools", "POST", "/"),  # POST without a method
        Resource(MCP, "tools", "POST", "/", mcp_method=""),
        Resource(MCP, "tools", "POST", "/", mcp_method="tools/call", tool=""),
    ]
    for r in bad:
        d = engine.authorize(p, r)
        assert not d.allowed and d.reason is DecisionReason.MALFORMED_REQUEST, r


def test_mcp_get_delete_need_any_grant(engine: PolicyEngine) -> None:
    for m in ("GET", "DELETE"):
        r = Resource(MCP, "tools", m, "/")
        assert engine.authorize(P("mcp:tools:read"), r).allowed
        assert engine.authorize(P("mcp:tools:admin"), r).allowed
        assert not engine.authorize(P("billing:read"), r).allowed
        assert not engine.authorize(P(), r).allowed
    assert not engine.authorize(P("mcp:tools:read"), Resource(MCP, "tools", "PUT", "/")).allowed


def test_mcp_deny_tool() -> None:
    e = make(
        [
            {"id": "a", "scopes_any": ["s"], "mcp": [{"upstream": "tools", "tools": ["*"]}]},
            {"id": "d", "effect": "deny", "mcp": [{"upstream": "tools", "tools": ["drop_*"]}]},
        ]
    )
    assert e.authorize(P("s"), tool("get_x")).allowed
    d = e.authorize(P("s"), tool("drop_table"))
    assert d.reason is DecisionReason.EXPLICIT_DENY and d.rule_id == "d"
    f = e.tool_filter(P("s"), "tools")
    assert f("get_x") and not f("drop_table")


def test_tool_filter(engine: PolicyEngine) -> None:
    f = engine.tool_filter(P("mcp:tools:read"), "tools")
    assert [t for t in ("search_a", "delete_a", "get_b", "list_c", "admin") if f(t)] == [
        "search_a",
        "get_b",
        "list_c",
    ]
    assert engine.tool_filter(P("mcp:tools:admin"), "tools")("anything")
    none = engine.tool_filter(P(), "tools")
    assert not none("search_a")
    assert not engine.tool_filter(P("mcp:tools:admin"), "ghost")("x")
    assert not engine.tool_filter(P("mcp:tools:admin"), "billing")("x")
    assert not f("")
    assert not f(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------- LLM


@pytest.mark.parametrize(
    ("scopes", "model", "ok", "cap"),
    [
        (("llm:claude",), "claude-haiku-4-5", True, 8192),
        (("llm:claude",), "claude-sonnet-4", True, 8192),
        (("llm:claude",), "claude-opus-4", False, None),
        (("llm:claude", "llm:premium"), "claude-opus-4", True, None),
        (("llm:premium",), "claude-opus-4", False, None),
        (("llm:claude",), "Claude-Haiku-4", False, None),
        (("llm:claude",), "gpt-4o-mini", False, None),
        (("llm:openai",), "claude-haiku-4", False, None),
        (("llm:claude",), "xclaude-haiku-4", False, None),
        ((), "claude-haiku-4", False, None),
    ],
)
def test_llm_models(
    engine: PolicyEngine, scopes: tuple[str, ...], model: str, ok: bool, cap: int | None
) -> None:
    d = engine.authorize(P(*scopes), llm(model))
    assert d.allowed is ok
    assert d.max_tokens_per_request == cap


def test_llm_per_upstream(engine: PolicyEngine) -> None:
    assert engine.authorize(P("llm:openai"), llm("gpt-4o-mini", "openai")).allowed
    assert not engine.authorize(P("llm:openai"), llm("gpt-4o", "openai")).allowed
    assert not engine.authorize(P("llm:claude"), llm("claude-haiku-4", "openai")).allowed
    assert not engine.authorize(P("llm:openai"), llm("gpt-4o-mini", "anthropic")).allowed


@pytest.mark.parametrize("model", [None, ""])
def test_llm_missing_model_denied(engine: PolicyEngine, model: str | None) -> None:
    d = engine.authorize(P("llm:claude", "llm:premium"), llm(model))
    assert not d.allowed and d.reason is DecisionReason.MALFORMED_REQUEST


def test_llm_cap_union_most_permissive() -> None:
    e = make(
        [
            {
                "id": "a",
                "scopes_any": ["x"],
                "llm": [{"upstream": "anthropic", "models": ["m-*"], "max_tokens_per_request": 100}],
            },
            {
                "id": "b",
                "scopes_any": ["y"],
                "llm": [{"upstream": "anthropic", "models": ["m-1"], "max_tokens_per_request": 500}],
            },
            {"id": "c", "scopes_any": ["z"], "llm": [{"upstream": "anthropic", "models": ["m-1"]}]},
        ]
    )
    assert e.authorize(P("x"), llm("m-1")).max_tokens_per_request == 100
    d = e.authorize(P("x", "y"), llm("m-1"))
    assert d.max_tokens_per_request == 500 and d.rule_id == "a" and d.matched_scopes == {"x", "y"}
    assert e.authorize(P("x", "z"), llm("m-1")).max_tokens_per_request is None
    assert e.authorize(P("x", "y"), llm("m-2")).max_tokens_per_request == 100


# ---------------------------------------------------------------- authorize_all


def test_authorize_all_empty_denied(engine: PolicyEngine) -> None:
    d = engine.authorize_all(P("mcp:tools:admin"), [])
    assert not d.allowed and d.reason is DecisionReason.MALFORMED_REQUEST


def test_authorize_all_every_resource_must_pass(engine: PolicyEngine) -> None:
    p = P("mcp:tools:read")
    ok = [rpc("initialize"), tool("search_x"), rpc("tools/list")]
    assert engine.authorize_all(p, ok).allowed
    bad = [rpc("initialize"), tool("delete_x"), tool("search_y")]
    d = engine.authorize_all(p, bad)
    assert not d.allowed and d.resource == bad[1]
    # a denial anywhere in the batch wins, even last
    d = engine.authorize_all(p, [tool("search_x"), rpc("resources/read")])
    assert not d.allowed and d.resource is not None and d.resource.mcp_method == "resources/read"


def test_authorize_all_returns_first_denial_and_last_allow(engine: PolicyEngine) -> None:
    p = P("mcp:tools:read")
    a, b = tool("search_x"), tool("get_y")
    assert engine.authorize_all(p, [a, b]).resource == b
    first, second = tool("nope"), rpc("resources/list")
    assert engine.authorize_all(p, [first, second]).resource == first


def test_authorize_all_single(engine: PolicyEngine) -> None:
    r = http("GET", "/invoices")
    assert engine.authorize_all(P("billing:read"), [r]) == engine.authorize(P("billing:read"), r)
