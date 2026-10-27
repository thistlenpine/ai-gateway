# ruff: noqa: E501, RUF001
"""Policy-engine adversarial cases and configuration / deployment hardening."""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from ai_gateway.config import (
    ConfigError,
    GatewayConfig,
    HttpGrant,
    IdPConfig,
    LimitPolicy,
    McpGrant,
    PolicyConfig,
    Rule,
    UpstreamConfig,
    load_config,
)
from ai_gateway.models import DecisionReason, Principal, Resource, UpstreamKind
from ai_gateway.policy import PolicyEngine
from ai_gateway.policy.matchers import name_match, normalize_path, path_match, scopes_satisfied

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "config" / "examples"


def principal(*scopes: str) -> Principal:
    return Principal(client_id="c", subject="s", issuer="i", scopes=frozenset(scopes))


def engine(*rules: Rule, extra_upstreams: dict[str, UpstreamConfig] | None = None) -> PolicyEngine:
    ups = {
        "api": UpstreamConfig(kind=UpstreamKind.HTTP, base_url="http://api.internal", prefix="/api"),
        "tools": UpstreamConfig(kind=UpstreamKind.MCP, base_url="http://mcp.internal/mcp", prefix="/mcp"),
        "tools2": UpstreamConfig(
            kind=UpstreamKind.MCP, base_url="http://mcp2.internal/mcp", prefix="/mcp/admin"
        ),
        **(extra_upstreams or {}),
    }
    return PolicyEngine(PolicyConfig(upstreams=ups, rules=list(rules)))


def http_res(path: str, method: str = "GET") -> Resource:
    return Resource(UpstreamKind.HTTP, "api", method, path)


# ------------------------------------------------------------------ routing


def test_route_prefix_confusion_and_longest_prefix_wins() -> None:
    e = engine()
    assert e.resolve_route("/apix/foo") is None
    assert e.resolve_route("/ap") is None
    assert e.resolve_route("/api")
    assert e.resolve_route("/api").relative_path == "/"  # type: ignore[union-attr]
    r = e.resolve_route("/mcp/admin/x")
    assert r and r.upstream == "tools2" and r.relative_path == "/x"
    r = e.resolve_route("/mcp/adminx")
    assert r and r.upstream == "tools"  # not tools2: segment-wise match
    for bad in ("api/x", "/api//x", "/api/./x", "/api/../x", "/API/x", "", "/api\x00/x"):
        assert e.resolve_route(bad) is None, bad


# ------------------------------------------------------------------ deny semantics


def test_deny_wins_over_any_number_of_allows_in_any_order() -> None:
    allow = Rule(
        id="a", scopes_any=["s"], http=[HttpGrant(upstream="api", methods=["GET", "DELETE"], paths=["/**"])]
    )
    deny_all = Rule(
        id="d", effect="deny", http=[HttpGrant(upstream="api", methods=["DELETE"], paths=["/**"])]
    )
    deny_scoped = Rule(
        id="d2",
        effect="deny",
        scopes_any=["banned"],
        http=[HttpGrant(upstream="api", methods=["GET"], paths=["/secret/**"])],
    )
    for rules in (
        (allow, deny_all, deny_scoped),
        (deny_scoped, deny_all, allow),
        (deny_all, allow, deny_scoped),
    ):
        e = engine(*rules)
        assert not e.authorize(principal("s"), http_res("/x", "DELETE")).allowed
        assert e.authorize(principal("s"), http_res("/x", "GET")).allowed
        assert not e.authorize(principal("s", "banned"), http_res("/secret/a")).allowed
        assert e.authorize(principal("s"), http_res("/secret/a")).allowed  # deny only binds banned principals
        d = e.authorize(principal("s", "banned"), http_res("/secret/a"))
        assert d.reason is DecisionReason.EXPLICIT_DENY


def test_unscoped_allow_rules_are_impossible_and_empty_scope_sets_never_satisfy() -> None:
    with pytest.raises(ValueError):
        Rule(id="x", http=[HttpGrant(upstream="api", paths=["/**"])])
    assert not scopes_satisfied({"a", "b"}, any_of=[], all_of=[])
    assert not scopes_satisfied(set(), any_of=[], all_of=[])
    assert not scopes_satisfied({"a"}, any_of=["b"], all_of=[])
    assert not scopes_satisfied({"a"}, any_of=["a"], all_of=["b"])
    assert scopes_satisfied({"a", "b"}, any_of=["a", "z"], all_of=["b"])


def test_scope_comparison_is_exact_not_glob_or_prefix() -> None:
    e = engine(
        Rule(id="a", scopes_any=["api:*", "api:read"], http=[HttpGrant(upstream="api", paths=["/**"])])
    )
    for s in ("api:write", "api:", "api", "API:read", "api:read ", "api:read\n", "api:rea", "xapi:read"):
        assert not e.authorize(principal(s), http_res("/x")).allowed, s
    assert e.authorize(
        principal("api:*"), http_res("/x")
    ).allowed  # literal star scope string matches only itself


def test_glob_semantics_do_not_overmatch() -> None:
    assert path_match("/invoices/*", "/invoices/42")
    assert not path_match("/invoices/*", "/invoices")
    assert not path_match("/invoices/*", "/invoices/42/items")
    assert not path_match("/invoices/*", "/invoices/")
    assert not path_match("/invoices/*", "/invoices//x")
    assert path_match("/a/**", "/a") and path_match("/a/**", "/a/b/c")
    assert not path_match("/a/**", "/ab")
    assert not path_match("/a/**/z", "/a/b/y")
    assert not path_match("/inv-*", "/inv-/x")
    assert not path_match("/x", "/X")
    assert not path_match("/a[b]c", "/abc") and path_match("/a[b]c", "/a[b]c")  # no char classes
    assert not path_match("/a.c", "/abc")  # '.' is literal
    assert not path_match("/a?c", "/a/c")
    assert (
        name_match("get_*", "get_x")
        and not name_match("get_*", "xget_x")
        and not name_match("get_*", "GET_x")
    )
    assert not name_match("get_?", "get_")
    assert not name_match("a.c", "abc") and not name_match("a[bc]", "ab")


@pytest.mark.parametrize(
    "path",
    ["/a/‮", "/a/%e2%80%ae", "/a/ ", "/a/ ", "/a/\u0085", "/a/﻿", "/a/​"],
)
def test_exotic_unicode_segments_are_never_treated_as_separators_or_dots(path: str) -> None:
    try:
        n = normalize_path(path)
    except ValueError:
        return
    assert n.count("/") == 2  # still exactly one extra segment


def test_normalize_path_exhaustive_hostile_set() -> None:
    bad = [
        "/a/%2e%2e/b",
        "/a/%2E%2e/b",
        "/a/%2f",
        "/a/%5C",
        "/a/\\b",
        "/a/%00",
        "/a/%252e",
        "/a/%25%32%65",
        "/a/;x",
        "/a/?x",
        "/a/#x",
        "/a/%",
        "/a/%g1",
        "/a/%e0%80%ae",
        "/../a",
        "/a/../../b",
        "/a/\t",
        "/a/\n",
        "/a/\x7f",
        "/a/．．",
        "/a/․․",
        "/a/ .. ",
        "/a/%20..%20",
        "/a/%2e",
        "/a/..%2f",
        "a",
        "",
    ]
    for p in bad:
        with pytest.raises(ValueError):
            normalize_path(p)
    assert normalize_path("/a//b/./c/") == "/a/b/c"
    assert normalize_path("/a/%41%62") == "/a/Ab"
    assert normalize_path("/") == "/"


def test_normalize_path_is_idempotent_and_fixpoint_safe() -> None:
    for p in ("/a//b", "/a/./b/../c", "/%41/%7e", "/a/b/", "/caf%C3%A9"):
        once = normalize_path(p)
        assert normalize_path(once) == once


# ------------------------------------------------------------------ mcp / tool filter consistency


def test_tool_filter_agrees_with_authorize_for_every_name() -> None:
    e = engine(
        Rule(id="a", scopes_any=["s"], mcp=[McpGrant(upstream="tools", tools=["get_*", "search"])]),
        Rule(id="d", effect="deny", mcp=[McpGrant(upstream="tools", tools=["get_secret*"])]),
    )
    p = principal("s")
    f = e.tool_filter(p, "tools")
    for name in (
        "get_a",
        "get_secret",
        "get_secret_2",
        "search",
        "searchx",
        "",
        "GET_a",
        "get_",
        "x",
        "get_\n",
    ):
        res = Resource(UpstreamKind.MCP, "tools", "POST", "/", mcp_method="tools/call", tool=name)
        assert f(name) == e.authorize(p, res).allowed, name
    assert not f(None) and not f(5)  # type: ignore[arg-type]


def test_mcp_methods_default_exclude_tools_call_and_resources() -> None:
    e = engine(Rule(id="a", scopes_any=["s"], mcp=[McpGrant(upstream="tools", tools=["*"])]))
    p = principal("s")

    def m(method: str) -> bool:
        return e.authorize(p, Resource(UpstreamKind.MCP, "tools", "POST", "/", mcp_method=method)).allowed

    assert m("initialize") and m("ping") and m("tools/list") and m("notifications/initialized")
    assert (
        not m("resources/read")
        and not m("prompts/get")
        and not m("sampling/createMessage")
        and not m("tools/call")
    )
    assert not m("notifications") and not m("Initialize") and not m("tools/list ")


def test_mcp_tool_resource_with_wrong_method_is_malformed_not_allowed() -> None:
    e = engine(Rule(id="a", scopes_any=["s"], mcp=[McpGrant(upstream="tools", tools=["*"], methods=["*"])]))
    p = principal("s")
    for res in (
        Resource(UpstreamKind.MCP, "tools", "POST", "/", mcp_method="ping", tool="x"),
        Resource(UpstreamKind.MCP, "tools", "POST", "/", mcp_method="tools/call", tool=None),
        Resource(UpstreamKind.MCP, "tools", "POST", "/", mcp_method=None, tool=None),
        Resource(UpstreamKind.MCP, "tools", "POST", "/", mcp_method="tools/call", tool=""),
        Resource(UpstreamKind.HTTP, "tools", "GET", "/"),
        Resource(UpstreamKind.MCP, "nope", "POST", "/", mcp_method="ping"),
        Resource(UpstreamKind.MCP, "tools", "POST", "/a/../b", mcp_method="ping"),
    ):
        assert not e.authorize(p, res).allowed, res


def test_authorize_all_denies_empty_and_mixed() -> None:
    e = engine(Rule(id="a", scopes_any=["s"], http=[HttpGrant(upstream="api", paths=["/ok"])]))
    assert not e.authorize_all(principal("s"), []).allowed
    assert not e.authorize_all(principal("s"), [http_res("/ok"), http_res("/no")]).allowed
    assert not e.authorize_all(principal("s"), [http_res("/no"), http_res("/ok")]).allowed
    assert e.authorize_all(principal("s"), [http_res("/ok"), http_res("/ok")]).allowed


def test_glob_matching_is_not_a_redos_vector() -> None:
    t = time.perf_counter()
    name_match("*a*a*a*b", "a" * 600)
    assert time.perf_counter() - t < 0.5


# ------------------------------------------------------------------ config loading


def _write_cfg(
    tmp_path: Path, gateway_extra: str = "", policy: str | None = None, clients: str | None = None
) -> Path:
    cfg = tmp_path / "cfg"
    (cfg / "policies").mkdir(parents=True)
    (cfg / "gateway.yaml").write_text(
        "idps:\n  - {name: t, issuer: 'https://idp.test', jwks_uri: 'https://idp.test/jwks', audiences: [gw]}\n"
        + gateway_extra
    )
    (cfg / "policies" / "p.yaml").write_text(policy or "upstreams: {}\nrules: []\n")
    if clients is not None:
        (cfg / "clients.yaml").write_text(clients)
    return cfg / "gateway.yaml"


def test_yaml_object_tags_are_refused_and_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "pwned"
    evil = f"!!python/object/apply:os.system ['touch {marker}']"
    for where in ("gateway", "policy", "clients"):
        gw = _write_cfg(
            tmp_path / where,
            gateway_extra=f"server: {evil}\n" if where == "gateway" else "",
            policy=f"upstreams: {evil}\n" if where == "policy" else None,
            clients=f"clients: {evil}\n" if where == "clients" else None,
        )
        with pytest.raises(ConfigError):
            load_config(gw)
    assert not marker.exists()


def test_inline_secrets_are_not_accepted_in_policy(tmp_path: Path) -> None:
    pol = "upstreams:\n  u: {kind: http, base_url: 'http://x', prefix: /u, credential: {env: X, value: hunter2}}\n"
    with pytest.raises(ConfigError):
        load_config(_write_cfg(tmp_path, policy=pol))
    pol = "upstreams:\n  u: {kind: http, base_url: 'http://x', prefix: /u, api_key: hunter2}\n"
    with pytest.raises(ConfigError):
        load_config(_write_cfg(tmp_path / "b", policy=pol))
    with pytest.raises(ConfigError):
        load_config(
            _write_cfg(
                tmp_path / "c", gateway_extra="telemetry: {otlp: {headers: 'Authorization=Bearer x'}}\n"
            )
        )


@pytest.mark.parametrize(
    "policy",
    [
        "rules: [{id: r, scope_any: [a], http: [{upstream: u, paths: ['/**']}]}]",  # typo'd key must not silently fail open
        "rules: [{id: r, effect: allow, scopes_any: [a], http: [{upstream: nope, paths: ['/**']}]}]",
        "rules: [{id: r, effect: permit, scopes_any: [a], http: [{upstream: u, paths: ['/**']}]}]",
        "rules: [{id: r, scopes_any: [a]}]",
        "rules: [{id: r, effect: allow, http: [{upstream: u, paths: ['/**']}]}]",
        "rules: [{id: r, scopes_any: [a], mcp: [{upstream: u}]}]",  # wrong kind
        "rules: [{id: r, scopes_any: [a], http: [{upstream: u, paths: ['/**']}]}, {id: r, scopes_any: [a], http: [{upstream: u, paths: ['/x']}]}]",
    ],
)
def test_invalid_or_ambiguous_policies_refuse_to_load(tmp_path: Path, policy: str) -> None:
    body = "upstreams:\n  u: {kind: http, base_url: 'http://x', prefix: /u}\n" + policy + "\n"
    with pytest.raises(ConfigError):
        load_config(_write_cfg(tmp_path, policy=body))


def test_env_expansion_fails_closed_and_cannot_inject_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SEC_UNSET_VAR", raising=False)
    with pytest.raises(ConfigError):
        load_config(_write_cfg(tmp_path, gateway_extra="telemetry: {environment: '${SEC_UNSET_VAR}'}\n"))
    monkeypatch.setenv("SEC_EVIL", "x\n  scopes_any: [admin]\nrules: []")
    pol = "upstreams:\n  u: {kind: http, base_url: 'http://x', prefix: /u}\nrules:\n  - {id: '${SEC_EVIL}', scopes_any: [a], http: [{upstream: u, paths: ['/x']}]}\n"
    cfg = load_config(_write_cfg(tmp_path / "b", policy=pol))
    assert len(cfg.policy.rules) == 1 and cfg.policy.rules[0].scopes_any == ["a"]


def test_secure_defaults() -> None:
    g = GatewayConfig(idps=[IdPConfig(name="t", issuer="https://i", audiences=["a"])])
    assert g.server.trust_forwarded_for is False
    assert set(g.idps[0].algorithms) <= {
        "RS256",
        "RS384",
        "RS512",
        "ES256",
        "ES384",
        "ES512",
        "PS256",
        "PS384",
        "PS512",
    }
    assert g.idps[0].leeway_s <= 60
    assert g.server.max_body_bytes <= 10_000_000
    assert g.idps[0].max_token_lifetime_s == 86_400  # no revocation: lifetime bounded by default
    assert g.defaults.unknown_clients == "deny"  # fail closed: clients must be registered
    assert g.server.ip_rate_limit.enabled
    with pytest.raises(ValueError):
        LimitPolicy(requests_per_window=0)
    with pytest.raises(ValueError):
        LimitPolicy(window_seconds=0)
    with pytest.raises(ValueError):
        LimitPolicy(delay_start=2, reject_at=1)
    with pytest.raises(ValueError):
        GatewayConfig(idps=[])


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    pol = (
        "upstreams:\n  u: {kind: http, base_url: 'http://x', prefix: /u}\n"
        "rules:\n  - id: r\n    effect: deny\n    effect: allow\n    scopes_any: [a]\n    http: [{upstream: u, paths: ['/x']}]\n"
    )
    with pytest.raises(ConfigError):
        load_config(_write_cfg(tmp_path, policy=pol))


@pytest.mark.parametrize(
    "kw",
    [
        {"audiences": [""]},
        {"issuer": ""},
        {"issuer": "idp.test"},
        {"jwks_min_refresh_interval_s": 0},
        {"jwks_min_refresh_interval_s": -1},
        {"jwks_cache_ttl_s": 0},
    ],
)
def test_idp_config_rejects_degenerate_values(kw: dict[str, Any]) -> None:
    base: dict[str, Any] = {"name": "t", "issuer": "https://i", "audiences": ["a"], **kw}
    with pytest.raises(ValueError):
        IdPConfig(**base)


def test_example_config_has_no_secrets_and_only_env_references() -> None:
    pat = re.compile(
        r"(sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN|eyJ[A-Za-z0-9_-]{10,}\.)"
    )
    for f in EXAMPLES.rglob("*.yaml"):
        text = f.read_text()
        assert not pat.search(text), f
        assert not re.search(r"(?im)^\s*(password|secret|api_key|token)\s*:\s*\S", text), f
    gateway = (EXAMPLES / "gateway.yaml").read_text()
    assert "headers_env" in gateway and "Authorization" not in gateway


def test_example_config_loads_and_is_default_deny(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for v in ("MCP_TOOLS_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(v, "x")
    shutil.copytree(EXAMPLES, tmp_path / "c")
    cfg = load_config(tmp_path / "c" / "gateway.yaml")
    e = PolicyEngine(cfg.policy)
    nobody = principal()
    for res in (
        http_res("/invoices"),
        Resource(UpstreamKind.HTTP, "billing", "GET", "/invoices"),
        Resource(UpstreamKind.HTTP, "billing", "DELETE", "/invoices/1"),
        Resource(UpstreamKind.MCP, "tools", "POST", "/", mcp_method="ping"),
        Resource(UpstreamKind.LLM, "anthropic", "POST", "/v1/messages", model="claude-sonnet-4"),
    ):
        assert not e.authorize(nobody, res).allowed
    admin = principal("billing:admin", "billing:write", "mcp:tools:admin", "llm:claude")
    assert not e.authorize(
        admin, Resource(UpstreamKind.HTTP, "billing", "DELETE", "/invoices/1")
    ).allowed  # deny wins
    assert not e.authorize(
        principal("llm:claude"),
        Resource(UpstreamKind.LLM, "anthropic", "POST", "/v1/messages", model="claude-opus-4"),
    ).allowed  # needs premium too


def test_deployment_artifacts_are_not_root_and_do_not_ship_secrets() -> None:
    docker = (ROOT / "Dockerfile").read_text()
    assert re.search(r"(?m)^USER\s+\S+", docker) and "USER root" not in docker
    assert "ENV" in docker and not re.search(r"(?i)(secret|password|api_key|token)\s*=", docker)
    ignore = (ROOT / ".dockerignore").read_text().split()
    assert ".git" in ignore and ".venv" in ignore
    gi = (ROOT / ".gitignore").read_text()
    assert ".env" in gi and "*.pem" in gi and "*.key" in gi


def test_dependencies_are_pinned_or_locked() -> None:
    py = (ROOT / "pyproject.toml").read_text()
    deps = re.findall(r'"([A-Za-z0-9_.\[\]-]+)(>=|==|~=)', py)
    locked = any(
        (ROOT / n).exists()
        for n in ("uv.lock", "requirements.lock", "poetry.lock", "pdm.lock", "requirements.txt")
    )
    assert locked or all(op in ("==", "~=") for _, op in deps)
