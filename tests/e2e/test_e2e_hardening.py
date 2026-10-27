"""Pre-auth IP limiter, unknown-client default, and the 403/404/429 no-oracle matrix."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ai_gateway.config import IpLimitConfig
from ai_gateway.observability.telemetry import NullTelemetry
from ai_gateway.ratelimit import IpRateLimiter
from ai_gateway.services import build_services
from gw_rig import make_config


def _enable_ip_limit(gw: Any, burst: int = 3) -> IpLimitConfig:
    cfg = IpLimitConfig(requests_per_window=60, window_seconds=60, burst=burst)
    gw.services.config.gateway.server.ip_rate_limit = cfg
    gw.services.ip_limiter = IpRateLimiter(cfg, clock=gw.clock)
    return cfg


def _client_from(gw: Any, ip: str) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=gw.app, client=(ip, 5555))
    return httpx.AsyncClient(transport=transport, base_url="http://gateway.test")


# --------------------------------------------------------------------------- IP limiter


async def test_ip_limit_applies_before_token_validation(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _enable_ip_limit(gw, burst=3)
        for _ in range(3):
            r = await gw.client.get("/billing/invoices", headers={"Authorization": "Bearer junk"})
            assert r.status_code == 401
        before = len(gw.obs.audit("authn.failure"))
        r = await gw.client.get("/billing/invoices", headers={"Authorization": "Bearer junk"})
        assert r.status_code == 429 and r.json()["error"]["code"] == "ip_rate_limited"
        assert int(r.headers["retry-after"]) >= 1
        assert "x-request-id" in r.headers
        # no token work, no authn audit record, no auth-failure metric for the throttled request
        assert len(gw.obs.audit("authn.failure")) == before
        assert sum(p.value for _, p in gw.obs.metric("gateway.auth.failures")) == 3
        scopes = {a["gateway.ratelimit.scope"] for a, _ in gw.obs.metric("gateway.ratelimit.decisions")}
        assert "ip" in scopes
        # the address itself never becomes a metric label
        assert all("127.0.0.1" not in str(a) for a, _ in gw.obs.metric("gateway.ratelimit.decisions"))


async def test_ip_limit_also_covers_valid_tokens_and_recovers(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _enable_ip_limit(gw, burst=2)
        gw.upstream.get("http://billing.internal/invoices").respond(json=[])
        h = gw.auth("billing:read", client_id="anyone")
        assert (await gw.client.get("/billing/invoices", headers=h)).status_code == 200
        assert (await gw.client.get("/billing/invoices", headers=h)).status_code == 200
        r = await gw.client.get("/billing/invoices", headers=h)
        assert r.status_code == 429 and r.json()["error"]["code"] == "ip_rate_limited"
        gw.clock.advance(1.5)
        assert (await gw.client.get("/billing/invoices", headers=h)).status_code == 200


async def test_ip_limit_is_per_source_ip(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _enable_ip_limit(gw, burst=1)
        async with _client_from(gw, "198.51.100.1") as a, _client_from(gw, "198.51.100.2") as b:
            assert (await a.get("/billing/x")).status_code == 401
            assert (await a.get("/billing/x")).status_code == 429
            assert (await b.get("/billing/x")).status_code == 401  # another IP is unaffected


async def test_ip_limit_ignores_spoofed_forwarded_for_by_default(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _enable_ip_limit(gw, burst=1)
        spoof = {"X-Forwarded-For": "203.0.113.7"}
        assert (await gw.client.get("/billing/x", headers=spoof)).status_code == 401
        spoof = {"X-Forwarded-For": "203.0.113.8"}  # rotating the header must not mint new buckets
        assert (await gw.client.get("/billing/x", headers=spoof)).status_code == 429


async def test_ip_limit_disabled_in_config_never_throttles(make_gateway: Any) -> None:
    async with make_gateway() as gw:  # the rig disables it
        assert not gw.services.ip_limiter.enabled
        for _ in range(50):
            assert (await gw.client.get("/billing/x")).status_code == 401


async def test_health_endpoints_are_not_ip_limited(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _enable_ip_limit(gw, burst=1)
        for _ in range(5):
            assert (await gw.client.get("/healthz")).status_code == 200


# --------------------------------------------------------------------------- unknown clients


async def test_unknown_clients_default_is_deny_and_open_mode_warns(monkeypatch: Any, caplog: Any) -> None:
    monkeypatch.setenv("E2E_MCP_TOKEN", "x")
    monkeypatch.setenv("E2E_LLM_KEY", "y")
    cfg = make_config()
    assert type(cfg.gateway.defaults)().unknown_clients == "deny"
    for mode, warned in (("deny", False), ("default_limits", True)):
        cfg.gateway.defaults.unknown_clients = mode  # type: ignore[assignment]
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            svc = build_services(
                cfg,
                telemetry=NullTelemetry(),
                jwks_http=httpx.AsyncClient(),
                upstream_http=httpx.AsyncClient(),
            )
        await svc.aclose()
        events = [getattr(r, "event", "") for r in caplog.records]
        assert ("startup.unknown_clients_open" in events) is warned, mode


async def test_unregistered_client_is_denied_with_default_config(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.services.config.gateway.defaults.unknown_clients = type(
            gw.services.config.gateway.defaults
        )().unknown_clients  # the shipped default
        r = await gw.client.get("/billing/invoices", headers=gw.auth("billing:read", client_id="stranger"))
        assert r.status_code == 403 and r.json()["error"]["code"] == "access_denied"


# --------------------------------------------------------------------------- 403/404/429 oracle matrix

_CASES: dict[str, tuple[str, str, str, dict[str, Any]]] = {
    # name: (method, path, scopes, extra request kwargs)
    "no_such_route": ("GET", "/nope/x", "billing:read", {}),
    "route_exists_path_not_granted": ("GET", "/billing/admin", "billing:read", {}),
    "route_exists_method_not_granted": ("DELETE", "/billing/invoices/1", "billing:read", {}),
    "http_no_scope": ("GET", "/billing/invoices", "", {}),
    "mcp_no_scope": (
        "POST",
        "/mcp/tools",
        "billing:read",
        {"json": {"jsonrpc": "2.0", "id": 1, "method": "ping"}},
    ),
    "llm_no_scope": ("POST", "/llm/anthropic/v1/messages", "billing:read", {"json": {"model": "claude-x"}}),
    "mcp_tool_not_granted": (
        "POST",
        "/mcp/tools",
        "mcp:tools:read",
        {"json": {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "delete_all"}}},
    ),
    "llm_model_not_granted": (
        "POST",
        "/llm/anthropic/v1/messages",
        "llm:claude",
        {"json": {"model": "gpt-4", "max_tokens": 5, "messages": []}},
    ),
}


async def _sequence(make_gateway: Any, case: str, n: int = 16) -> tuple[list[int], list[str]]:
    method, path, scopes, kw = _CASES[case]
    async with make_gateway(requests_per_window=6) as gw:
        gw.upstream.route().respond(404, json={"upstream": "not found"})
        statuses, codes = [], []
        for _ in range(n):
            r = await gw.client.request(method, path, headers=gw.auth(scopes), **kw)
            statuses.append(r.status_code)
            codes.append(r.json().get("error", {}).get("code", "upstream"))
        return statuses, codes


async def test_denial_kind_is_not_an_oracle_beyond_volume(make_gateway: Any) -> None:
    """Whether a route/scope/tool/model exists must not change which of 403/404/429 the caller sees,
    nor after how many requests the 429 starts: every denial kind is the same 403 and costs the same."""
    results = {c: await _sequence(make_gateway, c) for c in _CASES}
    # an authorised request that the upstream answers 404 (a real, granted route) for comparison
    async with make_gateway(requests_per_window=6) as gw:
        gw.upstream.route().respond(404, json={"upstream": "not found"})
        granted = []
        for _ in range(16):
            r = await gw.client.get("/billing/invoices", headers=gw.auth("billing:read"))
            granted.append(r.status_code)
    first_429 = {c: s.index(429) for c, (s, _) in results.items()}
    assert len(set(first_429.values())) == 1, first_429
    assert granted.index(429) == next(iter(first_429.values()))  # same cost as a granted request
    for case, (statuses, codes) in results.items():
        i = first_429[case]
        assert set(statuses[:i]) == {403}, (case, statuses)  # never 404/400/413/200 for a denial
        assert set(codes[:i]) == {"access_denied"}, case
        assert set(statuses[i:]) == {429}, (case, statuses)  # and 429 is sticky for every kind
    assert set(granted[: granted.index(429)]) == {404}  # only an authorised call reaches the upstream


async def test_denial_bodies_and_headers_are_identical_across_kinds(make_gateway: Any) -> None:
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for case, (method, path, scopes, kw) in _CASES.items():
        async with make_gateway() as gw:
            r = await gw.client.request(method, path, headers=gw.auth(scopes), **kw)
            err = r.json()["error"]
            assert r.status_code == 403, case
            hdrs = tuple(sorted(k for k in r.headers if k not in ("content-length",)))
            seen.add((err["code"], err["message"], hdrs))
    assert len(seen) == 1, seen
