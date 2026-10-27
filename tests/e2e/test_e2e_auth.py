"""Authentication and authorization end to end."""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest

from ai_gateway.config import ConfigError
from ai_gateway.services import build_services
from gw_rig import make_config
from support import KeyPair


def _ok_billing(gw: Any) -> Any:
    return gw.upstream.get("http://billing.internal/invoices/42").respond(json={"id": 42})


async def test_valid_token_http_happy_path(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = _ok_billing(gw)
        tok = gw.token("billing:read")
        r = await gw.client.get(
            "/billing/invoices/42?expand=1",
            headers={"Authorization": f"Bearer {tok}", "X-Request-ID": "req-abcdef123"},
        )
        assert r.status_code == 200 and r.json() == {"id": 42}
        assert r.headers["x-request-id"] == "req-abcdef123"
        assert r.headers["ratelimit-limit"] == "10"
        assert "ratelimit-remaining" in r.headers and "ratelimit-reset" in r.headers
        sent = route.calls.last.request
        assert sent.url.query == b"expand=1"
        assert "authorization" not in sent.headers  # the caller's token is never forwarded
        assert sent.headers["x-gateway-client-id"] == "client-a"
        assert sent.headers["x-request-id"] == "req-abcdef123"


async def test_request_id_minted_when_missing_or_unsafe(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        r = await gw.client.get("/billing/x", headers={"X-Request-ID": "bad id!"})
        assert r.status_code == 401
        rid = r.headers["x-request-id"]
        assert rid != "bad id!" and len(rid) >= 8
        assert r.json()["error"]["request_id"] == rid


@pytest.mark.parametrize(
    "kind", ["missing", "scheme", "garbage", "expired", "wrong_issuer", "bad_sig", "aud"]
)
async def test_401_variants(make_gateway: Any, kind: str) -> None:
    async with make_gateway() as gw:
        _ok_billing(gw)
        other = KeyPair("k1")  # same kid, different key
        headers: dict[str, str] = {}
        if kind == "scheme":
            headers = {"Authorization": f"Basic {gw.token('billing:read')}"}
        elif kind == "garbage":
            headers = {"Authorization": "Bearer not.a.jwt"}
        elif kind == "expired":
            headers = gw.auth("billing:read", iat=int(time.time()) - 900, exp=int(time.time()) - 600)
        elif kind == "wrong_issuer":
            headers = gw.auth("billing:read", issuer="https://evil.test")
        elif kind == "bad_sig":
            headers = {"Authorization": f"Bearer {other.mint(scopes='billing:read')}"}
        elif kind == "aud":
            headers = gw.auth("billing:read", audience="someone-else")
        r = await gw.client.get("/billing/invoices/42", headers=headers)
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "invalid_token"
        assert r.headers["www-authenticate"].startswith("Bearer")
        assert not gw.upstream.calls
        events = gw.obs.audit("authn.failure")
        assert events and events[-1]["decision"] == "deny"
        for tok in gw._tokens:
            assert tok not in r.text


async def test_403_wrong_scope_uniform_body_reason_only_in_audit(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _ok_billing(gw)
        wrong_scope = await gw.client.post(
            "/billing/invoices", headers=gw.auth("billing:read"), content=b"{}"
        )
        no_route = await gw.client.get("/nope/x", headers=gw.auth("billing:read"))
        no_scope = await gw.client.get("/billing/invoices/42", headers=gw.auth("other:scope"))
        for r in (wrong_scope, no_route, no_scope):
            assert r.status_code == 403
        bodies = []
        for r in (wrong_scope, no_route, no_scope):
            err = r.json()["error"]
            bodies.append({k: v for k, v in err.items() if k != "request_id"})
        assert bodies[0] == bodies[1] == bodies[2] == {"code": "access_denied", "message": "Access denied"}
        assert "scope" not in wrong_scope.text.lower()
        assert not gw.upstream.calls
        events = gw.obs.audit("authz.decision")
        assert [e["reason"] for e in events] == ["no_matching_rule", "no_route", "insufficient_scope"]
        assert all(e["decision"] == "deny" for e in events)
        assert events[2]["scopes_missing_hint"] == ["billing:read"]


async def test_path_traversal_denied(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = _ok_billing(gw)
        for path in ("/billing/invoices/%2e%2e/%2e%2e/admin", "/billing/invoices%2f42", "/billing/a/../../x"):
            r = await gw.client.get(path, headers=gw.auth("billing:read"))
            assert r.status_code in (403, 400), path
        assert route.call_count == 0


async def test_upstream_error_is_generic_502(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.upstream.get("http://billing.internal/invoices/42").mock(side_effect=httpx.ConnectError("boom"))
        r = await gw.client.get("/billing/invoices/42", headers=gw.auth("billing:read"))
        assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_error"
        assert "boom" not in r.text and "billing.internal" not in r.text


async def test_unknown_client_denied_when_configured(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.services.config.gateway.defaults.unknown_clients = "deny"
        r = await gw.client.get("/billing/invoices/42", headers=gw.auth("billing:read", client_id="stranger"))
        assert r.status_code == 403 and r.json()["error"]["code"] == "access_denied"


def test_missing_upstream_credential_refuses_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("E2E_MCP_TOKEN", raising=False)
    monkeypatch.delenv("E2E_LLM_KEY", raising=False)
    with pytest.raises(ConfigError):
        build_services(make_config())


async def test_health_and_ready(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        assert (await gw.client.get("/healthz")).status_code == 200
        r = await gw.client.get("/readyz")  # warmed by lifespan
        assert r.status_code == 200 and r.json()["status"] == "ready"


async def test_oversized_body_rejected_before_policy_work(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.upstream.post("http://mcp.internal/mcp").respond(json={})
        big = b"x" * (gw.services.config.gateway.server.max_body_bytes + 1)
        r = await gw.client.post("/mcp/tools", headers=gw.auth("mcp:tools:read"), content=big)
        assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
        assert not gw.upstream.calls


async def test_unsupported_method_uniform_error(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        r = await gw.client.request("TRACE", "/billing/x", headers=gw.auth("billing:read"))
        assert r.status_code == 405 and r.json()["error"]["code"] == "method_not_allowed"


async def test_readyz_503_when_jwks_unreachable(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.services.validator._caches.clear()
        gw.services.validator._caches["x"] = _Cold()
        assert (await gw.client.get("/readyz")).status_code == 503


class _Cold:
    ready = False
