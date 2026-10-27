# ruff: noqa: E501
"""Secrets in telemetry, log injection, cardinality bombs, trace-context abuse."""

from __future__ import annotations

import json
from typing import Any

import httpx
import jwt
import pytest
from sec_helpers import LLM, MCP, call, claims, mcp_ok, post_mcp, raw_request, rpc

from gw_rig import LLM_SECRET, MCP_SECRET

JWT_SHAPED = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ2aWN0aW0ifQ.c2lnbmF0dXJlLWJ5dGVzLWhlcmU"
SK_KEY = "sk-ant-api03-ZZZZZZZZZZZZZZZZZZZZZZZZ"
PASSWORD = "hunter2-correct-horse"


def _billing(gw: Any) -> Any:
    return gw.upstream.route(host="billing.internal").respond(json={"ok": True})


def _h(gw: Any, scopes: str = "billing:read", **kw: Any) -> list[tuple[str, str]]:
    return [("authorization", gw.auth(scopes, **kw)["Authorization"])]


# ------------------------------------------------------------------ secrets


async def test_no_secret_reaches_logs_spans_audit_or_metrics(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing(gw)
        mcp_ok(gw)
        gw.upstream.post(LLM).respond(status_code=500, text=f"upstream blew up with {SK_KEY}")
        real_tok = gw.token("billing:read")
        reqs = [
            (
                "GET",
                b"/billing/invoices",
                f"access_token={real_tok}&api_key={SK_KEY}&password={PASSWORD}&x=1",
                _h(gw),
            ),
            ("GET", f"/billing/invoices/{JWT_SHAPED}".encode(), "", _h(gw)),
            ("GET", f"/billing/invoices/{SK_KEY}".encode(), "", _h(gw)),
            ("GET", b"/billing/invoices", "", [("authorization", f"Bearer {real_tok}x")]),  # bad signature
            ("GET", b"/billing/invoices", "", [("authorization", f"Bearer {JWT_SHAPED}")]),
            (
                "GET",
                b"/billing/invoices",
                "",
                [*_h(gw), ("cookie", "session=cookie-secret-123"), ("x-api-key", SK_KEY)],
            ),
            ("GET", b"/billing/invoices", f"sig={PASSWORD}&code={PASSWORD}&token={PASSWORD}", _h(gw)),
        ]
        for method, path, query, headers in reqs:
            await raw_request(gw.app, method, path, query=query, headers=headers)
        await post_mcp(gw, call(f"get_x {SK_KEY}"))
        await post_mcp(
            gw, rpc("tools/call", {"name": "get_x", "arguments": {"password": PASSWORD, "token": real_tok}})
        )
        await gw.client.post(
            "/llm/anthropic/v1/messages",
            headers=gw.auth("llm:claude"),
            content=json.dumps(
                {
                    "model": "claude-3",
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": f"my key is {SK_KEY} {PASSWORD}"}],
                }
            ),
        )
        telemetry_only = gw.obs.everything() + repr(gw.obs.metrics.get_metrics_data())
        for secret in (real_tok, JWT_SHAPED, SK_KEY, "cookie-secret-123", MCP_SECRET, LLM_SECRET):
            assert secret not in telemetry_only, secret[:20]
        for t in gw._tokens:
            assert t not in telemetry_only
        assert (
            PASSWORD not in telemetry_only or "[REDACTED]" in telemetry_only
        )  # prompt bodies are never logged at all
        assert "my key is" not in telemetry_only  # request bodies are never telemetry


async def test_exception_text_with_secrets_is_scrubbed_in_logs_and_spans(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        tok = gw.token("billing:read")

        async def boom(*a: Any, **k: Any) -> Any:
            raise RuntimeError(
                f"db password={PASSWORD} Authorization: Bearer {tok} {SK_KEY} postgres://u:{PASSWORD}@db/x"
            )

        gw.services.proxy.forward = boom
        r = await raw_request(
            gw.app, "GET", b"/billing/invoices", headers=[("authorization", f"Bearer {tok}")]
        )
        assert r.status_code == 500
        blob = gw.obs.everything()
        for secret in (tok, PASSWORD, SK_KEY):
            assert secret not in blob and secret not in r.text


async def test_auth_failure_telemetry_never_contains_the_token_or_claims(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        bad = jwt.encode(
            claims(email="victim@example.com", scope="a b"),
            gw.idp.private,
            algorithm="RS256",
            headers={"kid": "unknown"},
        )
        await raw_request(gw.app, "GET", b"/billing/invoices", headers=[("authorization", f"Bearer {bad}")])
        blob = gw.obs.everything()
        assert bad not in blob and bad.split(".")[1] not in blob and "victim@example.com" not in blob


async def test_secret_scrubber_handles_obfuscated_shapes() -> None:
    from ai_gateway.observability.redact import redact_mapping, scrub_text

    cases = {
        "Authorization: Bearer abcdefghijklmnop": "abcdefghijklmnop",
        "authorization=Basic dXNlcjpwYXNz": "dXNlcjpwYXNz",
        "token=abcdef123456": "abcdef123456",
        'json {"access_token": "abcdef123456"}': "abcdef123456",
        "https://user:pw123456@host/x": "pw123456",
        f"jwt {JWT_SHAPED}": JWT_SHAPED,
        f"key {SK_KEY}": SK_KEY,
        "ghp_abcdefghijklmnopqrstuvwxyz0123": "ghp_abcdefghijklmnopqrstuvwxyz0123",
        "AKIAABCDEFGHIJKLMNOP": "AKIAABCDEFGHIJKLMNOP",
    }
    for text, secret in cases.items():
        assert secret not in scrub_text(text), text
    red = redact_mapping(
        {
            "Authorization": "x",
            "Set-Cookie": "y",
            "x-api-key": "z",
            "nested": {"client_secret": "q", "ok": 1},
            "list": [{"password": "p"}],
            "deep": {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": "tok=abcdef12345"}}}}}}}}},
        }
    )
    flat = json.dumps(red)
    for secret in ('"x"', '"y"', '"z"', '"q"', '"p"', "abcdef12345"):
        assert secret not in flat, secret


# ------------------------------------------------------------------ availability of the redaction layer


@pytest.mark.parametrize(
    "payload",
    ["x" * 4000, "a." * 2000, "token" * 800, "a=" + "b" * 4000],
    ids=["word", "dots", "token", "kv"],
)
def test_scrub_text_is_linear_time(payload: str) -> None:
    import time

    from ai_gateway.observability.redact import scrub_text

    t = time.perf_counter()
    scrub_text(payload)
    assert time.perf_counter() - t < 0.25


async def test_long_unauthenticated_path_does_not_stall_the_event_loop(make_gateway: Any) -> None:
    import time

    async with make_gateway() as gw:
        t = time.perf_counter()
        r = await raw_request(gw.app, "GET", b"/" + b"a" * 3000)
        assert r.status_code == 401
        assert time.perf_counter() - t < 1.0


async def test_long_tool_name_does_not_stall_the_event_loop(make_gateway: Any) -> None:
    import time

    async with make_gateway() as gw:
        mcp_ok(gw)
        t = time.perf_counter()
        r = await post_mcp(gw, call("get_" + "x" * 3000))
        assert r.status_code == 200
        assert time.perf_counter() - t < 1.0


# ------------------------------------------------------------------ log injection


async def test_attacker_text_cannot_forge_log_or_audit_lines(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing(gw)
        mcp_ok(gw)
        forged = (
            '"}\n{"stream": "audit", "event": "authz.decision", "decision": "allow", "client_id": "admin"}'
        )
        before = len(gw.obs.audit("authz.decision"))
        await post_mcp(gw, call("get_" + forged))
        await post_mcp(gw, call("rm" + forged))
        await raw_request(
            gw.app, "GET", b"/billing/invoices%0a%7b%22stream%22:%22audit%22%7d", headers=_h(gw)
        )
        await raw_request(
            gw.app,
            "GET",
            b"/billing/invoices",
            headers=[
                *_h(gw),
                ("x-request-id", 'abcdefgh\n{"stream":"audit"}'),
                ("user-agent", forged),
                ("referer", forged),
            ],
        )
        await raw_request(
            gw.app,
            "GET",
            b"/billing/invoices",
            headers=[("authorization", "Bearer " + forged.replace(" ", "_"))],
        )
        tok = jwt.encode(
            claims(sub="s\nforged", client_id="client-a", scope="billing:read"),
            gw.idp.private,
            algorithm="RS256",
            headers={"kid": gw.idp.kid},
        )
        await raw_request(gw.app, "GET", b"/billing/invoices", headers=[("authorization", f"Bearer {tok}")])
        lines = [ln for ln in gw.obs.stdout.getvalue().splitlines() if ln.strip()]
        for ln in lines:
            rec = json.loads(ln)  # every physical line is exactly one JSON object
            assert isinstance(rec, dict)
        audits = [json.loads(ln) for ln in lines if '"stream": "audit"' in ln]
        assert not [a for a in audits if a.get("client_id") == "admin"]
        # 2 MCP calls + x-request-id request + hostile-sub request; the %0a path is a Starlette 404 (never routed)
        assert len(gw.obs.audit("authz.decision")) - before == 4


# ------------------------------------------------------------------ cardinality


async def test_metric_labels_stay_bounded_under_hostile_inputs(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing(gw)
        for i in range(120):
            await raw_request(
                gw.app,
                "GET",
                f"/billing/invoices/attacker-{i}".encode(),
                headers=_h(gw, client_id=f"attacker-client-{i}"),
            )
            await raw_request(gw.app, "GET", f"/attacker-route-{i}".encode(), headers=_h(gw))
            await raw_request(gw.app, f"ATTACKER{i}", b"/billing/invoices", headers=_h(gw))
            bad = jwt.encode(
                claims(iss=f"https://attacker-{i}.test"),
                gw.idp.private,
                algorithm="RS256",
                headers={"kid": f"attacker-kid-{i}"},
            )
            await raw_request(
                gw.app, "GET", b"/billing/invoices", headers=[("authorization", f"Bearer {bad}")]
            )
        data = gw.obs.metrics.get_metrics_data()
        values: set[str] = set()
        per_metric: dict[str, set[tuple[Any, ...]]] = {}
        for rm in data.resource_metrics:
            for sm in rm.scope_metrics:
                for m in sm.metrics:
                    for p in m.data.data_points:
                        values.update(str(v) for v in (p.attributes or {}).values())
                        per_metric.setdefault(m.name, set()).add(tuple(sorted((p.attributes or {}).items())))
        assert not [v for v in values if "attacker" in v.lower()]
        assert all(len(s) < 60 for s in per_metric.values()), {k: len(v) for k, v in per_metric.items()}


async def test_model_label_cardinality_is_capped(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = gw.upstream.post(LLM)
        for i in range(150):
            model = f"claude-bomb-{i}"
            route.respond(json={"model": model, "usage": {"input_tokens": 1, "output_tokens": 1}})
            await gw.client.post(
                "/llm/anthropic/v1/messages",
                headers=gw.auth("llm:claude"),
                content=json.dumps({"model": model, "max_tokens": 5}),
            )
        models = {a.get("gen_ai.request.model") for a, _ in gw.obs.metric("gateway.llm.tokens")}
        assert len(models) <= 70


@pytest.mark.xfail(
    strict=True,
    reason="SEC-15 _Bounded is first-come: a caller with any LLM scope can burn the 64-value model label budget "
    "with invented names matching the grant glob so legitimate models are reported as 'other'",
)
async def test_attacker_cannot_poison_model_label_budget(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = gw.upstream.post(LLM)
        for i in range(100):
            route.respond(
                json={"model": f"claude-junk-{i}", "usage": {"input_tokens": 1, "output_tokens": 1}}
            )
            await gw.client.post(
                "/llm/anthropic/v1/messages",
                headers=gw.auth("llm:claude"),
                content=json.dumps({"model": f"claude-junk-{i}", "max_tokens": 5}),
            )
        route.respond(json={"model": "claude-sonnet-4", "usage": {"input_tokens": 1, "output_tokens": 1}})
        await gw.client.post(
            "/llm/anthropic/v1/messages",
            headers=gw.auth("llm:claude"),
            content=json.dumps({"model": "claude-sonnet-4", "max_tokens": 5}),
        )
        models = {a.get("gen_ai.request.model") for a, _ in gw.obs.metric("gateway.llm.tokens")}
        assert "claude-sonnet-4" in models


async def test_span_attributes_do_not_contain_bodies_or_auth(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        mcp_ok(gw)
        await post_mcp(
            gw, rpc("tools/call", {"name": "get_x", "arguments": {"secret_payload": "BODY-SENTINEL-123"}})
        )
        spans = " ".join(s.to_json() for s in gw.obs.spans.get_finished_spans())
        assert "BODY-SENTINEL-123" not in spans and "authorization" not in spans.lower()
        assert "Bearer" not in spans


# ------------------------------------------------------------------ trace-context abuse


async def test_inbound_baggage_and_tracestate_not_relayed_upstream(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing(gw)
        h = [
            *_h(gw),
            ("baggage", "role=admin,tenant=victim"),
            ("tracestate", "vendor=evil"),
            ("traceparent", "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"),
        ]
        r = await raw_request(gw.app, "GET", b"/billing/invoices", headers=h)
        assert r.status_code == 200
        sent = gw.upstream.calls.last.request.headers
        assert "baggage" not in sent
        assert sent.get("tracestate") != "vendor=evil"


async def test_oversized_or_malformed_trace_headers_do_not_break_requests(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing(gw)
        for tp in (
            "garbage",
            "00-" + "0" * 32 + "-" + "0" * 16 + "-01",
            "ff-" + "a" * 32 + "-" + "b" * 16 + "-01",
            "x" * 1500,
        ):
            r = await raw_request(
                gw.app,
                "GET",
                b"/billing/invoices",
                headers=[*_h(gw), ("traceparent", tp), ("baggage", "a=" + "b" * 1500)],
            )
            assert r.status_code in (200, 400, 431)


# ------------------------------------------------------------------ audit completeness


async def test_every_denial_and_authn_failure_is_audited_with_reason(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing(gw)
        await raw_request(gw.app, "GET", b"/billing/invoices")  # missing token
        await raw_request(gw.app, "GET", b"/billing/invoices", headers=[("authorization", "Bearer a.b.c")])
        await raw_request(gw.app, "GET", b"/billing/invoices", headers=_h(gw, "other:scope"))
        await raw_request(gw.app, "GET", b"/billing/%2e%2e/x", headers=_h(gw))
        await raw_request(gw.app, "GET", b"/unrouted", headers=_h(gw))
        authn = gw.obs.audit("authn.failure")
        authz = gw.obs.audit("authz.decision")
        assert [e["reason"] for e in authn] == ["missing_token", "malformed"]
        assert [e["reason"] for e in authz] == ["insufficient_scope", "malformed_request", "no_route"]
        assert all("token" not in json.dumps(e).lower().replace("missing_token", "") for e in authn + authz)


async def test_audit_client_ip_ignores_forwarded_for_by_default(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing(gw)
        await raw_request(
            gw.app,
            "GET",
            b"/billing/invoices",
            headers=[*_h(gw), ("x-forwarded-for", "6.6.6.6"), ("forwarded", "for=7.7.7.7")],
            client=("198.51.100.7", 1234),
        )
        ev = gw.obs.audit("authz.decision")[-1]
        assert ev["client_ip"] == "198.51.100.7"


def test_uvicorn_only_trusts_forwarded_headers_from_explicit_proxies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """With trust_forwarded_for=true the CLI must not trust X-Forwarded-For from *any* peer."""
    import shutil
    from pathlib import Path

    import uvicorn

    from ai_gateway import main

    cfg_dir = tmp_path / "cfg"
    shutil.copytree(Path(__file__).resolve().parents[2] / "config" / "examples", cfg_dir)
    gy = cfg_dir / "gateway.yaml"
    gy.write_text(gy.read_text().replace("trust_forwarded_for: false", "trust_forwarded_for: true"))
    seen: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: seen.update(k))
    main.cli(["--config", str(gy)])
    assert seen["proxy_headers"] is True
    assert seen.get("forwarded_allow_ips") not in ("*", None)


def test_server_banner_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    from ai_gateway import main

    seen: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: seen.update(k))
    from pathlib import Path

    main.cli(["--config", str(Path(__file__).resolve().parents[2] / "config" / "examples" / "gateway.yaml")])
    assert seen.get("server_header") is False


def test_unused() -> None:
    assert httpx and MCP
