"""In-process gateway rig for e2e tests (OWNER: core). Boots the whole gateway in-process:

client --ASGITransport--> create_app(services) -> mock upstreams (httpx.MockTransport + respx
router) and a local test IdP (KeyPair + respx-served JWKS). Rate limiting runs on a fake clock;
telemetry goes to in-memory OTel exporters.
"""

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import respx
from fastapi import FastAPI
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ai_gateway.app import create_app
from ai_gateway.config import (
    ClientConfig,
    ClientsConfig,
    DefaultsConfig,
    GatewayConfig,
    HttpGrant,
    IdPConfig,
    LimitPolicy,
    LlmGrant,
    LoadedConfig,
    McpGrant,
    PolicyConfig,
    Rule,
    TelemetryConfig,
    UpstreamConfig,
    UpstreamCredential,
)
from ai_gateway.models import UpstreamKind
from ai_gateway.observability.otel import OtelTelemetry
from ai_gateway.ratelimit import MemoryStateStore
from ai_gateway.services import Services, build_services
from support import FakeClock, KeyPair

ISSUER = "https://idp.test"
JWKS_URL = "https://idp.test/jwks.json"
MCP_SECRET = "mcp-upstream-secret-value"
LLM_SECRET = "sk-ant-upstream-secret-value"


def make_config(**limits: Any) -> LoadedConfig:
    """Gateway config used by the e2e suite. `client-a` has small, easily-exceeded limits."""
    policy = PolicyConfig(
        upstreams={
            "billing": UpstreamConfig(
                kind=UpstreamKind.HTTP, base_url="http://billing.internal", prefix="/billing"
            ),
            "tools": UpstreamConfig(
                kind=UpstreamKind.MCP,
                base_url="http://mcp.internal/mcp",
                prefix="/mcp/tools",
                credential=UpstreamCredential(env="E2E_MCP_TOKEN"),
            ),
            "anthropic": UpstreamConfig(
                kind=UpstreamKind.LLM,
                provider="anthropic",
                base_url="http://llm.internal",
                prefix="/llm/anthropic",
                credential=UpstreamCredential(env="E2E_LLM_KEY", header="x-api-key", scheme=None),
            ),
        },
        rules=[
            Rule(
                id="billing-read",
                scopes_any=["billing:read"],
                http=[
                    HttpGrant(upstream="billing", methods=["GET", "HEAD"], paths=["/invoices", "/invoices/*"])
                ],
            ),
            Rule(
                id="mcp-read",
                scopes_any=["mcp:tools:read"],
                mcp=[McpGrant(upstream="tools", tools=["search_*", "get_*"])],
            ),
            Rule(
                id="llm-claude",
                scopes_any=["llm:claude"],
                llm=[LlmGrant(upstream="anthropic", models=["claude-*"], max_tokens_per_request=8192)],
            ),
        ],
    )
    client_a = {
        "requests_per_window": 10,
        "window_seconds": 60,
        "delay_start": 0.5,
        "reject_at": 1.0,
        "block_at": 1.5,
        "block_sustained_seconds": 5,
        "block_duration_seconds": 60,
        "max_delay_ms": 1000,
        "max_concurrent_delayed": 10,
        **limits,
    }
    clients = ClientsConfig(
        clients={
            "client-a": ClientConfig(limits=client_a),
            "client-tokens": ClientConfig(limits={"tokens_per_window": 200, "token_window_seconds": 60}),
        }
    )
    gateway = GatewayConfig(
        idps=[IdPConfig(name="test", issuer=ISSUER, jwks_uri=JWKS_URL, audiences=["ai-gateway"])],
        telemetry=TelemetryConfig(),
        defaults=DefaultsConfig(rate_limit=LimitPolicy(requests_per_window=1000)),
    )
    return LoadedConfig(gateway, policy, clients)


@dataclass
class Telemetry:
    tel: OtelTelemetry
    spans: InMemorySpanExporter
    metrics: InMemoryMetricReader
    logs: InMemoryLogRecordExporter
    stdout: io.StringIO

    def metric(self, name: str) -> list[tuple[dict[str, Any], Any]]:
        data = self.metrics.get_metrics_data()
        out: list[tuple[dict[str, Any], Any]] = []
        for rm in data.resource_metrics if data else []:
            for sm in rm.scope_metrics:
                for m in sm.metrics:
                    if m.name == name:
                        out.extend((dict(p.attributes or {}), p) for p in m.data.data_points)
        return out

    def span_names(self) -> list[str]:
        return [s.name for s in self.spans.get_finished_spans()]

    def audit(self, event_type: str | None = None) -> list[dict[str, Any]]:
        import json

        out = []
        for line in self.stdout.getvalue().splitlines():
            if line.startswith("{") and '"stream": "audit"' in line.replace('":"', '": "'):
                rec = json.loads(line)
                if rec.get("stream") == "audit" and (event_type is None or rec.get("event") == event_type):
                    out.append(rec)
        return out

    def everything(self) -> str:
        parts = [self.stdout.getvalue()]
        parts += [s.to_json() for s in self.spans.get_finished_spans()]
        for ld in self.logs.get_finished_logs():
            parts.append(repr(dict(ld.log_record.attributes or {})) + repr(ld.log_record.body))
        return "\n".join(parts)


@dataclass
class Gateway:
    app: FastAPI
    client: httpx.AsyncClient
    idp: KeyPair
    idp_router: respx.Router
    upstream: respx.Router
    clock: FakeClock
    sleeps: list[float]
    services: Services
    obs: Telemetry
    _tokens: list[str] = field(default_factory=list)

    def token(self, scopes: str = "", client_id: str = "client-a", **kw: Any) -> str:
        tok = self.idp.mint(scopes=scopes, client_id=client_id, **kw)
        self._tokens.append(tok)
        return tok

    def auth(self, scopes: str = "", client_id: str = "client-a", **kw: Any) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token(scopes, client_id, **kw)}"}


@asynccontextmanager
async def boot_gateway(**limits: Any) -> AsyncIterator[Gateway]:
    cfg = make_config(**limits)
    spans = InMemorySpanExporter()
    reader = InMemoryMetricReader()
    logs = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]
    out = io.StringIO()
    tel = OtelTelemetry(
        cfg.gateway.telemetry, span_exporter=spans, metric_reader=reader, log_exporter=logs, stdout=out
    )
    idp = KeyPair()
    idp_router = respx.Router(assert_all_called=False)
    idp_router.get(JWKS_URL).respond(json=idp.jwks())
    upstream = respx.Router(assert_all_called=False)
    clock = FakeClock()
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    services = build_services(
        cfg,
        telemetry=tel,
        store=MemoryStateStore(clock=clock),
        jwks_http=httpx.AsyncClient(transport=httpx.MockTransport(idp_router.async_handler)),
        upstream_http=httpx.AsyncClient(
            transport=httpx.MockTransport(upstream.async_handler), follow_redirects=False
        ),
        clock=clock,
        sleep=fake_sleep,
    )
    app = create_app(services=services)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as client,
    ):
        yield Gateway(
            app, client, idp, idp_router, upstream, clock, sleeps, services,
            Telemetry(tel, spans, reader, logs, out),
        )  # fmt: skip
