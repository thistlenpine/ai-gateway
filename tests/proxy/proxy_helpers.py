"""Shared helpers for proxy tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from ai_gateway.config import DefaultsConfig, PolicyConfig, UpstreamConfig
from ai_gateway.models import (
    DecisionReason,
    PolicyDecision,
    Principal,
    RequestContext,
    RouteMatch,
    UpstreamKind,
)
from ai_gateway.observability.telemetry import NullTelemetry
from ai_gateway.proxy import Proxy, ProxyRequest, ProxyResponse

BASE = "https://up.internal:8443/api"


def upstream_cfg(kind: UpstreamKind = UpstreamKind.HTTP, **kw: Any) -> UpstreamConfig:
    if kind is UpstreamKind.LLM:
        kw.setdefault("provider", "anthropic")
    return UpstreamConfig(kind=kind, base_url=kw.pop("base_url", BASE), prefix=kw.pop("prefix", "/svc"), **kw)


class RecTelemetry(NullTelemetry):
    def __init__(self) -> None:
        super().__init__()
        self.upstream_calls: list[dict[str, Any]] = []

    def record_upstream(self, **kw: Any) -> None:
        self.upstream_calls.append(kw)


def make_proxy(
    handler: Callable[[httpx.Request], Any],
    cfg: UpstreamConfig | None = None,
    *,
    telemetry: RecTelemetry | None = None,
    **kw: Any,
) -> Proxy:
    cfg = cfg or upstream_cfg()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
    return Proxy(
        PolicyConfig(upstreams={"u": cfg}), client, telemetry or RecTelemetry(), DefaultsConfig(), **kw
    )


def route(kind: UpstreamKind = UpstreamKind.HTTP, rel: str = "/items", prefix: str = "/svc") -> RouteMatch:
    return RouteMatch("u", kind, prefix, rel)


def ctx(**kw: Any) -> RequestContext:
    c = RequestContext(request_id="req-12345678", method="GET", path="/svc/x")
    c.principal = Principal(
        client_id="client-a", subject="sub-1", issuer="https://idp", scopes=frozenset({"a"})
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def request(
    method: str = "GET",
    path: str = "/items",
    *,
    query: str = "",
    headers: list[tuple[str, str]] | None = None,
    body: bytes | AsyncIterator[bytes] | None = None,
    **ctx_kw: Any,
) -> ProxyRequest:
    return ProxyRequest(method, path, query, headers or [], body, ctx(**ctx_kw))


def decision(max_tokens: int | None = None) -> PolicyDecision:
    return PolicyDecision(True, DecisionReason.ALLOWED, max_tokens_per_request=max_tokens)


async def collect(resp: ProxyResponse) -> bytes:
    out = b""
    async for c in resp.body:
        out += c
    await resp.aclose()
    return out


async def chunks(*parts: bytes) -> AsyncIterator[bytes]:
    for p in parts:
        yield p


class AsyncStream(httpx.AsyncByteStream):
    def __init__(self, parts: list[bytes]) -> None:
        self.parts = parts

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for p in self.parts:
            yield p
