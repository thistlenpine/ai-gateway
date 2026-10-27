"""Rig for load / concurrency tests: the full ASGI gateway in-process (httpx.ASGITransport), with
a mock upstream, a local IdP, a fake clock (or the real one) and a controllable `sleep`.

Unlike gw_rig, *every* unregistered client id gets `defaults.rate_limit`, so N synthetic clients all
share one small policy without having to be registered, and the store cap is configurable."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import respx
from fastapi import FastAPI

from ai_gateway.app import create_app
from ai_gateway.config import LimitPolicy
from ai_gateway.observability.telemetry import NullTelemetry
from ai_gateway.ratelimit import MemoryStateStore
from ai_gateway.services import Services, build_services
from gw_rig import LLM_SECRET, MCP_SECRET, make_config
from support import FakeClock, KeyPair

# limiter keys are namespaced by IdP name (gw_rig configures the IdP as "test"): rl:<idp>:<client_id>:...
IDP_NAME = "test"
INVOICES = "http://billing.internal/invoices"
LLM_URL = "http://llm.internal/v1/messages"

DEFAULT_LIMITS: dict[str, Any] = {
    "requests_per_window": 10,
    "window_seconds": 60,
    "delay_start": 0.5,
    "reject_at": 1.0,
    "block_at": 1.5,
    "block_sustained_seconds": 5,
    "block_duration_seconds": 60,
    "max_delay_ms": 1000,
    "max_concurrent_delayed": 10,
}


class CountingTelemetry(NullTelemetry):
    """NullTelemetry + the in-flight gauge, so tests can assert nothing leaks."""

    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.completed = 0

    def add_active(self, delta: int, upstream: str | None = None) -> None:
        self.active += delta

    def record_request(self, ctx: Any) -> None:
        self.completed += 1


@dataclass
class LoadGateway:
    app: FastAPI
    client: httpx.AsyncClient
    idp: KeyPair
    upstream: respx.Router
    clock: FakeClock
    services: Services
    tel: CountingTelemetry
    sleeping: int = 0  # delay sleeps currently parked on `gate`
    sleeps: list[float] = field(default_factory=list)
    gate: asyncio.Event = field(default_factory=asyncio.Event)
    _tokens: dict[tuple[str, str], str] = field(default_factory=dict)

    def headers(self, client_id: str, scopes: str = "billing:read llm:claude") -> dict[str, str]:
        key = (client_id, scopes)
        if key not in self._tokens:
            self._tokens[key] = self.idp.mint(scopes=scopes, client_id=client_id, ttl=3600)
        return {"Authorization": f"Bearer {self._tokens[key]}"}

    async def get(self, client_id: str, path: str = "/billing/invoices") -> httpx.Response:
        return await self.client.get(path, headers=self.headers(client_id))

    def slots(self, client_id: str) -> int:
        return sum(int(v) for k, (v, _) in self._data().items() if k == f"rl:{IDP_NAME}:{client_id}:delayq")

    def tok(self, client_id: str) -> int:
        return sum(v for k, (v, _) in self._data().items() if k.startswith(f"rl:{IDP_NAME}:{client_id}:tok:"))

    def _data(self) -> dict[str, tuple[int, float]]:
        store: MemoryStateStore = self.services.store  # type: ignore[assignment]
        return store._data


@asynccontextmanager
async def boot_load_gateway(
    *,
    limits: dict[str, Any] | None = None,
    real_time: bool = False,
    gated_sleep: bool = False,
    max_keys: int = 100_000,
    tokens_per_window: int | None = None,
) -> AsyncIterator[LoadGateway]:
    """`gated_sleep`: delay sleeps park on `gw.gate` (set it to release them) so tests can observe
    concurrent waiters. Otherwise sleeps return instantly (fake) or really sleep (`real_time`)."""
    import os

    os.environ.setdefault("E2E_MCP_TOKEN", MCP_SECRET)
    os.environ.setdefault("E2E_LLM_KEY", LLM_SECRET)
    cfg = make_config()
    merged = {**DEFAULT_LIMITS, **(limits or {})}
    if tokens_per_window is not None:
        merged["tokens_per_window"] = tokens_per_window
    cfg.gateway.defaults.rate_limit = LimitPolicy(**merged)
    cfg.gateway.telemetry.enabled = False
    cfg.gateway.telemetry.log_stdout_json = False

    clock: Callable[[], float] = time.time if real_time else FakeClock()
    store = MemoryStateStore(clock=clock, max_keys=max_keys)
    idp = KeyPair()
    idp_router = respx.Router(assert_all_called=False)
    idp_router.get("https://idp.test/jwks.json").respond(json=idp.jwks())
    upstream = respx.Router(assert_all_called=False)
    upstream.get(INVOICES).respond(json={"ok": True})
    tel = CountingTelemetry()
    holder: dict[str, LoadGateway] = {}

    async def sleep(s: float) -> None:
        gw = holder["gw"]
        gw.sleeps.append(s)
        if gated_sleep:
            gw.sleeping += 1
            try:
                await gw.gate.wait()
            finally:
                gw.sleeping -= 1
        elif real_time:
            await asyncio.sleep(s)

    services = build_services(
        cfg,
        telemetry=tel,
        store=store,
        jwks_http=httpx.AsyncClient(transport=httpx.MockTransport(idp_router.async_handler)),
        upstream_http=httpx.AsyncClient(transport=httpx.MockTransport(upstream.async_handler)),
        clock=clock,
        sleep=sleep,
    )
    app = create_app(services=services)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw.test", timeout=60) as c,
    ):
        gw = LoadGateway(
            app, c, idp, upstream, clock if isinstance(clock, FakeClock) else FakeClock(), services, tel
        )
        holder["gw"] = gw
        yield gw


class HangingStream(httpx.AsyncByteStream):
    """Upstream SSE body: sends `first`, then waits until released or closed."""

    def __init__(self, first: list[bytes]) -> None:
        self.first = first
        self.release = asyncio.Event()
        self.closed = asyncio.Event()
        self.sent = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.first:
            yield chunk
        self.sent.set()
        await self.release.wait()
        yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    async def aclose(self) -> None:
        self.closed.set()


def anthropic_json(inp: int = 10, out: int = 20) -> dict[str, Any]:
    return {"id": "m", "model": "claude-sonnet-4", "usage": {"input_tokens": inp, "output_tokens": out}}


def llm_body(*, max_tokens: int = 100, text: str = "hello world", stream: bool = False) -> bytes:
    return json.dumps(
        {"model": "claude-sonnet-4", "max_tokens": max_tokens, "stream": stream,
         "messages": [{"role": "user", "content": text}]}
    ).encode()  # fmt: skip


async def settle(times: int = 5) -> None:
    for _ in range(times):
        await asyncio.sleep(0)
