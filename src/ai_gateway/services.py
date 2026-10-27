"""Dependency container. OWNER: core. Built once at startup; passed to the pipeline."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

from ai_gateway.auth import TokenValidator
from ai_gateway.config import LoadedConfig
from ai_gateway.observability import Telemetry, setup_telemetry
from ai_gateway.policy import PolicyEngine
from ai_gateway.proxy import Proxy, build_http_client
from ai_gateway.ratelimit import MemoryStateStore, RateLimiter, StateStore

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Services:
    config: LoadedConfig
    telemetry: Telemetry
    validator: TokenValidator
    policy: PolicyEngine
    store: StateStore
    limiter: RateLimiter
    proxy: Proxy
    jwks_http: httpx.AsyncClient  # used only for IdP/JWKS calls
    upstream_http: httpx.AsyncClient  # used only for upstream calls
    _closed: bool = field(default=False, repr=False)

    async def aclose(self) -> None:
        """Close http clients and the store; telemetry.shutdown() last (flush). Idempotent."""
        if self._closed:
            return
        self._closed = True
        for name, closer in (
            ("upstream_http", self.upstream_http.aclose),
            ("jwks_http", self.jwks_http.aclose),
            ("store", self.store.aclose),
        ):
            try:
                await closer()
            except Exception:
                log.exception("shutdown step failed", extra={"event": "services.close_error", "step": name})
        try:
            await self.telemetry.shutdown()
        except Exception:
            log.exception("telemetry shutdown failed", extra={"event": "services.close_error"})


def build_services(
    config: LoadedConfig,
    *,
    telemetry: Telemetry | None = None,
    store: StateStore | None = None,
    jwks_http: httpx.AsyncClient | None = None,
    upstream_http: httpx.AsyncClient | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Services:
    """Wire everything. Every argument override exists for tests (mock-transport clients,
    in-memory telemetry, fake-clock store). `clock`/`sleep` feed the rate limiter only.
    Defaults: setup_telemetry(cfg.telemetry), MemoryStateStore, build_http_client(...).
    Calls proxy.validate_credentials() (ConfigError -> process refuses to start)."""
    gw = config.gateway
    tel = telemetry if telemetry is not None else setup_telemetry(gw.telemetry)
    state = store if store is not None else MemoryStateStore()
    jwks = jwks_http if jwks_http is not None else build_http_client(gw.defaults)
    up = upstream_http if upstream_http is not None else build_http_client(gw.defaults)

    validator = TokenValidator(gw.idps, jwks, tel)
    policy = PolicyEngine(config.policy)
    limiter = RateLimiter(state, tel, clock=clock, sleep=sleep, on_error=gw.defaults.on_limiter_error)
    proxy = Proxy(config.policy, up, tel, gw.defaults, max_request_bytes=gw.server.max_body_bytes)
    try:
        proxy.validate_credentials()
    except BaseException:
        # Startup is aborting: do not leave exporters/threads (real OTLP) lingering.
        if telemetry is None:
            _shutdown_quietly(tel)
        raise
    return Services(
        config=config,
        telemetry=tel,
        validator=validator,
        policy=policy,
        store=state,
        limiter=limiter,
        proxy=proxy,
        jwks_http=jwks,
        upstream_http=up,
    )


_BG: set[asyncio.Task[None]] = set()


def _shutdown_quietly(tel: Telemetry) -> None:
    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(tel.shutdown())
            return
        task = asyncio.get_running_loop().create_task(tel.shutdown())
        _BG.add(task)
        task.add_done_callback(_BG.discard)
    except Exception:
        log.exception("telemetry shutdown failed", extra={"event": "services.close_error"})
