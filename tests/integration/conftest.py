"""Integration fixtures: real upstream server + real gateway server, both under uvicorn."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_gateway.app import create_app
from ai_gateway.observability.logging import shutdown_logging
from ai_gateway.observability.otel import uninstrument
from integration.servers import (
    AUDIENCE,
    ISSUER,
    LLM_SECRET,
    MCP_SECRET,
    OAI_SECRET,
    ThreadedServer,
    UpstreamState,
    build_upstream,
    write_gateway_config,
)
from support import KeyPair


@pytest.fixture
def secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INT_MCP_TOKEN", MCP_SECRET)
    monkeypatch.setenv("INT_LLM_KEY", LLM_SECRET)
    monkeypatch.setenv("INT_OAI_KEY", OAI_SECRET)


@pytest.fixture
def idp() -> KeyPair:
    return KeyPair("int-k1")


@pytest.fixture
def upstream_state() -> UpstreamState:
    return UpstreamState()


@pytest.fixture
def upstream(idp: KeyPair, upstream_state: UpstreamState) -> Iterator[ThreadedServer]:
    with ThreadedServer(build_upstream(idp, upstream_state)) as srv:
        yield srv


class Gw:
    def __init__(self, server: ThreadedServer, idp: KeyPair, cfg: Path) -> None:
        self.server, self.idp, self.cfg = server, idp, cfg
        self.url = server.url
        self.app: Any = server.server.config.app

    def headers(
        self, scopes: str = "api:read", client_id: str = "int-client", **extra: Any
    ) -> dict[str, str]:
        tok = self.idp.mint(issuer=ISSUER, audience=AUDIENCE, client_id=client_id, scopes=scopes, **extra)
        return {"Authorization": f"Bearer {tok}"}

    @property
    def services(self) -> Any:
        return self.app.state.services

    def store_keys(self, prefix: str) -> dict[str, int]:
        # limiter keys are namespaced by IdP name: rl:<idp>:<client_id>:<kind>
        scoped = prefix.replace("rl:", "rl:int:", 1)
        return {k: v[0] for k, v in self.services.store._data.items() if k.startswith(scoped)}


@pytest.fixture
def make_gw(
    secrets: None, upstream: ThreadedServer, idp: KeyPair, tmp_path: Path
) -> Iterator[Callable[..., Gw]]:
    started: list[ThreadedServer] = []

    def make(**cfg_kw: Any) -> Gw:
        cfg = write_gateway_config(tmp_path, upstream_url=upstream.url, **cfg_kw)
        app = create_app(str(cfg))  # the production path: YAML -> services -> app
        srv = ThreadedServer(app).__enter__()
        started.append(srv)
        return Gw(srv, idp, cfg)

    try:
        yield make
    finally:
        for s in started:
            s.stop()
        shutdown_logging()
        uninstrument()


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=15, trust_env=False) as c:
        yield c


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    assert os.environ["NO_PROXY"]
