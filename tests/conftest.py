"""Shared pytest fixtures (OWNER: core). The in-process gateway rig lives in gw_rig."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractAsyncContextManager

import pytest

from ai_gateway.observability.logging import shutdown_logging
from ai_gateway.observability.otel import uninstrument
from gw_rig import LLM_SECRET, MCP_SECRET, Gateway, boot_gateway


@pytest.fixture
def rig_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("E2E_MCP_TOKEN", MCP_SECRET)
    monkeypatch.setenv("E2E_LLM_KEY", LLM_SECRET)


@pytest.fixture
def make_gateway(rig_env: None) -> Iterator[Callable[..., AbstractAsyncContextManager[Gateway]]]:
    """Factory: `async with make_gateway(**limit_overrides) as gw`."""
    try:
        yield boot_gateway
    finally:
        shutdown_logging()
        uninstrument()
