"""Graceful shutdown of the real `ai-gateway` process (subprocess, SIGTERM) and in-process lifespan."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_gateway.main import cli
from integration.servers import (
    ISSUER,
    LLM_SECRET,
    MCP_SECRET,
    OAI_SECRET,
    ThreadedServer,
    UpstreamState,
    free_port,
    write_gateway_config,
)


class Proc:
    def __init__(self, popen: subprocess.Popen[str], port: int, idp: Any) -> None:
        self.p, self.port, self.idp = popen, port, idp
        self.url = f"http://127.0.0.1:{port}"

    def headers(self) -> dict[str, str]:
        tok = self.idp.mint(issuer=ISSUER, client_id="shutdown-client", scopes="api:read")
        return {"Authorization": f"Bearer {tok}"}

    def output(self) -> str:
        return ""


@pytest.fixture
def gateway_proc(upstream: ThreadedServer, idp: Any, tmp_path: Path) -> Iterator[Proc]:
    port = free_port()
    cfg = write_gateway_config(tmp_path, upstream_url=upstream.url, port=port)
    env = {
        **os.environ,
        "INT_MCP_TOKEN": MCP_SECRET,
        "INT_LLM_KEY": LLM_SECRET,
        "INT_OAI_KEY": OAI_SECRET,
        "NO_PROXY": "127.0.0.1",
        "PYTHONUNBUFFERED": "1",
    }
    popen = subprocess.Popen(
        [sys.executable, "-m", "ai_gateway.main", "--config", str(cfg)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc = Proc(popen, port, idp)
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                if httpx.get(f"{proc.url}/healthz", timeout=1, trust_env=False).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            assert popen.poll() is None, (
                f"gateway exited early: {popen.stdout.read() if popen.stdout else ''}"
            )
            assert time.monotonic() < deadline, "gateway did not start"
            time.sleep(0.05)
        yield proc
    finally:
        if popen.poll() is None:
            popen.kill()
        popen.wait(10)
        if popen.stdout:
            popen.stdout.close()


async def _wait_exit(p: subprocess.Popen[str], timeout: float) -> int | None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        rc = p.poll()
        if rc is not None:
            return rc
        await asyncio.sleep(0.05)
    return None


CLEAN_EXIT = (0, -signal.SIGTERM)  # uvicorn re-raises the captured signal after a graceful shutdown


async def test_idle_gateway_exits_cleanly_on_sigterm(gateway_proc: Proc) -> None:
    gateway_proc.p.send_signal(signal.SIGTERM)
    assert await _wait_exit(gateway_proc.p, 10) in CLEAN_EXIT


async def test_in_flight_stream_finishes_before_exit_and_new_connections_are_refused(
    gateway_proc: Proc, upstream_state: UpstreamState
) -> None:
    async with httpx.AsyncClient(timeout=15, trust_env=False) as http:
        events = 0
        async with http.stream(
            "GET", f"{gateway_proc.url}/api/sse?n=12&interval=0.1", headers=gateway_proc.headers()
        ) as r:
            assert r.status_code == 200
            sent = False
            async for chunk in r.aiter_text():
                events += chunk.count("event: tick")
                if not sent:
                    gateway_proc.p.send_signal(signal.SIGTERM)  # shut down while the stream is mid-flight
                    sent = True
                    await asyncio.sleep(0.3)
                    async with httpx.AsyncClient(timeout=2, trust_env=False) as fresh:
                        with pytest.raises(httpx.TransportError):  # listener already closed
                            await fresh.get(f"{gateway_proc.url}/healthz")
                    assert gateway_proc.p.poll() is None  # still draining
        assert events == 12  # the stream was allowed to finish
    assert upstream_state.sse_completed == 1 and upstream_state.sse_aborted == 0
    assert await _wait_exit(gateway_proc.p, 10) in CLEAN_EXIT


async def test_endless_stream_does_not_block_shutdown_forever(
    gateway_proc: Proc, upstream_state: UpstreamState
) -> None:
    """A never-ending SSE connection must not keep the process alive indefinitely."""
    upstream_state.hold_open = True
    async with httpx.AsyncClient(timeout=15, trust_env=False) as http:
        async with http.stream("GET", f"{gateway_proc.url}/api/sse?n=1", headers=gateway_proc.headers()) as r:
            async for _ in r.aiter_bytes():
                break
            gateway_proc.p.send_signal(signal.SIGTERM)
            rc = await _wait_exit(gateway_proc.p, 8)
    assert rc in CLEAN_EXIT, (
        f"rc={rc}: process still running 8s after SIGTERM with an open endless SSE stream"
    )


def test_cli_reports_config_errors_with_exit_code_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as ei:
        cli(["--config", str(tmp_path / "missing.yaml")])
    assert ei.value.code == 2
    assert "configuration error" in capsys.readouterr().err


def test_cli_wires_host_port_and_factory_into_uvicorn(
    tmp_path: Path, upstream: ThreadedServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    cfg = write_gateway_config(tmp_path, upstream_url=upstream.url, port=9191)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app, **kw))
    monkeypatch.delenv("GATEWAY_CONFIG", raising=False)
    cli(["--config", str(cfg)])
    assert seen["app"] == "ai_gateway.app:create_app" and seen["factory"] is True
    assert (seen["host"], seen["port"]) == ("127.0.0.1", 9191)
    assert seen["log_config"] is None
    seen.clear()
    cli(["--config", str(cfg), "--host", "0.0.0.0", "--port", "1234"])
    assert (seen["host"], seen["port"]) == ("0.0.0.0", 1234)
    assert os.environ["GATEWAY_CONFIG"] == str(cfg)
    monkeypatch.delenv("GATEWAY_CONFIG")


def test_lifespan_shutdown_closes_services_and_warms_jwks(make_gw: Any) -> None:
    gw = make_gw()
    svc = gw.services
    assert svc.validator.ready  # warmed from the real IdP during startup
    assert svc._closed is False
    gw.server.stop()
    assert svc._closed is True and svc.upstream_http.is_closed and svc.jwks_http.is_closed
