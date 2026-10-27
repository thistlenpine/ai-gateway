"""Real servers for integration tests: a combined IdP + upstream (HTTP/SSE, MCP, LLM) Starlette app and
helpers to run any ASGI app under a real uvicorn server on an ephemeral port in a background thread,
plus the gateway config written as YAML (the production loading path)."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from support import KeyPair

MCP_SECRET = "int-mcp-secret-value"
LLM_SECRET = "int-sk-ant-secret-value"
OAI_SECRET = "int-sk-openai-secret"
ISSUER = "https://idp.int.test"
AUDIENCE = "ai-gateway"
SESSION_ID = "sess-0123456789abcdef"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# --------------------------------------------------------------------------- upstream app


@dataclass
class Seen:
    method: str
    path: str
    query: str
    headers: dict[str, str]
    body: bytes


@dataclass
class UpstreamState:
    requests: list[Seen] = field(default_factory=list)
    sse_open: int = 0
    sse_completed: int = 0
    sse_aborted: int = 0
    tools_list_via_sse: bool = False
    llm_tokens: tuple[int, int] = (25, 40)  # (input, output) reported by the fake provider
    llm_delay_s: float = 0.0
    hold_open: bool = False  # SSE endpoint that never ends (for shutdown tests)

    def last(self, path_prefix: str) -> Seen:
        return next(r for r in reversed(self.requests) if r.path.startswith(path_prefix))

    def count(self, path_prefix: str) -> int:
        return sum(r.path.startswith(path_prefix) for r in self.requests)


def _sse(event: str | None, data: Any, eid: str | None = None) -> bytes:
    out = ""
    if eid:
        out += f"id: {eid}\n"
    if event:
        out += f"event: {event}\n"
    out += f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n"
    return out.encode()


MCP_TOOLS = ["get_weather", "search_docs", "delete_everything", "admin_reset"]


def build_upstream(idp: KeyPair, state: UpstreamState) -> Starlette:
    async def record(request: Request) -> bytes:
        body = await request.body()
        state.requests.append(
            Seen(request.method, request.url.path, request.url.query, dict(request.headers), body)
        )
        return body

    async def jwks(request: Request) -> Response:
        return JSONResponse(idp.jwks())

    async def echo(request: Request) -> Response:
        body = await record(request)
        return JSONResponse(
            {"method": request.method, "path": request.url.path, "query": request.url.query,
             "headers": dict(request.headers), "body_len": len(body)}
        )  # fmt: skip

    async def sse(request: Request) -> Response:
        await record(request)
        n = int(request.query_params.get("n", "5"))
        interval = float(request.query_params.get("interval", "0.1"))

        async def gen() -> AsyncIterator[bytes]:
            state.sse_open += 1
            finished = False
            try:
                for i in range(n):
                    yield _sse("tick", {"i": i, "t": time.monotonic()}, eid=str(i))
                    await asyncio.sleep(interval)
                if state.hold_open:
                    while True:
                        yield b": keepalive\n\n"
                        await asyncio.sleep(0.1)
                finished = True
            finally:
                state.sse_open -= 1
                if finished:
                    state.sse_completed += 1
                else:
                    state.sse_aborted += 1

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"cache-control": "no-cache"})

    async def chunked_upload(request: Request) -> Response:
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
        state.requests.append(Seen("POST", request.url.path, "", dict(request.headers), b""))
        return JSONResponse({"received": total})

    async def mcp(request: Request) -> Response:
        body = await record(request)
        sid = request.headers.get("mcp-session-id")
        if request.method == "DELETE":
            return Response(status_code=200 if sid == SESSION_ID else 404)
        if request.method == "GET":
            last = request.headers.get("last-event-id", "")

            async def stream() -> AsyncIterator[bytes]:
                start = int(last) + 1 if last.isdigit() else 0
                for i in range(start, start + 3):
                    yield _sse("message", {"jsonrpc": "2.0", "method": "notifications/message",
                                           "params": {"n": i, "last": last}}, eid=str(i))  # fmt: skip
                    await asyncio.sleep(0.02)

            return StreamingResponse(stream(), media_type="text/event-stream")
        msg = json.loads(body)
        if isinstance(msg, list):
            return JSONResponse(
                [{"jsonrpc": "2.0", "id": m.get("id"), "result": {}} for m in msg if "id" in m]
            )
        method, rid = msg.get("method"), msg.get("id")
        if rid is None:  # notification
            return Response(status_code=202)
        if method == "initialize":
            return JSONResponse(
                {"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}},
                                                         "serverInfo": {"name": "fake", "version": "1"}}},
                headers={"Mcp-Session-Id": SESSION_ID},
            )  # fmt: skip
        if sid != SESSION_ID:
            return JSONResponse(
                {"jsonrpc": "2.0", "id": rid, "error": {"code": -32600, "message": "no session"}},
                status_code=400,
            )
        if method == "tools/list":
            payload = {"jsonrpc": "2.0", "id": rid,
                       "result": {"tools": [{"name": t, "description": t} for t in MCP_TOOLS]}}  # fmt: skip
            if state.tools_list_via_sse:
                return StreamingResponse(
                    iter([_sse("message", payload, eid="0")]), media_type="text/event-stream"
                )
            return JSONResponse(payload)
        if method == "tools/call":
            return JSONResponse({"jsonrpc": "2.0", "id": rid,
                                 "result": {"content": [{"type": "text", "text": f"ran {msg['params']['name']}"}]}})  # fmt: skip
        return JSONResponse({"jsonrpc": "2.0", "id": rid, "result": {}})

    async def anthropic(request: Request) -> Response:
        body = await record(request)
        doc = json.loads(body)
        inp, out = state.llm_tokens
        if state.llm_delay_s:
            await asyncio.sleep(state.llm_delay_s)
        if not doc.get("stream"):
            return JSONResponse(
                {"id": "msg_1", "model": doc["model"], "usage": {"input_tokens": inp, "output_tokens": out}}
            )

        async def gen() -> AsyncIterator[bytes]:
            state.sse_open += 1
            done = False
            try:
                yield _sse("message_start", {"type": "message_start", "message": {"id": "msg_1", "model": doc["model"],
                                                                                  "usage": {"input_tokens": inp, "output_tokens": 1}}})  # fmt: skip
                for i in range(out // 10):
                    yield _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                       "delta": {"type": "text_delta", "text": f"t{i} "}})  # fmt: skip
                    await asyncio.sleep(0.02)
                yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                                             "usage": {"output_tokens": out}})  # fmt: skip
                yield _sse("message_stop", {"type": "message_stop"})
                done = True
            finally:
                state.sse_open -= 1
                state.sse_completed += done
                state.sse_aborted += not done

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def openai(request: Request) -> Response:
        body = await record(request)
        doc = json.loads(body)
        inp, out = state.llm_tokens
        include = (doc.get("stream_options") or {}).get("include_usage") is True
        if not doc.get("stream"):
            return JSONResponse(
                {"id": "c1", "model": doc["model"], "usage": {"prompt_tokens": inp, "completion_tokens": out}}
            )

        async def gen() -> AsyncIterator[bytes]:
            for i in range(3):
                yield _sse(
                    None, {"id": "c1", "model": doc["model"], "choices": [{"delta": {"content": f"w{i}"}}]}
                )
                await asyncio.sleep(0.01)
            if include:  # real OpenAI only sends usage when asked
                yield _sse(None, {"id": "c1", "model": doc["model"], "choices": [],
                                  "usage": {"prompt_tokens": inp, "completion_tokens": out}})  # fmt: skip
            yield _sse(None, "[DONE]")

        return StreamingResponse(gen(), media_type="text/event-stream")

    return Starlette(
        routes=[
            Route("/idp/jwks.json", jwks),
            Route("/api/echo", echo, methods=["GET", "POST", "PUT", "DELETE"]),
            Route("/api/echo/{rest:path}", echo, methods=["GET", "POST", "PUT", "DELETE"]),
            Route("/api/sse", sse),
            Route("/api/upload", chunked_upload, methods=["POST"]),
            Route("/mcp", mcp, methods=["GET", "POST", "DELETE"]),
            Route("/mcp/", mcp, methods=["GET", "POST", "DELETE"]),  # the gateway forwards "<base>/"
            Route("/anthropic/v1/messages", anthropic, methods=["POST"]),
            Route("/openai/v1/chat/completions", openai, methods=["POST"]),
        ]
    )


# --------------------------------------------------------------------------- uvicorn in a thread


class ThreadedServer:
    """Runs an ASGI app under a real uvicorn server on an ephemeral 127.0.0.1 port."""

    def __init__(self, app: Any, **config: Any) -> None:
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=0, log_config=None, lifespan="on", **config)
        )
        self._thread = threading.Thread(target=self._run, name="uvicorn-test", daemon=True)
        self.port = 0
        self.error: BaseException | None = None

    def _run(self) -> None:
        try:
            asyncio.run(self.server.serve())
        except BaseException as exc:
            self.error = exc

    def __enter__(self) -> ThreadedServer:
        self._thread.start()
        deadline = time.monotonic() + 15
        while not self.server.started:
            if self.error or time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError(f"server failed to start: {self.error!r}")
            time.sleep(0.01)
        self.port = self.server.servers[0].sockets[0].getsockname()[1]
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def stop(self, timeout: float = 15) -> None:
        self.server.should_exit = True
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "server thread did not stop"

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


# --------------------------------------------------------------------------- gateway config (YAML)


def write_gateway_config(
    root: Path,
    *,
    upstream_url: str,
    port: int = 0,
    limits: dict[str, Any] | None = None,
    clients: dict[str, Any] | None = None,
    max_body_bytes: int = 2_000_000,
) -> Path:
    import yaml

    (root / "policies").mkdir(parents=True, exist_ok=True)
    gateway = {
        "server": {"host": "127.0.0.1", "port": port, "max_body_bytes": max_body_bytes},
        "idps": [{"name": "int", "issuer": ISSUER, "jwks_uri": f"{upstream_url}/idp/jwks.json", "audiences": [AUDIENCE]}],
        "telemetry": {"enabled": False, "log_stdout_json": False},
        "defaults": {
            "upstream_timeout_s": 10,
            "rate_limit": {"requests_per_window": 1000, **(limits or {})},
        },
    }  # fmt: skip
    policy = {
        "upstreams": {
            "api": {"kind": "http", "base_url": f"{upstream_url}/api", "prefix": "/api"},
            "tools": {"kind": "mcp", "base_url": f"{upstream_url}/mcp", "prefix": "/mcp/tools",
                      "credential": {"env": "INT_MCP_TOKEN"}},
            "claude": {"kind": "llm", "provider": "anthropic", "base_url": f"{upstream_url}/anthropic",
                       "prefix": "/llm/anthropic",
                       "credential": {"env": "INT_LLM_KEY", "header": "x-api-key", "scheme": None}},
            "gpt": {"kind": "llm", "provider": "openai", "base_url": f"{upstream_url}/openai",
                    "prefix": "/llm/openai", "credential": {"env": "INT_OAI_KEY"}},
        },
        "rules": [
            {"id": "api-read", "scopes_any": ["api:read"],
             "http": [{"upstream": "api", "methods": ["GET", "HEAD"], "paths": ["/**"]}]},
            {"id": "api-write", "scopes_any": ["api:write"],
             "http": [{"upstream": "api", "methods": ["POST", "PUT", "DELETE"], "paths": ["/**"]}]},
            {"id": "mcp-ro", "scopes_any": ["mcp:ro"],
             "mcp": [{"upstream": "tools", "tools": ["get_*", "search_*"],
                      "methods": ["initialize", "notifications/*", "ping", "tools/list"]}]},
            {"id": "llm", "scopes_any": ["llm"],
             "llm": [{"upstream": "claude", "models": ["claude-*"], "max_tokens_per_request": 500},
                     {"upstream": "gpt", "models": ["gpt-*"]}]},
        ],
    }  # fmt: skip
    (root / "gateway.yaml").write_text(yaml.safe_dump(gateway))
    (root / "policies" / "p.yaml").write_text(yaml.safe_dump(policy))
    (root / "clients.yaml").write_text(yaml.safe_dump({"clients": clients or {}}))
    return root / "gateway.yaml"
