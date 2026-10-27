"""Helpers for the adversarial security suite (OWNER: security validation).

Everything here only builds attacker inputs; the assertions live in the test modules. Tests are
written to PASS when the gateway is secure. A test that currently FAILS because of a real defect
is marked `pytest.mark.xfail(strict=True, reason="SEC-nn ...")` so the finding stays executable
and the suite turns red (XPASS strict) the moment the defect is fixed.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from ai_gateway.auth import TokenValidator
from ai_gateway.errors import AuthenticationError, AuthFailure
from ai_gateway.observability.telemetry import NullTelemetry

ISSUER = "https://idp.test"
ISSUER_B = "https://other-idp.test"
JWKS_B = "https://other-idp.test/jwks.json"
MCP = "http://mcp.internal/mcp"
LLM = "http://llm.internal/v1/messages"
BILLING = "http://billing.internal"


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def forge(
    header: dict[str, Any], payload: dict[str, Any], *, secret: bytes | None = None, sig: bytes = b""
) -> str:
    """Hand-assemble a compact JWS (no library checks). HS256 signature if `secret` is given."""
    h = b64(json.dumps(header).encode())
    p = b64(json.dumps(payload).encode())
    if secret is not None:
        sig = hmac.new(secret, f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{b64(sig)}"


def forge_raw(header_json: str, payload_json: str, sig: bytes = b"sig") -> str:
    return f"{b64(header_json.encode())}.{b64(payload_json.encode())}.{b64(sig)}"


def claims(**over: Any) -> dict[str, Any]:
    now = int(time.time())
    base: dict[str, Any] = {
        "iss": ISSUER,
        "aud": "ai-gateway",
        "sub": "client-a",
        "client_id": "client-a",
        "scope": "billing:read",
        "iat": now,
        "nbf": now,
        "exp": now + 300,
    }
    base.update(over)
    return {k: v for k, v in base.items() if v is not None}


def rsa_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def rpc(method: str, params: dict[str, Any] | None = None, *, id_: int | None = 1) -> dict[str, Any]:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if id_ is not None:
        msg["id"] = id_
    if params is not None:
        msg["params"] = params
    return msg


def call(name: str) -> dict[str, Any]:
    return rpc("tools/call", {"name": name, "arguments": {}})


def mcp_ok(gw: Any, result: Any = None) -> Any:
    return gw.upstream.post(MCP).respond(json={"jsonrpc": "2.0", "id": 1, "result": result or {"ok": True}})


async def post_mcp(gw: Any, body: Any, scopes: str = "mcp:tools:read", **kw: Any) -> httpx.Response:
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    headers = {**gw.auth(scopes), "Content-Type": "application/json", **kw.pop("headers", {})}
    resp: httpx.Response = await gw.client.post(
        kw.pop("path", "/mcp/tools"), headers=headers, content=content, **kw
    )
    return resp


def only_failure(failure: AuthFailure | None = None) -> Any:
    """Context manager: the block must raise AuthenticationError (optionally a specific one)."""
    return _Fails(failure)


class _Fails:
    def __init__(self, failure: AuthFailure | None) -> None:
        self.failure = failure

    def __enter__(self) -> _Fails:
        self._ctx = pytest.raises(AuthenticationError)
        self.info = self._ctx.__enter__()
        return self

    def __exit__(self, *exc: Any) -> bool:
        suppressed = bool(self._ctx.__exit__(*exc))
        if suppressed and self.failure is not None:
            assert self.info.value.failure is self.failure
        return suppressed


class CountingNull(NullTelemetry):
    def __init__(self) -> None:
        super().__init__()
        self.failures: list[tuple[str, str | None]] = []

    def record_auth_failure(self, reason: str, *, issuer: str | None = None) -> None:
        self.failures.append((reason, issuer))


def validator(idps: list[Any], http: httpx.AsyncClient, **kw: Any) -> TokenValidator:
    return TokenValidator(idps, http, CountingNull(), **kw)


# ------------------------------------------------------------------ raw ASGI client


class RawResponse:
    def __init__(self, status: int, headers: list[tuple[bytes, bytes]], body: bytes) -> None:
        self.status_code = status
        self.raw_headers = headers
        self.content = body

    @property
    def headers(self) -> dict[str, str]:
        return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in self.raw_headers}

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", "replace")

    def json(self) -> Any:
        return json.loads(self.content)


async def raw_request(
    app: Any,
    method: str,
    raw_path: bytes | str,
    *,
    query: bytes | str = b"",
    headers: Sequence[tuple[bytes | str, bytes | str]] | None = None,
    body: bytes | list[bytes] = b"",
    client: tuple[str, int] = ("203.0.113.9", 40000),
) -> RawResponse:
    """Drive the ASGI app with a hand-built scope so no HTTP client library can 'fix' the request
    (dot-segment collapsing, header folding, CRLF stripping, duplicate headers, odd bytes...)."""
    from urllib.parse import unquote_to_bytes

    def b(x: bytes | str) -> bytes:
        return x if isinstance(x, bytes) else x.encode("latin-1")

    rp = b(raw_path)
    hdrs = [(b(k).lower(), b(v)) for k, v in (headers or [])]
    if not any(k == b"host" for k, _ in hdrs):
        hdrs.append((b"host", b"gateway.test"))
    chunks = body if isinstance(body, list) else [body]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": unquote_to_bytes(rp).decode("latin-1"),
        "raw_path": rp,
        "query_string": b(query),
        "root_path": "",
        "headers": hdrs,
        "client": client,
        "server": ("gateway.test", 80),
        "state": {},
    }
    pending = list(chunks)
    out: dict[str, Any] = {"status": 0, "headers": [], "body": b""}

    async def receive() -> dict[str, Any]:
        if pending:
            data = pending.pop(0)
            return {"type": "http.request", "body": data, "more_body": bool(pending)}
        return {"type": "http.disconnect"}

    async def send(msg: dict[str, Any]) -> None:
        if msg["type"] == "http.response.start":
            out["status"], out["headers"] = msg["status"], list(msg["headers"])
        elif msg["type"] == "http.response.body":
            out["body"] += msg.get("body", b"")

    await app(scope, receive, send)
    return RawResponse(out["status"], out["headers"], out["body"])
