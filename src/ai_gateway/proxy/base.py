"""Proxy data types + adapter protocol. OWNER: proxy."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from ai_gateway.config import UpstreamConfig
from ai_gateway.models import RequestContext, Resource, RouteMatch, UpstreamKind, UsageRecord

UsageCallback = Callable[[UsageRecord], Awaitable[None]]


@dataclass(slots=True)
class ProxyRequest:
    method: str
    path: str  # NORMALIZED upstream-relative path (same string authz saw)
    query: str  # raw query string without '?'
    headers: list[tuple[str, str]]  # inbound headers (adapter/client filter hop-by-hop + auth)
    body: bytes | AsyncIterator[bytes] | None  # bytes for mcp/llm (already buffered), stream for http
    ctx: RequestContext


@dataclass(slots=True)
class ProxyResponse:
    status: int
    headers: list[tuple[str, str]]
    body: AsyncIterator[bytes]  # always an async iterator; may be a single chunk
    is_stream: bool = False  # text/event-stream: no buffering, disable compression, flush per chunk
    _close: Callable[[], Awaitable[None]] | None = field(default=None, repr=False)

    async def aclose(self) -> None:
        """Release the upstream connection (idempotent). Must be called when client disconnects."""
        if self._close is not None:
            close, self._close = self._close, None
            await close()


class Adapter(Protocol):
    """Protocol-specific behaviour. One instance per UpstreamKind."""

    kind: UpstreamKind

    def classify(self, route: RouteMatch, method: str, body: bytes | None) -> list[Resource]:
        """Parse what the request touches for authz. HTTP: one Resource (no body read).
        MCP: JSON-RPC message(s) -> one Resource per message (tools/call -> tool name).
        LLM: `model` from JSON body. Unparsable/missing -> raise BadRequestError."""
        ...

    def estimate_tokens(self, body: bytes | None, cap: int | None = None) -> int:
        """LLM only: pre-flight cost = ceil(len(prompt_chars)/4) + max_tokens (or model default
        cap). Other kinds return 0."""
        ...

    def prepare(self, upstream: UpstreamConfig, req: ProxyRequest) -> ProxyRequest:
        """Rewrite the request just before sending: e.g. LLM streaming OpenAI -> inject
        stream_options.include_usage=true; cap max_tokens to grant limit."""
        ...

    def process_response(
        self,
        resp: ProxyResponse,
        *,
        tool_allowed: Callable[[str], bool] | None,
        on_usage: UsageCallback | None,
    ) -> ProxyResponse:
        """Wrap the response. MCP: filter `tools/list` results (JSON or SSE events) with
        tool_allowed. LLM: tap usage (JSON `usage`, or Anthropic message_start/message_delta,
        OpenAI final chunk) and await on_usage exactly once when the body finishes/aborts."""
        ...
