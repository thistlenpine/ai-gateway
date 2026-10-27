"""LLM provider adapter (kind=llm; Anthropic + OpenAI dialects). OWNER: proxy."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

from ai_gateway.config import UpstreamConfig
from ai_gateway.errors import BadRequestError
from ai_gateway.models import Resource, RouteMatch, UpstreamKind, UsageRecord
from ai_gateway.proxy._body import MAX_BUFFERED_BYTES, TapBody, with_body
from ai_gateway.proxy._json import parse_body, try_loads
from ai_gateway.proxy.base import ProxyRequest, ProxyResponse, UsageCallback
from ai_gateway.proxy.headers import get_header
from ai_gateway.proxy.streaming import SseParser

DEFAULT_MAX_TOKENS = 1024
_SKIP_KEYS = frozenset({"data", "image_url", "url", "file_data", "role", "type"})  # binary/base64 payloads
_CACHE_KEYS = ("cache_creation_input_tokens", "cache_read_input_tokens")


def _chars(node: Any) -> int:
    if isinstance(node, str):
        return len(node)
    if isinstance(node, list):
        return sum(_chars(n) for n in node)
    if isinstance(node, dict):
        return sum(_chars(v) for k, v in node.items() if k not in _SKIP_KEYS)
    return 0


def _positive_int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


class UsageAccumulator:
    """Folds usage objects from JSON bodies and SSE events (Anthropic, OpenAI chat/responses)."""

    def __init__(self) -> None:
        self.input: int | None = None
        self.output: int | None = None
        self.model: str | None = None

    def feed(self, obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        nested = [obj]
        for key in ("message", "response"):
            sub = obj.get(key)
            if isinstance(sub, dict):
                nested.append(sub)
        for node in nested:
            model = node.get("model")
            if isinstance(model, str) and model:
                self.model = model
            usage = node.get("usage")
            if isinstance(usage, dict):
                self._usage(usage)

    def _usage(self, usage: dict[str, Any]) -> None:
        inp = _int(usage.get("input_tokens"))
        if inp is None:
            inp = _int(usage.get("prompt_tokens"))
        out = _int(usage.get("output_tokens"))
        if out is None:
            out = _int(usage.get("completion_tokens"))
        if inp is not None:
            # Anthropic reports cached prompt tokens separately; they are still input.
            inp += sum(_int(usage.get(k)) or 0 for k in _CACHE_KEYS)
            self.input = inp
        if out is not None:
            self.output = out

    def feed_text(self, text: str | bytes) -> None:
        if text in ("", b"", "[DONE]"):
            return
        self.feed(try_loads(text))

    @property
    def seen(self) -> bool:
        return self.input is not None or self.output is not None

    def record(self) -> UsageRecord:
        return UsageRecord(
            input_tokens=self.input or 0,
            output_tokens=self.output or 0,
            model=self.model,
            estimated=not self.seen,
        )


class LlmAdapter:
    kind = UpstreamKind.LLM

    def classify(self, route: RouteMatch, method: str, body: bytes | None) -> list[Resource]:
        """POST with JSON body -> Resource(model=body["model"]). Non-POST (e.g. GET /v1/models)
        -> Resource(model=None) (authz grants must allow it explicitly via a models: ["*"] rule).
        Missing/non-string model on POST -> BadRequestError."""
        verb = method.upper()
        if verb != "POST":
            return [Resource(UpstreamKind.LLM, route.upstream, verb, route.relative_path)]
        doc = parse_body(body)
        model = doc.get("model") if isinstance(doc, dict) else None
        if not isinstance(model, str) or not model:
            raise BadRequestError("`model` is required")
        return [Resource(UpstreamKind.LLM, route.upstream, verb, route.relative_path, model=model)]

    def request_model(self, body: bytes | None) -> str | None:
        doc = try_loads(body) if body else None
        model = doc.get("model") if isinstance(doc, dict) else None
        return model if isinstance(model, str) else None

    def estimate_tokens(self, body: bytes | None, cap: int | None = None) -> int:
        """ceil(chars_in_messages_and_system/4) + (max_tokens | max_completion_tokens | 1024)."""
        doc = try_loads(body) if body else None
        if not isinstance(doc, dict):
            return 0
        chars = sum(_chars(doc.get(k)) for k in ("messages", "system", "prompt", "input", "tools"))
        max_out = (
            _positive_int(doc.get("max_tokens"))
            or _positive_int(doc.get("max_completion_tokens"))
            or _positive_int(doc.get("max_output_tokens"))
            or DEFAULT_MAX_TOKENS
        )
        if cap is not None:  # the upstream only ever sees the clamped value
            max_out = min(max_out, cap)
        return math.ceil(chars / 4) + max_out

    def prepare(self, upstream: UpstreamConfig, req: ProxyRequest) -> ProxyRequest:
        """Clamp max tokens to the grant cap (req.ctx.policy) and, for OpenAI streaming, inject
        stream_options.include_usage=true so the final chunk carries usage."""
        if not isinstance(req.body, bytes | bytearray) or req.method.upper() != "POST":
            return req
        doc = try_loads(bytes(req.body))
        if not isinstance(doc, dict):
            return req
        changed = False
        cap = req.ctx.policy.max_tokens_per_request if req.ctx.policy else None
        if cap is not None:
            present = [k for k in ("max_tokens", "max_completion_tokens", "max_output_tokens") if k in doc]
            if not present:
                present = ["max_tokens" if upstream.provider == "anthropic" else "max_completion_tokens"]
                doc[present[0]] = cap
                changed = True
            for key in present:
                val = _positive_int(doc.get(key))
                if val is None or val > cap:
                    doc[key] = cap
                    changed = True
        if upstream.provider == "openai" and doc.get("stream") is True:
            opts = doc.get("stream_options")
            if not isinstance(opts, dict):
                opts = {}
            if opts.get("include_usage") is not True:
                opts["include_usage"] = True
                doc["stream_options"] = opts
                changed = True
        if not changed:
            return req
        req.body = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode()
        return req

    def process_response(
        self,
        resp: ProxyResponse,
        *,
        tool_allowed: Callable[[str], bool] | None,
        on_usage: UsageCallback | None,
    ) -> ProxyResponse:
        """Tap usage (JSON `usage`, Anthropic message_start/message_delta, OpenAI final chunk)
        and await on_usage exactly once when the body finishes or is aborted. If the upstream
        reported no usage the record has zero tokens and estimated=True."""
        if on_usage is None:
            return resp
        acc = UsageAccumulator()
        on_chunk: Callable[[bytes], None] | None
        finalize: Callable[[], None] = lambda: None  # noqa: E731

        if resp.is_stream:
            parser = SseParser()

            def on_chunk(chunk: bytes) -> None:
                for ev in parser.feed(chunk):
                    acc.feed_text(ev.data)

            def finalize() -> None:
                for ev in parser.finish():
                    acc.feed_text(ev.data)

        elif "json" in (get_header(resp.headers, "content-type") or "").lower():
            buf = bytearray()
            overflow = False

            def on_chunk(chunk: bytes) -> None:
                nonlocal overflow
                if not overflow:
                    buf.extend(chunk)
                    if len(buf) > MAX_BUFFERED_BYTES:
                        overflow = True
                        buf.clear()

            def finalize() -> None:
                if buf and not overflow:
                    acc.feed_text(bytes(buf))

        else:
            on_chunk = None

        async def done() -> None:
            finalize()
            await on_usage(acc.record())

        return with_body(resp, TapBody(resp.body, on_chunk=on_chunk, on_done=done))
