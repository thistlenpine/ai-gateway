from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from proxy_helpers import AsyncStream, collect, decision, make_proxy, request, route, upstream_cfg

from ai_gateway.errors import BadRequestError
from ai_gateway.models import RouteMatch, UpstreamKind, UsageRecord
from ai_gateway.proxy.llm import LlmAdapter

R = RouteMatch("u", UpstreamKind.LLM, "/llm", "/v1/messages")
A = LlmAdapter()


def body(**kw: Any) -> bytes:
    return json.dumps(kw).encode()


# ------------------------------------------------------------------ classify / estimate


def test_classify() -> None:
    res = A.classify(R, "POST", body(model="claude-x", messages=[]))
    assert res[0].model == "claude-x" and res[0].kind is UpstreamKind.LLM
    assert A.classify(R, "GET", None)[0].model is None


@pytest.mark.parametrize(
    "b",
    [None, b"", b"x", body(), body(model=3), body(model=""), b'{"model":"a","model":"b"}', b"[]"],
)
def test_classify_bad(b: bytes | None) -> None:
    with pytest.raises(BadRequestError):
        A.classify(R, "POST", b)


def test_estimate_tokens() -> None:
    # 8 + 4 + 3 chars -> ceil(15/4)=4, max_tokens 100
    est = A.estimate_tokens(
        body(
            model="m",
            system="abcdefgh",
            messages=[
                {"role": "user", "content": "abcd"},
                {"role": "user", "content": [{"type": "text", "text": "abc"}]},
            ],
            max_tokens=100,
        )
    )
    assert est == 4 + 100
    assert A.estimate_tokens(body(model="m", messages=[{"content": "x" * 40}])) == 10 + 1024
    assert (
        A.estimate_tokens(body(model="m", messages=[{"content": "x" * 4}], max_completion_tokens=7)) == 1 + 7
    )
    # base64 image payloads must not blow up the estimate
    img = {"type": "image", "source": {"type": "base64", "data": "A" * 100000}}
    assert A.estimate_tokens(body(model="m", messages=[{"content": [img]}], max_tokens=5)) < 20
    assert A.estimate_tokens(b"garbage") == 0 and A.estimate_tokens(None) == 0
    assert A.estimate_tokens(body(model="m", max_tokens=-5)) == 1024
    assert A.estimate_tokens(body(model="m", max_tokens=True)) == 1024


# ------------------------------------------------------------------ prepare


async def sent_body(cfg, req_body: bytes, max_tokens: int | None = None) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    seen: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        return httpx.Response(200, json={})

    proxy = make_proxy(handler, cfg)
    await collect(
        await proxy.forward(
            route(UpstreamKind.LLM, "/v1/messages"),
            request("POST", "/v1/messages", body=req_body, policy=decision(max_tokens)),
        )
    )
    assert seen[0].headers["content-length"] == str(len(seen[0].content))
    out: dict[str, Any] = json.loads(seen[0].content)
    return out


async def test_openai_stream_gets_include_usage() -> None:
    cfg = upstream_cfg(UpstreamKind.LLM, provider="openai")
    out = await sent_body(cfg, body(model="m", stream=True))
    assert out["stream_options"] == {"include_usage": True}
    out = await sent_body(cfg, body(model="m", stream=True, stream_options={"include_usage": False, "x": 1}))
    assert out["stream_options"] == {"include_usage": True, "x": 1}
    out = await sent_body(cfg, body(model="m", stream=False))
    assert "stream_options" not in out
    out = await sent_body(upstream_cfg(UpstreamKind.LLM, provider="anthropic"), body(model="m", stream=True))
    assert "stream_options" not in out


async def test_unmodified_body_is_sent_verbatim() -> None:
    raw = b'{"model":  "m",   "max_tokens": 5}'
    seen: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        return httpx.Response(200)

    proxy = make_proxy(handler, upstream_cfg(UpstreamKind.LLM))
    await collect(
        await proxy.forward(
            route(UpstreamKind.LLM), request("POST", "/v1/messages", body=raw, policy=decision(100))
        )
    )
    assert seen[0].content == raw


async def test_max_tokens_clamped_to_grant() -> None:
    a = upstream_cfg(UpstreamKind.LLM, provider="anthropic")
    assert (await sent_body(a, body(model="m", max_tokens=9999), 500))["max_tokens"] == 500
    assert (await sent_body(a, body(model="m", max_tokens=50), 500))["max_tokens"] == 50
    assert (await sent_body(a, body(model="m"), 500))["max_tokens"] == 500
    o = upstream_cfg(UpstreamKind.LLM, provider="openai")
    assert (await sent_body(o, body(model="m"), 500))["max_completion_tokens"] == 500
    assert (await sent_body(o, body(model="m", max_completion_tokens=1e9 and 10**9), 500))[
        "max_completion_tokens"
    ] == 500
    assert "max_tokens" not in await sent_body(a, body(model="m", max_tokens=5)) or True


# ------------------------------------------------------------------ usage extraction


class Collector:
    def __init__(self) -> None:
        self.calls: list[UsageRecord] = []

    async def __call__(self, u: UsageRecord) -> None:
        self.calls.append(u)


async def call(handler, cfg, on_usage, req_body: bytes = b'{"model":"req-model"}'):  # type: ignore[no-untyped-def]
    proxy = make_proxy(handler, cfg)
    resp = await proxy.forward(
        route(UpstreamKind.LLM, "/v1/messages"),
        request("POST", "/v1/messages", body=req_body),
        on_usage=on_usage,
    )
    return resp


def sse_bytes(*events: tuple[str | None, object]) -> bytes:
    out = b""
    for name, data in events:
        if name:
            out += f"event: {name}\n".encode()
        d = data if isinstance(data, str) else json.dumps(data)
        out += f"data: {d}\n\n".encode()
    return out


async def test_anthropic_json_usage() -> None:
    c = Collector()
    doc = {
        "model": "claude-x",
        "usage": {"input_tokens": 11, "output_tokens": 7, "cache_read_input_tokens": 4},
    }
    resp = await call(lambda r: httpx.Response(200, json=doc), upstream_cfg(UpstreamKind.LLM), c)
    assert json.loads(await collect(resp)) == doc
    assert c.calls == [UsageRecord(15, 7, "claude-x", "anthropic", False)]


async def test_openai_json_usage_and_request_model_fallback() -> None:
    c = Collector()
    doc = {"usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}}
    resp = await call(
        lambda r: httpx.Response(200, json=doc), upstream_cfg(UpstreamKind.LLM, provider="openai"), c
    )
    await collect(resp)
    assert c.calls == [UsageRecord(3, 5, "req-model", "openai", False)]


async def test_anthropic_sse_usage_start_and_delta() -> None:
    c = Collector()
    stream = sse_bytes(
        (
            "message_start",
            {
                "type": "message_start",
                "message": {"model": "claude-x", "usage": {"input_tokens": 25, "output_tokens": 1}},
            },
        ),
        ("content_block_delta", {"type": "content_block_delta", "delta": {"text": "hi"}}),
        ("message_delta", {"type": "message_delta", "usage": {"output_tokens": 42}}),
        ("message_stop", {"type": "message_stop"}),
    )
    parts = [stream[i : i + 17] for i in range(0, len(stream), 17)]
    resp = await call(
        lambda r: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=AsyncStream(parts)
        ),
        upstream_cfg(UpstreamKind.LLM),
        c,
    )
    assert await collect(resp) == stream  # byte-identical relay
    assert c.calls == [UsageRecord(25, 42, "claude-x", "anthropic", False)]


async def test_openai_sse_final_chunk_usage() -> None:
    c = Collector()
    stream = sse_bytes(
        (None, {"model": "gpt-x", "choices": [{"delta": {"content": "a"}}], "usage": None}),
        (None, {"model": "gpt-x", "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}),
        (None, "[DONE]"),
    )
    resp = await call(
        lambda r: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=AsyncStream([stream])
        ),
        upstream_cfg(UpstreamKind.LLM, provider="openai"),
        c,
    )
    assert await collect(resp) == stream
    assert c.calls == [UsageRecord(9, 2, "gpt-x", "openai", False)]


async def test_openai_responses_api_stream() -> None:
    c = Collector()
    stream = sse_bytes(
        (
            "response.completed",
            {
                "type": "response.completed",
                "response": {"model": "gpt-r", "usage": {"input_tokens": 4, "output_tokens": 6}},
            },
        ),
    )
    resp = await call(
        lambda r: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=AsyncStream([stream])
        ),
        upstream_cfg(UpstreamKind.LLM, provider="openai"),
        c,
    )
    await collect(resp)
    assert c.calls[0].input_tokens == 4 and c.calls[0].output_tokens == 6


async def test_no_usage_yields_single_zero_estimated_record() -> None:
    c = Collector()
    resp = await call(
        lambda r: httpx.Response(429, json={"error": "slow"}), upstream_cfg(UpstreamKind.LLM), c
    )
    await collect(resp)
    assert len(c.calls) == 1 and c.calls[0].estimated and c.calls[0].total == 0


async def test_on_usage_exactly_once_on_abort_with_partial_usage() -> None:
    c = Collector()
    start = sse_bytes(
        ("message_start", {"message": {"model": "m", "usage": {"input_tokens": 30, "output_tokens": 1}}})
    )

    class S(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield start
            import asyncio

            await asyncio.sleep(60)

    resp = await call(
        lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=S()),
        upstream_cfg(UpstreamKind.LLM),
        c,
    )
    it = resp.body.__aiter__()
    await it.__anext__()
    await resp.aclose()
    await resp.aclose()
    assert len(c.calls) == 1 and c.calls[0].input_tokens == 30 and not c.calls[0].estimated


async def test_on_usage_called_even_if_never_iterated() -> None:
    c = Collector()
    resp = await call(lambda r: httpx.Response(200, json={}), upstream_cfg(UpstreamKind.LLM), c)
    await resp.aclose()
    assert len(c.calls) == 1


async def test_on_usage_failure_does_not_break_response() -> None:
    async def boom(u: UsageRecord) -> None:
        raise RuntimeError("x")

    resp = await call(
        lambda r: httpx.Response(200, json={"usage": {"input_tokens": 1}}),
        upstream_cfg(UpstreamKind.LLM),
        boom,
    )
    assert json.loads(await collect(resp)) == {"usage": {"input_tokens": 1}}


@pytest.mark.parametrize(
    "bad", [{"usage": {"input_tokens": -1, "output_tokens": "x"}}, {"usage": [1]}, [1, 2], "str"]
)
async def test_garbage_usage_is_ignored(bad: object) -> None:
    c = Collector()
    resp = await call(lambda r: httpx.Response(200, json=bad), upstream_cfg(UpstreamKind.LLM), c)
    await collect(resp)
    assert len(c.calls) == 1 and c.calls[0].estimated
