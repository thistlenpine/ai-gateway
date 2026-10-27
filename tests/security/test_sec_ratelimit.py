"""Rate-limit evasion / DoS attempts. Passing = the limiter holds."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import jwt
from sec_helpers import LLM, claims, mcp_ok, post_mcp, raw_request, rpc

from ai_gateway.ratelimit import MemoryStateStore


def _billing(gw: Any) -> Any:
    return gw.upstream.route(host="billing.internal").respond(json={"ok": True})


async def _get(gw: Any, headers: dict[str, str], path: str = "/billing/invoices") -> Any:
    return await gw.client.get(path, headers=headers)


# ------------------------------------------------------------------ identity spoofing


async def test_spoofed_identity_headers_do_not_change_the_bucket(make_gateway: Any) -> None:
    async with make_gateway() as gw:  # client-a: 10 requests / window
        _billing(gw)
        statuses = []
        for i in range(14):
            h = {
                **gw.auth("billing:read"),
                "X-Client-Id": f"c{i}",
                "X-Gateway-Client-Id": f"c{i}",
                "X-Forwarded-For": f"10.0.0.{i}",
                "X-Real-IP": f"10.0.1.{i}",
                "X-Api-Key": f"k{i}",
            }
            statuses.append((await _get(gw, h)).status_code)
        assert statuses[:10].count(200) == 10
        assert set(statuses[10:]) == {429}


async def test_many_tokens_for_the_same_client_share_one_bucket(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing(gw)
        statuses = [(await _get(gw, gw.auth("billing:read", jti=f"jti-{i}"))).status_code for i in range(14)]
        assert statuses.count(200) == 10 and statuses[-1] == 429


async def test_client_id_claim_precedence_cannot_be_used_to_hop_buckets(make_gateway: Any) -> None:
    """Same registered client: client_id claim wins over azp/sub, so varying sub/azp is useless."""
    async with make_gateway() as gw:
        _billing(gw)
        got = []
        for i in range(14):
            tok = jwt.encode(
                claims(client_id="client-a", sub="client-a", azp=f"app-{i}", scope="billing:read"),
                gw.idp.private,
                algorithm="RS256",
                headers={"kid": gw.idp.kid},
            )
            got.append((await _get(gw, {"Authorization": f"Bearer {tok}"})).status_code)
        assert got.count(200) == 10 and got[-1] == 429


async def test_path_method_and_query_variation_do_not_reset_counters(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing(gw)
        h = gw.auth("billing:read")
        paths = [
            "/billing/invoices",
            "/billing/invoices/1",
            "/billing//invoices",
            "/billing/invoices?x=1",
            "/billing/invoices/./",
            "/nope",
            "/billing/invoices/%41",
        ]
        codes = []
        for i in range(30):
            codes.append((await _get(gw, h, paths[i % len(paths)])).status_code)
        assert 429 in codes  # the limiter saw all of them as one client
        r = await _get(gw, h, "/billing/invoices/2")
        assert r.status_code in (429, 403)


async def test_blocked_client_is_blocked_on_every_path_and_before_body_parsing(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing(gw)
        mcp_ok(gw)
        h = gw.auth("billing:read mcp:tools:read")
        for _ in range(16):  # 160% of quota
            await _get(gw, h)
        gw.clock.advance(6)  # block_sustained_seconds=5
        for _ in range(16):
            await _get(gw, h)
        for path in ("/billing/invoices", "/nope", "/mcp/tools", "/billing/invoices/../x"):
            r = await _get(gw, h, path)
            assert r.status_code == 429 and r.json()["error"]["code"] == "client_blocked", path
        big = b"x" * (gw.services.config.gateway.server.max_body_bytes + 10)
        r = await gw.client.post("/mcp/tools", headers=h, content=big)
        assert r.status_code == 429  # not 413: no body was read
        assert int(r.headers["retry-after"]) >= 1
        gw.clock.advance(200)  # block (60s) and both counter buckets have expired
        assert (await _get(gw, gw.auth("billing:read"))).status_code == 200  # cooldown ends


async def test_block_does_not_leak_to_other_clients(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing(gw)
        for _ in range(16):
            await _get(gw, gw.auth("billing:read"))
        gw.clock.advance(6)
        for _ in range(16):
            await _get(gw, gw.auth("billing:read"))
        assert (await _get(gw, gw.auth("billing:read"))).status_code == 429
        assert (await _get(gw, gw.auth("billing:read", client_id="client-other"))).status_code == 200


# ------------------------------------------------------------------ design-level evasion


async def test_denied_request_spam_is_eventually_throttled(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        codes = []
        for _ in range(60):  # 6x the quota of client-a, every request denied (no such scope)
            codes.append((await _get(gw, gw.auth("nothing:here"))).status_code)
        assert 429 in codes


async def test_malformed_body_spam_is_eventually_throttled(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        codes = []
        for _ in range(60):
            codes.append((await post_mcp(gw, b"{not json")).status_code)
        assert 429 in codes


async def test_one_users_flood_must_not_block_other_users_of_same_client_id(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing(gw)
        for _ in range(16):
            await _get(gw, gw.auth("billing:read", sub="alice"))
        gw.clock.advance(6)
        for _ in range(16):
            await _get(gw, gw.auth("billing:read", sub="alice"))
        r = await _get(gw, gw.auth("billing:read", sub="bob"))
        assert r.status_code == 200


async def test_client_identity_is_namespaced_per_idp(make_gateway: Any) -> None:
    from sec_helpers import ISSUER_B, JWKS_B

    from ai_gateway.auth import TokenValidator
    from ai_gateway.config import IdPConfig
    from support import KeyPair

    async with make_gateway() as gw:
        _billing(gw)
        kp_b = KeyPair("kb")
        gw.idp_router.get(JWKS_B).respond(json=kp_b.jwks())
        idp_b = IdPConfig(name="b", issuer=ISSUER_B, jwks_uri=JWKS_B, audiences=["ai-gateway"])
        gw.services.validator = TokenValidator(
            [*gw.services.config.gateway.idps, idp_b], gw.services.jwks_http, gw.services.telemetry
        )
        for _ in range(10):  # A's real client-a uses up its quota
            assert (await _get(gw, gw.auth("billing:read"))).status_code == 200
        tok_b = kp_b.mint(issuer=ISSUER_B, scopes="billing:read", client_id="client-a")
        r = await _get(gw, {"Authorization": f"Bearer {tok_b}"})
        assert r.status_code != 429, "IdP B's 'client-a' consumed IdP A's client-a quota"


# ------------------------------------------------------------------ delay slots


async def test_delay_slots_are_per_client_released_and_cannot_starve_others(make_gateway: Any) -> None:
    async with make_gateway(max_concurrent_delayed=3, delay_start=0.3) as gw:
        _billing(gw)
        gate = asyncio.Event()
        started = 0

        async def slow_sleep(_: float) -> None:
            nonlocal started
            started += 1
            await gate.wait()

        gw.services.limiter._sleep = slow_sleep
        h = gw.auth("billing:read")
        for _ in range(3):  # below delay_start -> plain allows
            assert (await _get(gw, h)).status_code == 200
        pending = [asyncio.create_task(_get(gw, h)) for _ in range(6)]  # ratio 0.4..0.9: delay band
        for _ in range(50):
            await asyncio.sleep(0)
            if started >= 3:
                break
        assert started == 3  # exactly max_concurrent_delayed sleepers
        other = await _get(gw, gw.auth("billing:read", client_id="client-b"))  # other client unaffected
        assert other.status_code == 200
        gate.set()
        results = await asyncio.gather(*pending)
        codes = sorted(r.status_code for r in results)
        assert codes.count(429) >= 3 and codes.count(200) >= 3
        store = gw.services.store
        assert not [k for k in store._data if k.endswith(":delayq")]


async def test_delay_slot_released_when_client_disconnects(make_gateway: Any) -> None:
    async with make_gateway(max_concurrent_delayed=2, delay_start=0.1) as gw:
        _billing(gw)
        gate = asyncio.Event()

        async def slow_sleep(_: float) -> None:
            await gate.wait()

        gw.services.limiter._sleep = slow_sleep
        h = gw.auth("billing:read")
        await _get(gw, h)
        tasks = [asyncio.create_task(_get(gw, h)) for _ in range(2)]
        for _ in range(30):
            await asyncio.sleep(0)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        store = gw.services.store
        assert not [k for k in store._data if k.endswith(":delayq")]


async def test_delay_is_capped_by_max_wait(make_gateway: Any) -> None:
    async with make_gateway(max_delay_ms=10**9, delay_start=0.1) as gw:
        _billing(gw)
        h = gw.auth("billing:read")
        for _ in range(9):
            await _get(gw, h)
        assert gw.sleeps and max(gw.sleeps) <= 30.0


# ------------------------------------------------------------------ memory exhaustion


async def test_store_is_bounded_under_client_id_flood(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing(gw)
        store = MemoryStateStore(clock=gw.clock, max_keys=200)
        gw.services.store = store
        gw.services.limiter._store = store
        for i in range(400):
            await _get(gw, gw.auth("billing:read", client_id=f"flood-{i}"))
        assert len(store) <= 200


async def test_flood_of_client_ids_cannot_lift_an_existing_block(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing(gw)
        store = MemoryStateStore(clock=gw.clock, max_keys=300)
        gw.services.store = store
        gw.services.limiter._store = store
        h = gw.auth("billing:read")
        for _ in range(16):
            await _get(gw, h)
        gw.clock.advance(6)
        for _ in range(16):
            await _get(gw, h)
        assert (await _get(gw, h)).status_code == 429
        for i in range(600):  # attacker with many identities
            await _get(gw, gw.auth("billing:read", client_id=f"flood-{i}"))
        assert (await _get(gw, h)).status_code == 429


async def test_unregistered_clients_cannot_inflate_metric_labels(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing(gw)
        for i in range(80):
            await _get(gw, gw.auth("billing:read", client_id=f"stranger-{i}"))
        labels = {a.get("gateway.client_id") for a, _ in gw.obs.metric("gateway.requests")}
        assert labels <= {"client-a", "unregistered", "anonymous", "client-tokens"}
        labels = {a.get("gateway.client_id") for a, _ in gw.obs.metric("gateway.ratelimit.decisions")}
        assert not any(str(x).startswith("stranger") for x in labels)


# ------------------------------------------------------------------ token quota


def _llm_ok(gw: Any, *, i: int = 10, o: int = 10) -> Any:
    return gw.upstream.post(LLM).respond(
        json={"model": "claude-3", "usage": {"input_tokens": i, "output_tokens": o}}
    )


async def _ask(gw: Any, max_tokens: int = 50, client_id: str = "client-tokens", **extra: Any) -> Any:
    body = {
        "model": "claude-3",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }
    return await gw.client.post(
        "/llm/anthropic/v1/messages",
        headers=gw.auth("llm:claude", client_id=client_id),
        content=json.dumps(body),
    )


def _tok_total(gw: Any) -> int:
    store = gw.services.store
    return sum(v for k, (v, _) in store._data.items() if ":tok:" in k)


async def test_token_reservation_refunded_on_every_failure_path(make_gateway: Any) -> None:
    import httpx

    async with make_gateway() as gw:
        route = gw.upstream.post(LLM)
        for effect in (httpx.ConnectError("x"), httpx.ReadTimeout("x")):
            route.mock(side_effect=effect)
            await _ask(gw)
            assert _tok_total(gw) == 0, effect
        for status in (400, 401, 429, 500, 503):
            route.respond(status_code=status, json={"error": "x"})
            await _ask(gw)
            assert _tok_total(gw) == 0, status
        # denied or malformed requests never reserve anything
        r = await gw.client.post(
            "/llm/anthropic/v1/messages",
            headers=gw.auth("billing:read", client_id="client-tokens"),
            content=b'{"model":"claude-3","max_tokens":5}',
        )
        assert r.status_code == 403 and _tok_total(gw) == 0


async def test_token_quota_enforced_and_reconciled_to_actual_usage(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _llm_ok(gw, i=5, o=5)  # tiny real usage; quota is 200
        first = await _ask(gw, max_tokens=150)
        assert first.status_code == 200
        assert _tok_total(gw) == 10  # estimate replaced by actual
        big = await _ask(gw, max_tokens=500)  # estimate alone exceeds the whole quota
        assert big.status_code == 429 and big.json()["error"]["code"] == "token_quota_exceeded"
        assert _tok_total(gw) == 10  # rejected reservation rolled back


async def test_token_quota_not_bypassed_by_odd_max_tokens_values(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = _llm_ok(gw, i=1, o=1)
        bad_values: list[Any] = [None, 0, -5, "9" * 40, 1e30, True, [], {}]
        for mt in bad_values:
            gw.clock.advance(120)  # fresh window each time
            body: dict[str, Any] = {
                "model": "claude-3",
                "messages": [{"role": "user", "content": "x" * 4000}],
            }
            if mt is not None:
                body["max_tokens"] = mt
            r = await gw.client.post(
                "/llm/anthropic/v1/messages",
                headers=gw.auth("llm:claude", client_id="client-tokens"),
                content=json.dumps(body),
            )
            # estimate = 1000 (prompt) + >=1024 > 200 quota -> must be refused, not waved through
            assert r.status_code == 429, mt
        assert route.call_count == 0


async def test_token_quota_blocks_only_that_client(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _llm_ok(gw, i=100, o=100)
        assert (await _ask(gw, max_tokens=100)).status_code == 200
        assert (await _ask(gw, max_tokens=100)).status_code == 429
        assert (await _ask(gw, max_tokens=100, client_id="someone-else")).status_code == 200


async def test_malicious_usage_in_upstream_json_cannot_drive_counters_negative(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.upstream.post(LLM).respond(json={"usage": {"input_tokens": -(10**9), "output_tokens": -(10**9)}})
        await _ask(gw, max_tokens=50)
        assert _tok_total(gw) >= 0
        r = await _ask(gw, max_tokens=500)  # would exceed quota; must not be waved through
        assert r.status_code == 429


async def test_rate_limit_headers_have_sane_values(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing(gw)
        for _ in range(12):
            r = await _get(gw, gw.auth("billing:read"))
        assert r.status_code == 429
        assert 1 <= int(r.headers["retry-after"]) <= 3600
        assert r.headers["ratelimit-remaining"] == "0"
        r2 = await raw_request(
            gw.app,
            "GET",
            b"/billing/invoices",
            headers=[("authorization", gw.auth("billing:read")["Authorization"])],
        )
        assert r2.status_code == 429 and r2.headers["x-request-id"]


async def test_rpc_helper_import_used() -> None:
    assert rpc("ping")["method"] == "ping"


# ------------------------------------------------------------------ long-lived streams / shared pool


class _Hang(httpx.AsyncByteStream):
    """Upstream SSE body that emits one event then stays open until `gate` is set."""

    def __init__(self, gate: asyncio.Event, ticks: int = 1, interval: float = 0.0) -> None:
        self.gate, self.ticks, self.interval = gate, ticks, interval

    async def __aiter__(self) -> Any:
        for _ in range(self.ticks):
            yield b"data: x\n\n"
            if self.interval:
                await asyncio.sleep(self.interval)
        await self.gate.wait()


async def test_one_client_cannot_hold_unbounded_concurrent_upstream_streams(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gate = asyncio.Event()
        route = gw.upstream.get("http://mcp.internal/mcp")
        route.mock(
            side_effect=lambda req: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=_Hang(gate)
            )
        )
        h = gw.auth("mcp:tools:read", client_id="bulk-client")
        tasks = [asyncio.create_task(gw.client.get("/mcp/tools", headers=h)) for _ in range(250)]
        for _ in range(200):
            await asyncio.sleep(0)
        in_flight = route.call_count
        gate.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert in_flight < 200, f"{in_flight} concurrent upstream streams held by a single client"


async def test_stream_does_not_outlive_the_token(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.services.config.gateway.idps[0].leeway_s = 0
        gate = asyncio.Event()
        gate.set()
        route = gw.upstream.get("http://mcp.internal/mcp")
        route.mock(
            side_effect=lambda req: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=_Hang(gate, ticks=6, interval=0.6)
            )
        )
        r = await gw.client.get("/mcp/tools", headers=gw.auth("mcp:tools:read", ttl=2))
        assert r.status_code == 200
        assert r.content.count(b"data: x") < 6, "stream kept flowing 1+ s after the 2 s token expired"
