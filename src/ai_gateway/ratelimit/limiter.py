"""RateLimiter: per-client phased limiting over a StateStore. OWNER: ratelimit.

Algorithm for check_request (all keys prefixed `rl:{client_id}:`):

  0. blocked = ttl(`block`) -> if present: BLOCK decision immediately (no counter writes).
  1. Sliding window counter over `req:{bucket}` / previous bucket; EVERY attempt counts
     (including ones that end up rejected) so abusive clients climb toward block_at.
  2. Sustained-abuse timer (`abuse_since`): started when ratio >= block_at, cleared when it
     drops below; held for block_sustained_seconds -> `block` key with TTL block_duration.
  3. phases.classify(): ALLOW | DELAY | REJECT.
  4. DELAY: acquire the per-client `delayq` slot or escalate to REJECT; apply_delay() sleeps
     (bounded, cancel-safe) and always releases the slot.

Token quota (LLM) is a separate fixed-window counter `tok:{bucket}`; it never delays or blocks.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import time
from collections.abc import Awaitable, Callable

from ai_gateway.config import LimitPolicy
from ai_gateway.context import get_ctx
from ai_gateway.models import LimitDecision, LimitScope, Phase
from ai_gateway.observability.audit import AuditEvent, AuditType
from ai_gateway.observability.telemetry import Telemetry
from ai_gateway.ratelimit import phases
from ai_gateway.ratelimit.store import StateStore

log = logging.getLogger(__name__)

AUDIT_THROTTLE_S = 10.0
DEFAULT_MAX_WAIT_S = 30.0  # hard ceiling on any single apply_delay(), regardless of policy


class RateLimiter:
    def __init__(
        self,
        store: StateStore,
        telemetry: Telemetry,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        on_error: str = "fail_open",
        max_wait_s: float = DEFAULT_MAX_WAIT_S,
    ) -> None:
        self._store = store
        self._telemetry = telemetry
        self._clock = clock
        self._sleep = sleep
        self._on_error = on_error
        self._max_wait_s = max_wait_s

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _key(client_id: str, suffix: str) -> str:
        return f"rl:{client_id}:{suffix}"

    async def _audit(self, decision: LimitDecision, *, kind: AuditType | None = None) -> None:
        try:
            if kind is None:
                kind = AuditType.RATELIMIT_DECISION
                first = await self._store.set_if_absent(
                    self._key(decision.client_id, f"audit:{decision.phase.value}"),
                    1,
                    ttl_s=AUDIT_THROTTLE_S,
                )
                if not first:
                    return
            ctx = get_ctx()
            self._telemetry.audit(
                AuditEvent(
                    type=kind,
                    request_id=ctx.request_id if ctx else "",
                    trace_id=ctx.trace_id if ctx else "",
                    client_id=decision.client_id,
                    decision=decision.phase.value,
                    reason=decision.reason or None,
                    method=ctx.method if ctx else None,
                    path=ctx.path if ctx else None,
                    client_ip=(ctx.client_ip or None) if ctx else None,
                    extra={
                        "scope": decision.scope.value,
                        "usage_ratio": round(decision.usage_ratio, 3),
                        "retry_after_s": round(decision.retry_after_s, 1),
                    },
                )
            )
        except Exception:
            log.exception("ratelimit audit failed", extra={"event": "ratelimit.audit_error"})

    async def _finish(self, decision: LimitDecision, *, audit: bool = True) -> LimitDecision:
        try:
            self._telemetry.record_rate_limit(decision, get_ctx())
        except Exception:
            log.exception("ratelimit telemetry failed", extra={"event": "ratelimit.telemetry_error"})
        if audit and decision.phase is not Phase.ALLOW:
            await self._audit(decision)
        return decision

    def _error_decision(self, client_id: str, policy: LimitPolicy | None, scope: LimitScope) -> LimitDecision:
        limit = 0
        window = 60.0
        if policy is not None:
            window = policy.window_seconds
            limit = (
                policy.requests_per_window
                if scope is LimitScope.REQUESTS
                else (policy.tokens_per_window or 0)
            )
        fail_closed = self._on_error == "fail_closed"
        return LimitDecision(
            phase=Phase.REJECT if fail_closed else Phase.ALLOW,
            scope=scope,
            client_id=client_id,
            limit=limit,
            remaining=0,
            reset_s=1.0,
            retry_after_s=1.0 if fail_closed else 0.0,
            window_s=window,
            reason="limiter_error",
        )

    async def _incr_clamped(self, key: str, amount: int, ttl_s: float) -> int:
        """incr that never leaves the counter negative (late refunds after a window roll)."""
        value = await self._store.incr(key, amount, ttl_s=ttl_s)
        if value < 0:
            await self._store.incr(key, -value, ttl_s=ttl_s)
            return 0
        return value

    # ------------------------------------------------------------------ requests

    async def is_blocked(self, client_id: str) -> LimitDecision | None:
        """Fast path used right after authn: BLOCK decision if in cooldown else None."""
        try:
            remaining = await self._store.ttl(self._key(client_id, "block"))
        except Exception:
            log.exception("ratelimit store error", extra={"event": "ratelimit.store_error"})
            if self._on_error == "fail_closed":
                return await self._finish(self._error_decision(client_id, None, LimitScope.REQUESTS))
            return None
        if remaining is None:
            return None
        return await self._finish(self._block_decision(client_id, remaining, None), audit=False)

    @staticmethod
    def _block_decision(client_id: str, remaining: float, policy: LimitPolicy | None) -> LimitDecision:
        return LimitDecision(
            phase=Phase.BLOCK,
            scope=LimitScope.REQUESTS,
            client_id=client_id,
            limit=policy.requests_per_window if policy else 0,
            remaining=0,
            reset_s=remaining,
            usage_ratio=0.0,
            retry_after_s=remaining,
            window_s=policy.window_seconds if policy else 60.0,
            reason="client_blocked",
        )

    async def check_request(self, client_id: str, policy: LimitPolicy) -> LimitDecision:
        try:
            return await self._check_request(client_id, policy)
        except Exception:
            log.exception("ratelimit store error", extra={"event": "ratelimit.store_error"})
            return await self._finish(self._error_decision(client_id, policy, LimitScope.REQUESTS))

    async def _check_request(self, client_id: str, policy: LimitPolicy) -> LimitDecision:
        store = self._store
        block_key = self._key(client_id, "block")
        remaining_block = await store.ttl(block_key)
        if remaining_block is not None:
            return await self._finish(self._block_decision(client_id, remaining_block, policy), audit=False)

        now = self._clock()
        window = policy.window_seconds
        quota = policy.requests_per_window
        bucket = math.floor(now / window)
        current = await store.incr(self._key(client_id, f"req:{bucket}"), ttl_s=2 * window)
        previous = await store.get(self._key(client_id, f"req:{bucket - 1}")) or 0
        elapsed = (now - bucket * window) / window
        count = phases.sliding_count(current, previous, elapsed)
        ratio = count / quota
        to_reset = max((bucket + 1) * window - now, 0.0)
        left = max(int(quota - count), 0)

        # Phase 3: sustained abuse.
        abuse_key = self._key(client_id, "abuse_since")
        if ratio >= policy.block_at:
            ttl = policy.block_sustained_seconds * 2 + window
            now_ms = int(now * 1000)
            await store.set_if_absent(abuse_key, now_ms, ttl_s=ttl)
            since_ms = await store.get(abuse_key)
            held = (now_ms - since_ms) / 1000.0 if since_ms is not None else 0.0
            if policy.block_sustained_seconds == 0 or held >= policy.block_sustained_seconds:
                created = await store.set_if_absent(block_key, 1, ttl_s=policy.block_duration_seconds)
                await store.delete(abuse_key)
                decision = dataclasses.replace(
                    self._block_decision(client_id, policy.block_duration_seconds, policy),
                    usage_ratio=ratio,
                    reason="sustained_abuse",
                )
                if created:
                    await self._audit(decision, kind=AuditType.CLIENT_BLOCKED)
                return await self._finish(decision, audit=False)
        else:
            await store.delete(abuse_key)

        phase = phases.classify(ratio, policy)
        base = LimitDecision(
            phase=phase,
            scope=LimitScope.REQUESTS,
            client_id=client_id,
            limit=quota,
            remaining=left,
            reset_s=to_reset,
            usage_ratio=ratio,
            window_s=window,
        )
        if phase is Phase.ALLOW:
            return await self._finish(base)
        if phase is Phase.DELAY:
            delay = min(phases.delay_for(ratio, policy), self._max_wait_s)
            if delay <= 0:
                return await self._finish(dataclasses.replace(base, delay_s=0.0))
            slot_key = self._key(client_id, "delayq")
            leak_guard = 2 * self._max_wait_s + 60.0
            if await store.acquire(slot_key, policy.max_concurrent_delayed, ttl_s=leak_guard):
                return await self._finish(dataclasses.replace(base, delay_s=delay, slot=slot_key))
            return await self._finish(self._reject(base, count, policy, "delay_capacity"))
        return await self._finish(self._reject(base, count, policy, "rate_limited"))

    @staticmethod
    def _reject(base: LimitDecision, count: float, policy: LimitPolicy, reason: str) -> LimitDecision:
        wait = phases.retry_after(count, policy)
        return dataclasses.replace(
            base, phase=Phase.REJECT, retry_after_s=wait, reset_s=wait, remaining=0, reason=reason
        )

    async def apply_delay(self, decision: LimitDecision) -> float:
        """Sleep decision.delay_s (no-op unless phase DELAY) and release the slot. Returns the
        seconds actually waited. Safe to call exactly once per DELAY decision."""
        if decision.phase is not Phase.DELAY:
            return 0.0
        start = self._clock()
        try:
            wait = min(max(decision.delay_s, 0.0), self._max_wait_s)
            if wait > 0:
                await self._sleep(wait)
        finally:
            await self.release(decision)
        return max(self._clock() - start, 0.0)

    async def release(self, decision: LimitDecision) -> None:
        """Release a held delay slot without waiting (e.g. later pipeline stage failed)."""
        if decision.slot is None:
            return
        try:
            await self._store.release(decision.slot)
        except Exception:
            log.exception("ratelimit slot release failed", extra={"event": "ratelimit.store_error"})

    # ------------------------------------------------------------------ concurrency (SEC-22)

    async def acquire_concurrency(self, client_id: str, policy: LimitPolicy) -> str | None:
        """Take one in-flight slot (`policy.max_concurrent_requests` per client identity).
        Returns the slot key to hand to release_concurrency(), "" when no slot is held (store
        error under fail_open), or None when the client is at its cap (or fail_closed error)."""
        key = self._key(client_id, "conc")
        try:
            ok = await self._store.acquire(
                key, policy.max_concurrent_requests, ttl_s=2 * policy.max_stream_seconds + 120.0
            )
        except Exception:
            log.exception("ratelimit store error", extra={"event": "ratelimit.store_error"})
            return "" if self._on_error != "fail_closed" else None
        return key if ok else None

    async def release_concurrency(self, key: str) -> None:
        if not key:
            return
        try:
            await self._store.release(key)
        except Exception:
            log.exception("ratelimit slot release failed", extra={"event": "ratelimit.store_error"})

    # ------------------------------------------------------------------ tokens

    async def reserve_tokens(
        self, client_id: str, policy: LimitPolicy, estimated_tokens: int
    ) -> LimitDecision:
        """ALLOW if policy.tokens_per_window is None or capacity remains, else REJECT."""
        quota = policy.tokens_per_window
        window = policy.token_window_seconds
        if quota is None:
            return await self._finish(
                LimitDecision(
                    phase=Phase.ALLOW,
                    scope=LimitScope.TOKENS,
                    client_id=client_id,
                    limit=0,
                    remaining=0,
                    reset_s=0.0,
                    window_s=window,
                    reason="unlimited",
                )
            )
        try:
            now = self._clock()
            bucket = math.floor(now / window)
            key = self._key(client_id, f"tok:{bucket}")
            estimate = max(int(estimated_tokens), 0)
            total = await self._incr_clamped(key, estimate, 2 * window)
            to_reset = max((bucket + 1) * window - now, 0.0)
            ratio = total / quota
            if total > quota:
                await self._incr_clamped(key, -estimate, 2 * window)
                wait = max(to_reset, 1.0)
                decision = LimitDecision(
                    phase=Phase.REJECT,
                    scope=LimitScope.TOKENS,
                    client_id=client_id,
                    limit=quota,
                    remaining=0,
                    reset_s=to_reset,
                    usage_ratio=ratio,
                    retry_after_s=wait,
                    window_s=window,
                    reason="token_quota_exceeded",
                )
            else:
                decision = LimitDecision(
                    phase=Phase.ALLOW,
                    scope=LimitScope.TOKENS,
                    client_id=client_id,
                    limit=quota,
                    remaining=max(quota - total, 0),
                    reset_s=to_reset,
                    usage_ratio=ratio,
                    window_s=window,
                )
        except Exception:
            log.exception("ratelimit store error", extra={"event": "ratelimit.store_error"})
            return await self._finish(self._error_decision(client_id, policy, LimitScope.TOKENS))
        return await self._finish(decision)

    async def reconcile_tokens(
        self, client_id: str, policy: LimitPolicy, estimated_tokens: int, actual_tokens: int
    ) -> None:
        """Apply actual - estimated to the current token window (actual=0 refunds in full).
        The counter is clamped at zero so a refund landing after a window roll cannot go
        negative and grant phantom capacity."""
        if policy.tokens_per_window is None:
            return
        delta = max(int(actual_tokens), 0) - max(int(estimated_tokens), 0)
        if delta == 0:
            return
        window = policy.token_window_seconds
        bucket = math.floor(self._clock() / window)
        try:
            await self._incr_clamped(self._key(client_id, f"tok:{bucket}"), delta, 2 * window)
        except Exception:
            log.exception("ratelimit store error", extra={"event": "ratelimit.store_error"})
