# ADR 0004: Three-phase per-client limiter (delay, reject, block)

Status: accepted

## Context
Hard 429s at the quota edge cause retry storms; abusive clients still consume CPU parsing requests. We want graceful degradation and a cheap fast path for abuse, plus LLM token budgets.

## Decision
Per client_id, sliding-window counter over all attempts; thresholds are fractions of quota (defaults 0.8 / 1.0 / 1.5) configurable per client. Phase 1 delays admission by a linearly scaled, capped async sleep with a per-client waiter cap (excess escalate to reject). Phase 2 returns 429 with `Retry-After` and `RateLimit-*`. Phase 3 triggers when the ratio stays at or above `block_at` for `block_sustained_seconds`: the client is blocked for `block_duration_seconds` and rejected (429 `client_blocked`) right after authn, before body parsing or policy. LLM routes add a separate token quota: pre-flight estimate reserved, reconciled from upstream usage; it only rejects (never delays or blocks).

## Consequences
- Rejected attempts count toward the ratio (needed to reach block_at), so a client hammering at 2x quota is blocked, while a client at 1.1x only gets 429s.
- Delay holds a connection; bounded by `max_delay_ms` and `max_concurrent_delayed`.
- Estimates can over/under-charge briefly; reconciliation corrects the counter.
- Rate limiting follows authz so denied probes do not consume quota (see DESIGN section 3 for the trade-off).
