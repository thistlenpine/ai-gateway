# ADR 0002: In-memory rate-limit state behind a StateStore protocol

Status: accepted

## Context
v1 runs as a single instance. A Redis-backed store will be needed for horizontal scale. Limiter logic must not change when that happens.

## Decision
`ratelimit.store.StateStore` is a small async protocol of int-valued primitives: `incr(ttl on create)`, `get`, `set`, `set_if_absent`, `ttl`, `delete`, `acquire/release` (concurrency slots), `aclose`. These map 1:1 to Redis (INCRBY+EXPIRE, GET, SET EX, SET NX EX, TTL, DEL, Lua cap). `MemoryStateStore` implements them with a dict, lazy expiry, a key cap and an injectable clock. The limiter holds no state of its own besides the store.

## Consequences
- Counters, blocks and delay slots reset on restart; documented limitation.
- Only one gateway instance may run until a Redis store exists (otherwise limits multiply by N).
- Multi-step limiter logic (sustained-abuse timer) uses only primitives; a Redis version may later optimize with Lua without changing the interface.
