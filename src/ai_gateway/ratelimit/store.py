"""StateStore interface + in-memory implementation. OWNER: ratelimit.

Designed so a Redis implementation maps 1:1 (INCRBY+EXPIRE, GET, SET EX, SET NX EX, TTL, DEL,
and a Lua/INCR-with-cap for acquire). All values are ints; keys are opaque strings built by the
limiter (`rl:{client}:req:{bucket}` ...). Methods must be safe under concurrent asyncio tasks.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

_PROTECTED_SUFFIXES = (":block", ":abuse_since")


@runtime_checkable
class StateStore(Protocol):
    async def incr(self, key: str, amount: int = 1, *, ttl_s: float) -> int:
        """Atomically add `amount` (may be negative); create at 0 with TTL `ttl_s` if absent
        (TTL is set only on creation). Returns the new value."""
        ...

    async def get(self, key: str) -> int | None: ...

    async def set(self, key: str, value: int, *, ttl_s: float) -> None: ...

    async def set_if_absent(self, key: str, value: int, *, ttl_s: float) -> bool:
        """True if the key was created (SET NX)."""
        ...

    async def ttl(self, key: str) -> float | None:
        """Remaining seconds, None if the key is missing/expired."""
        ...

    async def delete(self, key: str) -> None: ...

    async def acquire(self, key: str, limit: int, *, ttl_s: float) -> bool:
        """Concurrency slot: increment and return True if the new count <= limit, else
        roll back and return False. `ttl_s` is a leak guard refreshed on each acquire."""
        ...

    async def release(self, key: str) -> None:
        """Decrement a slot (floor 0)."""
        ...

    async def aclose(self) -> None: ...


class MemoryStateStore:
    """Single-process dict store with lazy expiry + sweeping (bounded memory).

    Every method is a synchronous critical section (no awaits inside), so concurrent asyncio
    tasks can never interleave within one operation. Expired keys are dropped lazily on access
    and by a sweep that runs when the store exceeds `max_keys` or every `sweep_interval_s`.
    If a sweep of expired keys still leaves more than `max_keys`, the idlest keys (those closest
    to expiry, never block/abuse keys) are evicted until the store is at 90% of the cap. Time source
    is injectable.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 100_000,
        sweep_interval_s: float = 60.0,
    ) -> None:
        if max_keys < 1:
            raise ValueError("max_keys must be >= 1")
        self._clock = clock
        self._max_keys = max_keys
        self._sweep_interval_s = sweep_interval_s
        self._next_sweep = clock() + sweep_interval_s
        self._data: dict[str, tuple[int, float]] = {}  # key -> (value, expires_at)

    def __len__(self) -> int:
        return len(self._data)

    # -- internals (synchronous) -------------------------------------------------------------

    def _live(self, key: str, now: float) -> tuple[int, float] | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        if entry[1] <= now:
            del self._data[key]
            return None
        return entry

    def _maybe_sweep(self, now: float) -> None:
        if len(self._data) <= self._max_keys and now < self._next_sweep:
            return
        self._next_sweep = now + self._sweep_interval_s
        for k in [k for k, (_, exp) in self._data.items() if exp <= now]:
            del self._data[k]
        if len(self._data) > self._max_keys:
            target = int(self._max_keys * 0.9)
            # Never evict enforcement state (blocks, abuse timers): a flood of distinct identities
            # must not be able to lift an existing block (SEC-13).
            victims = sorted(
                (k for k in self._data if not k.endswith(_PROTECTED_SUFFIXES)),
                key=lambda k: self._data[k][1],
            )
            for k in victims[: len(self._data) - target]:
                del self._data[k]

    # -- protocol ----------------------------------------------------------------------------

    async def incr(self, key: str, amount: int = 1, *, ttl_s: float) -> int:
        now = self._clock()
        entry = self._live(key, now)
        if entry is None:
            self._data[key] = (amount, now + ttl_s)
            self._maybe_sweep(now)
            return amount
        value = entry[0] + amount
        self._data[key] = (value, entry[1])
        return value

    async def get(self, key: str) -> int | None:
        entry = self._live(key, self._clock())
        return None if entry is None else entry[0]

    async def set(self, key: str, value: int, *, ttl_s: float) -> None:
        now = self._clock()
        self._data[key] = (value, now + ttl_s)
        self._maybe_sweep(now)

    async def set_if_absent(self, key: str, value: int, *, ttl_s: float) -> bool:
        now = self._clock()
        if self._live(key, now) is not None:
            return False
        self._data[key] = (value, now + ttl_s)
        self._maybe_sweep(now)
        return True

    async def ttl(self, key: str) -> float | None:
        now = self._clock()
        entry = self._live(key, now)
        return None if entry is None else entry[1] - now

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def acquire(self, key: str, limit: int, *, ttl_s: float) -> bool:
        now = self._clock()
        entry = self._live(key, now)
        current = 0 if entry is None else entry[0]
        if current + 1 > limit:
            return False
        self._data[key] = (current + 1, now + ttl_s)
        self._maybe_sweep(now)
        return True

    async def release(self, key: str) -> None:
        entry = self._live(key, self._clock())
        if entry is None:
            return
        if entry[0] <= 1:
            del self._data[key]  # slot count back to zero: free the key
        else:
            self._data[key] = (entry[0] - 1, entry[1])

    async def aclose(self) -> None:
        self._data.clear()
