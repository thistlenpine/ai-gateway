"""Strict JSON helpers shared by the adapters. OWNER: proxy.

Duplicate object keys are rejected: Python keeps the last one, other parsers keep the first, and
a gateway/upstream disagreement on e.g. `method` or `model` would be an authorization bypass.
"""

from __future__ import annotations

import json
from typing import Any

from ai_gateway.errors import BadRequestError


def _no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # Keys that collide after case-folding are rejected too: Go/.NET parsers match keys
    # case-insensitively (last wins), so `method`/`Method` would be a parser differential (SEC-01).
    out: dict[str, Any] = {}
    seen: set[str] = set()
    for key, value in pairs:
        folded = key.casefold()
        if folded in seen:
            raise ValueError("duplicate key")
        seen.add(folded)
        out[key] = value
    return out


def loads_strict(body: bytes | str) -> Any:
    """Parse JSON or raise ValueError (duplicate keys, bad UTF-8, NaN, depth blow-ups)."""
    try:
        if isinstance(body, bytes | bytearray):  # json.loads(bytes) sniffs UTF-16/32/BOM (SEC-10)
            body = bytes(body).decode("utf-8")
        if body.startswith("\ufeff"):
            raise ValueError("BOM not allowed")
        return json.loads(body, object_pairs_hook=_no_dupes, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise ValueError("too deeply nested") from exc


def _reject_constant(name: str) -> Any:
    raise ValueError(f"invalid constant {name}")


def parse_body(body: bytes | None) -> Any:
    """Parse a request body for classification; BadRequestError on anything unusable."""
    if not body:
        raise BadRequestError("Request body is required")
    try:
        return loads_strict(body)
    except ValueError as exc:
        raise BadRequestError("Request body is not valid JSON") from exc


def try_loads(body: bytes | str) -> Any | None:
    try:
        return loads_strict(body)
    except ValueError:
        return None
