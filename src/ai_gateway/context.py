"""Per-request context propagation via ContextVar. OWNER: core."""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar

from ai_gateway.models import RequestContext

_current: ContextVar[RequestContext | None] = ContextVar("ai_gateway_request_ctx", default=None)

_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._\-]{8,128}")
REQUEST_ID_HEADER = "X-Request-ID"


def get_ctx() -> RequestContext | None:
    """Current request's context, or None outside a request (startup, background tasks)."""
    return _current.get()


def set_ctx(ctx: RequestContext) -> object:
    """Bind ctx; returns a token to pass to reset_ctx()."""
    return _current.set(ctx)


def reset_ctx(token: object) -> None:
    _current.reset(token)  # type: ignore[arg-type]


def new_request_id(inbound: str | None = None) -> str:
    """Accept a well-formed inbound X-Request-ID, otherwise mint a uuid4 hex."""
    if inbound and _SAFE_REQUEST_ID.fullmatch(inbound):
        return inbound
    return uuid.uuid4().hex
