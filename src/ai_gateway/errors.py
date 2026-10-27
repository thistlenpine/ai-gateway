"""Gateway exception hierarchy. Every client-visible failure is a GatewayError subclass.

OWNER: core. The pipeline converts GatewayError -> JSON response:
    {"error": {"code": "<code>", "message": "<safe message>", "request_id": "<id>"}}
Messages must be safe to show to the client: no token contents, no policy internals.
"""

from __future__ import annotations

import enum


class AuthFailure(enum.StrEnum):
    """Why authentication failed (metric attribute `gateway.auth.failure_reason`; audit only)."""

    MISSING_TOKEN = "missing_token"
    MALFORMED = "malformed"
    UNKNOWN_ISSUER = "unknown_issuer"
    ALG_NOT_ALLOWED = "alg_not_allowed"
    UNKNOWN_KID = "unknown_kid"
    BAD_SIGNATURE = "bad_signature"
    EXPIRED = "expired"
    NOT_YET_VALID = "not_yet_valid"
    BAD_AUDIENCE = "bad_audience"
    MISSING_CLAIM = "missing_claim"
    JWKS_UNAVAILABLE = "jwks_unavailable"
    CLIENT_NOT_REGISTERED = "client_not_registered"


class GatewayError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    message: str = "Internal error"

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message or self.message)
        self.message = message or self.message
        self.code = code or self.code
        self.headers: dict[str, str] = headers or {}


class BadRequestError(GatewayError):
    status_code = 400
    code = "bad_request"
    message = "Bad request"


class PayloadTooLargeError(GatewayError):
    status_code = 413
    code = "payload_too_large"
    message = "Request body too large"


class AuthenticationError(GatewayError):
    """401 (or 503 when the IdP is unreachable and no cached keys exist)."""

    status_code = 401
    code = "invalid_token"
    message = "Authentication required"

    def __init__(self, failure: AuthFailure, message: str | None = None) -> None:
        super().__init__(
            message,
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'}
            if failure is not AuthFailure.MISSING_TOKEN
            else {"WWW-Authenticate": "Bearer"},
        )
        self.failure = failure
        if failure is AuthFailure.JWKS_UNAVAILABLE:
            self.status_code = 503
            self.code = "idp_unavailable"
            self.headers = {}


class AuthorizationError(GatewayError):
    """403. Same body for every deny reason (no route enumeration); detail goes to audit."""

    status_code = 403
    code = "access_denied"
    message = "Access denied"


class RateLimitedError(GatewayError):
    """429. `headers` carries Retry-After + RateLimit-*; code is rate_limited | client_blocked
    | token_quota_exceeded."""

    status_code = 429
    code = "rate_limited"
    message = "Rate limit exceeded"


class UpstreamError(GatewayError):
    status_code = 502
    code = "upstream_error"
    message = "Upstream request failed"


class UpstreamTimeoutError(UpstreamError):
    status_code = 504
    code = "upstream_timeout"
    message = "Upstream timed out"
