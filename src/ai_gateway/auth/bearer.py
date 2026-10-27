"""Authorization header parsing. OWNER: auth."""

from __future__ import annotations

from ai_gateway.errors import AuthenticationError, AuthFailure


def extract_bearer(authorization: str | None) -> str:
    """Return the token from `Authorization: Bearer <token>` (scheme case-insensitive).
    Raises AuthenticationError(MISSING_TOKEN) if absent, (MALFORMED) if not a single-token
    Bearer value. Never logs the value."""
    if not authorization:
        raise AuthenticationError(AuthFailure.MISSING_TOKEN)
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        raise AuthenticationError(AuthFailure.MALFORMED)
    return parts[1]
