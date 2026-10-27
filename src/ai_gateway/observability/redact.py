"""Redaction layer. OWNER: observability.

Single place that decides what may be written to logs, span attributes and audit events.
Two complementary mechanisms:

* key based: any mapping key that names a credential (``schema.REDACT_KEYS`` plus a suffix
  heuristic such as ``*_token``, ``*_secret``) has its value replaced wholesale;
* value based: free text (messages, exception text, URLs, string values) is scrubbed for bearer
  tokens, JWTs, well-known API key shapes, ``key=value`` credentials and URL userinfo.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Final

from ai_gateway.observability import schema

REDACTED: Final = "[REDACTED]"

_MAX_DEPTH: Final = 8
# Hard cap on any string handed to the regexes below (SEC-00): several patterns can rescan long runs
# of word characters from every start offset, so input length is bounded before any scan. Content
# beyond the cap is dropped (never passed through), so truncation cannot leak a secret.
MAX_SCRUB_CHARS: Final = 2048
_TRUNC: Final = "...[truncated]"


def _cap(text: str) -> str:
    return text if len(text) <= MAX_SCRUB_CHARS else text[:MAX_SCRUB_CHARS] + _TRUNC


def _norm(key: str) -> str:
    return key.strip().lower().replace("-", "_").replace(".", "_")


_EXACT: Final = frozenset(_norm(k) for k in schema.REDACT_KEYS)
_SUFFIXES: Final = (
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "credentials",
    "private_key",
    "signature",
)

# Query parameter names (normalized) whose values are credentials.
_QUERY_KEYS: Final = frozenset({"sig", "key", "code", "auth", "jwt", "bearer"})

_URL_ATTR_KEYS: Final = frozenset({"http_url", "url_full", "http_target", "url_original"})


def is_sensitive_key(key: object) -> bool:
    """True when a mapping/attribute key names a credential (case/dash/dot-insensitive)."""
    if not isinstance(key, str):
        return False
    n = _norm(key)
    if n in _EXACT:
        return True
    return any(n == s or n.endswith("_" + s) for s in _SUFFIXES)


_KV = re.compile(
    r"""(?ix)
    (?P<name>(?:access_token|refresh_token|id_token|token|api[_-]?key|apikey|client_secret|
        secret|password|passwd|authorization|signature|sig|private_key))
    (?P<q1>["']?)
    (?P<sep>\s*[=:]\s*)
    (?P<q2>["']?)
    (?P<scheme>(?:bearer|basic|token)\s+)?
    (?P<val>[^\s&"',;]+)
    """
)
_BEARER = re.compile(r"(?i)\b(bearer|basic)(\s+)[A-Za-z0-9._~+/=\-]{8,}")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}(?:\.[A-Za-z0-9_\-]*)?")
_KEY_SHAPES = re.compile(
    r"\b(?:(?:sk|pk|rk)-[A-Za-z0-9_\-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"
    r"|xox[abprs]-[A-Za-z0-9\-]{10,})"
)
_USERINFO = re.compile(r"(?i)(\b[a-z][a-z0-9+.\-]*://)[^/\s?#@]+@")


def _kv_sub(m: re.Match[str]) -> str:
    if m.group("val") == REDACTED:
        return m.group(0)
    return f"{m.group('name')}{m.group('q1')}{m.group('sep')}{m.group('q2')}{REDACTED}"


def scrub_text(text: str) -> str:
    """Remove credential-looking substrings from free text."""
    if not text:
        return text
    text = _cap(text)
    out = _USERINFO.sub(lambda m: f"{m.group(1)}{REDACTED}@", text)
    out = _KV.sub(_kv_sub, out)
    out = _BEARER.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", out)
    out = _JWT.sub(REDACTED, out)
    return _KEY_SHAPES.sub(REDACTED, out)


def scrub_query(query: str) -> str:
    """Redact credential parameters in a raw query string, preserving the rest verbatim."""
    parts: list[str] = []
    for piece in _cap(query).split("&"):
        name, sep, _value = piece.partition("=")
        if sep and (is_sensitive_key(name) or _norm(name) in _QUERY_KEYS):
            parts.append(f"{name}={REDACTED}")
        else:
            parts.append(scrub_text(piece))
    return "&".join(parts)


def scrub_url(url: str) -> str:
    """Drop userinfo and redact credential query parameters / fragments of a URL or target."""
    url = _cap(url)
    base, qmark, rest = url.partition("?")
    base = _USERINFO.sub(lambda m: f"{m.group(1)}{REDACTED}@", base)
    if not qmark:
        return scrub_text(base)
    query, hash_, _fragment = rest.partition("#")
    return f"{scrub_text(base)}?{scrub_query(query)}" + (f"#{REDACTED}" if hash_ else "")


def strip_query(path: str) -> str:
    """Path without query string or fragment (for audit/log fields that need no query)."""
    return path.split("?", 1)[0].split("#", 1)[0]


def redact_value(key: object, value: Any, _depth: int = 0) -> Any:
    """Redact ``value`` given its mapping ``key``; recurses through mappings and sequences."""
    if is_sensitive_key(key):
        return REDACTED
    if isinstance(value, str):
        if isinstance(key, str) and _norm(key) in _URL_ATTR_KEYS:
            return scrub_url(value)
        return scrub_text(value)
    if _depth >= _MAX_DEPTH:
        return value if isinstance(value, int | float | bool | None) else REDACTED
    if isinstance(value, Mapping):
        return {str(k): redact_value(k, v, _depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [redact_value(None, v, _depth + 1) for v in value]
    return value


def redact_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of ``data`` with credential keys masked and string values scrubbed."""
    return {str(k): redact_value(k, v) for k, v in data.items()}
