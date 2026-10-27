"""Pure matching helpers. OWNER: policy. No I/O, no telemetry."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection
from functools import lru_cache

_PCT = re.compile(r"%([0-9A-Fa-f]{2})")
_BAD_PCT = re.compile(r"%(?![0-9A-Fa-f]{2})")
# Percent-escapes that must never be accepted as such: encoded slash / backslash / NUL.
_ENCODED_SEPARATORS = re.compile(r"%(2f|5c|00)", re.IGNORECASE)
_FORBIDDEN_CHARS = frozenset("\\?#;")
_UNICODE_SPECIALS = frozenset("/\\.%;?#")


def _reject(reason: str) -> ValueError:
    return ValueError(f"invalid path: {reason}")


@lru_cache(maxsize=1024)
def _compile_glob(pattern: str) -> tuple[tuple[re.Pattern[str], int], ...]:
    """Split a glob on '*' into fixed-width chunks ('?' = any one char). Each chunk is a
    backtracking-free regex (every position matches exactly one character), so a match is
    O(len(name) * len(pattern)) in the worst case, never exponential/cubic (SEC-18)."""
    chunks = []
    for part in pattern.split("*"):
        rx = "".join("." if ch == "?" else re.escape(ch) for ch in part)
        chunks.append((re.compile(rx, re.DOTALL), len(part)))
    return tuple(chunks)


def _glob(pattern: str, name: str) -> bool:
    chunks = _compile_glob(pattern)
    if len(chunks) == 1:  # no '*'
        rx, width = chunks[0]
        return len(name) == width and rx.match(name) is not None
    first, last = chunks[0], chunks[-1]
    end = len(name) - last[1]
    if end < first[1] or first[0].match(name) is None or last[0].match(name, end) is None:
        return False
    pos = first[1]
    for rx, _width in chunks[1:-1]:
        m = rx.search(name, pos, end)
        if m is None:
            return False
        pos = m.end()
    return True


def _split(path: str) -> list[str]:
    return [] if path == "/" else path[1:].split("/")


def path_match(pattern: str, path: str) -> bool:
    """Glob match on a '/'-separated path. `*` = exactly one segment (may be partial, e.g.
    'inv-*'), `**` = zero or more whole segments. Case-sensitive. `path` must already be
    normalized by normalize_path(). Pattern '/' matches only '/'."""
    if not pattern.startswith("/") or not path.startswith("/"):
        return False
    pats = _split(pattern)
    segs = _split(path)
    states = {0}
    for pat in pats:
        nxt: set[int] = set()
        if pat == "**":
            lo = min(states)
            nxt.update(range(lo, len(segs) + 1))
        else:
            for j in states:
                if j < len(segs) and segs[j] != "" and _glob(pat, segs[j]):
                    nxt.add(j + 1)
        if not nxt:
            return False
        states = nxt
    return len(segs) in states


def normalize_path(path: str) -> str:
    """Canonicalize before matching: percent-decode ONCE, collapse '//' and resolve '.'/'..'
    segments; reject (raise ValueError) encoded slashes, NULs and '..' escaping root. The proxy
    must forward this exact normalized path so authz and upstream see the same path.

    Also rejected (fail closed): paths not starting with '/', malformed or residual
    (double-encoded) percent escapes, invalid UTF-8, control characters, backslash, ';', '?',
    '#', dot-segments disguised with whitespace, and non-ASCII characters whose compatibility
    form contains a path-significant character. A trailing slash is dropped. Case is preserved
    (matching is case-sensitive)."""
    if not isinstance(path, str):  # pyright/mypy-safe guard for untyped callers
        raise _reject("not a string")
    if "\x00" in path:
        raise _reject("NUL byte")
    if not path.startswith("/"):
        raise _reject("must start with '/'")
    if _ENCODED_SEPARATORS.search(path):
        raise _reject("encoded slash or NUL")
    if _BAD_PCT.search(path):
        raise _reject("malformed percent escape")
    try:
        decoded = _decode_utf8(path)  # single pass, strict UTF-8 (rejects overlong forms)
    except UnicodeDecodeError as exc:
        raise _reject("invalid UTF-8") from exc
    for raw_seg in path.split("/"):
        if "%" in raw_seg and _decode_utf8(raw_seg).strip() in {".", ".."}:
            raise _reject("encoded dot segment")
    for ch in decoded:
        if ord(ch) < 0x20 or ord(ch) == 0x7F:
            raise _reject("control character")
        if ch in _FORBIDDEN_CHARS:
            raise _reject(f"forbidden character {ch!r}")
        if ord(ch) > 0x7F and _UNICODE_SPECIALS & set(unicodedata.normalize("NFKC", ch)):
            raise _reject("unicode look-alike of a path character")
    if _PCT.search(decoded):
        raise _reject("double-encoded percent escape")

    out: list[str] = []
    for seg in decoded.split("/"):
        if seg == "" or seg == ".":
            continue
        if seg == "..":
            if not out:
                raise _reject("traversal above root")
            out.pop()
            continue
        if seg.strip() in {".", ".."}:
            raise _reject("disguised dot segment")
        out.append(seg)
    return "/" + "/".join(out)


def _decode_utf8(s: str) -> str:
    """Percent-decode exactly once into bytes (literal text passes through), then strict UTF-8."""
    buf = bytearray()
    i = 0
    while i < len(s):
        if s[i] == "%" and _PCT.fullmatch(s, i, i + 3):
            buf.append(int(s[i + 1 : i + 3], 16))
            i += 3
        else:
            buf.extend(s[i].encode("utf-8"))
            i += 1
    return buf.decode("utf-8")


def name_match(pattern: str, name: str) -> bool:
    """fnmatch-style glob for tool/model names (`*`, `?`), case-sensitive."""
    return _glob(pattern, name)


def scopes_satisfied(granted: Collection[str], *, any_of: Collection[str], all_of: Collection[str]) -> bool:
    """True iff (any_of empty or intersects granted) and all_of subset of granted
    and at least one of any_of/all_of is non-empty."""
    if not any_of and not all_of:
        return False
    if any_of and not any(s in granted for s in any_of):
        return False
    return all(s in granted for s in all_of)
