"""JWKS fetching, caching and key rotation. OWNER: auth."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import socket
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import ec, ed448, ed25519, rsa

from ai_gateway.config import IdPConfig
from ai_gateway.errors import AuthenticationError, AuthFailure
from ai_gateway.observability.schema import SPAN_JWKS_REFRESH
from ai_gateway.observability.telemetry import Telemetry

logger = logging.getLogger(__name__)

MAX_JWKS_BYTES = 1_000_000
MIN_RSA_BITS = 2048
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

_KTY_FOR_ALG_PREFIX = (("RS", "RSA"), ("PS", "RSA"), ("ES", "EC"), ("Ed", "OKP"))
_CURVE_FOR_ALG = {
    "ES256": "secp256r1",
    "ES384": "secp384r1",
    "ES512": "secp521r1",
    "ES256K": "secp256k1",
}


def kty_for_alg(alg: str) -> str | None:
    for prefix, kty in _KTY_FOR_ALG_PREFIX:
        if alg.startswith(prefix):
            return kty
    return None


def key_matches_alg(key: Any, alg: str) -> bool:
    """True only if the key object's type (and curve / size) is valid for `alg`.

    This is the guard against algorithm/key confusion: an RSA key is never accepted for ES*,
    an EC key never for RS*, and no symmetric/unknown key object is ever accepted.
    """
    kty = kty_for_alg(alg)
    if kty == "RSA":
        return isinstance(key, rsa.RSAPublicKey) and key.key_size >= MIN_RSA_BITS
    if kty == "EC":
        return isinstance(key, ec.EllipticCurvePublicKey) and key.curve.name == _CURVE_FOR_ALG.get(alg)
    if kty == "OKP":
        return isinstance(key, ed25519.Ed25519PublicKey | ed448.Ed448PublicKey)
    return False


class _FetchError(Exception):
    """Internal: a JWKS/discovery fetch failed. Never carries upstream response content."""


@dataclass(frozen=True, slots=True)
class _Key:
    kid: str | None
    kty: str
    alg: str | None  # alg declared in the JWK, if any
    key: Any


_METADATA_ADDRS = frozenset(
    ipaddress.ip_address(a) for a in ("169.254.169.254", "fd00:ec2::254", "100.100.100.200")
)


def _is_local_host(host: str) -> bool:
    return host in _LOCAL_HOSTS or host.endswith(".localhost")


def _forbidden_address(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Loopback, link-local (incl. 169.254.169.254 / fe80::), unspecified, multicast, reserved and
    well-known cloud metadata addresses. RFC1918 space is allowed (internal IdPs are normal)."""
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return (
        addr in _METADATA_ADDRS
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_unspecified
        or addr.is_multicast
        or addr.is_reserved
    )


async def _resolve_host(host: str) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [str(i[4][0]).split("%", 1)[0] for i in infos]


class JWKSCache:
    """Per-IdP signing-key cache.

    Rules (all required):
    - jwks_uri from config, else OIDC discovery of `{issuer}/.well-known/openid-configuration`
      (discovered `issuer` must equal config issuer exactly; the discovered jwks_uri must be https
      and its host the issuer's or in idp.jwks_allowed_hosts). Every endpoint must be https and
      must not be/resolve to loopback, link-local or metadata addresses; http + loopback is only
      possible with the explicit idp.jwks_allow_insecure_local opt-in (tests/dev).
    - Responses are streamed: oversize Content-Length is refused up front and the read aborts
      once MAX_JWKS_BYTES is exceeded.
    - Cache keyed by `kid`; TTL = idp.jwks_cache_ttl_s; refresh is single-flight (asyncio.Lock).
    - Unknown `kid` triggers ONE forced refresh, rate-limited to idp.jwks_min_refresh_interval_s
      (a flood of bogus kids must not hammer the IdP); still unknown -> AuthFailure.UNKNOWN_KID.
    - If a refresh fails, serve cached keys up to idp.jwks_stale_if_error_s old; with no usable
      cache raise AuthenticationError(JWKS_UNAVAILABLE).
    - Only keys with use in (None,"sig") and a kty/alg compatible with idp.algorithms are kept.
    - Emits telemetry.record_jwks_refresh(outcome=hit|refreshed|stale|failed) and a
      SPAN_JWKS_REFRESH span on network fetches.

    Every fetch *attempt* (successful or not) counts against the min refresh interval, so an
    unreachable IdP is also not hammered by a stream of requests.
    """

    def __init__(
        self,
        idp: IdPConfig,
        http: httpx.AsyncClient,
        telemetry: Telemetry,
        *,
        clock: Callable[[], float] = time.monotonic,
        resolver: Callable[[str], Awaitable[list[str]]] = _resolve_host,
    ) -> None:
        self.idp = idp
        self._resolver = resolver
        self._http = http
        self._telemetry = telemetry
        self._clock = clock
        self._lock = asyncio.Lock()
        self._keys: list[_Key] = []
        self._fetched_at: float | None = None
        self._last_attempt: float | None = None
        self._generation = 0
        self._jwks_uri: str | None = idp.jwks_uri

    # ------------------------------------------------------------------ public API

    async def get_key(self, kid: str | None, alg: str) -> Any:
        """Return a cryptography public key object usable by PyJWT for `alg`.
        `kid` may be None only if the JWKS holds exactly one candidate key."""
        if alg not in self.idp.algorithms:
            raise AuthenticationError(AuthFailure.ALG_NOT_ALLOWED)

        fresh = self._is_fresh()
        if not fresh:
            await self._try_refresh()
            if not self._is_usable():
                raise AuthenticationError(AuthFailure.JWKS_UNAVAILABLE)
            fresh = self._is_fresh()

        found = self._lookup(kid, alg)
        if found is None and kid is None and self._ambiguous_no_kid(alg):
            raise AuthenticationError(AuthFailure.UNKNOWN_KID)  # fail fast: no refresh can fix it
        if found is None and kid is not None and self._kid_known_but_incompatible(kid):
            raise AuthenticationError(AuthFailure.ALG_NOT_ALLOWED)
        if found is None:
            # Possible key rotation. Rate limited by the min refresh interval; a refresh that
            # just happened in this call counts as the attempt, so no second fetch occurs.
            await self._try_refresh()
            if not self._is_usable():
                raise AuthenticationError(AuthFailure.JWKS_UNAVAILABLE)
            fresh = self._is_fresh()
            found = self._lookup(kid, alg)
            if found is None:
                if kid is not None and self._kid_known_but_incompatible(kid):
                    raise AuthenticationError(AuthFailure.ALG_NOT_ALLOWED)
                raise AuthenticationError(AuthFailure.UNKNOWN_KID)

        if fresh:
            self._telemetry.record_jwks_refresh(issuer=self.idp.issuer, outcome="hit")
        else:
            self._telemetry.record_jwks_refresh(issuer=self.idp.issuer, outcome="stale")
        return found.key

    async def refresh(self, *, force: bool = False) -> None:
        """Fetch + replace the key set (respects min refresh interval unless force).

        Raises AuthenticationError(JWKS_UNAVAILABLE) if no usable keys are cached afterwards."""
        await self._try_refresh(force=force)
        if not self._is_usable():
            raise AuthenticationError(AuthFailure.JWKS_UNAVAILABLE)

    async def warm(self) -> None:
        """Best-effort prefetch at startup; swallow errors (readiness reports status)."""
        try:
            await self._try_refresh(force=True)
        except Exception:
            logger.warning("jwks.warm_failed", extra={"event": "jwks.warm_failed", "idp": self.idp.name})

    @property
    def ready(self) -> bool:
        """True if at least one usable key is cached (used by /readyz)."""
        return self._is_usable()

    # ------------------------------------------------------------------ cache state

    def _age(self) -> float | None:
        return None if self._fetched_at is None else self._clock() - self._fetched_at

    def _is_fresh(self) -> bool:
        age = self._age()
        return bool(self._keys) and age is not None and age <= self.idp.jwks_cache_ttl_s

    def _is_usable(self) -> bool:
        age = self._age()
        limit = max(self.idp.jwks_cache_ttl_s, self.idp.jwks_stale_if_error_s)
        return bool(self._keys) and age is not None and age <= limit

    def _lookup(self, kid: str | None, alg: str) -> _Key | None:
        """Key for (kid, alg). With a kid: kid AND kty/alg compatibility (a duplicate kid across
        RSA/EC must resolve to the compatible key). Without: exactly one compatible key."""
        candidates = [k for k in self._keys if self._compatible(k, alg) and (kid is None or k.kid == kid)]
        if kid is None:
            return candidates[0] if len(candidates) == 1 else None
        return candidates[0] if candidates else None

    def _ambiguous_no_kid(self, alg: str) -> bool:
        """No-kid token and several compatible keys: a refresh cannot make this unambiguous."""
        return sum(1 for k in self._keys if self._compatible(k, alg)) > 1

    def _kid_known_but_incompatible(self, kid: str) -> bool:
        return any(k.kid == kid for k in self._keys)

    @staticmethod
    def _compatible(k: _Key, alg: str) -> bool:
        return (k.alg is None or k.alg == alg) and key_matches_alg(k.key, alg)

    # ------------------------------------------------------------------ refresh

    async def _try_refresh(self, *, force: bool = False) -> None:
        """Single-flight, rate-limited refresh. Never raises; failures keep the old cache."""
        generation = self._generation
        async with self._lock:
            if self._generation != generation:
                return  # another task refreshed while we waited
            now = self._clock()
            if (
                not force
                and self._last_attempt is not None
                and now - self._last_attempt < self.idp.jwks_min_refresh_interval_s
            ):
                return
            self._last_attempt = now
            self._generation += 1
            try:
                with self._telemetry.span(SPAN_JWKS_REFRESH, {"idp.issuer": self.idp.issuer}):
                    keys = await self._fetch()
            except _FetchError:
                self._telemetry.record_jwks_refresh(issuer=self.idp.issuer, outcome="failed")
                return
            except Exception:
                logger.warning("jwks.refresh_error", extra={"event": "jwks.refresh_error"})
                self._telemetry.record_jwks_refresh(issuer=self.idp.issuer, outcome="failed")
                return
            self._keys = keys
            self._fetched_at = self._clock()
            self._telemetry.record_jwks_refresh(issuer=self.idp.issuer, outcome="refreshed")

    def _check_scheme(self, url: str) -> str:
        """Hostname of an acceptable endpoint URL, else _FetchError. https always; plain http only
        to a loopback name and only with the explicit jwks_allow_insecure_local opt-in."""
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if not host or parts.username is not None or parts.password is not None:
            raise _FetchError("bad url")
        if parts.scheme == "https":
            return host
        if parts.scheme == "http" and self.idp.jwks_allow_insecure_local and _is_local_host(host):
            return host
        raise _FetchError("insecure url")

    async def _check_address(self, host: str) -> None:
        """Refuse hosts that are, or resolve to, loopback/link-local/metadata addresses (SSRF).
        The connection re-resolves, so a DNS-rebinding race remains; fetches are rate limited and
        redirects are never followed. An unresolvable host just fails at connect time."""
        if self.idp.jwks_allow_insecure_local:
            return
        if _is_local_host(host):
            raise _FetchError("forbidden host")
        try:
            addrs = [ipaddress.ip_address(host)]
        except ValueError:
            try:
                addrs = [ipaddress.ip_address(a) for a in await self._resolver(host)]
            except (OSError, ValueError):
                return
        if any(_forbidden_address(a) for a in addrs):
            raise _FetchError("forbidden address")

    def _check_discovered_host(self, uri: str) -> None:
        host = self._check_scheme(uri)
        allowed = {(urlsplit(self.idp.issuer).hostname or "").lower()}
        allowed.update(h.lower() for h in self.idp.jwks_allowed_hosts)
        if host not in allowed:
            raise _FetchError("jwks_uri host not allowed")

    async def _get_json(self, url: str) -> dict[str, Any]:
        host = self._check_scheme(url)
        await self._check_address(host)
        buf = bytearray()
        try:
            async with self._http.stream(
                "GET",
                url,
                timeout=self.idp.http_timeout_s,
                headers={"Accept": "application/json"},
                follow_redirects=False,
            ) as resp:
                if resp.status_code != 200:
                    raise _FetchError(f"status {resp.status_code}")
                declared = resp.headers.get("content-length", "")
                if declared.isascii() and declared.isdigit() and int(declared) > MAX_JWKS_BYTES:
                    raise _FetchError("too large")
                async for chunk in resp.aiter_bytes():
                    buf += chunk
                    if len(buf) > MAX_JWKS_BYTES:
                        raise _FetchError("too large")  # abort: never buffer the rest
        except httpx.HTTPError as exc:
            raise _FetchError(type(exc).__name__) from None
        try:
            data = json.loads(bytes(buf))
        except ValueError:
            raise _FetchError("invalid json") from None
        if not isinstance(data, dict):
            raise _FetchError("unexpected json")
        return data

    async def _resolve_uri(self) -> str:
        if self._jwks_uri is None:
            base = self.idp.issuer.rstrip("/")
            doc = await self._get_json(f"{base}/.well-known/openid-configuration")
            if doc.get("issuer") != self.idp.issuer:
                raise _FetchError("issuer mismatch")
            uri = doc.get("jwks_uri")
            if not isinstance(uri, str):
                raise _FetchError("no jwks_uri")
            self._check_discovered_host(uri)  # never cache an unacceptable discovered URI
            self._jwks_uri = uri
        return self._jwks_uri

    async def _fetch(self) -> list[_Key]:
        uri = await self._resolve_uri()
        doc = await self._get_json(uri)
        raw = doc.get("keys")
        if not isinstance(raw, list):
            raise _FetchError("no keys")
        keys = [k for item in raw if (k := self._parse_jwk(item)) is not None]
        if not keys:
            raise _FetchError("no usable keys")  # never replace a good cache with nothing
        return keys

    def _parse_jwk(self, item: Any) -> _Key | None:
        if not isinstance(item, dict):
            return None
        kty, use, alg, kid = item.get("kty"), item.get("use"), item.get("alg"), item.get("kid")
        if not isinstance(kty, str) or use not in (None, "sig"):
            return None
        if kid is not None and not isinstance(kid, str):
            return None
        if alg is not None and (not isinstance(alg, str) or alg not in self.idp.algorithms):
            return None
        if any(p in item for p in ("d", "p", "q", "k")):  # private / symmetric material: ignore
            return None
        usable_algs = [alg] if alg else [a for a in self.idp.algorithms if kty_for_alg(a) == kty]
        if not usable_algs:
            return None
        try:
            key = jwt.get_algorithm_by_name(usable_algs[0]).from_jwk(json.dumps(item))
        except Exception:
            return None
        if not any(key_matches_alg(key, a) for a in usable_algs):
            return None
        return _Key(kid=kid, kty=kty, alg=alg, key=key)
