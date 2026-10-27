"""JWKS fetching, caching and key rotation. OWNER: auth."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
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


def _https_or_local(url: str) -> bool:
    parts = urlsplit(url)
    if parts.scheme == "https" and parts.hostname:
        return True
    return parts.scheme == "http" and parts.hostname in _LOCAL_HOSTS


class JWKSCache:
    """Per-IdP signing-key cache.

    Rules (all required):
    - jwks_uri from config, else OIDC discovery of `{issuer}/.well-known/openid-configuration`
      (discovered `issuer` must equal config issuer exactly; https required unless localhost).
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
    ) -> None:
        self.idp = idp
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
        if found is None:
            # Possible key rotation. Rate limited by the min refresh interval; a refresh that
            # just happened in this call counts as the attempt, so no second fetch occurs.
            await self._try_refresh()
            if not self._is_usable():
                raise AuthenticationError(AuthFailure.JWKS_UNAVAILABLE)
            fresh = self._is_fresh()
            found = self._lookup(kid, alg)
            if found is None:
                raise AuthenticationError(AuthFailure.UNKNOWN_KID)

        if not self._compatible(found, alg):
            raise AuthenticationError(AuthFailure.ALG_NOT_ALLOWED)
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
        kty = kty_for_alg(alg)
        if kid is not None:
            for k in self._keys:
                if k.kid == kid:
                    return k
            return None
        candidates = [k for k in self._keys if k.kty == kty and k.alg in (None, alg)]
        return candidates[0] if len(candidates) == 1 else None

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

    async def _get_json(self, url: str) -> dict[str, Any]:
        try:
            resp = await self._http.get(
                url,
                timeout=self.idp.http_timeout_s,
                headers={"Accept": "application/json"},
                follow_redirects=False,
            )
        except httpx.HTTPError as exc:
            raise _FetchError(type(exc).__name__) from None
        if resp.status_code != 200 or len(resp.content) > MAX_JWKS_BYTES:
            raise _FetchError(f"status {resp.status_code}")
        try:
            data = json.loads(resp.content)
        except ValueError:
            raise _FetchError("invalid json") from None
        if not isinstance(data, dict):
            raise _FetchError("unexpected json")
        return data

    async def _resolve_uri(self) -> str:
        if self._jwks_uri is None:
            base = self.idp.issuer.rstrip("/")
            if not _https_or_local(base):
                raise _FetchError("insecure issuer")
            doc = await self._get_json(f"{base}/.well-known/openid-configuration")
            if doc.get("issuer") != self.idp.issuer:
                raise _FetchError("issuer mismatch")
            uri = doc.get("jwks_uri")
            if not isinstance(uri, str):
                raise _FetchError("no jwks_uri")
            self._jwks_uri = uri
        if not _https_or_local(self._jwks_uri):
            self._jwks_uri = self.idp.jwks_uri  # do not cache an unacceptable discovered URI
            if self._jwks_uri is None or not _https_or_local(self._jwks_uri):
                self._jwks_uri = None
                raise _FetchError("insecure jwks_uri")
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
