"""JWT validation -> Principal. OWNER: auth.

See docs/DESIGN.md section 4. Every failure is an AuthenticationError(AuthFailure.X); neither
PyJWT exception text nor any token fragment is ever logged or put into an error message.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import jwt

from ai_gateway.auth.jwks import JWKSCache
from ai_gateway.config import FORBIDDEN_ALGS, IdPConfig
from ai_gateway.errors import AuthenticationError, AuthFailure
from ai_gateway.models import Principal
from ai_gateway.observability.telemetry import Telemetry

MAX_TOKEN_BYTES = 8192
MAX_KID_LENGTH = 256
SAFE_CLAIMS = ("azp", "client_id", "jti", "iat", "nbf", "exp")


def _is_number(v: Any) -> bool:
    # finite only: json.loads accepts NaN/Infinity, which would make `exp <= now` always false (SEC-04)
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


class TokenValidator:
    """Validates bearer JWTs from any configured IdP.

    validate() algorithm (order matters):
      1. Parse header WITHOUT verifying (jwt.get_unverified_header); reject non-JWS, `crit` we
         don't understand, and alg not in the IdP allow-list -> ALG_NOT_ALLOWED. `none` and HS*
         are never allowed (config forbids them; re-check here defensively) so a symmetric
         "key confusion" with a public key is impossible: the key type MUST match alg family.
      2. Peek `iss` from the unverified payload ONLY to select the IdP by exact string match;
         unknown -> UNKNOWN_ISSUER. (No fetching based on token-supplied URLs; jku/x5u/jwk
         headers are ignored.)
      3. key = await JWKSCache.get_key(kid, alg); jwt.decode with algorithms=[alg],
         audience=idp.audiences, issuer=idp.issuer, leeway=idp.leeway_s,
         options require exp (+iss, aud); nbf/iat verified if present.
      4. Extract client_id via idp.client_id_claims (first present), scopes via
         idp.scope_claims (space-separated string or list), enforce idp.max_token_lifetime_s.
      5. Return Principal; is_client_credentials = (sub == client_id) or no end-user claim.
    Map every PyJWT exception to AuthFailure (never leak exception text or token fragments).
    Records telemetry.record_auth_failure on failure and starts SPAN_AUTHN-child spans only
    for JWKS network fetches (the pipeline owns the SPAN_AUTHN span).

    Time checks (exp/nbf/iat, leeway) use the injected `clock` (epoch seconds), not PyJWT's.
    """

    def __init__(
        self,
        idps: Sequence[IdPConfig],
        http: httpx.AsyncClient,
        telemetry: Telemetry,
        *,
        clock: Callable[[], float] = time.time,
        jwks_clock: Callable[[], float] | None = None,
    ) -> None:
        self._idps = list(idps)
        self._http = http
        self._telemetry = telemetry
        self._clock = clock
        self._caches: dict[str, JWKSCache] = {}
        for idp in self._idps:
            if idp.issuer in self._caches:
                raise ValueError(f"duplicate issuer configured: {idp.issuer}")
            self._caches[idp.issuer] = (
                JWKSCache(idp, http, telemetry, clock=jwks_clock)
                if jwks_clock is not None
                else JWKSCache(idp, http, telemetry)
            )
        self._by_issuer = {idp.issuer: idp for idp in self._idps}

    async def validate(self, token: str) -> Principal:
        """Raise ai_gateway.errors.AuthenticationError(AuthFailure.X) on any failure."""
        issuer: list[str | None] = [None]  # known (configured) issuer once selected
        try:
            return await self._validate(token, issuer)
        except AuthenticationError as exc:
            self._telemetry.record_auth_failure(exc.failure.value, issuer=issuer[0])
            raise

    async def warm(self) -> None:
        """Prefetch JWKS for all IdPs (startup)."""
        await asyncio.gather(*(c.warm() for c in self._caches.values()))

    @property
    def ready(self) -> bool:
        """All IdPs have usable cached keys (for /readyz)."""
        return all(c.ready for c in self._caches.values())

    # ------------------------------------------------------------------ internals

    async def _validate(self, token: str, issuer_out: list[str | None]) -> Principal:
        if not isinstance(token, str) or not token or len(token) > MAX_TOKEN_BYTES:
            raise AuthenticationError(AuthFailure.MALFORMED)
        if token.count(".") != 2 or not token.isascii():
            raise AuthenticationError(AuthFailure.MALFORMED)  # not a compact JWS

        # 1. header
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            raise AuthenticationError(AuthFailure.MALFORMED) from None
        alg = header.get("alg")
        if not isinstance(alg, str) or alg.lower() == "none" or alg.upper() in FORBIDDEN_ALGS:
            raise AuthenticationError(AuthFailure.ALG_NOT_ALLOWED)
        if "crit" in header:
            raise AuthenticationError(AuthFailure.MALFORMED)  # we understand no critical ext
        kid = header.get("kid")
        if kid is not None and (not isinstance(kid, str) or not kid or len(kid) > MAX_KID_LENGTH):
            raise AuthenticationError(AuthFailure.MALFORMED)

        # 2. select the IdP by exact issuer match
        try:
            unverified = jwt.decode(token, options={"verify_signature": False})
        except jwt.PyJWTError:
            raise AuthenticationError(AuthFailure.MALFORMED) from None
        iss = unverified.get("iss")
        if not isinstance(iss, str):
            raise AuthenticationError(AuthFailure.MISSING_CLAIM if iss is None else AuthFailure.MALFORMED)
        idp = self._by_issuer.get(iss)
        if idp is None:
            raise AuthenticationError(AuthFailure.UNKNOWN_ISSUER)
        issuer_out[0] = idp.issuer
        if alg not in idp.algorithms:
            raise AuthenticationError(AuthFailure.ALG_NOT_ALLOWED)

        # 3. signature + registered claims
        key = await self._caches[idp.issuer].get_key(kid, alg)
        claims = self._decode(token, key, alg, idp)
        self._check_times(claims, idp)

        # 4/5. principal
        return self._principal(claims, idp)

    @staticmethod
    def _decode(token: str, key: Any, alg: str, idp: IdPConfig) -> dict[str, Any]:
        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                key,
                algorithms=[alg],
                audience=idp.audiences,
                issuer=idp.issuer,
                options={
                    "require": ["exp", "iss", "aud"],
                    # time claims are checked by us against the injected clock
                    "verify_exp": False,
                    "verify_nbf": False,
                    "verify_iat": False,
                },
            )
        except jwt.InvalidSignatureError:
            raise AuthenticationError(AuthFailure.BAD_SIGNATURE) from None
        except jwt.InvalidAudienceError:
            raise AuthenticationError(AuthFailure.BAD_AUDIENCE) from None
        except jwt.InvalidIssuerError:
            raise AuthenticationError(AuthFailure.UNKNOWN_ISSUER) from None
        except jwt.MissingRequiredClaimError:
            raise AuthenticationError(AuthFailure.MISSING_CLAIM) from None
        except (jwt.InvalidAlgorithmError, jwt.InvalidKeyError):
            raise AuthenticationError(AuthFailure.ALG_NOT_ALLOWED) from None
        except jwt.PyJWTError:
            raise AuthenticationError(AuthFailure.MALFORMED) from None
        except Exception:  # defensive: a crypto backend error must never become a 500 or leak
            raise AuthenticationError(AuthFailure.BAD_SIGNATURE) from None
        return claims

    def _check_times(self, claims: dict[str, Any], idp: IdPConfig) -> None:
        now = self._clock()
        leeway = idp.leeway_s
        exp = claims["exp"]
        if not _is_number(exp):
            raise AuthenticationError(AuthFailure.MALFORMED)
        if exp <= now - leeway:
            raise AuthenticationError(AuthFailure.EXPIRED)
        for name in ("nbf", "iat"):
            value = claims.get(name)
            if value is None:
                continue
            if not _is_number(value):
                raise AuthenticationError(AuthFailure.MALFORMED)
            if value > now + leeway:
                raise AuthenticationError(AuthFailure.NOT_YET_VALID)
        if idp.max_token_lifetime_s is not None:
            iat = claims.get("iat")
            if iat is None:
                raise AuthenticationError(AuthFailure.MISSING_CLAIM)
            if exp - iat > idp.max_token_lifetime_s:
                raise AuthenticationError(AuthFailure.MALFORMED)

    @staticmethod
    def _scopes(claims: dict[str, Any], idp: IdPConfig) -> frozenset[str]:
        scopes: set[str] = set()
        for name in idp.scope_claims:
            value = claims.get(name)
            if value is None:
                continue
            if isinstance(value, str):
                scopes.update(value.split())
            elif isinstance(value, list) and all(isinstance(s, str) for s in value):
                scopes.update(part for s in value for part in s.split())
            else:
                raise AuthenticationError(AuthFailure.MALFORMED)
        return frozenset(scopes)

    def _principal(self, claims: dict[str, Any], idp: IdPConfig) -> Principal:
        client_id = next(
            (v for name in idp.client_id_claims if isinstance(v := claims.get(name), str) and v),
            None,
        )
        if client_id is None:
            raise AuthenticationError(AuthFailure.MISSING_CLAIM)
        sub = claims.get("sub")
        if sub is not None and not isinstance(sub, str):
            raise AuthenticationError(AuthFailure.MALFORMED)
        aud = claims["aud"]
        audience = (aud,) if isinstance(aud, str) else tuple(a for a in aud if isinstance(a, str))
        jti = claims.get("jti")
        return Principal(
            client_id=client_id,
            subject=sub or client_id,
            issuer=idp.issuer,
            scopes=self._scopes(claims, idp),
            audience=audience,
            token_id=jti if isinstance(jti, str) else None,
            expires_at=float(claims["exp"]),
            idp=idp.name,
            is_client_credentials=not sub or sub == client_id,
            claims={k: claims[k] for k in SAFE_CLAIMS if k in claims},
        )
