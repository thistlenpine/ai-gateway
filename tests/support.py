"""Shared test helpers (OWNER: core). Mint RS256 tokens + JWKS for any module's tests."""

from __future__ import annotations

import json
import time
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm


class FakeClock:
    """Callable epoch clock that only moves when told to (rate limiter / store tests)."""

    def __init__(self, t: float = 1_000_020.0) -> None:  # multiple of 60: starts at a window edge
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class KeyPair:
    def __init__(self, kid: str = "k1") -> None:
        self.kid = kid
        self.private = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def jwk(self) -> dict[str, Any]:
        d: dict[str, Any] = json.loads(RSAAlgorithm.to_jwk(self.private.public_key()))
        d.update(kid=self.kid, use="sig", alg="RS256")
        return d

    def jwks(self) -> dict[str, Any]:
        return {"keys": [self.jwk()]}

    def mint(
        self,
        *,
        issuer: str = "https://idp.test",
        audience: str = "ai-gateway",
        client_id: str = "client-a",
        scopes: str = "",
        ttl: int = 300,
        alg: str = "RS256",
        **extra: Any,
    ) -> str:
        now = int(time.time())
        claims = {
            "iss": issuer,
            "aud": audience,
            "sub": client_id,
            "client_id": client_id,
            "scope": scopes,
            "iat": now,
            "nbf": now,
            "exp": now + ttl,
            **extra,
        }
        return jwt.encode(claims, self.private, algorithm=alg, headers={"kid": self.kid})
