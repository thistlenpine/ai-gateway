"""Configuration models and YAML loader. OWNER: core.

Files (all YAML, `${VAR}` / `${VAR:-default}` env expansion on every string):
  gateway.yaml  - server, idps, telemetry, defaults, pointers to the other files
  policies/*.yaml - upstreams + scope rules (merged; duplicate upstream/rule ids are errors)
  clients.yaml  - per-client rate-limit overrides
Secrets (upstream API keys) are NEVER stored in config; credentials name an env var.
"""

from __future__ import annotations

import glob
import ipaddress
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_gateway.models import UpstreamKind

FORBIDDEN_ALGS = frozenset({"NONE", "HS256", "HS384", "HS512"})
DEFAULT_ALGS = ["RS256", "ES256", "PS256"]


class ConfigError(Exception):
    """Raised for any invalid/missing configuration; startup must abort."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- gateway.yaml


class IpLimitConfig(_Strict):
    """Cheap per-IP token bucket applied BEFORE token validation (in-memory, per instance)."""

    enabled: bool = True
    requests_per_window: int = Field(600, gt=0)  # sustained rate
    window_seconds: float = Field(60.0, gt=0)
    burst: int | None = Field(None, gt=0)  # bucket size; None -> requests_per_window
    max_tracked_ips: int = Field(100_000, ge=100)  # memory bound; LRU eviction beyond this


class ServerConfig(_Strict):
    host: str = "0.0.0.0"
    port: int = 8080
    max_body_bytes: int = 2_000_000  # buffered bodies (MCP/LLM); plain HTTP routes stream
    trust_forwarded_for: bool = False
    # Peers (IPs/CIDRs) whose X-Forwarded-* headers uvicorn may trust. Required when
    # trust_forwarded_for is true; "*" is refused (SEC-16).
    trusted_proxies: list[str] = Field(default_factory=list)
    # Pre-authentication limiter keyed on the client IP (the one uvicorn derived, i.e. honouring
    # trust_forwarded_for/trusted_proxies). Behind a proxy without trust_forwarded_for every caller
    # shares the proxy's IP: enable forwarding, raise the limit or set `enabled: false`.
    ip_rate_limit: IpLimitConfig = Field(default_factory=IpLimitConfig)

    @field_validator("trusted_proxies")
    @classmethod
    def _explicit_proxies(cls, v: list[str]) -> list[str]:
        cleaned = [p.strip() for p in v]
        for p in cleaned:
            if not p or "*" in p:
                raise ValueError("trusted_proxies entries must be explicit IPs/CIDRs (no wildcards)")
            try:
                ipaddress.ip_network(p, strict=False)
            except ValueError as exc:
                raise ValueError(f"invalid trusted proxy {p!r}") from exc
        return cleaned

    @model_validator(mode="after")
    def _forwarded_needs_proxies(self) -> ServerConfig:
        if self.trust_forwarded_for and not self.trusted_proxies:
            raise ValueError("trust_forwarded_for: true requires server.trusted_proxies (explicit IPs/CIDRs)")
        return self


class IdPConfig(_Strict):
    """One trusted issuer. Tokens are matched to an IdP by exact `iss` string."""

    name: str = Field(min_length=1)
    issuer: str = Field(min_length=1)
    jwks_uri: str | None = None  # None -> OIDC discovery at {issuer}/.well-known/openid-configuration
    audiences: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1
    )  # token `aud` must contain at least one
    algorithms: list[str] = Field(default_factory=lambda: list(DEFAULT_ALGS))
    leeway_s: int = Field(30, ge=0, le=300)  # clock skew for exp/nbf
    jwks_cache_ttl_s: int = Field(600, ge=1)
    jwks_min_refresh_interval_s: int = Field(30, ge=1)  # unknown-kid refresh cooldown (anti-DoS)
    jwks_stale_if_error_s: int = 3600  # keep serving cached keys if IdP is down
    http_timeout_s: float = 5.0
    client_id_claims: list[str] = Field(default_factory=lambda: ["client_id", "azp", "sub"])
    scope_claims: list[str] = Field(default_factory=lambda: ["scope", "scp"])  # str (space) or list
    # Reject tokens whose lifetime is longer than this: exp - iat, or exp - now when `iat` is absent.
    # There is no revocation list, so this bounds how long a leaked token stays usable. 0 or null
    # = unbounded (explicit opt-in; not recommended).
    max_token_lifetime_s: int | None = Field(86_400, ge=0)
    # Hosts (besides the issuer's own) a *discovered* jwks_uri may point at. Exact, case-insensitive.
    jwks_allowed_hosts: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)
    # Test/dev opt-in: allow plain http and loopback/link-local/metadata addresses for the issuer
    # and JWKS endpoints. Never enable in production.
    jwks_allow_insecure_local: bool = False

    @field_validator("max_token_lifetime_s")
    @classmethod
    def _zero_is_unbounded(cls, v: int | None) -> int | None:
        return None if v == 0 else v

    @field_validator("issuer")
    @classmethod
    def _issuer_is_url(cls, v: str) -> str:
        u = urlsplit(v)
        if not u.netloc or u.scheme not in ("https", "http"):
            raise ValueError(
                "issuer must be a URL such as https://idp.example.com (plain http is refused at runtime)"
            )
        return v

    @field_validator("algorithms")
    @classmethod
    def _no_weak_algs(cls, v: list[str]) -> list[str]:
        bad = FORBIDDEN_ALGS.intersection(a.upper() for a in v)
        if bad or not v:
            raise ValueError(f"algorithms must be non-empty asymmetric; forbidden: {sorted(bad)}")
        return v


class OtlpConfig(_Strict):
    endpoint: str = "http://localhost:4317"
    protocol: Literal["grpc", "http/protobuf"] = "grpc"
    headers_env: str | None = None  # env var holding "k=v,k2=v2" (auth headers); never inline
    insecure: bool = True
    timeout_s: float = 10.0


class AuditConfig(_Strict):
    enabled: bool = True
    stdout: bool = True  # JSON lines with "stream":"audit"
    otlp_endpoint: str | None = None  # None -> reuse telemetry.otlp


class TelemetryConfig(_Strict):
    service_name: str = "ai-gateway"
    service_version: str = "0.1.0"
    environment: str = "dev"
    enabled: bool = True  # False -> NullTelemetry (tests)
    otlp: OtlpConfig = Field(default_factory=OtlpConfig)
    trace_sampler_ratio: float = Field(1.0, ge=0, le=1)
    metrics_export_interval_s: float = 15.0
    log_level: str = "INFO"
    log_stdout_json: bool = True
    audit: AuditConfig = Field(default_factory=AuditConfig)


class LimitPolicy(_Strict):
    """Per-client limits. Fractions are of `requests_per_window` (sliding window)."""

    requests_per_window: int = Field(600, gt=0)
    window_seconds: float = Field(60.0, gt=0)
    delay_start: float = Field(0.8, gt=0)  # phase 1 starts at this fraction of quota
    reject_at: float = Field(1.0, gt=0)  # phase 2: ratio > reject_at -> 429
    block_at: float = Field(1.5, gt=0)  # phase 3 trigger ratio ...
    block_sustained_seconds: float = Field(30.0, ge=0)  # ... held for this long
    block_duration_seconds: float = Field(300.0, gt=0)  # cooldown length
    max_delay_ms: int = Field(2000, ge=0)  # cap on phase-1 wait; 0 disables phase 1
    max_concurrent_delayed: int = Field(10, ge=0)  # waiters per client; excess -> phase 2
    tokens_per_window: int | None = Field(None, gt=0)  # LLM token quota; None = unlimited
    token_window_seconds: float = Field(60.0, gt=0)
    # Concurrency / lifetime caps (SEC-22/23): in-flight proxied requests per client identity, and
    # the longest any one response (incl. SSE) may stay open. Streams also end at token expiry.
    max_concurrent_requests: int = Field(32, gt=0)
    max_stream_seconds: float = Field(900.0, gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> LimitPolicy:
        if not (0 < self.delay_start <= self.reject_at <= self.block_at):
            raise ValueError("require 0 < delay_start <= reject_at <= block_at")
        return self


class DefaultsConfig(_Strict):
    rate_limit: LimitPolicy = Field(default_factory=LimitPolicy)
    # "deny": a valid token whose client_id is not in clients.yaml gets 403. "default_limits" admits
    # any valid token from a trusted IdP with defaults.rate_limit (fail-open; warned at startup).
    unknown_clients: Literal["default_limits", "deny"] = "deny"
    upstream_timeout_s: float = 30.0
    upstream_connect_timeout_s: float = 5.0
    on_limiter_error: Literal["fail_open", "fail_closed"] = "fail_open"


class GatewayConfig(_Strict):
    server: ServerConfig = Field(default_factory=ServerConfig)
    idps: list[IdPConfig] = Field(min_length=1)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    policy_files: list[str] = Field(default_factory=lambda: ["policies/*.yaml"])
    clients_file: str | None = "clients.yaml"


# --------------------------------------------------------------------------- policy files

DEFAULT_MCP_METHODS = ["initialize", "notifications/*", "ping", "tools/list"]


class UpstreamCredential(_Strict):
    """How the gateway authenticates TO the upstream. The caller's token is never forwarded."""

    header: str = "Authorization"
    scheme: str | None = "Bearer"  # None -> raw value (e.g. x-api-key)
    env: str  # env var with the secret

    def resolve(self) -> str:
        """Header value for the upstream. Raises ConfigError if env var is unset."""
        secret = os.environ.get(self.env)
        if not secret:
            raise ConfigError(f"credential env var {self.env!r} is not set")
        return f"{self.scheme} {secret}" if self.scheme else secret


class UpstreamConfig(_Strict):
    kind: UpstreamKind
    base_url: str
    prefix: str  # public path prefix, e.g. "/billing"; longest prefix wins
    strip_prefix: bool = True
    timeout_s: float | None = None  # None -> defaults.upstream_timeout_s
    credential: UpstreamCredential | None = None
    forward_identity_headers: bool = True  # X-Gateway-Client-Id / -Subject / X-Request-ID
    provider: Literal["anthropic", "openai"] | None = None  # LLM usage/body dialect

    @model_validator(mode="after")
    def _check(self) -> UpstreamConfig:
        if not self.prefix.startswith("/") or (len(self.prefix) > 1 and self.prefix.endswith("/")):
            raise ValueError("prefix must start with '/' and not end with '/'")
        if self.kind is UpstreamKind.LLM and self.provider is None:
            raise ValueError("llm upstreams require `provider`")
        return self


class HttpGrant(_Strict):
    upstream: str
    methods: list[str] = Field(default_factory=lambda: ["GET", "HEAD"])
    paths: list[str]  # globs on the upstream-relative path: `*` one segment, `**` any depth


class McpGrant(_Strict):
    upstream: str
    tools: list[str] = Field(default_factory=list)  # globs; required to call tools/call
    methods: list[str] = Field(default_factory=lambda: list(DEFAULT_MCP_METHODS))  # JSON-RPC


class LlmGrant(_Strict):
    upstream: str
    models: list[str]  # globs on the request `model`
    max_tokens_per_request: int | None = None


class Rule(_Strict):
    """allow rules need scopes; deny rules may omit them (= applies to every principal)."""

    id: str
    effect: Literal["allow", "deny"] = "allow"
    scopes_any: list[str] = Field(default_factory=list)
    scopes_all: list[str] = Field(default_factory=list)
    http: list[HttpGrant] = Field(default_factory=list)
    mcp: list[McpGrant] = Field(default_factory=list)
    llm: list[LlmGrant] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Rule:
        if self.effect == "allow" and not (self.scopes_any or self.scopes_all):
            raise ValueError(f"allow rule {self.id!r} must require at least one scope")
        if not (self.http or self.mcp or self.llm):
            raise ValueError(f"rule {self.id!r} has no grants")
        return self


class PolicyConfig(_Strict):
    upstreams: dict[str, UpstreamConfig] = Field(default_factory=dict)
    rules: list[Rule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _cross_validate(self) -> PolicyConfig:
        prefixes = [u.prefix for u in self.upstreams.values()]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("duplicate upstream prefix")
        ids = [r.id for r in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate rule id")
        for r in self.rules:
            for kind, grants in (
                (UpstreamKind.HTTP, r.http),
                (UpstreamKind.MCP, r.mcp),
                (UpstreamKind.LLM, r.llm),
            ):
                for g in grants:
                    up = self.upstreams.get(g.upstream)
                    if up is None or up.kind is not kind:
                        raise ValueError(
                            f"rule {r.id!r}: {kind.value} grant references unknown/"
                            f"wrong-kind upstream {g.upstream!r}"
                        )
        return self


# --------------------------------------------------------------------------- clients.yaml


class ClientConfig(_Strict):
    description: str = ""
    enabled: bool = True
    issuer: str | None = None  # when set, only tokens from this issuer match this registration
    limits: dict[str, Any] = Field(default_factory=dict)  # partial LimitPolicy overrides


class ClientsConfig(_Strict):
    clients: dict[str, ClientConfig] = Field(default_factory=dict)


# --------------------------------------------------------------------------- loader


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    gateway: GatewayConfig
    policy: PolicyConfig
    clients: ClientsConfig

    def client_for(self, client_id: str, issuer: str | None = None) -> ClientConfig | None:
        """Registration for `client_id`; when it pins an issuer, the token's issuer must match."""
        client = self.clients.clients.get(client_id)
        if client is None or (client.issuer is not None and issuer is not None and client.issuer != issuer):
            return None
        return client

    def is_registered(self, client_id: str, issuer: str | None = None) -> bool:
        return self.client_for(client_id, issuer) is not None

    def limits_for(self, client_id: str, issuer: str | None = None) -> LimitPolicy:
        """defaults.rate_limit overlaid with the client's overrides (re-validated)."""
        base = self.gateway.defaults.rate_limit.model_dump()
        client = self.client_for(client_id, issuer)
        if client is None:
            return self.gateway.defaults.rate_limit
        try:
            return LimitPolicy.model_validate({**base, **client.limits})
        except ValueError as e:  # pragma: no cover - guarded at load time
            raise ConfigError(f"client {client_id!r}: {e}") from e


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(obj: Any) -> Any:
    if isinstance(obj, str):

        def sub(m: re.Match[str]) -> str:
            val = os.environ.get(m.group(1), m.group(2))
            if val is None:
                raise ConfigError(f"environment variable {m.group(1)!r} is not set")
            return val

        return _ENV.sub(sub, obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


class _UniqueKeyLoader(yaml.SafeLoader):
    """safe_load that refuses duplicate mapping keys (PyYAML silently keeps the last; SEC-19)."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            try:
                dup = key in seen
            except TypeError:
                continue  # unhashable: the base class raises its own error
            if dup:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.load(path.read_text(), Loader=_UniqueKeyLoader) or {}
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return _expand(data)  # type: ignore[no-any-return]


def load_config(gateway_yaml: str | Path) -> LoadedConfig:
    """Load gateway.yaml + referenced policy/clients files; validate everything or raise."""
    from pydantic import ValidationError

    path = Path(gateway_yaml).resolve()
    base = path.parent
    try:
        gateway = GatewayConfig.model_validate(_read_yaml(path))

        upstreams: dict[str, Any] = {}
        rules: list[Any] = []
        for pattern in gateway.policy_files:
            for f in sorted(glob.glob(str(base / pattern))):
                doc = _read_yaml(Path(f))
                for name, up in (doc.get("upstreams") or {}).items():
                    if name in upstreams:
                        raise ConfigError(f"{f}: duplicate upstream {name!r}")
                    upstreams[name] = up
                rules.extend(doc.get("rules") or [])
        policy = PolicyConfig.model_validate({"upstreams": upstreams, "rules": rules})

        clients = ClientsConfig()
        if gateway.clients_file and (base / gateway.clients_file).exists():
            clients = ClientsConfig.model_validate(_read_yaml(base / gateway.clients_file))
        if len(gateway.idps) > 1:
            unpinned = sorted(c for c, v in clients.clients.items() if v.issuer is None)
            if unpinned:
                raise ConfigError(
                    f"clients {unpinned} must set `issuer` when more than one IdP is configured "
                    "(a client_id is only unique within its issuer)"
                )
        loaded = LoadedConfig(gateway, policy, clients)
        for cid in clients.clients:  # fail at startup, not at first request
            loaded.limits_for(cid)
    except ValidationError as e:
        raise ConfigError(str(e)) from e
    return loaded
