# AI Gateway - Design

Status: v1 contract. Python 3.13, FastAPI, httpx, pydantic v2, PyJWT, OpenTelemetry (OTLP only).
Skeleton in `src/ai_gateway/` is the source of truth for types; this document is the source of truth for behaviour.

## 1. Goals and non-goals

One gateway in front of (a) internal HTTP APIs, (b) MCP servers (streamable HTTP) and (c) LLM providers (Anthropic/OpenAI style).
It authenticates callers (OAuth2/OIDC JWTs), authorizes them by scope against a declarative **default-deny** policy, applies per-client phased rate limiting / token quotas, proxies (with streaming), and emits first-class OpenTelemetry telemetry.

Non-goals (v1): multi-instance shared limits (Redis later, behind `StateStore`), request/response body transformation beyond MCP `tools/list` filtering and LLM usage tapping, token exchange / on-behalf-of, mTLS, a management API, hot reload (restart to reconfigure; engine is immutable so reload is a later swap).

## 2. Architecture

```
client --TLS--> [ASGI: request-id] -> FastAPI catch-all -> GatewayPipeline
                                       |
     authn (auth/) -> authz (policy/) -> rate limit (ratelimit/) -> proxy (proxy/) -> accounting
         |                |                   |                         |
         +---------- Telemetry facade (observability/) -- OTLP --> collector (traces, metrics, logs, audit)
                                              |
                              StateStore (MemoryStateStore | future Redis)
```

Modules (one owner each, see section 11): `config`, `models`, `errors`, `context`, `services`, `pipeline`, `app`, `main` (core); `auth/`; `policy/`; `ratelimit/`; `observability/`; `proxy/`.
Dependency rule (no cycles): `models, errors, context, config` <- `observability` <- `auth, policy, ratelimit, proxy` <- `services` <- `pipeline` <- `app`.
Only `observability/otel.py` imports the OTel SDK; everyone else uses the `Telemetry` facade (exception: `proxy/headers.py` may call `opentelemetry.propagate.inject`).

## 3. Request pipeline

| # | Stage | Span | What happens | Failure |
|---|-------|------|--------------|---------|
| 0 | request-id | (server span from FastAPI instrumentation) | accept inbound `X-Request-ID` if `^[A-Za-z0-9._-]{8,128}$` else uuid4; bind `RequestContext` to ContextVar; `trace_id` from active span; `requests.active +1`; echo header on every response | - |
| 1 | authn | `gateway.authn` | `Authorization: Bearer` -> `TokenValidator.validate` -> `Principal`; client label for metrics (registered id / `unregistered`) | 401 `invalid_token` (+`WWW-Authenticate`), 503 `idp_unavailable`, 403 if `unknown_clients: deny` |
| 1b | block check | (in `gateway.ratelimit`) | `limiter.is_blocked(client_id)` before reading body or touching policy | 429 `client_blocked` + `Retry-After` |
| 2 | authz | `gateway.authz` | normalize path; `resolve_route`; buffer body for MCP/LLM (<= `max_body_bytes`); `adapter.classify` -> `Resource[]`; `authorize_all`; audit event (allow and deny) | 403 `access_denied` (identical body for all reasons), 400 malformed, 413 too large |
| 3 | rate limit | `gateway.ratelimit` (+ `.delay`) | request limiter phases; for LLM also `reserve_tokens(estimate)` | DELAY then continue; 429 `rate_limited` / `token_quota_exceeded` |
| 4 | proxy | `gateway.proxy` (parent of httpx CLIENT span) | build upstream request, inject `traceparent`, stream response back | 502 `upstream_error`, 504 `upstream_timeout` |
| 5 | accounting | `gateway.accounting` | after body completes/aborts: reconcile tokens, token + request metrics, access log | never alters the response |

Notes:
- Authz runs before rate limiting so denied probes do not consume quota; the cost is that a client can spam 403s freely. Mitigation (v1): every deny is audited and counted in `gateway.authz.decisions`; (v2) count denies against a separate small limiter bucket.
- Path normalization (`policy.matchers.normalize_path`) happens once; the proxy forwards the normalized path so authz and upstream always agree (no `..`/`%2f`/`//` bypass).
- Allowed responses carry `RateLimit-Limit/Remaining/Reset/Policy` headers from the request-scope decision.
- Client disconnect: cancel -> `ProxyResponse.aclose()`, release any delay slot, reconcile tokens with `actual=0` unless usage was already observed.

## 4. Authentication (`auth/`)

- Multiple IdPs; selected by **exact** match of unverified `iss` against `idps[].issuer`. No `jku/x5u/jwk` header is ever honored; JWKS URL comes only from config or OIDC discovery of the configured issuer (discovered `issuer` must equal configured).
- Algorithm allow-list per IdP (default RS256/ES256/PS256). Config validation forbids `none` and `HS*`; the validator re-checks and requires key type to match alg family, closing the HS/RS key-confusion hole.
- Required: signature, `exp` (+ `leeway_s`), `iss`, `aud` (intersects `audiences`); `nbf`/`iat` checked if present; optional `max_token_lifetime_s`.
- JWKS cache: per IdP, keyed by `kid`, TTL `jwks_cache_ttl_s`, single-flight refresh, unknown `kid` => at most one forced refresh per `jwks_min_refresh_interval_s`, stale-if-error for `jwks_stale_if_error_s`, `/readyz` reflects cache state.
- Client credentials: no special grant handling at the gateway (the IdP issues the token). `client_id` = first present of `client_id_claims` (default `client_id`, `azp`, `sub`); `is_client_credentials = (sub == client_id)`. Scopes from `scope` (space string) or `scp` (list).
- `Principal` never contains the raw token; `claims` holds only an allow-listed subset.

## 5. Authorization policy (`policy/`)

Policy YAML (see `config/examples/policies/default.yaml`):

```yaml
upstreams: {<name>: {kind: http|mcp|llm, base_url, prefix, strip_prefix, credential{env,header,scheme}, provider, timeout_s}}
rules:
  - id: <unique>
    effect: allow|deny            # default allow; deny may omit scopes (applies to everyone)
    scopes_any: [..]              # principal has at least one
    scopes_all: [..]              # principal has all
    http: [{upstream, methods, paths}]       # paths: globs on upstream-relative path; * = segment, ** = any depth
    mcp:  [{upstream, tools, methods}]       # tools: globs for tools/call; methods: JSON-RPC method globs
    llm:  [{upstream, models, max_tokens_per_request}]
```

Semantics: **default deny**; any matching deny grant wins; otherwise any matching allow grant (from a rule whose scope condition the principal satisfies) allows; routes with no matching upstream prefix are denied (`NO_ROUTE`).
MCP specifics: `tools/call` is authorized by tool name; other JSON-RPC methods by `methods` globs (default: `initialize`, `notifications/*`, `ping`, `tools/list`; `resources/*`, `prompts/*`, `sampling/*` are off unless granted); batch requests need every message allowed; `tools/list` responses are filtered to tools the principal may call; GET/DELETE on the MCP endpoint need any MCP grant for the upstream.
LLM specifics: `model` from the JSON body must match a grant; missing model on POST is rejected (400); `max_tokens_per_request` caps/clamps the request.
The client always receives the same 403 body; the specific `DecisionReason` and `missing_scopes_hint` go only to audit.

## 6. Rate limiting and backpressure (`ratelimit/`)

Per `client_id`; policy = `defaults.rate_limit` overlaid with `clients.<id>.limits` (re-validated; fractions must satisfy `0 < delay_start <= reject_at <= block_at`).

Counting: sliding-window counter (current + previous fixed bucket weighted) of **all attempts** (also rejected ones). `ratio = estimate / requests_per_window`.

| Phase | Condition | Behaviour |
|-------|-----------|-----------|
| ALLOW | `ratio < delay_start` | pass |
| 1 DELAY | `delay_start <= ratio <= reject_at` | `asyncio.sleep(delay_for(ratio))`, linear 0 -> `max_delay_ms` (cap); at most `max_concurrent_delayed` concurrent waiters per client, excess escalate to phase 2. `max_delay_ms: 0` disables the phase |
| 2 REJECT | `ratio > reject_at` | 429 `rate_limited`, `Retry-After`, `RateLimit-*` |
| 3 BLOCK | `ratio >= block_at` continuously for `block_sustained_seconds` | store `block` key with TTL `block_duration_seconds`; every request from that client returns 429 `client_blocked` + `Retry-After` straight after authn, before any body read, policy or upstream work; no counters are touched while blocked |

Sustained-abuse timer: `set_if_absent(abuse_since)` when `ratio >= block_at`; cleared when ratio drops below. `block_sustained_seconds: 0` blocks immediately.
Token quota (LLM): separate fixed-window counter over `token_window_seconds`; pre-flight `reserve_tokens(estimate)` where `estimate = ceil(prompt_chars/4) + max_tokens` (default 1024); on overflow roll back and 429 `token_quota_exceeded`; after the response `reconcile_tokens(estimate, actual)` using upstream `usage` (Anthropic `usage.input/output_tokens`; OpenAI `prompt/completion_tokens`, with `stream_options.include_usage` injected). Token scope never delays and never triggers a block.
Store failure: `defaults.on_limiter_error` (`fail_open` default: allow + log + metric; `fail_closed`: 429).
Unknown clients get `defaults.rate_limit` (or 403 when `unknown_clients: deny`). The in-memory store is per-process; run a single instance until a Redis `StateStore` exists.

## 7. Proxy (`proxy/`)

- One shared `httpx.AsyncClient` for upstreams (separate one for JWKS), no redirects, no automatic retries, per-upstream timeouts.
- Upstream URL = `base_url` + relative path (+ prefix if `strip_prefix: false`) + query.
- Headers: drop hop-by-hop, inbound `Authorization`/`Cookie`, and `X-Gateway-*`; add upstream credential from env var, `X-Request-ID`, and (optional) `X-Gateway-Client-Id` / `X-Gateway-Subject`. The caller's token is **never** forwarded. `traceparent`/`tracestate` injected.
- Streaming: SSE and chunked bodies are relayed chunk by chunk with no buffering; MCP `Mcp-Session-Id`/`Last-Event-ID` pass through; HTTP routes stream request bodies, MCP/LLM routes buffer (bounded) because the body is needed for authz.
- Adapters (`Adapter` protocol): `HttpAdapter`, `McpAdapter`, `LlmAdapter` implement `classify`, `estimate_tokens`, `prepare`, `process_response`.

## 8. Observability (OTel only; no Prometheus)

Resource: `service.name`, `service.version`, `deployment.environment`. Export: OTLP (grpc or http/protobuf) for traces, metrics, logs. W3C `traceparent` extracted from inbound requests (FastAPI instrumentation) and injected upstream (httpx instrumentation + explicit inject). Names live in `observability/schema.py`.

Spans (children of the server span): `gateway.authn` (+`gateway.authn.jwks_refresh`), `gateway.authz`, `gateway.ratelimit` (+`.delay`), `gateway.proxy` (parent of httpx CLIENT span), `gateway.accounting`. Common attributes: `gateway.request_id`, `gateway.client_id`, `gateway.upstream`, `gateway.upstream.kind`, `gateway.policy.rule_id`, `gateway.ratelimit.phase`, `gen_ai.request.model`, `gateway.mcp.tool`. Never `Authorization`, tokens or bodies.

Metrics:

| Name | Type/unit | Attributes |
|------|-----------|------------|
| `gateway.requests` | counter | method, status, upstream, upstream.kind, client_id |
| `gateway.request.duration` | histogram s | same |
| `gateway.requests.active` | up-down counter | upstream |
| `gateway.ratelimit.decisions` | counter | client_id, ratelimit.phase, ratelimit.scope |
| `gateway.ratelimit.delay.duration` | histogram s | client_id |
| `gateway.auth.failures` | counter | auth.failure_reason, idp.issuer |
| `gateway.authz.decisions` | counter | authz.decision, authz.reason, upstream |
| `gateway.upstream.duration` | histogram s | upstream, kind, status |
| `gateway.upstream.errors` | counter | upstream, kind, error.type |
| `gateway.llm.tokens` | counter {token} | gen_ai.token.type (input/output), gen_ai.request.model, gen_ai.system, client_id, upstream |
| `gateway.jwks.refreshes` | counter | idp.issuer, jwks.outcome |

Cardinality: `client_id` label is the registered id, else `unregistered` / `anonymous`; paths are never labels.

Logs: one JSON object per line on stdout and via OTLP logs. Fields: `timestamp, level, message, logger, event, trace_id, span_id, request_id, client_id, stream` plus event-specific keys. Access log event `request.completed` carries `method, route(upstream), status, duration_ms, bytes, rl_phase`. Redaction of `schema.REDACT_KEYS` is mandatory in the formatter; JWT library exceptions are mapped to `AuthFailure` codes and never logged verbatim.
Audit stream: logger scope `ai_gateway.audit`, `stream="audit"`, optional own OTLP endpoint, unsampled. Event types: `authn.failure`, `authz.decision` (allow and deny), `ratelimit.decision` (delay/reject/block; throttled to 1 per client per phase per 10 s), `ratelimit.client_blocked`, `ratelimit.client_unblocked`. Fields: see `observability/audit.py::AuditEvent` (`request_id, trace_id, client_id, subject, issuer, token_id(jti), decision, reason, rule_id, upstream, resource_kind, method, path, mcp_method, tool, model, scopes_matched, scopes_missing_hint, client_ip`).

## 9. Failure modes

| Failure | Behaviour |
|---------|-----------|
| IdP/JWKS unreachable | serve cached keys until stale limit; else 503 `idp_unavailable`; never accept unverified tokens |
| Unknown `kid` flood | one refresh per cooldown; rest get 401 |
| Clock skew | `leeway_s` (<=300) |
| Policy/config invalid | process refuses to start (`ConfigError`) |
| Upstream timeout / connect error | 504 / 502, `upstream.errors` metric, no retry; delay slot + token reservation released/refunded |
| Upstream 5xx | passed through; counted as upstream error |
| Client disconnects mid-stream | upstream closed, slots released, tokens reconciled with observed usage (else refunded) |
| Store error | per `on_limiter_error` |
| Collector down | OTel SDK batches/drops; gateway never blocks on telemetry |
| Memory growth | `MemoryStateStore` TTL + key cap sweep; block/abuse keys expire by TTL |
| Process restart | in-memory counters and blocks reset (documented limitation until Redis) |

## 10. Security considerations

- Default deny at every layer: unknown route, unmatched rule, missing model, unparsable MCP body, unknown client (optional).
- Token handling: never logged, never in spans/metrics/audit, never forwarded; upstream credentials come from env vars, not config files.
- Confused deputy: gateway uses its own upstream credentials; identity is conveyed via `X-Gateway-*` headers only to upstreams that opt in, and inbound `X-Gateway-*` is stripped.
- Path normalization before authz; same path forwarded; reject encoded slashes and NULs.
- Body limits (413) before buffering; per-upstream timeouts; no redirect following (SSRF via upstream 3xx).
- Upstream URLs only come from config; clients cannot choose a destination.
- `X-Forwarded-For` trusted only if `server.trust_forwarded_for` and only from `server.trusted_proxies` (explicit IPs/CIDRs; `*` refused).
- Error bodies are uniform (no policy internals, no route enumeration); detail only in audit.
- Block fast-path denies abusive clients before parsing bodies, protecting the process.
- Supply chain: pin versions in lockfile at implementation time; CI runs `ruff`, `mypy`, `pytest`.

## 11. Ownership and public contracts

Rules: stay inside your files; import others only through the signatures below; change a shared type via the core owner. Each owner writes tests under `tests/<module>/` (shared helpers in `tests/support.py`: `KeyPair.mint/jwks`).

| Owner | Files | Public surface other modules may call |
|-------|-------|----------------------------------------|
| **core** | `models.py`, `errors.py`, `context.py`, `config.py`, `services.py`, `pipeline.py`, `app.py`, `main.py`, `tests/support.py`, `tests/conftest.py` | `load_config(path) -> LoadedConfig`; `LoadedConfig.limits_for(client_id) -> LimitPolicy`; `LoadedConfig.is_registered(client_id) -> bool`; `get_ctx() -> RequestContext \| None`; `new_request_id(inbound: str \| None) -> str`; `build_services(config, *, telemetry=None, store=None, jwks_http=None, upstream_http=None) -> Services`; `create_app(config_path=None, *, services=None) -> FastAPI` |
| **auth** | `auth/*` | `extract_bearer(authorization: str \| None) -> str`; `TokenValidator(idps, http, telemetry, *, clock).validate(token: str) -> Principal` (raises `AuthenticationError`); `.warm() -> None`; `.ready -> bool`; `JWKSCache(idp, http, telemetry, *, clock).get_key(kid, alg) -> key` |
| **policy** | `policy/*` | `PolicyEngine(policy: PolicyConfig)`: `.resolve_route(path: str) -> RouteMatch \| None`; `.authorize(principal, resource) -> PolicyDecision`; `.authorize_all(principal, resources) -> PolicyDecision`; `.tool_filter(principal, upstream) -> Callable[[str], bool]`; `matchers.normalize_path(path) -> str` (raises `ValueError`) |
| **ratelimit** | `ratelimit/*` | `StateStore` protocol (`incr, get, set, set_if_absent, ttl, delete, acquire, release, aclose`); `MemoryStateStore(*, clock, max_keys)`; `RateLimiter(store, telemetry, *, clock, sleep, on_error)`: `await .is_blocked(client_id) -> LimitDecision \| None`; `await .check_request(client_id, policy) -> LimitDecision`; `await .apply_delay(decision) -> float`; `await .release(decision)`; `await .reserve_tokens(client_id, policy, estimated) -> LimitDecision`; `await .reconcile_tokens(client_id, policy, estimated, actual)` |
| **observability** | `observability/*` | `Telemetry` protocol + `NullTelemetry` (done): `span(name, attrs)`, `add_active`, `record_request(ctx)`, `record_auth_failure(reason, *, issuer)`, `record_authz(decision, ctx)`, `record_rate_limit(decision, ctx)`, `record_upstream(*, upstream, kind, duration_s, status, error_type)`, `record_llm_tokens(usage, *, upstream, client_label)`, `record_jwks_refresh(*, issuer, outcome)`, `audit(event)`, `await shutdown()`; `setup_telemetry(cfg) -> Telemetry`; `instrument_app(app, cfg)`; `AuditEvent`, `authz_event(ctx, decision)`, `ratelimit_event(ctx, decision)`; `schema.*` constants |
| **proxy** | `proxy/*` | `build_http_client(defaults) -> httpx.AsyncClient`; `Proxy(policy, http, telemetry, defaults)`: `.adapter(kind) -> Adapter`; `await .forward(route, req: ProxyRequest, *, tool_allowed=None, on_usage=None) -> ProxyResponse`; `.validate_credentials()`; `Adapter.classify(route, method, body) -> list[Resource]`; `Adapter.estimate_tokens(body) -> int`; `ProxyResponse.aclose()` |

Contract invariants: `Principal`, `Resource`, `PolicyDecision`, `LimitDecision`, `UsageRecord`, `RequestContext` in `models.py` are frozen except `RequestContext`; errors raised across module boundaries are only `GatewayError` subclasses; no module reads environment variables except `UpstreamCredential.resolve`, `config` env expansion, `main`, and the OTLP auth headers read by `observability/otel.py` from the env var named in `telemetry.otlp.headers_env` (i.e. only env vars explicitly named in config); no module imports `app`, `pipeline` or `services` except `app`/`pipeline`.

Implementation order that avoids blocking: observability(`NullTelemetry` ready) / policy / ratelimit / auth / proxy are independent and can start now; core wires them in last (`build_services`, `pipeline`, `app`) and owns end-to-end tests with `respx` upstreams and `KeyPair` tokens.
