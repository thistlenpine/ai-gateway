# ai-gateway

An OAuth2/OIDC-secured gateway in front of internal HTTP APIs, MCP servers (streamable HTTP) and
LLM providers (Anthropic / OpenAI style). It authenticates callers with JWTs, authorizes them by
scope against a declarative **default-deny** policy, applies per-client phased rate limiting and
token quotas, proxies (including SSE streaming) and emits OpenTelemetry traces, metrics, logs and
an audit stream. Behaviour is specified in [docs/DESIGN.md](docs/DESIGN.md); decisions are in
[docs/adr](docs/adr).

## Quickstart

```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# upstream credentials are read from env vars named in the policy (never stored in config)
export MCP_TOOLS_TOKEN=... ANTHROPIC_API_KEY=... OPENAI_API_KEY=...

ai-gateway --config config/examples/gateway.yaml        # or: GATEWAY_CONFIG=... ai-gateway
curl -s localhost:8080/healthz                          # liveness
curl -s localhost:8080/readyz                           # 200 once JWKS of every IdP is cached
curl -s -H "Authorization: Bearer $TOKEN" localhost:8080/billing/invoices
```

Docker: `docker build -t ai-gateway . && docker run -p 8080:8080 -e ANTHROPIC_API_KEY=... -v $PWD/config:/etc/ai-gateway ai-gateway`.

Development: `pytest -q` (30 s per-test timeout), `ruff check .`, `ruff format --check .`, `mypy src tests`.
Run the suite with plain `pytest` (directory discovery); passing individual test files in arbitrary
interleaved order can confuse pytest because several test directories have a `conftest.py`.
End-to-end tests (`tests/e2e`) run the whole app in-process against a mock upstream and a local
test IdP (RSA key pair + mocked JWKS) with a fake clock for the rate limiter.

## Configuration

All YAML, `${VAR}` / `${VAR:-default}` expansion on every string; any invalid config aborts startup.

| File | Purpose |
|------|---------|
| `gateway.yaml` | `server`, `idps[]`, `telemetry`, `server` (incl. `ip_rate_limit`), `defaults` (rate limit, timeouts, `unknown_clients` = `deny`, `on_limiter_error`), pointers to the files below |
| `policies/*.yaml` | `upstreams` (kind `http`/`mcp`/`llm`, `base_url`, public `prefix`, optional `credential: {env, header, scheme}`) and scope `rules` |
| `clients.yaml` | per-client overrides of the rate-limit policy (`requests_per_window`, `tokens_per_window`, ...) |

See `config/examples/` for a complete set. Policy semantics: a request is allowed only if some
allow rule whose scope condition the caller satisfies has a grant matching the request (HTTP
method + path glob, MCP method/tool glob, LLM model glob with optional `max_tokens_per_request`
clamp). Any matching deny rule wins. Unknown prefixes are denied.

## Request pipeline and phases

`request-id -> ip limit -> authn -> block check -> authz -> rate limit -> proxy -> accounting`

Rate limiting is per `client_id` over a sliding window of **all attempts**; `ratio = attempts / requests_per_window`:

1. **Delay** (`delay_start <= ratio <= reject_at`): the request waits, linearly from 0 up to `max_delay_ms`, at most `max_concurrent_delayed` waiters per client (excess escalates to phase 2).
2. **Reject** (`ratio > reject_at`): `429 rate_limited` with `Retry-After` and `RateLimit-Limit/Remaining/Reset/Policy`.
3. **Block** (`ratio >= block_at` for `block_sustained_seconds`): the client is blocked for `block_duration_seconds`; every request is answered `429 client_blocked` right after authn, before the body is read or policy evaluated.

LLM routes also reserve `estimate = ceil(prompt_chars/4) + max_tokens` against `tokens_per_window`
(`429 token_quota_exceeded`) and reconcile with the provider-reported usage afterwards (refund
when the upstream fails).

**Rate limits are per instance.** All limiter state (per-client windows, blocks, token quotas,
concurrency slots and the per-IP limiter) lives in process memory (`StateStore` interface; Redis
later). With N replicas behind a load balancer every limit is effectively multiplied by up to N
(and a block only applies on the replica that issued it): divide `requests_per_window` /
`tokens_per_window` / `ip_rate_limit` by the replica count, or run a single instance, until a
shared store exists.

### Pre-auth per-IP limiter

Before any token work (so floods of garbage/expired tokens cannot burn CPU, audit volume or JWKS
traffic) every request passes a cheap in-memory token bucket keyed on the client IP: default
`server.ip_rate_limit` = 600 requests/min/IP with a burst of 600. Over the limit: `429
ip_rate_limited` with `Retry-After`, counted in `gateway.ratelimit.decisions{scope="ip"}`
(the address is never a metric label). Memory is bounded (`max_tracked_ips`, LRU; IPv6 collapses
to /64). `/healthz` and `/readyz` are exempt. Configure or disable with `server.ip_rate_limit`
(`enabled: false`).

The IP is the one uvicorn derives: the TCP peer, or `X-Forwarded-For` only when
`server.trust_forwarded_for` is set for peers in `server.trusted_proxies`. **Behind a load
balancer without `trust_forwarded_for`, every caller shares the balancer's IP**: enable
forwarding, raise the limit or disable the limiter. This limiter is a backstop, not a WAF: in
production still put an L7 rate limiter / WAF / DDoS protection in front of the gateway.

## Observability

OpenTelemetry only (no Prometheus): OTLP traces, metrics and logs, W3C `traceparent` extracted
inbound and injected upstream. Stage spans `gateway.authn|authz|ratelimit|proxy|accounting`;
metrics `gateway.requests`, `.request.duration`, `.ratelimit.decisions`, `.authz.decisions`,
`.auth.failures`, `.upstream.*`, `.llm.tokens`, `.jwks.refreshes` (names in
`observability/schema.py`). JSON logs on stdout with `trace_id`, `request_id`, `client_id`; an
unsampled **audit** stream (`stream="audit"`) records every authn failure, every authz decision
(allow and deny, with reason and rule) and rate-limit delay/reject/block events. OTLP auth
headers come from the env var named in `telemetry.otlp.headers_env`.

## Security model

- Tokens: exact-`iss` IdP selection, asymmetric algorithms only (`none`/`HS*` rejected), required `exp`/`iss`/`aud`, JWKS only from config/discovery of the configured issuer, rate-limited key refresh, stale-if-error.
- Token lifetime: there is no revocation list, so `idps[].max_token_lifetime_s` (default **86400**) rejects tokens whose lifetime is longer: `exp - iat`, or `exp - now` when `iat` is absent. `0`/`null` disables the bound (explicit opt-in, not recommended); prefer short-lived tokens and rotate signing keys to cut off a leaked token.
- Unknown clients: `defaults.unknown_clients` defaults to **`deny`** (a valid token whose `client_id` is not in `clients.yaml` gets 403). `default_limits` admits any valid token from a trusted IdP and logs a startup warning.
- JWKS fetching: a `kid` selects only keys whose `kty`/`alg` fit the token (a duplicate `kid` across RSA/EC is fine); a token without `kid` is only accepted when exactly one key fits and never forces a refresh. Responses are streamed and capped at 1 MB (up-front `Content-Length` check, read aborted at the cap). Endpoints must be `https`; a discovered `jwks_uri` must be on the issuer's host or in `idps[].jwks_allowed_hosts`; hosts that are or resolve to loopback/link-local/metadata addresses (169.254.169.254, `fe80::`, ...) are refused. Local `http` IdPs for dev/tests need `idps[].jwks_allow_insecure_local: true`. (DNS can still change between the check and the connect; redirects are never followed and fetches are rate limited.)
- Default deny at every layer; the client always gets the same `403 access_denied` body - the reason and missing scopes go to the audit stream only.
- The caller's token is never logged, traced, put in metrics/audit, or forwarded; the gateway authenticates to upstreams with its own credentials (env vars). Identity reaches upstreams via `X-Gateway-*` headers, and inbound `X-Gateway-*`/hop-by-hop/cookie headers are stripped.
- Paths are normalized once (encoded slashes, `..`, NUL rejected) and the same path is forwarded. Upstream URLs come only from config; redirects are not followed.
- Body limits (`server.max_body_bytes`, 413), per-upstream timeouts, error bodies carry no internals (`{"error": {"code", "message", "request_id"}}`).
- Known limits: per-instance rate-limit state (see above); no token revocation (bounded by `max_token_lifetime_s`); denied/malformed requests are audited and count against the caller's window, so the 403/429 a caller sees leaks nothing about which routes or scopes exist; `X-Forwarded-For` is trusted only with `server.trust_forwarded_for`.
