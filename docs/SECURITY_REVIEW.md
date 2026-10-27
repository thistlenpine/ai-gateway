# Security review - AI gateway v1

Reviewer: security validation agent. Method: read `docs/DESIGN.md` and ADR 0001-0004, then every file in
`src/ai_gateway` adversarially (unauthenticated attacker, and attacker holding a valid low-privilege token),
then wrote executable exploit attempts in `tests/security/` and ran them against the real ASGI app (rig:
`tests/gw_rig.py`, `tests/support.py`) and, for the DoS items, against a live `ai-gateway` (uvicorn + httptools)
process. No source file was modified; defects are reported only. Line numbers are as of this review.

Run: `.venv/bin/python -m pytest tests/security -q` (about 60 s; after remediation: all pass except 2 xfail (SEC-15, SEC-21)).
A test **passes when the gateway behaves securely**. Every test that currently fails because of a real defect
is `pytest.mark.xfail(strict=True, reason="SEC-nn ...")`: the suite goes red (XPASS) the moment a defect is fixed,
which is the signal to delete the marker.

## Summary

| Severity | Count | IDs |
|----------|-------|-----|
| Critical | 1 | SEC-00 |
| High | 1 | SEC-22 |
| Medium | 8 | SEC-01, SEC-03, SEC-05, SEC-08, SEC-16, SEC-18, SEC-23, SEC-24 |
| Low | 11 | SEC-02, SEC-04, SEC-06, SEC-07, SEC-10, SEC-11, SEC-12, SEC-13, SEC-14, SEC-19, SEC-20 |
| Info | 4 | SEC-09, SEC-15, SEC-17, SEC-21 (plus the observations in section 3) |

Headline: the cryptographic and authorization core held up (see section 4 for what was attacked and did not
break), but the **redaction layer is a pre-authentication denial of service**, and nothing bounds per-client
concurrency, so one low-privilege client can starve every other client.

## 1. Findings

Status legend: **Fixed** = remediated, regression test now passes (marker removed). **Open** = reproduced, regression test is an `xfail(strict)` in `tests/security`.

| ID | Sev | Location | Exploit (verified) | Recommended fix | Status |
|----|-----|----------|--------------------|-----------------|--------|
| SEC-00 | **Critical** | `observability/redact.py:61` (`_KV`), called from `scrub_text` :87 via `audit.py:74` (`to_dict`), `logging.py:83`, `otel.py` span hooks | `_KV` starts with `[\w.\-]*` then a keyword, so every start offset rescans the whole word run: **quadratic ReDoS**. `scrub_text("x"*8000)` = 4 s, 16 000 = 16 s, 20 000 = 28 s on the event loop. Unauthenticated: `GET /aaaa...` (6 000 chars) -> 401 in 18 s because `authn.failure` audits the raw path through `scrub_text`. Live uvicorn: 20 KB path = 28 s, and `/healthz` from another connection stalled 4 s behind a single 8 KB request (httptools accepted a 20 KB request line and refused 70 KB, so a single request costs on the order of a minute). Post-auth variant: an allowed MCP tool name of a few KB (`get_xxxx...`, any granted glob) hits the audit `tool` field. OTel warnings that echo an attacker `baggage` header go through the same path. | Anchor the key name with `\b` and bound it (`\b[\w.\-]{0,64}?(?:...)`) or drop the prefix and match keywords only; hard-cap every string handed to `scrub_text` (e.g. 2 KB, truncate with `...`) before any regex; never scrub on the event loop for unbounded input. Add a timing regression test (provided). | Fixed - every string is capped at 2 KB before any regex; `_KV` lost its backtracking prefix (`redact.py`). |
| SEC-22 | **High** | `proxy/client.py:42-55` (`build_http_client`: one shared pool, `max_connections=200`), `pipeline.py` (no concurrency accounting), `ratelimit/limiter.py` (counts request *starts* only) | Any principal with one MCP grant opens `GET /mcp/<x>` (SSE) repeatedly; the limiter only counts starts per window, streams are unbounded (`DEFAULT_MAX_RESPONSE_BYTES` exempts SSE) and nothing caps concurrent streams. 250 concurrent held-open streams from one client were all admitted. Past 200 the shared pool starves **every upstream and every client** (pool timeout -> 502/504). Slow-reading clients or slow LLM completions do the same. | Per-client (and global) cap on in-flight proxied requests/streams via the `StateStore` (`acquire`/`release` already exist), max stream lifetime, optional per-upstream pool limits so one upstream cannot starve the others. | Fixed - per-client in-flight cap `max_concurrent_requests` (default 32, 429 `too_many_concurrent_requests`) + `max_stream_seconds` (default 900). Not done: a global/per-upstream pool cap. |
| SEC-01 | Medium | `proxy/_json.py:24` (`loads_strict`), `proxy/mcp.py` `_resource`, `proxy/llm.py` `classify`; body forwarded verbatim (`McpAdapter.prepare` is identity) | Duplicate-key defence is exact-match only. `{"method":"ping","Method":"tools/call","params":{"name":"delete_all"}}` is classified as `ping` (allowed) and forwarded unchanged; Go `encoding/json` / Newtonsoft match keys case-insensitively and the LAST wins, so the upstream runs `delete_all`. Same for `model`/`Model` (LLM grant bypass) and `params.name`/`Name`. | Reject any object whose keys collide after `str.casefold()` (all levels), and/or re-serialise the parsed, validated document and forward *that* (so authz and upstream parse identical bytes). | Fixed - keys colliding after `casefold()` are rejected at every level (`proxy/_json.py`). |
| SEC-03 | Medium | `pipeline.py:_run` (authz before `_rate_limit`), `docs/DESIGN.md` s3 note | Authz denials (403), malformed MCP/LLM bodies (400) and 413s never touch the limiter, so a client can send unlimited denied requests: free audit-log + CPU amplification, never reaches `block`. 60 denied requests against a 10/min client were all answered 403, none 429. Documented v1 trade-off, but the stated mitigation (a deny bucket) is not implemented. | Charge denials to a small separate bucket (design's v2) and apply the block fast path to it; at minimum count `400/403/413` toward `block_at`. | Fixed - 403/400/413 after authentication are charged to the client's window and can reach 429/block. Unauthenticated (401) floods remain unthrottled by design (accepted risk; terminate with an IP limiter at the edge). |
| SEC-05 | Medium | `pipeline.py:340-369` (`_build_response`: `st.streaming = True` at :342, then `v.encode("latin-1")` at :363) | An upstream response header containing non-Latin-1 text (httpx decodes UTF-8; e.g. `Content-Disposition`/`X-Filename` reflecting a request value, `Location: /é`) raises `UnicodeEncodeError` after `st.streaming` was set. Result: generic 500, the upstream connection is never closed, `_finish` never runs: `requests.active` drifts, and for LLM routes the token reservation is **never refunded** (quota burned for the window). A client that can make the upstream reflect input can leak pool connections one request at a time. | Build/encode headers before `st.streaming = True`, or wrap in try/except that calls `presp.aclose()` and `_finish`; encode with `latin-1` + `errors="replace"` or drop invalid headers in `filter_response_headers`. | Fixed - response headers are encoded before `streaming` is set; unencodable ones are dropped; the upstream response is closed if building the response fails. |
| SEC-08 | Medium | `pipeline.py:_authenticate` (:214 `registered = principal.client_id in clients`), `auth/validator.py:214` (`_principal`), limiter keys `rl:{client_id}:*` | The rate-limit/registry identity is the bare `client_id`/`azp`/`sub` string. (a) All end users of one OAuth app (same `azp`) share a bucket and a **block**: alice's flood blocks bob for 5 minutes (reproduced). (b) With several IdPs, a token from IdP B whose `client_id` equals a registered client of IdP A lands in A's bucket/registration (reproduced: B's request got 429 from A's usage) - quota theft/DoS and impersonation of A's limits (e.g. an unlimited `agent-prod`). | Namespace the identity `"{idp.name}:{client_id}"` for registry, limiter and metrics; key per-user buckets by `(client_id, sub)` for non-client-credentials tokens (`is_client_credentials` already exists) with the app bucket as a parent. | Fixed - limiter identity is `{idp}:{client_id}` (+ `|{sub}` for end-user tokens); clients.yaml entries may pin `issuer` (required with >1 IdP). |
| SEC-16 | Medium (when `trust_forwarded_for: true`) | `main.py:34` (`forwarded_allow_ips="*"`) | Enabling the flag makes uvicorn trust `X-Forwarded-For/Proto` from **any** TCP peer, so any client forges `client_ip` in audit records and scheme. Default (`false`) is safe. | Add `server.trusted_proxies: [cidr]` and pass it to uvicorn; refuse `trust_forwarded_for: true` without it. | Fixed - `server.trusted_proxies` (explicit IPs/CIDRs) is required with `trust_forwarded_for: true`; `*` refused. |
| SEC-18 | Medium (config dependent) | `policy/matchers.py:23-48` (`_segment_regex`, `_name_regex`: `*` -> `.*`/`[^/]*` backtracking) | A grant glob with >=3 wildcards is cubic on attacker-controlled input: `name_match("*a*a*a*b", "a"*1000)` = 33 s, `"*a*a*a*a*b"` x300 = 21 s; tool names/models come from a 2 MB body. Needs an operator-written pattern such as `*_read_*_v*`, but nothing rejects it. | Replace with an iterative two-pointer glob (O(n*m)) or `fnmatch`-style non-backtracking matcher; cap wildcards per pattern at load. | Fixed - glob matcher is chunk based (no backtracking, O(n*m)). |
| SEC-23 | Medium | `pipeline.py:_authenticate` (single validation), `proxy/streaming.py`/`client.py` (unbounded SSE) | Authentication happens once. A stream (MCP GET/SSE, LLM streaming) keeps flowing after the token's `exp` (and after scope revocation at the IdP). Reproduced: 2 s token, 3.6 s stream delivered in full. | Close streams at `min(Principal.expires_at + leeway, max_stream_s)`; document revocation latency = stream lifetime. | Fixed - streams end at token `exp` or `max_stream_seconds`; SSE ends cleanly, other bodies are aborted. |
| SEC-24 | Medium | `pipeline.py:244-283` (`_authorize` -> `_read_body` before any scope check), `adapter.classify` errors -> 400 | Body is buffered (<= 2 MB) and parsed, and 400/413 returned, **before** the principal's grants are consulted. An authenticated caller with no MCP/LLM scope (a) learns which prefixes exist and their kind (400 vs the uniform 403 - contradicts DESIGN s5 "no route enumeration"), (b) forces a 2 MB buffer + JSON parse per request. | Pre-check "does the principal have *any* grant for this upstream" (pure scope test) and return the uniform 403 before reading the body. | Fixed - a scope-only `has_any_grant` check returns the uniform 403 before the body is read or parsed. |
| SEC-02 | Low | `context.py:13,33` (`re.match` + `$`) | `^...{8,128}$` with `.match` accepts a trailing `\n`: `X-Request-ID: valid-id-1\n` is accepted and echoed into the response header. uvicorn rejects the header value and returns 500 *after* the upstream call already happened (reproduced at ASGI level only; a real socket cannot carry a raw LF in a header value because parsers split on it, so header/response splitting is not reachable). | `fullmatch` / `\Z`. | Fixed - `fullmatch`. |
| SEC-04 | Low | `auth/validator.py:182` (`exp <= now - leeway`) | `exp: NaN` or `Infinity` (`json.loads` accepts them) makes `exp <= x` false: a validly signed token never expires. Requires a signed token from a misbehaving IdP, so defence in depth. `-Infinity`, `nbf`/`iat` non-finite are rejected correctly. | `math.isfinite()` check on `exp`/`nbf`/`iat`; pass `parse_constant` that raises to the JWT decode. | Fixed - non-finite `exp`/`nbf`/`iat` are rejected. |
| SEC-06 | Low | `proxy/headers.py:127` (`filter_response_headers`) | Upstream `Set-Cookie` (and `Server`, `X-Powered-By`, `Via`, `WWW-Authenticate`, `Access-Control-*`) are relayed verbatim: one upstream can plant cookies scoped to the gateway host that other upstreams' responses/browsers then honour. | Response header allow-list per upstream kind (default deny `Set-Cookie`). | Fixed - `Set-Cookie`, `Set-Cookie2`, `Server`, `X-Powered-By`, `Via` dropped. `Access-Control-*` and `WWW-Authenticate` are still relayed (dropping them would break browser/MCP-OAuth clients). |
| SEC-07 | Low | `proxy/headers.py:127`, `proxy/client.py:80` | Upstream 3xx `Location` (internal hostnames/ports) is passed to the caller. Combined with SEC-07b below it is easy to trigger on the MCP root. Redirects are correctly **not followed** (verified; `169.254.169.254` never contacted). | Rewrite or strip `Location` that points at the upstream origin (nginx `proxy_redirect` behaviour). | Fixed - upstream-origin `Location` is rewritten to a gateway-relative path under the public prefix; foreign absolute `Location` is dropped. |
| SEC-07b | Low (correctness) | `proxy/client.py:66-81` (`build_upstream_url`) | **Known item - trailing slash.** `rel == "/"` is appended to `base_url`, so `base_url: http://mcp/mcp` + request to the prefix root is called as `http://mcp/mcp/`. Not a security issue: the path the gateway authorised and the path it forwards are the same normalised string (verified for 7 normalisation forms), there is no authz bypass and no SSRF. It is a **correctness bug**: the MCP spec's single endpoint `/mcp` is a different resource from `/mcp/` on Starlette/FastMCP/Go servers, which answer 307 to the internal origin; the gateway does not follow redirects (correct) and relays the 307 (SEC-07), so standard MCP clients cannot initialise. `tests/e2e` asserts the slash form, so fixing it needs those tests updated. Related nit: HTTP routes always stream `request.stream()`, so even bodiless GETs go upstream as `Transfer-Encoding: chunked`. | When `relative_path == "/"` forward exactly `base_url` (no added slash); only send chunked when a body exists. | Fixed - the prefix root is forwarded as exactly `base_url` (no added slash); bodiless HTTP requests are no longer sent chunked. e2e/proxy tests updated. |
| SEC-10 | Low | `proxy/_json.py:27` (`json.loads(bytes)` sniffs UTF-16/32/BOM) | Bodies encoded as UTF-16/UTF-32/UTF-8-BOM are accepted for classification and forwarded verbatim: parser differential against a strict UTF-8 upstream. | Decode explicitly as UTF-8 (`body.decode("utf-8")`, reject BOM) before `json.loads`. | Fixed - bodies are decoded as strict UTF-8, BOM rejected. |
| SEC-11 | Low | `pipeline.py:424-428`, `app.py:45` (`raw_path.decode("latin-1")`) | Raw non-ASCII bytes in the request target are reinterpreted as Latin-1 (0xFF -> `U+00FF`; UTF-8 `é` -> mojibake) and then forwarded percent-encoded, so authz and upstream agree with each other but not with what the client sent. Browsers always percent-encode, so low impact. | Reject any byte >= 0x80 in `raw_path` (400). | Fixed - outermost guard returns 400 for any non-ASCII byte in the raw path. |
| SEC-12 | Low | `observability/otel.py` `instrument_app` (third-party `opentelemetry/instrumentation/asgi` decodes `query_string` as UTF-8), `proxy/client.py:81` (`.encode("ascii")`) | Non-ASCII bytes in the query string (`?a=%E9` sent raw as 0xE9; h11 rejects, httptools accepts) raise `UnicodeDecodeError` in the OTel ASGI wrapper **before authentication**: uvicorn answers 500 and logs a full traceback per request (log amplification, no data leak). Authenticated requests would hit a second failure in `build_upstream_url`. | Catch/normalise in `RequestContextMiddleware` (reject non-ASCII query bytes with 400 before the instrumented app) and validate ASCII in `build_upstream_url` -> `BadRequestError`. | Fixed - same guard covers the query string (runs ahead of the OTel wrapper). |
| SEC-13 | Low | `ratelimit/store.py:87-96` (`_maybe_sweep` evicts "closest to expiry") | At the key cap (100 000) the keys that expire first are evicted; a `block` key (TTL 300 s) expires before fresh counters (TTL 2*window), so a flood of distinct `client_id`s (needs many valid identities; e.g. an IdP with open client registration plus `unknown_clients: default_limits`) lifts an existing block and resets counters. Reproduced with a 300-key store. | Evict by LRU/lowest-priority, never evict `block`/`abuse_since` keys; prefer `unknown_clients: deny`; keep `max_keys` well above expected clients. | Fixed - `block`/`abuse_since` keys are never evicted by the key cap. |
| SEC-14 | Low | `proxy/headers.py:30-54,120`, `observability/otel.py:255` | `baggage` and `tracestate` are on the drop list, but the global propagator is `CompositePropagator([TraceContext, W3CBaggage])` and the httpx instrumentation re-injects them: attacker-chosen `baggage: role=admin,...`, `tracestate` and trace-id reach every upstream (reproduced). Upstreams that trust baggage are spoofable; attacker also controls sampling/trace id. | Use `TraceContextTextMapPropagator` only; extract inbound context only from a trusted proxy or start a new trace with a link. | Fixed - propagator is traceparent-only; baggage and tracestate are never relayed. |
| SEC-19 | Low | `config.py:300-306` (`yaml.safe_load`) | PyYAML keeps the **last** duplicate mapping key silently: a duplicated `effect`, `scopes_any`, `unknown_clients` changes policy with no error and is easy to miss in review. (`safe_load` itself is correct: `!!python/object` payloads are refused and never executed - verified.) | Loader subclass that raises on duplicate keys. | Fixed - duplicate YAML mapping keys are a config error. |
| SEC-20 | Low | `config.py:36-55` (`IdPConfig`) | Accepts empty `issuer`/`audiences` strings (e.g. from an empty `${ENV}` expansion) and `jwks_min_refresh_interval_s <= 0`, `jwks_cache_ttl_s <= 0` (removes the unknown-`kid` cooldown, so a `kid` flood hits the IdP on every request). | `min_length=1` on strings, `https://` issuer pattern, `ge=1` on intervals. | Fixed - non-empty issuer/audiences, issuer must be a URL, JWKS intervals >= 1. |
| SEC-09 | Info | `config.py:66` | The forbidden-algorithm check upper-cases everything except the literal `none`, so `algorithms: [None]`/`[NONE]` pass config validation. Harmless at runtime (the validator rejects any `alg.lower() == "none"` and the key-family check backs it up) but defeats the documented "config refuses none". | Compare `a.lower()`. | Fixed - algorithm check is case-insensitive for `none`. |
| SEC-15 | Info | `observability/metrics.py:75-80` (`_Bounded`) | First-come label budget: a caller with any LLM scope burns the 64 `gen_ai.request.model` slots with invented names matching the grant glob; real models are then reported as `other`. Cardinality is correctly bounded (verified at 150 models, 120 issuers/clients/methods/paths). | Seed the budget from the policy's configured models; collapse unknown to `other` instead of first-come. | Open (deliberately not fixed) - low value: cardinality is already bounded; the cost is only mislabelled model names. Needs the policy's model list plumbed into metrics. |
| SEC-17 | Info | `main.py:27` | uvicorn sends `Server: uvicorn` (fingerprinting). | `server_header=False`. | Fixed - `server_header=False`. |
| SEC-21 | Info | `pyproject.toml`, `.github/workflows/ci.yml` | Dependencies are `>=` only, no lockfile/hashes/`pip-audit`, Actions pinned by tag, base images by tag - DESIGN s10 requires pinned versions. | Lockfile (uv/pip-tools with hashes), `pip-audit` in CI, pin Actions by SHA, image digests. | Open (deliberately not fixed) - needs a lockfile/CI/registry decision (uv/pip-tools hashes, pip-audit, SHA-pinned Actions); not a code change. |

### Functional defect found on the way (not security)
`PolicyEngine._precheck` (`policy/engine.py`) rejects every LLM resource with `model is None`, so non-POST calls such
as `GET /v1/models` can never be granted (docstring of `LlmAdapter.classify` promises a `models: ["*"]` grant works).
Fails closed; fix either side.

## 2. Reading the xfail markers

`xfail(strict=True)` ids map 1:1 to the table. Parametrised cases that share an id (e.g. SEC-01 has four payloads) must
all fail today. When you fix an item, run the suite: the strict XPASS tells you to remove the marker.

## 3. Observations and accepted risks (no test, or documented behaviour)

- **Defaults that fail open** (by design, documented): `unknown_clients: default_limits` (any valid token from a
  trusted IdP is admitted with default limits; use `deny` in production), `on_limiter_error: fail_open`,
  `server.host: 0.0.0.0`, no TLS in the process (terminate in front; tokens are bearer secrets).
- **`Mcp-Session-Id` is passed through unbound** to the principal (required by the transport). If a session id leaks, any
  principal with an MCP grant can use or `DELETE` it. The gateway does send the *current* caller's
  `X-Gateway-Client-Id`, so MCP servers must bind sessions to it (documented in
  `test_mcp_session_header_passthrough_documented`). `GET`/`DELETE` on the MCP endpoint need only *any* MCP grant.
- **Unauthenticated floods** are not throttled (no pre-auth/IP limiter) and each 401 emits an audit record and metric
  (log amplification). JWT validation is cheap and rejects oversize/garbage before any crypto or JWKS traffic (tested).
- **Deny rules are case-sensitive and path-literal**: against a case-insensitive upstream (IIS) `/ADMIN` bypasses a deny
  on `/admin/**`. Prefer allow-lists; do not rely on deny globs for such upstreams.
- Upstream 5xx bodies are relayed verbatim (design choice) and may expose stack traces of internal APIs.
- Client-chosen `X-Request-ID` is accepted if well-formed and not unique: log-correlation spoofing.
- Scrubber evasion: `\b`-anchored key patterns miss secrets glued to word characters (`get_sk-ant-...`); inputs are
  attacker-chosen so no foreign secret leaks, but do not treat the scrubber as a guarantee.
- `SseParser` buffers an unterminated event without bound (trusted upstream only).
- A discovered `jwks_uri` may name any `https` host (SSRF-lite through a *trusted* IdP); plain `http` non-localhost
  and issuer mismatches are refused (tested).
- `/readyz` is unauthenticated and reveals JWKS availability.
- Slow request bodies (buffered MCP/LLM routes) have no per-request read deadline beyond the server's.
- Trace context: inbound `traceparent` ids/sampled flag are honoured (see SEC-14).

## 4. What was attacked and held (all covered by passing tests)

- **JWT**: `alg` none (4 case variants), HS256/384/512 with PEM/JWK/`n` as secret, `kty:oct` key in JWKS, ES256 header
  over RSA key, wrong EC curve, 1024-bit RSA, `jku`/`x5u`/`jwk`/`x5c` headers (never fetched), `crit`, kid injection
  (path, SQLi, NUL, 5 KB, non-string types), empty/extra/short signature, payload tampering, 8 KB+ tokens (no JWKS
  traffic), non-ASCII, 3 000-deep JSON in header and payload (clean 401), wrong-typed `exp/iss/aud/sub/scope`, leeway
  boundaries (exclusive), `nbf`/`iat` future, `max_token_lifetime_s`, multi-IdP signature/audience/issuer
  cross-use, issuer exact match (7 variants), unknown issuer never fetched nor used as metric label.
- **JWKS**: unknown-`kid` flood = at most one fetch per cooldown, single-flight cold start (20 concurrent -> 1 fetch),
  rotated key rejected after refresh, stale-if-error honoured and expires, empty/poisoned JWKS cannot evict a good
  cache, private-material/`use:enc` keys ignored, >1 MB JWKS ignored, redirects not followed, discovery issuer
  mismatch / plain-http `jwks_uri` / plain-http issuer refused, IdP down + cold cache = 503, never accept.
- **Authz**: 40+ hostile paths (dot segments in every encoding, double encoding, encoded slash/backslash/NUL, overlong
  UTF-8, fullwidth/dot-leader look-alikes, `;`, `?`, `#`, whitespace dots, case, prefix confusion, absolute-form,
  authority tricks) - only exact normalised allowed paths ever reach the upstream and the forwarded path equals the
  authorised path; method override headers; odd methods; scope exactness (wildcards, prefix/suffix, case, zero-width,
  homoglyph, wrong claims); MCP batches hiding denied calls (any position), malformed/nested/oversized batches, 20
  method/name shapes, exact-key duplicate keys at every level (incl. `\u` escapes); `tools/list` filtering over
  JSON/SSE/batch/chunked/multi-line and malformed entries; LLM model matrix, `max_tokens` clamp for 13 value types;
  bodies: NaN, 5 000-digit ints, 1.9 M-deep nesting, trailing data, concatenated objects, lone surrogates; chunked
  and lying `Content-Length` bodies; disabled client; duplicate `Authorization` headers.
- **Rate limiting**: spoofed identity headers, many tokens/`sub`/`azp` for one client, path/method/query variation,
  block applies to every path and before body read (429 not 413), block isolated per client, delay slots capped per
  client, released on completion and on client disconnect, never starve other clients, delay hard-capped (30 s),
  token reservation refunded on connect error/timeout/4xx/5xx and never taken on denied requests, quota enforced
  against odd `max_tokens` values, hostile negative `usage` cannot create phantom capacity, store stays bounded.
- **Proxy**: caller `Authorization`/`Cookie`/API-key headers (including duplicates) never forwarded; 21 identity /
  `X-Forwarded-*` / `Forwarded` / `X-Original-URL` / baggage-class headers cannot be spoofed (identity headers always
  derive from the token, hostile `sub` sanitised); hop-by-hop and `Connection`-listed headers stripped; CRLF/NUL in
  header names and values dropped; upstream credential replaces caller-supplied values; credential absent from every
  response/log/span/metric; missing credential at runtime = generic 502 and nothing sent; redirects not followed;
  host cannot be changed via path or headers; response framing headers neutralised; generic upstream errors; bounded
  request/response sizes.
- **Observability**: tokens, API keys, passwords, cookies in query/path/headers/bodies/exception text never reach
  logs, spans, audit or metrics; every log line stays one JSON object under log-forging payloads; labels bounded under
  120-way client/issuer/method/path/model fuzzing; audit records every authn failure and authz decision with reason;
  client IP ignores `X-Forwarded-For` by default.
- **Config**: unsafe YAML tags refused, inline secrets refused (`extra=forbid`), typo'd/ambiguous policies refuse to
  load (fail closed), env expansion fails closed and cannot inject YAML structure, example config has no secrets and is
  default-deny (deny wins), Dockerfile runs as a non-root user, `.env`/keys git-ignored.

## 5. Remediation notes (config / behaviour changes)

- New limits (`defaults.rate_limit` or per client): `max_concurrent_requests` (32), `max_stream_seconds` (900).
- New `server.trusted_proxies`; `trust_forwarded_for: true` without it is a startup error.
- `clients.yaml` entries accept `issuer`; it is mandatory when more than one IdP is configured. Rate-limit
  keys are now `rl:{idp}:{client_id}[|{sub}]:*` (previously bare `client_id`).
- Denied/malformed authenticated requests now consume rate-limit quota.
- MCP prefix-root requests go to `base_url` without a trailing slash.
- Config validation is stricter: duplicate YAML keys, empty issuer/audience, `jwks_*_s < 1`.
- Case-variant duplicate JSON keys, non-UTF-8 bodies and non-ASCII request targets are 400s.
- Not mitigated: unauthenticated flood throttling, global upstream pool partitioning (SEC-22 residual), SEC-15, SEC-21.
