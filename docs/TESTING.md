# Testing

Status: v1. Everything runs offline (no network, no external services) with the project venv.

```bash
uv pip install --python .venv/bin/python -e '.[dev]'     # pytest, pytest-asyncio, respx, pytest-cov, hypothesis, pytest-randomly
```

## How to run

| Goal | Command |
|------|---------|
| Everything (random order, `pytest-randomly` is a dev dependency) | `.venv/bin/pytest` |
| Fixed order / reproduce a failing order | `.venv/bin/pytest -p no:randomly` / `-p randomly --randomly-seed=<seed from the header>` |
| Fast loop (skip real-time + real-server tests) | `.venv/bin/pytest -m "not slow and not integration" --ignore=tests/security` |
| One layer | `pytest tests/property`, `tests/load`, `tests/integration`, `tests/e2e`, ... |
| Coverage (line + branch) | `.venv/bin/pytest --cov=ai_gateway --cov-branch --cov-report=term-missing --ignore=tests/security` |
| More Hypothesis examples | `HYPOTHESIS_PROFILE=thorough .venv/bin/pytest tests/property --timeout=900` (2000 examples/property, ~1 min) |
| Static checks | `.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy` |

Notes:
- Default per-test timeout is 30 s (`pytest-timeout`). Integration tests start real servers and need a free loopback port; they unset proxy env vars themselves.
- `tests/security/` contains regression tests that deliberately feed pathological inputs (very long names/paths). Under `coverage` (python tracing) they are 10x+ slower, which is why the coverage recipe above ignores that directory; run it separately without `--cov`.
- Keep pytest's default import mode (`prepend`). `--import-mode=importlib` breaks the shared helper modules (`gw_rig`, `support`, `obs_rig`, `load.load_rig`, `integration.servers`) because they are imported by name from `pythonpath = ["tests"]`.
- Test directories that need helpers import them with absolute names; do not add a second plain `conftest.py` module name without a package `__init__.py` (duplicate module basenames were the cause of an earlier "fixture not found").

## Layout and what each layer proves

| Layer | Path | Real things | Fake things | Proves |
|-------|------|-------------|-------------|--------|
| Unit | `tests/{auth,policy,ratelimit,observability,proxy}/` | the module under test | collaborators (respx, `FakeClock`, in-memory OTel exporters) | each module's contract in isolation: JWT validation matrix, JWKS cache, policy decisions, limiter phases/store, redaction, adapters |
| In-process e2e | `tests/e2e/`, `tests/gw_rig.py` | full ASGI app, pipeline, policy, limiter | upstreams (`httpx.MockTransport`/respx), IdP, clock | the pipeline order and its observable effects (status codes, headers, spans, metrics, audit) |
| Security | `tests/security/` (separate owner) | as e2e | as e2e | abuse cases: bypass attempts, header smuggling, secret leakage, DoS |
| Property | `tests/property/` | `normalize_path`, `PolicyEngine.resolve_route`, `RateLimiter` + `MemoryStateStore`, `TokenValidator` | clock | invariants for all inputs Hypothesis can find (below) |
| Load / concurrency | `tests/load/` | full ASGI app under `asyncio.gather` of dozens of clients, real tasks | upstream, IdP, clock (fake by default; one real-time smoke) | phases per client without bleed, no leaked slots/tasks/reservations, bounded memory |
| Integration | `tests/integration/` | **uvicorn gateway on an ephemeral port, uvicorn upstream/IdP server, real sockets, subprocess `ai-gateway`**, YAML config path | nothing in the data path (a Starlette app plays the upstream/IdP) | wire-level behaviour: streaming, MCP sessions, LLM usage accounting, graceful shutdown, OIDC discovery |

### Property tests (`tests/property`, Hypothesis profile `ci` = 150 examples)
- `test_prop_paths.py`: `normalize_path` output is either rejected or canonical (starts with `/`, no `//`, no `.`/`..` segments, no encoded separators, idempotent); agrees with an independent reference model; routing a normalized path equals pure literal-prefix semantics (longest whole-segment prefix, `/billing` never matches `/billingx`, `prefix + relative_path` reassembles the path the proxy forwards); un-normalized paths never route; `*` never crosses a segment.
- `test_prop_limiter.py`: allowed requests per fixed window never exceed `reject_at * quota`; phase is monotonic with load; phase is never milder than the classification of the reported ratio; a block lasts exactly `block_duration_seconds` and touches no counters; clients are isolated (a client's decisions in an interleaved run equal its solo run); delay slots are always released (apply, release or cancelled sleep) and never exceed `max_concurrent_delayed`; token counters never go negative across window rolls, reservations never exceed the quota, full refunds restore capacity.
- `test_prop_jwt_scopes.py`: scope parsing (space string, list, `scope`+`scp` union, no empty/whitespace members, wrong types rejected as `malformed`, end to end through a real signed RS256 token).

### Load tests (`tests/load`)
`load_rig.boot_load_gateway` gives every unregistered client the same small policy so N synthetic clients need no registration; `gated_sleep=True` parks delay sleeps on an event so tests can count concurrent waiters.
- `test_load_phases.py`: 24 abusive + 24 polite clients concurrently: exact phase 1 (delays), phase 2 (429), phase 3 (`client_blocked`) per client, polite clients never affected, blocked clients short-circuit before the upstream, block/history expiry; a 100-way simultaneous burst admits exactly the quota; the delay concurrency cap escalates excess to 429.
- `test_load_leaks.py`: cancelling requests while parked in the delay, after upstream failure, mid-LLM-stream and at random points leaves zero delay slots, zero in-flight gauge, zero extra asyncio tasks, closed upstream streams, non-negative token counters; the store stays at its cap with thousands of client ids and expires idle ones.
- `test_load_concurrency_cap.py`: the per-client in-flight cap rejects excess concurrent streams with `too_many_concurrent_requests`, slots return on disconnect and stream expiry, `max_stream_seconds` ends an SSE body cleanly, store failures follow `on_limiter_error`.
- `test_load_realtime.py` (marked `slow`): real clock, real sleeps, 20 clients, ~5 s.

### Integration tests (`tests/integration`)
`servers.py` runs a combined upstream/IdP (SSE, chunked upload, MCP with sessions, Anthropic and OpenAI style streaming) and a `ThreadedServer` helper (uvicorn on port 0). The gateway under test is created through the production path `create_app(path/to/gateway.yaml)`.
- `test_int_http_sse.py`: header hygiene both ways, SSE relayed incrementally (first event arrives long before the stream ends), client disconnect closes the upstream and frees the delay slot, streamed request body, real-time phases over the wire.
- `test_int_mcp_llm.py`: full MCP session (initialize, session id, notification, filtered `tools/list` over JSON and SSE, allowed/denied `tools/call`, GET stream with `Last-Event-ID`, DELETE), batch authorization; LLM streaming with reserve then reconcile to actual usage, `max_tokens` clamp, quota exhaustion/rollback, OpenAI `include_usage` injection, abort mid-stream settled at observed usage.
- `test_int_shutdown.py`: SIGTERM to the real process: idle exit; an in-flight stream completes while new connections are refused; an endless stream does not block exit past 8 s; CLI wiring and config-error exit code 2; lifespan closes services.
- `test_int_idp_real.py`: OIDC discovery against a real HTTP IdP, key rotation, hostile/garbage JWKS, discovery issuer mismatch, stale-if-error.
- `test_int_gaps_inproc.py`: failure paths of `pipeline.py`/`app.py`/`validator.py` that the happy path never reaches (unhandled error to uniform 500, credential leak masking, body limits, store failure fail-open/closed, accounting failure never alters the response, backend crypto errors mapped without leaking).

## Coverage

Baseline before this work: 93 % lines+branches (492 tests, most of it in `pipeline.py`/`validator.py`/`limiter.py` already above 87 %). After: see the numbers below (regenerate with the command above); the `security` directory is excluded because tracing makes its pathological-input tests take minutes. Lines still uncovered are defensive branches (PyJWT paths unreachable because PyJWT rejects first, OTel exporter wiring for OTLP gRPC/HTTP, config error messages) and are listed by `--cov-report=term-missing`.

Measured at the end of this work (`--ignore=tests/security`, 574 tests, random order): **92 % total** (lines+branches); `pipeline.py` 94 %, `ratelimit/limiter.py` 99 %, `auth/validator.py` 96 %, `policy/engine.py` 99 %, `phases.py` 100 %. `store.py`/`app.py` dipped below their earlier figures only because other owners added code (eviction, error handlers) while this ran. Subprocess and uvicorn-thread code paths are measured in-process only.

## Known issues surfaced by these tests
Strict `xfail` tests fail loudly (XPASS) the moment the defect is fixed; delete the marker then.
- `tests/load/test_load_leaks.py::test_cancelled_stream_closes_upstream_and_refunds_reserved_tokens` (open): a stream aborted before any usage event keeps the whole pre-flight estimate charged instead of refunding (DESIGN sections 3 and 9).
- `tests/integration/test_int_mcp_llm.py::test_token_estimate_uses_requested_max_tokens_before_the_grant_clamp` (characterization, not xfail): the pre-flight estimate uses `max_tokens` as sent, before the grant clamp, so an oversized `max_tokens` can be refused by a small token quota although the upstream would only see the clamped value.
- Fixed while this suite was being written: eviction of `block` keys when the store is at `max_keys` (`test_block_survives_flood_when_store_is_at_capacity` is now a plain regression test).

## Flake investigation (earlier reports)

| Report | Finding |
|--------|---------|
| Full suite "hung" once | Reproduced the cause, not a deadlock: two tests (`e2e/test_e2e_auth.py::test_missing_upstream_credential_refuses_to_start`, `e2e/test_e2e_boot.py::test_missing_credential_env_aborts_startup`) call `build_services`/`create_app` with default telemetry; the real `OtelTelemetry` (OTLP exporters pointed at `localhost:4317`, global log handler) is created, `validate_credentials()` then raises `ConfigError` and nothing ever shuts the telemetry down (src defect: `build_services` leaks it on failure). Later tests log through the leaked handler, and at interpreter exit the batch processors retry against the closed port with back-off: the process lingers 13-22 s after "N passed" and prints `Logging error` / `Failed to export` noise (`pytest tests/e2e/test_e2e_boot.py::test_missing_credential_env_aborts_startup tests/proxy` reproduces it). With a black-holed collector instead of a refused port this can exceed any CI step timeout, i.e. looks like a hang. Fix in src: shut telemetry down when `build_services` fails; in tests: pass `telemetry.enabled: false` or `telemetry=NullTelemetry()`. |
| `test_method_mismatch_with_right_scope` failed once | Not reproducible: passes under 60 different `PYTHONHASHSEED`s and 5+ random orders; the test is pure (immutable module-scoped engine over `config/examples/policies/default.yaml`). `test_engine.py` and `policy/engine.py` were being edited (13:41-13:42) shortly after `default.yaml` (13:37) by the policy owner, so the failure coincided with a half-applied change (the `EXPLICIT_DENY` DELETE expectation). Treated as a concurrent-edit transient. |
| "fixture `rig` not found" in shuffled order | The observability fixture used to be defined in a plain `conftest.py` whose module name collides with `tests/conftest.py`; it was moved to `obs_rig.py` and registered from `observability/conftest.py` (rewritten 13:58). Not reproducible after that in 8 random seeds, each directory alone, directory pairs and `-p no:cacheprovider`. Still fragile under `--import-mode=importlib` (documented above). |

`pytest-randomly` is now installed so order dependence shows up immediately; use the printed seed to reproduce.
