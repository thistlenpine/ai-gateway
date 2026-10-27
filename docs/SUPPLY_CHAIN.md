# Supply-chain controls (SEC-21)

| Control | Where |
|---|---|
| Resolved + hashed lock | `uv.lock` (source of truth); exported `requirements.lock` (runtime), `requirements-dev.lock` (runtime + dev), `requirements-build.lock` (hatchling) |
| Hash-verified installs | CI: `uv sync --frozen --all-extras`; Docker: `pip install --require-hashes --no-deps -r requirements.lock`; project installed with `--no-deps` |
| Base image pinned | `Dockerfile` `PYTHON_IMAGE` = `python:3.13-slim@sha256:...` (non-root uid 10001 kept) |
| Actions pinned to SHA | `.github/workflows/ci.yml` (version in trailing comment) |
| Vulnerability scan | CI job `pip-audit` on `requirements-dev.lock` and `requirements-build.lock`; pip-audit itself is hash-locked in `.github/requirements-audit.lock` |
| Automated updates | `.github/dependabot.yml` (pip, uv, docker, github-actions) |
| Least privilege | workflow `permissions: contents: read`, `persist-credentials: false` |
| Security/load tests | dedicated CI job `pytest tests/security tests/load` |

## Updating dependencies

```
uv lock --upgrade-package <name>     # or: uv lock --upgrade
uv export --format requirements.txt --no-emit-project --no-dev --no-header -o requirements.lock
uv export --format requirements.txt --no-emit-project --no-dev --all-extras --no-header -o requirements-dev.lock
```

CI (`lockfiles up to date`) fails if `uv.lock` or the exported files drift from `pyproject.toml`.
Regenerate `requirements-build.lock` / `.github/requirements-audit.lock` with
`uv pip compile --generate-hashes --no-header <in-file> -o <lock>` (inputs: `hatchling`, `pip-audit`).

## Bumping the base image digest

`docker buildx imagetools inspect python:3.13-slim` and replace the digest (Dependabot does this too).
