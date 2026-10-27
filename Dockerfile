# syntax=docker/dockerfile:1
# Base image pinned by digest (supply-chain hardening, SEC-21). Tag kept for humans;
# Dependabot (docker ecosystem) proposes digest bumps. Resolved 2026-10-27.
# python:3.13-slim
ARG PYTHON_IMAGE=python:3.13-slim@sha256:b3575dcf816cdacc5292df77a6e3e59b8941631078373a045efa2ac1e9c5c3b2

FROM ${PYTHON_IMAGE} AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
WORKDIR /src
# 1) third-party deps: exact versions, hash-verified, no resolver surprises
COPY requirements.lock requirements-build.lock ./
RUN /opt/venv/bin/pip install --require-hashes --no-deps -r requirements.lock -r requirements-build.lock
# 2) the project itself, without dependencies (all already installed above)
COPY pyproject.toml README.md ./
COPY src ./src
# hatchling is hash-pinned in requirements-build.lock, so build isolation is disabled
RUN /opt/venv/bin/pip install --no-deps --no-build-isolation .

FROM ${PYTHON_IMAGE}
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH \
    GATEWAY_CONFIG=/etc/ai-gateway/gateway.yaml
RUN useradd --system --uid 10001 --no-create-home gateway
COPY --from=build /opt/venv /opt/venv
COPY config/examples /etc/ai-gateway
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status == 200 else 1)"
ENTRYPOINT ["ai-gateway"]
CMD ["--config", "/etc/ai-gateway/gateway.yaml"]
