# syntax=docker/dockerfile:1
FROM python:3.13-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels .

FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    GATEWAY_CONFIG=/etc/ai-gateway/gateway.yaml
RUN useradd --system --uid 10001 --no-create-home gateway
COPY --from=build /wheels /wheels
RUN pip install --no-index --find-links /wheels ai-gateway && rm -rf /wheels
COPY config/examples /etc/ai-gateway
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status == 200 else 1)"
ENTRYPOINT ["ai-gateway"]
CMD ["--config", "/etc/ai-gateway/gateway.yaml"]
