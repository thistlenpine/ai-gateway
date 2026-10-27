"""create_app() from YAML (the production path), without network."""

from __future__ import annotations

import shutil
from pathlib import Path

import httpx
import pytest

from ai_gateway.app import create_app

EXAMPLES = Path(__file__).resolve().parents[2] / "config" / "examples"


async def test_create_app_from_yaml_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copytree(EXAMPLES, tmp_path / "cfg")
    gateway_yaml = tmp_path / "cfg" / "gateway.yaml"
    gateway_yaml.write_text(
        gateway_yaml.read_text().replace("telemetry:\n", "telemetry:\n  enabled: false\n", 1)
    )
    for var in ("MCP_TOOLS_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(var, "dummy-secret")
    monkeypatch.setenv("GATEWAY_CONFIG", str(gateway_yaml))
    app = create_app()  # config path from GATEWAY_CONFIG
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw.test") as c:
        assert (await c.get("/healthz")).status_code == 200
        assert (await c.get("/readyz")).status_code == 503  # JWKS never fetched (no lifespan/network)
        r = await c.get("/billing/invoices")
        assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_token"
        assert (await c.get("/docs")).status_code == 401  # no docs/openapi surface
    await app.state.services.aclose()


def test_missing_credential_env_aborts_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_gateway.config import ConfigError

    shutil.copytree(EXAMPLES, tmp_path / "cfg")
    for var in ("MCP_TOOLS_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ConfigError):
        create_app(str(tmp_path / "cfg" / "gateway.yaml"))
