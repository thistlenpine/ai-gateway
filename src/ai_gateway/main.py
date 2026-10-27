"""Process entrypoint: `ai-gateway --config config/gateway.yaml`. OWNER: core."""

from __future__ import annotations

import argparse
import os
import sys


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai-gateway")
    parser.add_argument("--config", default=os.environ.get("GATEWAY_CONFIG", "config/gateway.yaml"))
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    import uvicorn

    from ai_gateway.config import ConfigError, load_config

    try:
        cfg = load_config(args.config).gateway.server
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    os.environ["GATEWAY_CONFIG"] = args.config
    uvicorn.run(
        "ai_gateway.app:create_app",
        factory=True,
        host=args.host or cfg.host,
        port=args.port or cfg.port,
        log_config=None,  # logging is configured by observability (JSON), not uvicorn
        proxy_headers=cfg.trust_forwarded_for,
        forwarded_allow_ips=",".join(cfg.trusted_proxies) if cfg.trust_forwarded_for else None,
        server_header=False,
    )


if __name__ == "__main__":
    cli()
