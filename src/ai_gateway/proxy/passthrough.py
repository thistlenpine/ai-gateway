"""Generic HTTP adapter (kind=http). OWNER: proxy."""

from __future__ import annotations

from collections.abc import Callable

from ai_gateway.config import UpstreamConfig
from ai_gateway.models import Resource, RouteMatch, UpstreamKind
from ai_gateway.proxy.base import ProxyRequest, ProxyResponse, UsageCallback


class HttpAdapter:
    kind = UpstreamKind.HTTP

    def classify(self, route: RouteMatch, method: str, body: bytes | None) -> list[Resource]:
        return [Resource(UpstreamKind.HTTP, route.upstream, method.upper(), route.relative_path)]

    def estimate_tokens(self, body: bytes | None, cap: int | None = None) -> int:
        return 0

    def prepare(self, upstream: UpstreamConfig, req: ProxyRequest) -> ProxyRequest:
        return req

    def process_response(
        self,
        resp: ProxyResponse,
        *,
        tool_allowed: Callable[[str], bool] | None,
        on_usage: UsageCallback | None,
    ) -> ProxyResponse:
        return resp
