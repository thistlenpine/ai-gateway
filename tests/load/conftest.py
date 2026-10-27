from __future__ import annotations

from collections.abc import Iterator

import pytest

from ai_gateway.observability.logging import shutdown_logging
from ai_gateway.observability.otel import uninstrument


@pytest.fixture(autouse=True)
def _clean_logging_and_instrumentation() -> Iterator[None]:
    yield
    shutdown_logging()
    uninstrument()
