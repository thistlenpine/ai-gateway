"""Hypothesis profiles: `ci` (default, deterministic-ish and fast) and `thorough` (HYPOTHESIS_PROFILE=thorough)."""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci", max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow], print_blob=True
)
settings.register_profile("thorough", max_examples=2000, deadline=None, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
