"""Fixture registration only: shared helpers live in obs_rig (a module named `conftest` is
ambiguous when several directories have one)."""

from obs_rig import rig

__all__ = ["rig"]
