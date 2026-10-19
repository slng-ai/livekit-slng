"""Shared fixtures for the test suite."""

import pytest


@pytest.fixture(autouse=True)
def _slng_region(monkeypatch):
    """The plugin has no default host, so the suite provides one the way a deployment would."""
    monkeypatch.setenv("SLNG_BASE_URL", "us-east.api.slng.ai")
