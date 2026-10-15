import os

import pytest

from src.config import get_settings

# Ensure module-scoped fixtures default to stub during pytest collection/setup
os.environ.setdefault("LLM_PROVIDER", "stub")


@pytest.fixture(autouse=True)
def configure_test_environment(monkeypatch):
    """Enforce offline stub LLM provider and clear settings cache for unit tests."""
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    get_settings.cache_clear()