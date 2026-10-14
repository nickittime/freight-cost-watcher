import pytest

from src.config import get_settings


@pytest.fixture(autouse=True)
def configure_test_environment(monkeypatch):
    """Enforce offline fake LLM provider and clear settings cache for all tests."""
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    get_settings.cache_clear()