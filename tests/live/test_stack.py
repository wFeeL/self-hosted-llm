import os

import httpx
import pytest

# Read at import, before the autouse clean_env fixture runs.
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "qwen3.5-2b-q4km")

pytestmark = pytest.mark.live


def test_health() -> None:
    response = httpx.get(f"{API_URL}/health", timeout=5)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready() -> None:
    response = httpx.get(f"{API_URL}/ready", timeout=5)
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model": MODEL_ALIAS}
