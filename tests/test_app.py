import logging

import pytest
from conftest import Serve
from pydantic import ValidationError

from error_classifier.api.app import create_app
from error_classifier.concurrency import ConcurrencyLimiter
from error_classifier.config import Settings
from error_classifier.errors import LlmUnavailableError, QueueFullError

pytestmark = pytest.mark.anyio


def _settings() -> Settings:
    return Settings(llm_model="m", llm_base_url="http://127.0.0.1:9/v1")


async def test_health_ok_without_runtime(serve: Serve) -> None:
    async with serve(create_app(_settings())) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_service_error_envelope(serve: Serve) -> None:
    app = create_app(_settings())

    async def boom() -> None:
        raise LlmUnavailableError("down")

    app.add_api_route("/boom", boom)
    async with serve(app) as client:
        response = await client.get("/boom")
    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "llm_unavailable",
            "message": "down",
            "type": "server_error",
            "param": None,
        }
    }


def test_create_app_fails_on_bad_env() -> None:
    with pytest.raises(ValidationError, match="llm_model"):
        create_app()


def test_create_app_sets_uvicorn_logger_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    names = ("uvicorn", "uvicorn.error", "uvicorn.access")
    for name in names:
        monkeypatch.setattr(logging.getLogger(name), "level", logging.INFO)
    create_app(Settings(llm_model="m", log_level="ERROR"))
    assert [logging.getLogger(name).level for name in names] == [logging.ERROR] * 3


async def test_lifespan_builds_the_limiter_from_settings(serve: Serve) -> None:
    settings = Settings(llm_model="m", max_concurrency=3, max_queue=0)
    app = create_app(settings)
    async with serve(app):
        limiter = app.state.limiter
        assert isinstance(limiter, ConcurrencyLimiter)
        async with limiter.slot(), limiter.slot(), limiter.slot():
            assert limiter.active == 3
            with pytest.raises(QueueFullError):
                async with limiter.slot():
                    pass
