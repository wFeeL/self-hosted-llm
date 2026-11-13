import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from error_classifier.api.routes import router
from error_classifier.config import Settings, load_settings
from error_classifier.errors import ServiceError
from error_classifier.llm.client import LlmClient, build_http_client


async def _service_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ServiceError)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = load_settings()
    logging.basicConfig(level=settings.log_level)
    # uvicorn configures its loggers before calling the factory, so set the level again here.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with build_http_client(settings) as http:
            app.state.llm = LlmClient(http, settings.llm_model, settings.llm_probe_timeout_s)
            yield

    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings
    app.include_router(router)
    app.add_exception_handler(ServiceError, _service_error_handler)
    return app
