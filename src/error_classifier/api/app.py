import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from error_classifier.api.routes import describe_invalid, router
from error_classifier.classify.service import Classifier, ClassifierOptions
from error_classifier.concurrency import ConcurrencyLimiter
from error_classifier.config import Settings, load_settings
from error_classifier.errors import ServiceError
from error_classifier.llm.client import LlmClient, build_http_client


async def _service_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ServiceError)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
        headers=exc.headers(),
    )


async def _validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return JSONResponse(
        status_code=400,
        content={"error": {"code": "invalid_input", "message": describe_invalid(exc.errors())}},
    )


async def _unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": "Internal error"}},
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
            app.state.classifier = Classifier(
                app.state.llm, ClassifierOptions.from_settings(settings)
            )
            app.state.limiter = ConcurrencyLimiter.from_settings(settings)
            yield

    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings
    app.include_router(router)
    app.add_exception_handler(ServiceError, _service_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(Exception, _unexpected_error_handler)
    return app
