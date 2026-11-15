import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from error_classifier.api.routes import describe_invalid, router
from error_classifier.classify.service import Classifier, ClassifierOptions
from error_classifier.concurrency import ConcurrencyLimiter
from error_classifier.config import Settings, load_settings
from error_classifier.errors import ServiceError, error_body
from error_classifier.llm.client import LlmClient, build_http_client


async def _service_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ServiceError)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(exc.code, exc.message, exc.status_code, exc.param),
        headers=exc.headers(),
    )


async def _validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return JSONResponse(
        status_code=400,
        content=error_body("invalid_input", describe_invalid(exc.errors()), 400),
    )


async def _http_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
    message = {404: "Not found", 405: "Method not allowed"}.get(exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(code, message, exc.status_code),
        headers=exc.headers,
    )


async def _unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=error_body("internal_error", "Internal error", 500),
        headers={"x-should-retry": "false"},
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
    app.add_exception_handler(StarletteHTTPException, _http_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(Exception, _unexpected_error_handler)
    return app
