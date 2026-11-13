import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from error_classifier.api.routes import router
from error_classifier.config import Settings, load_settings
from error_classifier.errors import ServiceError


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
    app = FastAPI()
    app.state.settings = settings
    app.include_router(router)
    app.add_exception_handler(ServiceError, _service_error_handler)
    return app
