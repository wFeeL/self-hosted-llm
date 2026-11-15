import hmac
from typing import cast

from fastapi import Request

from error_classifier.config import Settings
from error_classifier.errors import UnauthorizedError


async def require_api_key(request: Request) -> None:
    key = cast(Settings, request.app.state.settings).api_key
    if key is None:
        return
    scheme, _, given = request.headers.get("authorization", "").partition(" ")
    # Starlette decodes headers as latin-1, so this restores the bytes the client sent.
    sent = given.strip().encode("latin-1", "replace")
    # compare_digest always runs, so a wrong scheme does not answer faster than a wrong key.
    matches = hmac.compare_digest(sent, key.get_secret_value().encode())
    if scheme.lower() != "bearer" or not matches:
        raise UnauthorizedError()
