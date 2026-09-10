"""OpenAI-shaped error envelope, and the handlers that put every failure inside it.

The official SDKs decide which exception to raise from the ``error`` object in the
body, not from the status line alone, so a bare FastAPI ``{"detail": ...}`` reaches
the caller as an opaque APIStatusError.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class ApiError(Exception):
    """A failure to report to the client in OpenAI's error envelope."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code


def _envelope(
    message: str,
    error_type: str,
    param: str | None,
    code: str | None,
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def _param_of(location: tuple[Any, ...]) -> str | None:
    """Field name from a pydantic error location, dropping the leading "body"."""
    parts = [str(p) for p in location if p != "body"]
    return ".".join(parts) if parts else None


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.message, exc.error_type, exc.param, exc.code),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        return JSONResponse(
            status_code=400,
            content=_envelope(
                str(first.get("msg", "invalid request")),
                "invalid_request_error",
                _param_of(tuple(first.get("loc", ()))),
                None,
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(str(exc.detail), "invalid_request_error", None, None),
            headers=getattr(exc, "headers", None),
        )
