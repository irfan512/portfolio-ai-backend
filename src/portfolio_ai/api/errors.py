"""One error envelope for every failure. Messages are safe for the public:
no exception text, stack traces, hostnames or configuration details."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("portfolio_ai.api")


class ApiError(Exception):
    def __init__(
        self, status: int, code: str, message: str, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}


def error_body(code: str, message: str, request: Request, **extra: object) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": getattr(request.state, "request_id", None),
            **extra,
        }
    }


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            error_body(exc.code, exc.message, request), status_code=exc.status, headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Field + reason only: never echo the submitted input back.
        details = [
            {"field": ".".join(str(p) for p in err["loc"] if p != "body"), "issue": err["msg"]}
            for err in exc.errors()
        ]
        return JSONResponse(
            error_body("invalid_request", "The request is invalid.", request, details=details),
            status_code=422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        message = {404: "Not found.", 405: "Method not allowed."}.get(
            exc.status_code, "Request failed."
        )
        return JSONResponse(error_body(code, message, request), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error", extra={"path": request.url.path})
        return JSONResponse(
            error_body("internal_error", "Something went wrong. Please try again.", request),
            status_code=500,
        )
