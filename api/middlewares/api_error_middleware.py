import httpx
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.logger import log


class APIErrorMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        log.info(f"Request received: {request}")
        try:
            response = await call_next(request)
            log.info(f"Middleware: Response sent")
            return response
        except httpx.HTTPError as e:
            log.error(f"Middleware: HTTP error occurred: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={
                    "Middleware: detail": "External service error",
                    "error": str(e),
                },
            )
        except HTTPException as e:
            log.exception(f"Middleware: HTTP exception occurred: {e}")
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
        except Exception as e:
            log.exception(
                "Middleware: expected error occurred in middleware: {e}",
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "error": str(e)},
            )
