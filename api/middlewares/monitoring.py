from typing import Awaitable, Callable
import time
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from api.core.logger import log


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging, tracing, and monitoring."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = str(uuid4())
        start_time = time.time()
        
        # Add trace ID to request state
        request.state.trace_id = request_id
        
        # Log request details
        log.info(
            "Request started",
            extra={
                "trace_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_host": request.client.host if request.client else None,
                "headers": dict(request.headers),
            },
        )
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Calculate request duration
            duration = time.time() - start_time
            
            # Log response details
            log.info(
                "Request completed",
                extra={
                    "trace_id": request_id,
                    "status_code": response.status_code,
                    "duration": duration,
                },
            )
            
            # Add monitoring headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = str(duration)
            
            return response
            
        except Exception as e:
            # Log error details
            log.error(
                "Request failed",
                extra={
                    "trace_id": request_id,
                    "error": str(e),
                    "duration": time.time() - start_time,
                },
                exc_info=True,
            )
            raise


def setup_monitoring(app: FastAPI) -> None:
    """Set up monitoring middleware and other monitoring tools."""
    app.add_middleware(MonitoringMiddleware) 
