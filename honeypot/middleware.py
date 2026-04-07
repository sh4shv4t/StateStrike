from __future__ import annotations

"""Telemetry middleware and stream endpoint for honeypot observations."""

import json
import time
from collections import deque
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

TELEMETRY_BUFFER: deque[dict[str, object]] = deque(maxlen=500)


class TelemetryMiddleware(BaseHTTPMiddleware):
    """Capture request latency and expose response timing metadata.

    Args:
        app: Wrapped ASGI application.

    Returns:
        None. Middleware mutates response headers and side effects telemetry buffer.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process an incoming request and append telemetry entry.

        Args:
            request: Starlette request object.
            call_next: Next middleware/app callable.

        Returns:
            The downstream response with X-Process-Time-Ms header attached.
        """

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.3f}"

        TELEMETRY_BUFFER.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "path": request.url.path,
                "method": request.method,
                "status": response.status_code,
                "latency_ms": round(elapsed_ms, 3),
            }
        )
        return response


def create_telemetry_router() -> APIRouter:
    """Create telemetry SSE routes.

    Returns:
        A FastAPI router exposing telemetry streaming endpoint.
    """

    router = APIRouter(prefix="/telemetry", tags=["telemetry"])

    @router.get("/stream")
    async def stream_recent_entries() -> StreamingResponse:
        """Emit the latest telemetry entries over Server-Sent Events.

        Returns:
            StreamingResponse configured with text/event-stream media type.
        """

        async def event_source() -> AsyncIterator[str]:
            payload = json.dumps(list(TELEMETRY_BUFFER)[-100:])
            yield f"data: {payload}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    return router
