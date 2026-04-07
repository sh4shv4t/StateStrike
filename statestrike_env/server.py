from __future__ import annotations

"""OpenEnv-style WebSocket environment server for StateStrike."""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

try:
    import openenv_core  # noqa: F401
except ImportError:  # pragma: no cover - optional import for compatibility signaling.
    openenv_core = None

from statestrike_env.constants import (
    ACTION_TIMEOUT_SECONDS,
    DEFAULT_BASELINE_LATENCY_MS,
    EPISODE_LENGTH,
    RewardConstants,
)
from statestrike_env.grader import compute_reward
from statestrike_env.models import ActionType, PayloadStrategy, StateStrikeAction, StateStrikeObservation, StateStrikeState
from statestrike_env.session import StateStrikeSession

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

HONEYPOT_URL = os.getenv("HONEYPOT_URL", "http://localhost:8000")
HOST = os.getenv("STATESTRIKE_ENV_HOST", "0.0.0.0")
PORT = int(os.getenv("STATESTRIKE_ENV_PORT", "8001"))


async def wait_for_honeypot(url: str, max_wait: int = 30) -> None:
    """Block until honeypot is reachable or raise RuntimeError.

    Args:
        url: Honeypot base URL.
        max_wait: Maximum wait time in seconds.

    Raises:
        RuntimeError: If honeypot is not reachable before timeout.
    """

    deadline = asyncio.get_event_loop().time() + max_wait
    delay = 1.0
    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                response = await client.get(f"{url}/health", timeout=3.0)
                if response.status_code == 200:
                    LOGGER.info("Honeypot is ready at %s", url)
                    return
                LOGGER.warning(
                    "Honeypot health returned status=%s, retrying in %.1fs...",
                    response.status_code,
                    delay,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Honeypot not ready (%s), retrying in %.1fs...", exc, delay)

            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 5.0)

    raise RuntimeError(f"Honeypot at {url} did not become ready within {max_wait}s")


class StateStrikeEnvironment:
    """Core reset/step/state implementation.

    Theory:
        OpenEnv training loops benefit from persistent transport: WebSocket-based
        sessions amortize handshake overhead and preserve episode-local state,
        which aligns with OpenEnv architecture guidance (Burtenshaw, 2025).
    """

    def __init__(self, honeypot_url: str, constants: RewardConstants | None = None) -> None:
        """Initialize environment service.

        Args:
            honeypot_url: Base URL for vulnerable honeypot API.
            constants: Optional reward constants override.
        """

        self.honeypot_url = honeypot_url.rstrip("/")
        self.constants = constants or RewardConstants()

    async def reset(self, session: StateStrikeSession) -> StateStrikeObservation:
        """Reset session and return initial observation.

        Args:
            session: Session object tied to one client connection.

        Returns:
            Initial observation with zero reward.
        """

        status, latency_ms, _ = await self._request_honeypot("GET", "/health")
        baseline = latency_ms if latency_ms > 0 else DEFAULT_BASELINE_LATENCY_MS
        session.reset(baseline_latency=baseline)

        action = StateStrikeAction(action_type=ActionType.HEALTH_CHECK, payload_strategy=PayloadStrategy.VALID)
        obs = StateStrikeObservation(
            step=0,
            action_taken=action,
            http_status=status,
            latency_ms=latency_ms,
            reward=0.0,
            cumulative_reward=0.0,
            baseline_latency_ms=session.baseline_latency,
            order_count=0,
            triggered_vulns=[],
            done=False,
            info={"event": "reset"},
        )
        return obs

    async def step(self, session: StateStrikeSession, action: StateStrikeAction) -> StateStrikeObservation:
        """Execute one environment transition.

        Args:
            session: Session object tied to one client connection.
            action: Agent action.

        Returns:
            Updated observation with reward and terminal signal.
        """

        request_method, request_path, params, payload = self._translate_action(action, session)
        status, latency_ms, body = await self._request_honeypot(request_method, request_path, params=params, payload=payload)

        session.step_count += 1
        if action.action_type == ActionType.POST_ORDER:
            session.order_count += 1
        session.append_action(action)

        provisional = StateStrikeObservation(
            step=session.step_count,
            action_taken=action,
            http_status=status,
            latency_ms=latency_ms,
            reward=0.0,
            cumulative_reward=session.cumulative_reward,
            baseline_latency_ms=session.baseline_latency,
            order_count=session.order_count,
            triggered_vulns=sorted(session.triggered_vulns),
            done=False,
            info={"response": body},
        )

        reward, breakdown = compute_reward(provisional, session, self.constants)
        session.cumulative_reward += reward

        done = (
            session.step_count >= EPISODE_LENGTH
            or session.cumulative_reward < self.constants.EARLY_TERMINATION_REWARD
        )
        obs = StateStrikeObservation(
            step=session.step_count,
            action_taken=action,
            http_status=status,
            latency_ms=latency_ms,
            reward=reward,
            cumulative_reward=session.cumulative_reward,
            baseline_latency_ms=session.baseline_latency,
            order_count=session.order_count,
            triggered_vulns=sorted(session.triggered_vulns),
            done=done,
            info={"reward_breakdown": breakdown, "response": body},
        )
        return obs

    async def state(self, session: StateStrikeSession) -> StateStrikeState:
        """Return serializable state snapshot.

        Args:
            session: Session object tied to one client connection.

        Returns:
            Current state model.
        """

        return session.as_state()

    def _translate_action(
        self,
        action: StateStrikeAction,
        session: StateStrikeSession,
    ) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
        """Translate action schema into honeypot HTTP request details.

        Args:
            action: Agent action.
            session: Session used for contextual defaults.

        Returns:
            Tuple of method, path, query params, and JSON payload.
        """

        target_user_id = action.target_user_id or 1

        if action.action_type == ActionType.POST_USER:
            email = self._payload_email(action.payload_strategy)
            return "POST", "/users", None, {"email": email}
        if action.action_type == ActionType.GET_USER:
            return "GET", f"/users/{target_user_id}", None, None
        if action.action_type == ActionType.POST_ORDER:
            item = self._payload_item(action.payload_strategy)
            return "POST", "/orders", None, {"user_id": target_user_id, "item": item}
        if action.action_type == ActionType.GET_ORDERS:
            return "GET", "/orders", {"user_id": target_user_id}, None
        return "GET", "/health", None, None

    @staticmethod
    def _payload_email(strategy: PayloadStrategy) -> str:
        """Build email-like payload for POST /users action.

        Args:
            strategy: Payload strategy enum.

        Returns:
            Strategy-specific string payload.
        """

        if strategy == PayloadStrategy.REDOS_ATTACK:
            return "a" * 39 + "!"
        if strategy == PayloadStrategy.OVERSIZED:
            return "A" * 4096
        if strategy == PayloadStrategy.MALFORMED:
            return "@@@"
        return "validuser123"

    @staticmethod
    def _payload_item(strategy: PayloadStrategy) -> str:
        """Build order item payload.

        Args:
            strategy: Payload strategy enum.

        Returns:
            Strategy-specific order item string.
        """

        if strategy == PayloadStrategy.OVERSIZED:
            return "item_" + ("X" * 2048)
        if strategy == PayloadStrategy.MALFORMED:
            return ""
        return "standard_item"

    async def _request_honeypot(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, float, dict[str, Any]]:
        """Execute honeypot request and normalize response metadata.

        Args:
            method: HTTP method.
            path: Relative path.
            params: Optional query parameters.
            payload: Optional JSON body.

        Returns:
            Tuple of status code, latency milliseconds, and parsed response body.
        """

        url = f"{self.honeypot_url}{path}"
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=ACTION_TIMEOUT_SECONDS) as client:
                response = await client.request(method, url, params=params, json=payload)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            header_latency = response.headers.get("X-Process-Time-Ms")
            latency_ms = float(header_latency) if header_latency else elapsed_ms
            body = response.json() if response.content else {}
            return response.status_code, latency_ms, body
        except (httpx.RequestError, ValueError) as exc:
            LOGGER.warning("Honeypot request failed method=%s path=%s error=%s", method, path, exc)
            return 0, 0.0, {"error": str(exc), "synthetic": True}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Block API startup until honeypot health endpoint is reachable."""

    await wait_for_honeypot(HONEYPOT_URL, max_wait=30)
    yield


app = FastAPI(title="StateStrike OpenEnv Server", version="1.0.0", lifespan=lifespan)
env_service = StateStrikeEnvironment(HONEYPOT_URL)
http_debug_session = StateStrikeSession.new_session()


# OpenEnv uses WebSocket (/ws) for persistent sessions rather than
# stateless HTTP. Each step() is a lightweight frame over an existing
# connection (~0.1ms overhead vs ~10-50ms TCP handshake per HTTP call).
# Reference: openenv-course module-5, burtenshaw/openenv-scaling
# This architecture enables high-frequency RL training loops.
@app.websocket("/ws")
async def websocket_env(websocket: WebSocket) -> None:
    """Run one isolated environment loop per WebSocket client.

    Args:
        websocket: Connected client transport.
    """

    await websocket.accept()
    session = StateStrikeSession.new_session()
    LOGGER.info("WebSocket session started session_id=%s", session.session_id)

    try:
        while True:
            frame = await websocket.receive_text()
            request = json.loads(frame)
            method = request.get("method")

            if method == "reset":
                obs = await env_service.reset(session)
                await websocket.send_json({"ok": True, "observation": obs.model_dump()})
                continue

            if method == "step":
                action_payload = request.get("action", {})
                action = StateStrikeAction.model_validate(action_payload)
                obs = await env_service.step(session, action)
                await websocket.send_json({"ok": True, "observation": obs.model_dump()})
                continue

            if method == "state":
                state = await env_service.state(session)
                await websocket.send_json({"ok": True, "state": state.model_dump()})
                continue

            await websocket.send_json({"ok": False, "error": f"Unknown method: {method}"})
    except (WebSocketDisconnect, json.JSONDecodeError):
        LOGGER.info("WebSocket session ended session_id=%s", session.session_id)


@app.get("/reset")
async def reset_http() -> JSONResponse:
    """HTTP debug endpoint for reset semantics.

    Returns:
        JSON response containing reset observation.
    """

    obs = await env_service.reset(http_debug_session)
    return JSONResponse(obs.model_dump())


@app.post("/step")
async def step_http(action: StateStrikeAction) -> JSONResponse:
    """HTTP debug endpoint for step semantics.

    Args:
        action: Action payload.

    Returns:
        JSON response containing post-step observation.
    """

    obs = await env_service.step(http_debug_session, action)
    return JSONResponse(obs.model_dump())


@app.get("/state")
async def state_http() -> JSONResponse:
    """HTTP debug endpoint for state semantics.

    Returns:
        JSON response containing current session state.
    """

    state = await env_service.state(http_debug_session)
    return JSONResponse(state.model_dump())


def main() -> None:
    """Entrypoint for running environment server via python -m."""

    import uvicorn

    uvicorn.run("statestrike_env.server:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
