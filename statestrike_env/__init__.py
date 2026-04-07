from __future__ import annotations

"""StateStrike OpenEnv-compatible client exports."""

import json
from contextlib import AbstractContextManager
from typing import Any

from websockets.sync.client import ClientConnection, connect

from statestrike_env.models import StateStrikeAction, StateStrikeObservation, StateStrikeState


class _SyncStateStrikeClient(AbstractContextManager["_SyncStateStrikeClient"]):
    """Synchronous WebSocket client wrapper for reset/step/state calls."""

    def __init__(self, base_url: str) -> None:
        """Initialize client.

        Args:
            base_url: WebSocket URL including `/ws` path.
        """

        normalized = base_url.rstrip("/")
        self.base_url = normalized if normalized.endswith("/ws") else f"{normalized}/ws"
        self._conn: ClientConnection | None = None

    def __enter__(self) -> "_SyncStateStrikeClient":
        """Open WebSocket connection for environment operations.

        Returns:
            Connected client instance.
        """

        self._conn = connect(self.base_url)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close WebSocket connection.

        Args:
            exc_type: Exception type if raised in context block.
            exc: Exception value if raised in context block.
            tb: Traceback object if raised in context block.
        """

        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def reset(self) -> StateStrikeObservation:
        """Request environment reset.

        Returns:
            Initial observation.

        Raises:
            RuntimeError: If the server response is malformed or unsuccessful.
        """

        frame = self._request({"method": "reset"})
        return StateStrikeObservation.model_validate(frame["observation"])

    def step(self, action: StateStrikeAction) -> StateStrikeObservation:
        """Execute one environment step.

        Args:
            action: Action payload.

        Returns:
            Updated observation.

        Raises:
            RuntimeError: If the server response is malformed or unsuccessful.
        """

        frame = self._request({"method": "step", "action": action.model_dump()})
        return StateStrikeObservation.model_validate(frame["observation"])

    def state(self) -> StateStrikeState:
        """Retrieve current environment state.

        Returns:
            Current state model.

        Raises:
            RuntimeError: If the server response is malformed or unsuccessful.
        """

        frame = self._request({"method": "state"})
        return StateStrikeState.model_validate(frame["state"])

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send request frame and parse server response.

        Args:
            payload: JSON-serializable request payload.

        Returns:
            Parsed response object.

        Raises:
            RuntimeError: If connection is closed or server reports failure.
        """

        if self._conn is None:
            raise RuntimeError("WebSocket connection is not open")

        self._conn.send(json.dumps(payload))
        raw = self._conn.recv()
        frame = json.loads(raw)
        if not frame.get("ok"):
            raise RuntimeError(frame.get("error", "Unknown server error"))
        return frame


class StateStrikeEnv:
    """Environment client namespace matching OpenEnv SDK usage patterns."""

    def __init__(self, base_url: str = "ws://localhost:8001/ws") -> None:
        """Store base URL for later sync client creation.

        Args:
            base_url: Environment WebSocket endpoint.
        """

        self.base_url = base_url

    def sync(self) -> _SyncStateStrikeClient:
        """Create synchronous context-managed client.

        Returns:
            A synchronous environment client implementing reset/step/state.
        """

        return _SyncStateStrikeClient(self.base_url)


__all__ = ["StateStrikeEnv", "StateStrikeAction", "StateStrikeObservation", "StateStrikeState"]
