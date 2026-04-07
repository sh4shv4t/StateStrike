from __future__ import annotations

"""Session state manager for per-agent environment isolation."""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque
from uuid import uuid4

from statestrike_env.constants import BASELINE_WINDOW, DEFAULT_BASELINE_LATENCY_MS, MAX_ACTION_HISTORY
from statestrike_env.models import StateStrikeAction, StateStrikeState


@dataclass
class StateStrikeSession:
    """Mutable per-WebSocket environment session.

    Attributes:
        session_id: Current episode UUID.
        step_count: Number of steps taken in current episode.
        cumulative_reward: Running reward total.
        order_count: Number of POST /orders actions issued.
        baseline_latency: Rolling average latency used in reward normalization.
        action_history: Most recent action history window.
        triggered_vulns: Vulnerabilities discovered in current episode.
        latency_window: Sliding window for baseline computation.
    """

    session_id: str
    step_count: int = 0
    cumulative_reward: float = 0.0
    order_count: int = 0
    baseline_latency: float = DEFAULT_BASELINE_LATENCY_MS
    action_history: list[StateStrikeAction] = field(default_factory=list)
    triggered_vulns: set[str] = field(default_factory=set)
    latency_window: Deque[float] = field(default_factory=lambda: deque(maxlen=BASELINE_WINDOW))

    @classmethod
    def new_session(cls) -> StateStrikeSession:
        """Create a new initialized session.

        Returns:
            Newly initialized StateStrikeSession instance.
        """

        return cls(session_id=str(uuid4()))

    def reset(self, baseline_latency: float = DEFAULT_BASELINE_LATENCY_MS) -> None:
        """Reset session in-place for a new episode.

        Args:
            baseline_latency: Fresh baseline latency in milliseconds.
        """

        self.session_id = str(uuid4())
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.order_count = 0
        self.baseline_latency = baseline_latency
        self.action_history.clear()
        self.triggered_vulns.clear()
        self.latency_window.clear()
        self.latency_window.append(baseline_latency)

    def record_latency(self, latency_ms: float) -> float:
        """Update rolling baseline latency.

        Args:
            latency_ms: Observed latency for the current step.

        Returns:
            Updated baseline latency.
        """

        self.latency_window.append(max(latency_ms, 1.0))
        self.baseline_latency = sum(self.latency_window) / len(self.latency_window)
        return self.baseline_latency

    def append_action(self, action: StateStrikeAction) -> None:
        """Append action while enforcing history length constraints.

        Args:
            action: Action to append.
        """

        self.action_history.append(action)
        if len(self.action_history) > MAX_ACTION_HISTORY:
            self.action_history.pop(0)

    def as_state(self) -> StateStrikeState:
        """Convert mutable session internals to external state model.

        Returns:
            Immutable API-safe state representation.
        """

        return StateStrikeState(
            session_id=self.session_id,
            step_count=self.step_count,
            cumulative_reward=self.cumulative_reward,
            order_count=self.order_count,
            baseline_latency_ms=self.baseline_latency,
            action_history=list(self.action_history),
            triggered_vulns=sorted(self.triggered_vulns),
        )
