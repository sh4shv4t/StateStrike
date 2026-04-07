from __future__ import annotations

"""Typed action, observation, and state models for StateStrike.

Theory:
    Explicit state/action schemas reduce ambiguity in RL interfaces and improve
    reproducibility when evaluating policies across different backends.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Discrete actions available to the StateStrike agent."""

    POST_USER = "post_user"
    GET_USER = "get_user"
    POST_ORDER = "post_order"
    GET_ORDERS = "get_orders"
    HEALTH_CHECK = "health_check"


class PayloadStrategy(str, Enum):
    """Payload generation strategies used by the fuzzing policy."""

    VALID = "valid"
    REDOS_ATTACK = "redos"
    OVERSIZED = "oversized"
    MALFORMED = "malformed"


class StateStrikeAction(BaseModel):
    """Action frame sent by the RL agent.

    Args:
        action_type: Target endpoint operation.
        payload_strategy: Payload mutation strategy.
        target_user_id: Optional user identifier override.
    """

    action_type: ActionType
    payload_strategy: PayloadStrategy
    target_user_id: Optional[int] = None


class StateStrikeObservation(BaseModel):
    """Step-level feedback returned by the environment.

    Args:
        step: Current step index within the episode.
        action_taken: Action executed during the step.
        http_status: HTTP status code from honeypot response.
        latency_ms: End-to-end processing latency in milliseconds.
        reward: Scalar reward at this step.
        cumulative_reward: Running reward sum for the episode.
        baseline_latency_ms: Rolling latency baseline used for normalization.
        order_count: Number of POST /orders calls in this episode.
        triggered_vulns: Vulnerability labels discovered so far.
        done: Terminal signal for episode completion.
        info: Arbitrary metadata, including reward breakdown.
    """

    step: int
    action_taken: StateStrikeAction
    http_status: int
    latency_ms: float
    reward: float
    cumulative_reward: float
    baseline_latency_ms: float
    order_count: int
    triggered_vulns: list[str]
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class StateStrikeState(BaseModel):
    """Persistent session state exposed by state().

    Args:
        session_id: Unique identifier for current environment episode.
        step_count: Number of actions executed in current session.
        cumulative_reward: Running reward sum for current session.
        order_count: Number of POST /orders calls in session.
        baseline_latency_ms: Rolling baseline latency in milliseconds.
        action_history: Most recent action history window.
        triggered_vulns: Vulnerabilities discovered in this session.
    """

    session_id: str
    step_count: int
    cumulative_reward: float
    order_count: int
    baseline_latency_ms: float
    action_history: list[StateStrikeAction]
    triggered_vulns: list[str]
