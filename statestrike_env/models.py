from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EndpointChoice(str, Enum):
    POST_USERS = "POST /users"
    GET_USER = "GET /users/{id}"
    POST_ORDERS = "POST /orders"
    GET_ORDERS = "GET /orders"
    HEALTH = "GET /health"


class PayloadStrategy(str, Enum):
    VALID = "valid"
    REDOS_ATTACK = "redos"
    OVERSIZED = "oversized"
    MALFORMED = "malformed"


class StateStrikeAction(BaseModel):
    """Action space for StateStrike environment."""

    endpoint: EndpointChoice
    payload_strategy: PayloadStrategy = PayloadStrategy.VALID
    target_user_id: Optional[int] = None

    class Config:
        use_enum_values = True


class StateStrikeObservation(BaseModel):
    """Observation returned after each step."""

    step: int
    endpoint_called: str
    http_status: int
    latency_ms: float
    response_body: dict[str, Any] = Field(default_factory=dict)
    session_order_count: int = 0
    endpoints_discovered: list[str] = Field(default_factory=list)
    vulnerabilities_found: list[str] = Field(default_factory=list)
    task_progress: float = 0.0


class StepResult(BaseModel):
    """Top-level return from step()."""

    observation: StateStrikeObservation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class StateStrikeState(BaseModel):
    """Full session state, returned by state()."""

    session_id: str
    task_name: str
    step_count: int
    cumulative_reward: float
    order_count: int
    baseline_latency_ms: float
    endpoints_discovered: list[str]
    vulnerabilities_found: list[str]
    task_specific_state: dict[str, Any] = Field(default_factory=dict)
