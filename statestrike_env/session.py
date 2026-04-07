from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from statestrike_env.constants import DEFAULT_BASELINE_LATENCY_MS
from statestrike_env.models import StateStrikeState


@dataclass
class StateStrikeSession:
    session_id: str
    task_name: str = "endpoint_discovery"
    step_count: int = 0
    cumulative_reward: float = 0.0
    order_count: int = 0
    baseline_latency: float = DEFAULT_BASELINE_LATENCY_MS

    endpoints_discovered: set[str] = field(default_factory=set)
    vulnerabilities_found: set[str] = field(default_factory=set)
    task_specific_state: dict[str, Any] = field(default_factory=dict)
    steps_history: list[dict[str, Any]] = field(default_factory=list)

    user_created: bool = False
    previous_task_score: float = 0.0
    last_action_signature: str | None = None

    redos_bounty_awarded: bool = False
    db_degradation_bounty_awarded: bool = False
    last_chain_bonus_step: int = -10
    post_count_at_last_chain: int = 0
    baseline_sample_count: int = 0

    @classmethod
    def new_session(cls, task_name: str = "endpoint_discovery") -> StateStrikeSession:
        return cls(session_id=str(uuid4()), task_name=task_name)

    def reset(
        self,
        task_name: str,
        baseline_latency: float = DEFAULT_BASELINE_LATENCY_MS,
    ) -> None:
        self.session_id = str(uuid4())
        self.task_name = task_name
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.order_count = 0
        self.baseline_latency = baseline_latency

        self.endpoints_discovered.clear()
        self.vulnerabilities_found.clear()
        self.task_specific_state = {}
        self.steps_history.clear()

        self.user_created = False
        self.previous_task_score = 0.0
        self.last_action_signature = None

        self.redos_bounty_awarded = False
        self.db_degradation_bounty_awarded = False
        self.last_chain_bonus_step = -10
        self.post_count_at_last_chain = 0
        self.baseline_sample_count = 1 if baseline_latency > 0 else 0

    def as_state(self) -> StateStrikeState:
        return StateStrikeState(
            session_id=self.session_id,
            task_name=self.task_name,
            step_count=self.step_count,
            cumulative_reward=self.cumulative_reward,
            order_count=self.order_count,
            baseline_latency_ms=self.baseline_latency,
            endpoints_discovered=sorted(self.endpoints_discovered),
            vulnerabilities_found=sorted(self.vulnerabilities_found),
            task_specific_state=dict(self.task_specific_state),
        )

    def as_grader_state(self) -> dict[str, Any]:
        return {
            "endpoints_discovered": sorted(self.endpoints_discovered),
            "vulnerabilities_found": sorted(self.vulnerabilities_found),
            "steps_history": list(self.steps_history),
            "order_count": self.order_count,
            "user_created": self.user_created,
        }
