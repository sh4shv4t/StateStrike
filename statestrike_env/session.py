from __future__ import annotations

"""Session state manager for per-agent environment isolation."""

from dataclasses import dataclass, field
from uuid import uuid4

from statestrike_env.constants import DEFAULT_BASELINE_LATENCY_MS, MAX_ACTION_HISTORY
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
        redos_bounty_awarded: One-time ReDoS bounty guard.
        db_degradation_bounty_awarded: One-time DB degradation bounty guard.
        last_chain_bonus_step: Last step where chain bonus was awarded.
        post_count_at_last_chain: Order count snapshot at last chain award.
        baseline_sample_count: Number of successful baseline samples seen.
    """

    session_id: str
    step_count: int = 0
    cumulative_reward: float = 0.0
    order_count: int = 0
    baseline_latency: float = DEFAULT_BASELINE_LATENCY_MS
    action_history: list[StateStrikeAction] = field(default_factory=list)
    triggered_vulns: set[str] = field(default_factory=set)
    # Anti-hacking: one-time flags so each bounty fires exactly once per episode.
    redos_bounty_awarded: bool = False
    db_degradation_bounty_awarded: bool = False
    # Anti-hacking: chain bonus can only fire once between meaningful progress windows.
    last_chain_bonus_step: int = -10
    post_count_at_last_chain: int = 0
    # Baseline integrity: updated only on successful (non-zero latency) steps.
    baseline_sample_count: int = 0

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
        self.redos_bounty_awarded = False
        self.db_degradation_bounty_awarded = False
        self.last_chain_bonus_step = -10
        self.post_count_at_last_chain = 0
        self.baseline_sample_count = 1 if baseline_latency > 0 else 0

    def record_latency(self, latency_ms: float) -> float:
        """Update baseline latency using EMA from successful samples.

        Args:
            latency_ms: Observed latency for the current step.

        Returns:
            Updated baseline latency.
        """

        sample = max(latency_ms, 1.0)
        alpha_ema = 2.0 / (10 + 1)
        if self.baseline_sample_count == 0:
            self.baseline_latency = sample
        else:
            self.baseline_latency = alpha_ema * sample + (1 - alpha_ema) * self.baseline_latency
        self.baseline_sample_count += 1
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
