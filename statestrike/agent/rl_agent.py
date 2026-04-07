from __future__ import annotations

"""Optional PPO/GRPO extension stub for future RL training."""

from dataclasses import dataclass

from statestrike_env.models import StateStrikeAction, StateStrikeObservation


@dataclass
class PPOConfig:
    """Configuration skeleton for PPO/GRPO training experiments."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    rollout_steps: int = 1024


class PPOAgent:
    """Minimal RL agent stub preserving extension points for future work."""

    def __init__(self, config: PPOConfig | None = None) -> None:
        """Store configuration.

        Args:
            config: Optional PPO hyperparameter bundle.
        """

        self.config = config or PPOConfig()

    def act(self, obs: StateStrikeObservation) -> StateStrikeAction:
        """Return action for observation.

        Args:
            obs: Latest environment observation.

        Returns:
            Chosen action.

        Raises:
            NotImplementedError: Always, until policy network is implemented.
        """

        raise NotImplementedError("Implement PPO policy inference for competition extension")
