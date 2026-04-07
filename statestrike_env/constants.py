from __future__ import annotations

"""Centralized constants for StateStrike environment and reward grading.

Theory:
    Consolidating reward and episode hyperparameters avoids hidden magic numbers,
    supports reproducibility, and aligns with RL experiment hygiene guidance from
    Sutton & Barto (2018).
"""

from dataclasses import dataclass


ALPHA = 1.0
BETA = 10.0
GAMMA = 500.0
DELTA = 1.0
BASELINE_WINDOW = 10
EPISODE_LENGTH = 200
REDOS_LATENCY_THRESHOLD = 1500.0
DB_TIMEOUT_THRESHOLD = 3000.0
CHAIN_REQUIRED_ORDERS = 20
MAX_ACTION_HISTORY = 20
ACTION_TIMEOUT_SECONDS = 8.0
DEFAULT_BASELINE_LATENCY_MS = 50.0
EARLY_TERMINATION_REWARD = -200.0


@dataclass(frozen=True)
class RewardConstants:
    """Typed reward constants passed into the reward grader.

    Attributes:
        alpha: Latency reward weight.
        beta: State-chain completion bonus.
        gamma: Exploitation bounty for severe degradation/failure.
        delta: Penalty magnitude for low-value fuzzing requests.
        redos_latency_threshold: Latency threshold used to infer ReDoS impact.
        db_timeout_threshold: Latency threshold used for DB timeout exploitation.
        chain_required_orders: Minimum order count before GET /orders chain bonus.
    """

    alpha: float = ALPHA
    beta: float = BETA
    gamma: float = GAMMA
    delta: float = DELTA
    redos_latency_threshold: float = REDOS_LATENCY_THRESHOLD
    db_timeout_threshold: float = DB_TIMEOUT_THRESHOLD
    chain_required_orders: int = CHAIN_REQUIRED_ORDERS
