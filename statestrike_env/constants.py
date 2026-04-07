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
# Minimum steps between chain bonus awards.
# Prevents POST->GET->POST->GET cycling for free +10 every 2 steps.
# With cooldown=10, max chain bonus rate is +10/10 steps = +1/step.
# Agent must discover actual vulnerabilities to beat this rate.
CHAIN_COOLDOWN_STEPS = 10
MAX_ACTION_HISTORY = 20
ACTION_TIMEOUT_SECONDS = 8.0
DEFAULT_BASELINE_LATENCY_MS = 50.0
# Raised from -50 to -200 to give the agent runway past startup variance.
# At delta=1 and 100% fast-400 rate, takes 200 steps to hit this floor.
# In practice an RL agent should never approach this on valid inputs.
EARLY_TERMINATION_REWARD = -200.0


@dataclass(frozen=True)
class RewardConstants:
    """Typed reward constants passed into the reward grader.

    Attributes:
        ALPHA: Latency reward weight.
        BETA: State-chain completion bonus.
        GAMMA: Exploitation bounty for severe degradation/failure.
        DELTA: Penalty magnitude for low-value fuzzing requests.
        REDOS_LATENCY_THRESHOLD: Latency threshold used to infer ReDoS impact.
        DB_TIMEOUT_THRESHOLD: Latency threshold used for DB timeout exploitation.
        CHAIN_REQUIRED_ORDERS: Minimum order count before GET /orders chain bonus.
        CHAIN_COOLDOWN_STEPS: Minimum steps between chain bonus awards.
        EARLY_TERMINATION_REWARD: Episode early-stop reward floor.
        BASELINE_WINDOW: EMA window used for baseline latency updates.
    """

    ALPHA: float = ALPHA
    BETA: float = BETA
    GAMMA: float = GAMMA
    DELTA: float = DELTA
    REDOS_LATENCY_THRESHOLD: float = REDOS_LATENCY_THRESHOLD
    DB_TIMEOUT_THRESHOLD: float = DB_TIMEOUT_THRESHOLD
    CHAIN_REQUIRED_ORDERS: int = CHAIN_REQUIRED_ORDERS
    CHAIN_COOLDOWN_STEPS: int = CHAIN_COOLDOWN_STEPS
    EARLY_TERMINATION_REWARD: float = EARLY_TERMINATION_REWARD
    BASELINE_WINDOW: int = BASELINE_WINDOW
