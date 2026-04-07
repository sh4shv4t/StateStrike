from __future__ import annotations

"""Reward grading logic for StateStrike.

Theory:
    The reward function follows standard MDP shaping principles from Sutton &
    Barto (2018): combine dense shaping signals (latency ratio), sparse goal
    rewards (exploit bounty), and penalties (invalid spam suppression). It also
    borrows stateful-sequence ideas from RESTler (Atlidakis et al., ICSE 2019)
    while rewarding infrastructure effects (e.g., ReDoS latency spikes) inspired
    by Davis et al. (USENIX Security 2018).
"""

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from statestrike_env.constants import RewardConstants
    from statestrike_env.models import StateStrikeObservation
    from statestrike_env.session import StateStrikeSession

logger = logging.getLogger(__name__)


def compute_reward(
    obs: "StateStrikeObservation",
    session: "StateStrikeSession",
    constants: "RewardConstants",
) -> tuple[float, dict[str, float]]:
    """
    Compute R_t = α·log(L_t/L_base) + β·S_t + γ·E_t − δ·P_t

    Theory (Sutton & Barto, 2018, Ch. 3 — Finite MDPs):
    The reward signal must be designed so the ONLY way to maximize cumulative
    reward is to achieve the TRUE objective. Each term is chosen to prevent a
    specific reward-hacking strategy:

    TERM 1 — α·log(L_t/L_base): Logarithmic latency reward.
      Why log? Linear reward incentivizes the agent to find ONE massive spike
      and repeat it. Logarithmic reward gives diminishing returns per repeated
      exploitation, pushing the agent to discover NEW vulnerabilities.
      Why ratio? Prevents baseline-anchoring attacks where agent engineers a
      low baseline then makes normal requests look like spikes.
      Anti-hack: baseline ONLY updates from successful (latency>0) steps.

    TERM 2 — β·S_t: State-chain bonus.
      Fires at most once per CHAIN_COOLDOWN_STEPS steps, and only if
      order_count has increased since the last award. This prevents the
      POST→GET farming loop that would yield +5 reward/step for free.
      Anti-hack: last_chain_bonus_step and post_count_at_last_chain guards.

    TERM 3 — γ·E_t: Exploitation bounty.
      Fires EXACTLY ONCE per vulnerability type per episode (one-time flag).
      Without this, an agent discovering db_degradation would spam GET /orders
      for +500/step indefinitely. The one-time award correctly signals
      "you found it" without incentivizing repeated triggering.
      Anti-hack: redos_bounty_awarded and db_degradation_bounty_awarded flags.

    TERM 4 — δ·P_t: Fuzzing penalty.
      Applied only to genuinely fast 400s (latency < 100ms), not to slow 400s
      (which may indicate actual CPU burn from ReDoS parsing).
      Threshold tightened from 200ms to 100ms to avoid penalizing legitimate
      slow-failing payloads.
      Anti-hack: latency threshold ensures ReDoS probes are not penalized.

    Reference:
      - Sutton & Barto (2018): reward shaping and sparse reward design
      - Atlidakis et al. (ICSE 2019): stateful API exploration objectives
      - Davis et al. (USENIX 2018): ReDoS computational complexity

    Args:
        obs: The observation from the current step.
        session: The mutable session state (modified in-place for flags).
        constants: Reward weight constants from constants.py.

    Returns:
        Tuple of (scalar_reward, breakdown_dict) where breakdown_dict
        contains each term's contribution for telemetry and dashboard display.
    """

    reward = 0.0
    breakdown: dict[str, float] = {
        "latency_reward": 0.0,
        "chain_bonus": 0.0,
        "exploit_bounty": 0.0,
        "fuzz_penalty": 0.0,
        "total": 0.0,
    }

    # Guard: connection failure -> neutral observation, no reward signal.
    if obs.http_status == 0 or obs.latency_ms == 0.0:
        breakdown["error"] = 1.0
        logger.debug("Step %d: connection failure, returning zero reward", obs.step)
        return 0.0, breakdown

    # Update rolling baseline only from successful steps.
    _update_baseline(session, obs.latency_ms, constants.BASELINE_WINDOW)

    # TERM 1: Logarithmic latency reward.
    if session.baseline_latency > 0:
        latency_ratio = obs.latency_ms / session.baseline_latency
        latency_ratio = max(0.01, min(latency_ratio, 100.0))
        latency_reward = constants.ALPHA * math.log(latency_ratio)
        reward += latency_reward
        breakdown["latency_reward"] = round(latency_reward, 4)
        logger.debug(
            "Step %d: latency=%.1fms baseline=%.1fms ratio=%.2f reward=%.3f",
            obs.step,
            obs.latency_ms,
            session.baseline_latency,
            latency_ratio,
            latency_reward,
        )

    # TERM 2: State-chain bonus (anti-farming guards).
    chain_bonus = 0.0
    if _should_award_chain_bonus(obs, session, constants):
        chain_bonus = constants.BETA
        session.last_chain_bonus_step = obs.step
        session.post_count_at_last_chain = session.order_count
        logger.info(
            "Step %d: Chain bonus awarded (+%.1f). order_count=%d",
            obs.step,
            chain_bonus,
            session.order_count,
        )
    reward += chain_bonus
    breakdown["chain_bonus"] = chain_bonus

    # TERM 3: Exploitation bounties (one-time per episode).
    exploit_bounty = 0.0

    if (
        not session.db_degradation_bounty_awarded
        and (obs.http_status >= 500 or obs.latency_ms > constants.DB_TIMEOUT_THRESHOLD)
        and obs.action_taken.action_type.value == "get_orders"
    ):
        exploit_bounty += constants.GAMMA
        session.db_degradation_bounty_awarded = True
        session.triggered_vulns.add("db_degradation")
        logger.info(
            "Step %d: DB_DEGRADATION bounty awarded (+%.1f). latency=%.1fms",
            obs.step,
            constants.GAMMA,
            obs.latency_ms,
        )

    if (
        not session.redos_bounty_awarded
        and obs.latency_ms > constants.REDOS_LATENCY_THRESHOLD
        and obs.http_status == 400
        and obs.action_taken.action_type.value == "post_user"
    ):
        redos_bounty = constants.GAMMA * 0.8
        exploit_bounty += redos_bounty
        session.redos_bounty_awarded = True
        session.triggered_vulns.add("redos")
        logger.info(
            "Step %d: REDOS bounty awarded (+%.1f). latency=%.1fms",
            obs.step,
            redos_bounty,
            obs.latency_ms,
        )

    reward += exploit_bounty
    breakdown["exploit_bounty"] = round(exploit_bounty, 4)

    # TERM 4: Fuzzing penalty (only genuine fast-fail syntax errors).
    fuzz_penalty = 0.0
    if obs.http_status == 400 and obs.latency_ms < 100.0:
        fuzz_penalty = -constants.DELTA
        logger.debug("Step %d: Fuzz penalty applied (fast 400, %.1fms)", obs.step, obs.latency_ms)
    reward += fuzz_penalty
    breakdown["fuzz_penalty"] = round(fuzz_penalty, 4)

    breakdown["total"] = round(reward, 4)
    return reward, breakdown


def _update_baseline(session: "StateStrikeSession", latency_ms: float, window: int) -> None:
    """Update rolling baseline latency using exponential moving average."""

    alpha_ema = 2.0 / (window + 1)
    if session.baseline_sample_count == 0:
        session.baseline_latency = latency_ms
    else:
        session.baseline_latency = alpha_ema * latency_ms + (1 - alpha_ema) * session.baseline_latency
    session.baseline_sample_count += 1


def _should_award_chain_bonus(
    obs: "StateStrikeObservation",
    session: "StateStrikeSession",
    constants: "RewardConstants",
) -> bool:
    """Determine if the state-chain bonus should be awarded this step."""

    if obs.action_taken.action_type.value != "get_orders":
        return False
    if session.order_count < constants.CHAIN_REQUIRED_ORDERS:
        return False
    steps_since_last = obs.step - session.last_chain_bonus_step
    if steps_since_last < constants.CHAIN_COOLDOWN_STEPS:
        return False
    if session.order_count <= session.post_count_at_last_chain:
        return False
    return True
