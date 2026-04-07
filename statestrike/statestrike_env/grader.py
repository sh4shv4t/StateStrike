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

import math
from typing import TYPE_CHECKING

from statestrike_env.constants import RewardConstants
from statestrike_env.models import ActionType, StateStrikeObservation

if TYPE_CHECKING:
    from statestrike_env.session import StateStrikeSession


def _detects_post_get_chain(session: StateStrikeSession, constants: RewardConstants) -> bool:
    """Detect a POST /orders -> GET /orders chain completion.

    Args:
        session: Mutable per-episode session state.
        constants: Reward constants used for threshold checks.

    Returns:
        True if the latest action completed a valid stateful chain.
    """

    if len(session.action_history) < 2:
        return False

    previous_action = session.action_history[-2]
    current_action = session.action_history[-1]
    return (
        previous_action.action_type == ActionType.POST_ORDER
        and current_action.action_type == ActionType.GET_ORDERS
        and session.order_count > constants.chain_required_orders
    )


def compute_reward(
    obs: StateStrikeObservation,
    session: StateStrikeSession,
    constants: RewardConstants,
) -> tuple[float, dict[str, float]]:
    """Compute reward with latency shaping, chain bonus, and exploit bounties.

    Computes:
        R_t = alpha*log(L_t/L_base) + beta*S_t + gamma*E_t - delta*P_t

    Args:
        obs: Current environment observation.
        session: Session state for action history and discovered vulnerabilities.
        constants: Tunable reward weights and thresholds.

    Returns:
        A tuple containing the scalar reward and a component breakdown map.

    Theory:
        - Log latency term normalizes by rolling baseline to avoid rewarding raw
          high latency alone.
        - Chain bonus rewards meaningful state transitions rather than random
          endpoint hits.
        - Exploit bounty provides sparse objective completion feedback.
        - Fast-400 penalty suppresses random malformed-input spam while still
          allowing costly 400 responses (e.g., ReDoS-induced CPU burn).
    """

    reward = 0.0
    breakdown: dict[str, float] = {}

    if session.baseline_latency > 0:
        latency_ratio = obs.latency_ms / session.baseline_latency
        latency_reward = constants.alpha * math.log(max(latency_ratio, 0.01))
        reward += latency_reward
        breakdown["latency_reward"] = round(latency_reward, 3)
    else:
        breakdown["latency_reward"] = 0.0

    chain_bonus = 0.0
    if _detects_post_get_chain(session, constants):
        chain_bonus = constants.beta
    reward += chain_bonus
    breakdown["chain_bonus"] = chain_bonus

    exploit_bounty = 0.0
    if obs.http_status >= 500 or obs.latency_ms > constants.db_timeout_threshold:
        if "db_degradation" not in session.triggered_vulns:
            exploit_bounty = constants.gamma
            session.triggered_vulns.add("db_degradation")

    if obs.latency_ms > constants.redos_latency_threshold and obs.http_status == 400:
        if "redos" not in session.triggered_vulns:
            exploit_bounty = max(exploit_bounty, constants.gamma * 0.8)
            session.triggered_vulns.add("redos")

    reward += exploit_bounty
    breakdown["exploit_bounty"] = exploit_bounty

    fuzz_penalty = 0.0
    if obs.http_status == 400 and obs.latency_ms < 200:
        fuzz_penalty = -constants.delta
    reward += fuzz_penalty
    breakdown["fuzz_penalty"] = fuzz_penalty

    breakdown["total"] = round(reward, 4)
    return reward, breakdown
