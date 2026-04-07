from __future__ import annotations

import logging
from typing import Any

from statestrike_env.constants import RewardConstants
from statestrike_env.models import StateStrikeObservation
from statestrike_env.session import StateStrikeSession
from statestrike_env.tasks import TASK_REGISTRY

logger = logging.getLogger(__name__)


def compute_task_score(session: StateStrikeSession, task_name: str) -> float:
    task_config, grader = TASK_REGISTRY[task_name]
    del task_config
    return float(grader.score(session.as_grader_state()))


def _update_baseline_ema(session: StateStrikeSession, latency_ms: float, window: int) -> None:
    alpha_ema = 2.0 / (window + 1)
    if session.baseline_sample_count == 0:
        session.baseline_latency = latency_ms
    else:
        session.baseline_latency = alpha_ema * latency_ms + (1 - alpha_ema) * session.baseline_latency
    session.baseline_sample_count += 1


def compute_task_reward(
    obs: StateStrikeObservation,
    session: StateStrikeSession,
    task_name: str,
    constants: RewardConstants,
) -> tuple[float, dict[str, Any]]:
    """
    Compute step reward in [0.0, 1.0] based on task progress delta.

    Theory (reward shaping, Ng et al. 1999):
      R_shaped(s, a, s') = R(s, a, s') + gamma*Phi(s') - Phi(s)
      where Phi(s) = task_score(s) is the potential function.

    The terminal bonus (+0.20) is a sparse goal reward layered on top of
    the shaped reward, following the hybrid approach in Sutton & Barto (2018).
    """

    task_config, _ = TASK_REGISTRY[task_name]
    breakdown: dict[str, Any] = {
        "score_delta": 0.0,
        "new_endpoint_bonus": 0.0,
        "new_vulnerability_bonus": 0.0,
        "repeat_penalty": 0.0,
        "terminal_bonus": 0.0,
        "total": 0.0,
    }

    if obs.http_status == 0 or obs.latency_ms == 0.0:
        breakdown["error"] = "connection_failed"
        return 0.0, breakdown

    _update_baseline_ema(session, obs.latency_ms, constants.BASELINE_WINDOW)

    current_task_score = compute_task_score(session, task_name)
    previous_task_score = session.previous_task_score
    score_delta = max(0.0, current_task_score - previous_task_score)
    score_delta = min(score_delta, constants.STEP_DELTA_MAX)
    breakdown["score_delta"] = round(score_delta, 4)

    reward = score_delta

    if bool(session.task_specific_state.get("new_endpoint_discovered", False)):
        reward += constants.NEW_ENDPOINT_BONUS
        breakdown["new_endpoint_bonus"] = constants.NEW_ENDPOINT_BONUS

    if bool(session.task_specific_state.get("new_vulnerability_found", False)):
        reward += constants.NEW_VULNERABILITY_BONUS
        breakdown["new_vulnerability_bonus"] = constants.NEW_VULNERABILITY_BONUS

    if bool(session.task_specific_state.get("repeated_action", False)):
        reward -= constants.REPEATED_ACTION_PENALTY
        breakdown["repeat_penalty"] = -constants.REPEATED_ACTION_PENALTY

    terminal = session.step_count >= task_config.max_steps or current_task_score >= task_config.success_threshold
    if terminal and current_task_score >= task_config.success_threshold:
        reward += constants.TERMINAL_BONUS
        breakdown["terminal_bonus"] = constants.TERMINAL_BONUS

    reward = max(0.0, min(1.0, reward))
    breakdown["total"] = round(reward, 4)
    breakdown["task_score"] = round(current_task_score, 4)

    session.previous_task_score = max(previous_task_score, current_task_score)
    logger.debug(
        "task=%s step=%d score=%.3f reward=%.3f",
        task_name,
        obs.step,
        current_task_score,
        reward,
    )
    return reward, breakdown
