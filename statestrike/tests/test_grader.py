from __future__ import annotations

"""Unit tests for reward grading logic."""

from statestrike_env.constants import RewardConstants
from statestrike_env.grader import compute_reward
from statestrike_env.models import ActionType, PayloadStrategy, StateStrikeAction, StateStrikeObservation
from statestrike_env.session import StateStrikeSession


def _make_obs(
    *,
    step: int,
    action: StateStrikeAction,
    status: int,
    latency: float,
    cumulative: float = 0.0,
) -> StateStrikeObservation:
    return StateStrikeObservation(
        step=step,
        action_taken=action,
        http_status=status,
        latency_ms=latency,
        reward=0.0,
        cumulative_reward=cumulative,
        baseline_latency_ms=100.0,
        order_count=0,
        triggered_vulns=[],
        done=False,
        info={},
    )


def test_normal_response_small_positive_reward() -> None:
    """Rewards mild latency increase to encourage exploration of stateful load patterns.

    Why it matters:
        Demonstrates that the dense shaping term provides gradient-like feedback
        for RL optimization even before sparse exploit events occur.
    """

    session = StateStrikeSession.new_session()
    session.baseline_latency = 100.0
    action = StateStrikeAction(action_type=ActionType.HEALTH_CHECK, payload_strategy=PayloadStrategy.VALID)
    session.action_history.append(action)

    obs = _make_obs(step=1, action=action, status=200, latency=120.0)
    reward, breakdown = compute_reward(obs, session, RewardConstants())

    assert reward > 0
    assert breakdown["latency_reward"] > 0


def test_redos_trigger_awards_bounty_and_marks_vulnerability() -> None:
    """Awards a large positive signal when high-latency 400 indicates ReDoS behavior.

    Why it matters:
        Captures the core hackathon objective: discovering semantic exploit paths
        rather than only syntactic API failures.
    """

    session = StateStrikeSession.new_session()
    session.baseline_latency = 80.0
    action = StateStrikeAction(action_type=ActionType.POST_USER, payload_strategy=PayloadStrategy.REDOS_ATTACK)
    session.action_history.append(action)

    obs = _make_obs(step=2, action=action, status=400, latency=1800.0)
    reward, breakdown = compute_reward(obs, session, RewardConstants())

    assert reward > 300
    assert breakdown["exploit_bounty"] >= 400
    assert "redos" in session.triggered_vulns


def test_db_degradation_awards_exploit_bounty() -> None:
    """Grants exploitation bounty on server error/timeout-class DB degradation.

    Why it matters:
        Verifies scoring alignment with real infrastructure impact, which is a
        major evaluation criterion for practical security relevance.
    """

    session = StateStrikeSession.new_session()
    session.baseline_latency = 100.0
    action = StateStrikeAction(action_type=ActionType.GET_ORDERS, payload_strategy=PayloadStrategy.VALID)
    session.action_history.append(action)

    obs = _make_obs(step=3, action=action, status=500, latency=3500.0)
    reward, breakdown = compute_reward(obs, session, RewardConstants())

    assert reward > 450
    assert breakdown["exploit_bounty"] == RewardConstants().gamma
    assert "db_degradation" in session.triggered_vulns


def test_fast_bad_request_gets_negative_reward() -> None:
    """Penalizes low-latency malformed requests that indicate random spam.

    Why it matters:
        Prevents reward hacking where an agent could farm invalid requests
        instead of learning meaningful state transitions.
    """

    session = StateStrikeSession.new_session()
    session.baseline_latency = 100.0
    action = StateStrikeAction(action_type=ActionType.POST_USER, payload_strategy=PayloadStrategy.MALFORMED)
    session.action_history.append(action)

    obs = _make_obs(step=4, action=action, status=400, latency=50.0)
    reward, breakdown = compute_reward(obs, session, RewardConstants())

    assert reward < 0
    assert breakdown["fuzz_penalty"] < 0


def test_chain_completion_bonus_fires_exactly_once() -> None:
    """Applies state-chain bonus only when POST->GET sequence is newly completed.

    Why it matters:
        Confirms the environment rewards logical CRUD sequencing, mirroring
        stateful fuzzing goals highlighted by RESTler-style research.
    """

    session = StateStrikeSession.new_session()
    session.baseline_latency = 100.0
    session.order_count = 21

    post_action = StateStrikeAction(action_type=ActionType.POST_ORDER, payload_strategy=PayloadStrategy.VALID)
    get_action = StateStrikeAction(action_type=ActionType.GET_ORDERS, payload_strategy=PayloadStrategy.VALID)

    session.action_history.extend([post_action, get_action])
    obs_chain = _make_obs(step=22, action=get_action, status=200, latency=130.0)
    _, breakdown_chain = compute_reward(obs_chain, session, RewardConstants())

    session.action_history.append(get_action)
    obs_repeat = _make_obs(step=23, action=get_action, status=200, latency=130.0)
    _, breakdown_repeat = compute_reward(obs_repeat, session, RewardConstants())

    assert breakdown_chain["chain_bonus"] == RewardConstants().beta
    assert breakdown_repeat["chain_bonus"] == 0.0


def test_repeated_get_orders_does_not_refire_exploit_bounty() -> None:
    """Prevents repeated bounty farming from identical already-discovered failures.

    Why it matters:
        Ensures reward logic remains robust against exploitation loops, a key
        requirement for stable and meaningful RL training.
    """

    session = StateStrikeSession.new_session()
    session.baseline_latency = 100.0
    action = StateStrikeAction(action_type=ActionType.GET_ORDERS, payload_strategy=PayloadStrategy.VALID)
    session.action_history.append(action)

    obs_first = _make_obs(step=30, action=action, status=500, latency=3200.0)
    _, breakdown_first = compute_reward(obs_first, session, RewardConstants())

    session.action_history.append(action)
    obs_second = _make_obs(step=31, action=action, status=500, latency=3300.0)
    _, breakdown_second = compute_reward(obs_second, session, RewardConstants())

    assert breakdown_first["exploit_bounty"] == RewardConstants().gamma
    assert breakdown_second["exploit_bounty"] == 0.0
