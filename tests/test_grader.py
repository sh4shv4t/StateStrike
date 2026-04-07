from __future__ import annotations

from statestrike_env.constants import RewardConstants
from statestrike_env.grader import compute_task_reward
from statestrike_env.models import StateStrikeObservation
from statestrike_env.session import StateStrikeSession


def _obs(step: int, endpoint: str, status: int, latency: float, progress: float = 0.0) -> StateStrikeObservation:
    return StateStrikeObservation(
        step=step,
        endpoint_called=endpoint,
        http_status=status,
        latency_ms=latency,
        response_body={},
        session_order_count=0,
        endpoints_discovered=[],
        vulnerabilities_found=[],
        task_progress=progress,
    )


def test_reward_uses_progress_delta() -> None:
    session = StateStrikeSession.new_session("endpoint_discovery")
    session.endpoints_discovered.add("GET /health")
    session.previous_task_score = 0.0
    session.task_specific_state["new_endpoint_discovered"] = False
    session.task_specific_state["new_vulnerability_found"] = False
    session.task_specific_state["repeated_action"] = False
    session.step_count = 1

    reward, breakdown = compute_task_reward(
        _obs(1, "GET /health", 200, 10.0),
        session,
        "endpoint_discovery",
        RewardConstants(),
    )

    assert reward > 0.0
    assert breakdown["score_delta"] > 0.0


def test_connection_failure_yields_zero_reward() -> None:
    session = StateStrikeSession.new_session("endpoint_discovery")
    session.baseline_latency = 50.0
    session.baseline_sample_count = 5

    reward, breakdown = compute_task_reward(
        _obs(1, "GET /health", 0, 0.0),
        session,
        "endpoint_discovery",
        RewardConstants(),
    )

    assert reward == 0.0
    assert breakdown.get("error") == "connection_failed"
    assert session.baseline_latency == 50.0


def test_terminal_bonus_applies_on_success() -> None:
    session = StateStrikeSession.new_session("endpoint_discovery")
    session.endpoints_discovered.update(
        {
            "POST /users",
            "GET /users/{id}",
            "POST /orders",
            "GET /orders",
            "GET /health",
        }
    )
    session.previous_task_score = 0.8
    session.task_specific_state["new_endpoint_discovered"] = False
    session.task_specific_state["new_vulnerability_found"] = False
    session.task_specific_state["repeated_action"] = False
    session.step_count = 20

    reward, breakdown = compute_task_reward(
        _obs(20, "GET /health", 200, 11.0, progress=1.0),
        session,
        "endpoint_discovery",
        RewardConstants(),
    )

    assert breakdown["terminal_bonus"] > 0.0
    assert 0.0 <= reward <= 1.0
