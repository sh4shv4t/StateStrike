from __future__ import annotations

"""Integration tests for environment reset/step/state interface."""

import pytest

from statestrike_env.models import ActionType, PayloadStrategy, StateStrikeAction
from statestrike_env.server import StateStrikeEnvironment
from statestrike_env.session import StateStrikeSession


@pytest.mark.asyncio
async def test_reset_returns_initial_observation() -> None:
    """Ensure reset initializes an episode with valid observation payload.

    Why it matters:
        OpenEnv compliance depends on deterministic reset semantics for every
        policy rollout and evaluator run.
    """

    env = StateStrikeEnvironment(honeypot_url="http://127.0.0.1:65533")

    async def _mock_request(*args, **kwargs):
        del args, kwargs
        return 200, 15.0, {"status": "ok"}

    env._request_honeypot = _mock_request  # type: ignore[method-assign]
    session = StateStrikeSession.new_session()

    obs = await env.reset(session)

    assert obs.step == 0
    assert obs.reward == 0.0
    assert obs.action_taken.action_type == ActionType.HEALTH_CHECK


@pytest.mark.asyncio
async def test_step_updates_state_even_when_target_unavailable() -> None:
    """Verify step robustness with synthetic fallback when honeypot is unreachable.

    Why it matters:
        Training jobs should not crash on transient infra faults; robust fallback
        keeps episodes analyzable and aligns with production reliability criteria.
    """

    env = StateStrikeEnvironment(honeypot_url="http://127.0.0.1:65533")

    async def _mock_request(*args, **kwargs):
        del args, kwargs
        return 0, 0.0, {"error": "connection_failed", "synthetic": True}

    env._request_honeypot = _mock_request  # type: ignore[method-assign]
    session = StateStrikeSession.new_session()
    await env.reset(session)

    action = StateStrikeAction(action_type=ActionType.POST_ORDER, payload_strategy=PayloadStrategy.VALID)
    obs = await env.step(session, action)

    assert obs.step == 1
    assert obs.http_status == 0
    assert obs.latency_ms == 0.0
    assert session.order_count == 1


@pytest.mark.asyncio
async def test_state_returns_serializable_session_snapshot() -> None:
    """Ensure state() exposes all required session fields for evaluators.

    Why it matters:
        Reproducible scoring and debugging require transparent environment state,
        especially for stateful exploit-chain discovery.
    """

    env = StateStrikeEnvironment(honeypot_url="http://127.0.0.1:65533")

    async def _mock_request(*args, **kwargs):
        del args, kwargs
        return 200, 8.0, {"status": "ok"}

    env._request_honeypot = _mock_request  # type: ignore[method-assign]
    session = StateStrikeSession.new_session()
    await env.reset(session)

    state = await env.state(session)

    assert state.session_id
    assert state.step_count == 0
    assert isinstance(state.action_history, list)
