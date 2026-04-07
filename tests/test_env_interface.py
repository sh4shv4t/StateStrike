from __future__ import annotations

import pytest

from statestrike_env.environment import StateStrikeEnv
from statestrike_env.models import EndpointChoice, PayloadStrategy, StateStrikeAction


@pytest.mark.asyncio
async def test_reset_returns_initial_observation() -> None:
    env = StateStrikeEnv(honeypot_url="http://127.0.0.1:65533")

    async def _mock_request(*args, **kwargs):
        del args, kwargs
        return 200, 12.0, {"status": "ok"}

    env._request_honeypot = _mock_request  # type: ignore[method-assign]

    result = await env.reset(task_name="endpoint_discovery")
    assert result.done is False
    assert result.reward == 0.0
    assert result.observation.step == 0


@pytest.mark.asyncio
async def test_step_updates_task_progress() -> None:
    env = StateStrikeEnv(honeypot_url="http://127.0.0.1:65533")

    async def _mock_request(method, path, **kwargs):
        del method, path, kwargs
        return 200, 15.0, {"status": "ok"}

    env._request_honeypot = _mock_request  # type: ignore[method-assign]
    await env.reset(task_name="endpoint_discovery")

    action = StateStrikeAction(endpoint=EndpointChoice.HEALTH, payload_strategy=PayloadStrategy.VALID)
    result = await env.step(action)

    assert result.observation.step == 1
    assert result.observation.http_status == 200
    assert result.reward >= 0.0
    assert "GET /health" in result.observation.endpoints_discovered


@pytest.mark.asyncio
async def test_state_returns_serializable_snapshot() -> None:
    env = StateStrikeEnv(honeypot_url="http://127.0.0.1:65533")

    async def _mock_request(method, path, **kwargs):
        del method, path, kwargs
        return 200, 8.0, {"status": "ok"}

    env._request_honeypot = _mock_request  # type: ignore[method-assign]
    await env.reset(task_name="endpoint_discovery")

    state = await env.state()
    assert state.session_id
    assert state.task_name == "endpoint_discovery"
    assert isinstance(state.task_specific_state, dict)
