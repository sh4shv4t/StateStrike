from __future__ import annotations

import asyncio
import os
import subprocess
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Body, FastAPI

from statestrike_env.constants import RewardConstants
from statestrike_env.models import (
    EndpointChoice,
    PayloadStrategy,
    StateStrikeAction,
    StateStrikeObservation,
    StateStrikeState,
    StepResult,
)
from statestrike_env.session import StateStrikeSession
from statestrike_env.tasks import TASK_REGISTRY


def _compute_current_task_score(session: StateStrikeSession, task_name: str) -> float:
    _, grader = TASK_REGISTRY[task_name]
    state_dict = {
        "endpoints_discovered": list(session.endpoints_discovered),
        "vulnerabilities_found": list(session.triggered_vulns),
        "steps_history": session.steps_history,
        "user_created": session.user_created,
        "order_count": session.order_count,
    }
    return float(grader.score(state_dict))


class StateStrikeEnv:
    """Unified OpenEnv-compatible runtime for StateStrike."""

    def __init__(
        self,
        honeypot_url: str | None = None,
        constants: RewardConstants | None = None,
    ) -> None:
        self.honeypot_url = (honeypot_url or os.getenv("HONEYPOT_URL", "http://localhost:8000")).rstrip("/")
        self.constants = constants or RewardConstants()
        self.session = StateStrikeSession.new_session("endpoint_discovery")
        self._managed_container_id: str | None = None

    @classmethod
    async def from_docker_image(cls, image_name: str) -> StateStrikeEnv:
        env = cls(honeypot_url=os.getenv("HONEYPOT_URL", "http://localhost:8000"))
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "run",
            "-d",
            "-p",
            "8000:8000",
            image_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to start docker image: {stderr.decode().strip()}")
        env._managed_container_id = stdout.decode().strip()

        for _ in range(30):
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(f"{env.honeypot_url}/health")
                    if response.status_code == 200:
                        return env
            except Exception:
                pass
            await asyncio.sleep(1)

        await env.close()
        raise RuntimeError("Timed out waiting for honeypot container to become ready")

    async def close(self) -> None:
        if self._managed_container_id:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "-f",
                self._managed_container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
            self._managed_container_id = None

    async def reset(self, task_name: str = "endpoint_discovery") -> StepResult:
        if task_name not in TASK_REGISTRY:
            task_name = "endpoint_discovery"

        status, latency_ms, _ = await self._request_honeypot("GET", "/health")
        baseline = latency_ms if latency_ms > 0 else self.constants.DEFAULT_BASELINE_LATENCY_MS
        self.session.reset(task_name=task_name, baseline_latency=baseline)

        observation = StateStrikeObservation(
            step=0,
            endpoint_called=EndpointChoice.HEALTH.value,
            http_status=status,
            latency_ms=latency_ms,
            response_body={"status": "reset"},
            session_order_count=0,
            endpoints_discovered=list(self.session.endpoints_discovered),
            vulnerabilities_found=list(self.session.triggered_vulns),
            task_progress=_compute_current_task_score(self.session, self.session.task_name),
        )
        return StepResult(observation=observation, reward=0.0, done=False, info={"task": task_name})

    async def step(self, action: StateStrikeAction) -> StepResult:
        prev_score = _compute_current_task_score(self.session, self.session.task_name)
        endpoint_before = len(self.session.endpoints_discovered)
        vulns_before = len(self.session.triggered_vulns)

        method, path, params, payload = self._translate_action(action, self.session)
        status, latency_ms, body = await self._request_honeypot(method, path, params=params, payload=payload)

        endpoint_str = action.endpoint if isinstance(action.endpoint, str) else action.endpoint.value
        strategy_value = (
            action.payload_strategy if isinstance(action.payload_strategy, str) else action.payload_strategy.value
        )

        self.session.step_count += 1

        if status in (200, 201, 400, 422):
            self.session.endpoints_discovered.add(endpoint_str)

        if endpoint_str == EndpointChoice.POST_ORDERS.value and status in (200, 201):
            self.session.order_count += 1

        action_signature = f"{endpoint_str}|{strategy_value}|{action.target_user_id}"
        repeated_action = action_signature == self.session.last_action_str
        self.session.last_action_str = action_signature
        self.session.last_action_signature = action_signature

        if (
            not self.session.redos_bounty_awarded
            and endpoint_str == EndpointChoice.POST_USERS.value
            and strategy_value == PayloadStrategy.REDOS_ATTACK.value
            and status == 400
            and latency_ms > self.constants.REDOS_LATENCY_THRESHOLD_MS
        ):
            self.session.redos_bounty_awarded = True
            self.session.triggered_vulns.add("redos")

        chain_cooldown_ready = (
            self.session.step_count - self.session.last_chain_bonus_step
        ) >= self.constants.CHAIN_COOLDOWN_STEPS
        chain_progressed = self.session.order_count > self.session.post_count_at_last_chain
        if (
            not self.session.db_degradation_bounty_awarded
            and endpoint_str == EndpointChoice.GET_ORDERS.value
            and self.session.order_count >= self.constants.CHAIN_REQUIRED_ORDERS
            and latency_ms > self.constants.DB_TIMEOUT_THRESHOLD_MS
            and chain_cooldown_ready
            and chain_progressed
        ):
            self.session.db_degradation_bounty_awarded = True
            self.session.triggered_vulns.add("db_degradation")
            self.session.last_chain_bonus_step = self.session.step_count
            self.session.post_count_at_last_chain = self.session.order_count

        if (
            endpoint_str == EndpointChoice.POST_USERS.value
            and strategy_value == PayloadStrategy.VALID.value
            and status in (200, 201)
        ):
            self.session.user_created = True
            if isinstance(body, dict):
                user_id = body.get("id")
                if isinstance(user_id, int):
                    self.session.user_id = user_id

        self.session.steps_history.append(
            {
                "endpoint": endpoint_str,
                "payload_strategy": strategy_value,
                "http_status": status,
                "latency_ms": latency_ms,
            }
        )
        if len(self.session.steps_history) > 200:
            self.session.steps_history.pop(0)

        new_endpoint_found = len(self.session.endpoints_discovered) > endpoint_before
        new_vulnerability_found = len(self.session.triggered_vulns) > vulns_before
        self.session.task_specific_state["new_endpoint_discovered"] = new_endpoint_found
        self.session.task_specific_state["new_vulnerability_found"] = new_vulnerability_found
        self.session.task_specific_state["repeated_action"] = repeated_action

        current_score = _compute_current_task_score(self.session, self.session.task_name)
        observation = StateStrikeObservation(
            step=self.session.step_count,
            endpoint_called=endpoint_str,
            http_status=status,
            latency_ms=latency_ms,
            response_body=body,
            session_order_count=self.session.order_count,
            endpoints_discovered=sorted(self.session.endpoints_discovered),
            vulnerabilities_found=sorted(self.session.triggered_vulns),
            task_progress=current_score,
        )

        task_cfg, _ = TASK_REGISTRY[self.session.task_name]
        done = self.session.step_count >= task_cfg.max_steps or current_score >= task_cfg.success_threshold

        score_delta = max(0.0, current_score - prev_score)
        score_delta = min(score_delta, self.constants.STEP_DELTA_MAX)

        step_reward = score_delta
        step_reward += self.constants.NEW_ENDPOINT_BONUS if new_endpoint_found else 0.0
        step_reward += self.constants.NEW_VULNERABILITY_BONUS if new_vulnerability_found else 0.0
        step_reward -= self.constants.REPEATED_ACTION_PENALTY if repeated_action else 0.0
        if done and current_score >= task_cfg.success_threshold:
            step_reward += self.constants.TERMINAL_BONUS

        step_reward = round(min(max(step_reward, 0.0), 1.0), 4)
        self.session.previous_task_score = max(self.session.previous_task_score, current_score)
        self.session.cumulative_reward += step_reward

        breakdown = {
            "score_delta": round(score_delta, 4),
            "new_endpoint_bonus": self.constants.NEW_ENDPOINT_BONUS if new_endpoint_found else 0.0,
            "new_vulnerability_bonus": self.constants.NEW_VULNERABILITY_BONUS if new_vulnerability_found else 0.0,
            "repeat_penalty": -self.constants.REPEATED_ACTION_PENALTY if repeated_action else 0.0,
            "terminal_bonus": self.constants.TERMINAL_BONUS if (done and current_score >= task_cfg.success_threshold) else 0.0,
            "total": step_reward,
            "task_score": round(current_score, 4),
        }

        return StepResult(
            observation=observation,
            reward=step_reward,
            done=done,
            info={
                "reward_breakdown": breakdown,
                "task": self.session.task_name,
            },
        )

    async def state(self) -> StateStrikeState:
        return self.session.as_state()

    def reset_sync(self, task_name: str = "endpoint_discovery") -> StepResult:
        return asyncio.run(self.reset(task_name=task_name))

    def step_sync(self, action: StateStrikeAction) -> StepResult:
        return asyncio.run(self.step(action))

    def state_sync(self) -> StateStrikeState:
        return asyncio.run(self.state())

    def _translate_action(
        self,
        action: StateStrikeAction,
        session: StateStrikeSession,
    ) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
        endpoint_value = action.endpoint if isinstance(action.endpoint, str) else action.endpoint.value
        strategy_value = (
            action.payload_strategy if isinstance(action.payload_strategy, str) else action.payload_strategy.value
        )
        target_user_id = action.target_user_id if action.target_user_id is not None else (session.user_id or 1)

        if endpoint_value == EndpointChoice.POST_USERS.value:
            return "POST", "/users", None, {"email": self._email_for_strategy(strategy_value)}
        if endpoint_value == EndpointChoice.GET_USER.value:
            return "GET", f"/users/{target_user_id}", None, None
        if endpoint_value == EndpointChoice.POST_ORDERS.value:
            return "POST", "/orders", None, {
                "user_id": target_user_id,
                "item": self._item_for_strategy(strategy_value),
            }
        if endpoint_value == EndpointChoice.GET_ORDERS.value:
            return "GET", "/orders", {"user_id": target_user_id}, None
        return "GET", "/health", None, None

    @staticmethod
    def _email_for_strategy(strategy: str) -> str:
        if strategy == PayloadStrategy.REDOS_ATTACK.value:
            return "a" * 39 + "!"
        if strategy == PayloadStrategy.OVERSIZED.value:
            return "A" * 4096
        if strategy == PayloadStrategy.MALFORMED.value:
            return "@@@"
        return "validuser123"

    @staticmethod
    def _item_for_strategy(strategy: str) -> str:
        if strategy == PayloadStrategy.OVERSIZED.value:
            return "item_" + ("X" * 2048)
        if strategy == PayloadStrategy.MALFORMED.value:
            return ""
        return "standard_item"

    async def _request_honeypot(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, float, dict[str, Any]]:
        url = f"{self.honeypot_url}{path}"
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.constants.ACTION_TIMEOUT_SECONDS) as client:
                response = await client.request(method, url, params=params, json=payload)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            header_latency = response.headers.get("X-Process-Time-Ms")
            latency_ms = float(header_latency) if header_latency else elapsed_ms
            body = response.json() if response.content else {}
            return response.status_code, latency_ms, body
        except Exception as exc:
            return 0, 0.0, {"error": str(exc), "synthetic": True}


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title="StateStrike", lifespan=lifespan)
_global_env = StateStrikeEnv()


@app.post("/reset")
async def reset_endpoint(body: dict = Body(default={})):
    task = body.get("task", "endpoint_discovery")
    result = await _global_env.reset(task_name=task)
    return result.model_dump()


@app.post("/step")
async def step_endpoint(action: StateStrikeAction):
    result = await _global_env.step(action)
    return result.model_dump()


@app.get("/state")
async def state_endpoint():
    return (await _global_env.state()).model_dump()


@app.get("/health")
async def health():
    return {"status": "ok"}


def main() -> None:
    import uvicorn

    host = os.getenv("STATESTRIKE_ENV_HOST", "0.0.0.0")
    port = int(os.getenv("STATESTRIKE_ENV_PORT", "7860"))
    uvicorn.run("statestrike_env.environment:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
