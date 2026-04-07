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
from statestrike_env.grader import compute_task_reward, compute_task_score
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
            endpoints_discovered=[],
            vulnerabilities_found=[],
            task_progress=0.0,
        )
        return StepResult(observation=observation, reward=0.0, done=False, info={"task": task_name})

    async def step(self, action: StateStrikeAction) -> StepResult:
        method, path, params, payload = self._translate_action(action)
        status, latency_ms, body = await self._request_honeypot(method, path, params=params, payload=payload)

        endpoint_value = action.endpoint if isinstance(action.endpoint, str) else action.endpoint.value
        strategy_value = (
            action.payload_strategy if isinstance(action.payload_strategy, str) else action.payload_strategy.value
        )

        self.session.step_count += 1
        if endpoint_value == EndpointChoice.POST_ORDERS.value and status in (200, 201):
            self.session.order_count += 1

        endpoint_name = endpoint_value
        new_endpoint = False
        if status > 0 and endpoint_name not in self.session.endpoints_discovered:
            self.session.endpoints_discovered.add(endpoint_name)
            new_endpoint = True

        signature = f"{endpoint_value}|{strategy_value}|{action.target_user_id}"
        repeated_action = signature == self.session.last_action_signature
        self.session.last_action_signature = signature

        new_vulnerability = False

        if (
            not self.session.redos_bounty_awarded
            and endpoint_value == EndpointChoice.POST_USERS.value
            and strategy_value == PayloadStrategy.REDOS_ATTACK.value
            and status == 400
            and latency_ms > self.constants.REDOS_LATENCY_THRESHOLD_MS
        ):
            self.session.redos_bounty_awarded = True
            self.session.vulnerabilities_found.add("redos")
            new_vulnerability = True

        chain_cooldown_ready = (
            self.session.step_count - self.session.last_chain_bonus_step
        ) >= self.constants.CHAIN_COOLDOWN_STEPS
        chain_progressed = self.session.order_count > self.session.post_count_at_last_chain
        if (
            not self.session.db_degradation_bounty_awarded
            and endpoint_value == EndpointChoice.GET_ORDERS.value
            and self.session.order_count >= self.constants.CHAIN_REQUIRED_ORDERS
            and latency_ms > self.constants.DB_TIMEOUT_THRESHOLD_MS
            and chain_cooldown_ready
            and chain_progressed
        ):
            self.session.db_degradation_bounty_awarded = True
            self.session.vulnerabilities_found.add("db_degradation")
            self.session.last_chain_bonus_step = self.session.step_count
            self.session.post_count_at_last_chain = self.session.order_count
            new_vulnerability = True

        if (
            endpoint_value == EndpointChoice.POST_USERS.value
            and strategy_value == PayloadStrategy.VALID.value
            and status in (200, 201)
        ):
            self.session.user_created = True

        self.session.steps_history.append(
            {
                "endpoint": endpoint_value,
                "payload_strategy": strategy_value,
                "target_user_id": action.target_user_id,
                "http_status": status,
                "latency_ms": latency_ms,
            }
        )
        if len(self.session.steps_history) > 200:
            self.session.steps_history.pop(0)

        self.session.task_specific_state["new_endpoint_discovered"] = new_endpoint
        self.session.task_specific_state["new_vulnerability_found"] = new_vulnerability
        self.session.task_specific_state["repeated_action"] = repeated_action

        task_score = compute_task_score(self.session, self.session.task_name)
        observation = StateStrikeObservation(
            step=self.session.step_count,
            endpoint_called=endpoint_value,
            http_status=status,
            latency_ms=latency_ms,
            response_body=body,
            session_order_count=self.session.order_count,
            endpoints_discovered=sorted(self.session.endpoints_discovered),
            vulnerabilities_found=sorted(self.session.vulnerabilities_found),
            task_progress=task_score,
        )

        reward, breakdown = compute_task_reward(
            observation,
            self.session,
            self.session.task_name,
            self.constants,
        )
        self.session.cumulative_reward += reward

        task_cfg, _ = TASK_REGISTRY[self.session.task_name]
        done = self.session.step_count >= task_cfg.max_steps or task_score >= task_cfg.success_threshold

        return StepResult(
            observation=observation,
            reward=reward,
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
    ) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
        endpoint_value = action.endpoint if isinstance(action.endpoint, str) else action.endpoint.value
        strategy_value = (
            action.payload_strategy if isinstance(action.payload_strategy, str) else action.payload_strategy.value
        )
        target_user_id = action.target_user_id or 1

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
