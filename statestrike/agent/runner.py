from __future__ import annotations

"""Random baseline StateStrike agent runner."""

import argparse
import logging
import os
import random
import time
from typing import Callable

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from agent.telemetry import TelemetryWriter
from statestrike_env import StateStrikeEnv
from statestrike_env.models import ActionType, PayloadStrategy, StateStrikeAction, StateStrikeObservation

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)
CONSOLE = Console()

WS_URL = os.getenv("STATESTRIKE_ENV_WS_URL", "ws://localhost:8001/ws")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_BASE_SECONDS = float(os.getenv("RETRY_BACKOFF_BASE_SECONDS", "0.25"))
TELEMETRY_FILE = os.getenv("TELEMETRY_FILE", "telemetry.json")
RESET_STARTUP_ATTEMPTS = 5
RESET_STARTUP_DELAY_SECONDS = 2
INITIAL_STARTUP_DELAY_SECONDS = 3


def random_policy(obs: StateStrikeObservation, target_user_id: int | None = None) -> StateStrikeAction:
    """Sample weighted baseline action distribution.

    Args:
        obs: Latest observation (currently unused but preserved for extensibility).
        target_user_id: Preferred user id for user/order actions.

    Returns:
        A sampled StateStrikeAction.

    Theory:
        A uniform random policy almost never discovers vuln 2 (requires >20 POST
        /orders). This weighted policy has a meaningful probability of triggering
        it within modest horizons, providing a stronger baseline for RL comparison.
    """

    del obs

    weighted_actions: list[tuple[float, StateStrikeAction]] = [
        (
            0.40,
            StateStrikeAction(
                action_type=ActionType.POST_ORDER,
                payload_strategy=PayloadStrategy.VALID,
                target_user_id=target_user_id,
            ),
        ),
        (
            0.20,
            StateStrikeAction(
                action_type=ActionType.GET_ORDERS,
                payload_strategy=PayloadStrategy.VALID,
                target_user_id=target_user_id,
            ),
        ),
        (
            0.15,
            StateStrikeAction(
                action_type=ActionType.POST_USER,
                payload_strategy=PayloadStrategy.REDOS_ATTACK,
            ),
        ),
        (
            0.15,
            StateStrikeAction(
                action_type=ActionType.POST_USER,
                payload_strategy=PayloadStrategy.VALID,
            ),
        ),
        (
            0.10,
            StateStrikeAction(
                action_type=ActionType.GET_USER,
                payload_strategy=PayloadStrategy.VALID,
                target_user_id=target_user_id,
            ),
        ),
    ]

    threshold = random.random()
    cumulative = 0.0
    for weight, action in weighted_actions:
        cumulative += weight
        if threshold <= cumulative:
            return action
    return weighted_actions[-1][1]


def _call_with_retry(operation: Callable[[], StateStrikeObservation], label: str) -> StateStrikeObservation:
    """Execute environment call with retry and exponential backoff.

    Args:
        operation: Zero-arg callable invoking env.reset or env.step.
        label: Operation label for logs.

    Returns:
        Observation produced by successful operation.

    Raises:
        RuntimeError: If retries are exhausted.
    """

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            delay = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            LOGGER.warning("%s failed attempt=%s/%s error=%s", label, attempt, MAX_RETRIES, exc)
            time.sleep(delay)

    raise RuntimeError(f"{label} failed after {MAX_RETRIES} attempts") from last_error


def _print_progress(obs: StateStrikeObservation) -> None:
    """Render periodic rich progress table.

    Args:
        obs: Latest observation.
    """

    table = Table(title="StateStrike Baseline Agent")
    table.add_column("Step")
    table.add_column("Cumulative Reward")
    table.add_column("Last Action")
    table.add_column("Last Latency (ms)")
    table.add_column("Vulns Found")

    table.add_row(
        str(obs.step),
        f"{obs.cumulative_reward:.2f}",
        obs.action_taken.action_type.value,
        f"{obs.latency_ms:.2f}",
        ", ".join(obs.triggered_vulns) if obs.triggered_vulns else "none",
    )
    CONSOLE.print(table)


def _reset_with_startup_retry(env: object) -> StateStrikeObservation:
    """Run reset with fixed retry policy for cold-start stabilization.

    Args:
        env: Synchronous environment client implementing reset().

    Returns:
        Initial reset observation.

    Raises:
        RuntimeError: If all startup reset attempts fail.
    """

    last_error: Exception | None = None
    for attempt in range(1, RESET_STARTUP_ATTEMPTS + 1):
        try:
            return env.reset()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            LOGGER.warning(
                "Startup reset failed attempt=%s/%s error=%s",
                attempt,
                RESET_STARTUP_ATTEMPTS,
                exc,
            )
            if attempt < RESET_STARTUP_ATTEMPTS:
                time.sleep(RESET_STARTUP_DELAY_SECONDS)

    raise RuntimeError("env.reset failed during startup retries") from last_error


def run(steps: int = 500) -> None:
    """Run baseline agent against StateStrike environment.

    Args:
        steps: Number of action steps to execute.
    """

    telemetry = TelemetryWriter(file_path=TELEMETRY_FILE)
    target_user_id: int | None = 1

    # Allow env service to complete startup readiness checks.
    time.sleep(INITIAL_STARTUP_DELAY_SECONDS)

    with StateStrikeEnv(base_url=WS_URL).sync() as env:
        obs = _reset_with_startup_retry(env)

        bootstrap_action = StateStrikeAction(
            action_type=ActionType.POST_USER,
            payload_strategy=PayloadStrategy.VALID,
        )
        bootstrap_obs = _call_with_retry(lambda: env.step(bootstrap_action), "env.step.bootstrap")
        telemetry.record(bootstrap_obs)
        response = bootstrap_obs.info.get("response", {})
        if isinstance(response, dict) and isinstance(response.get("id"), int):
            target_user_id = int(response["id"])

        for step_index in range(steps):
            action = random_policy(obs, target_user_id=target_user_id)

            try:
                obs = _call_with_retry(lambda: env.step(action), "env.step")
            except RuntimeError as exc:
                LOGGER.warning("Returning synthetic observation after retry exhaustion: %s", exc)
                obs = StateStrikeObservation(
                    step=step_index,
                    action_taken=action,
                    http_status=0,
                    latency_ms=0.0,
                    reward=0.0,
                    cumulative_reward=obs.cumulative_reward,
                    baseline_latency_ms=obs.baseline_latency_ms,
                    order_count=obs.order_count,
                    triggered_vulns=obs.triggered_vulns,
                    done=False,
                    info={"synthetic": True, "warning": str(exc)},
                )

            telemetry.record(obs)

            if step_index % 10 == 0:
                _print_progress(obs)

            if obs.done:
                LOGGER.info("Episode ended at step=%s reward=%.3f", obs.step, obs.cumulative_reward)
                obs = _call_with_retry(env.reset, "env.reset")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed namespace.
    """

    parser = argparse.ArgumentParser(description="Run StateStrike random baseline agent")
    parser.add_argument("--steps", type=int, default=500, help="Total steps to execute")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(steps=args.steps)
