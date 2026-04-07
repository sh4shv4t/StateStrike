from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
from typing import Iterable

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from agent.telemetry import TelemetryWriter
from statestrike_env.environment import StateStrikeEnv
from statestrike_env.models import EndpointChoice, PayloadStrategy, StateStrikeAction

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)
CONSOLE = Console()

TELEMETRY_FILE = os.getenv("TELEMETRY_FILE", "telemetry.json")


def random_policy() -> StateStrikeAction:
    choices: Iterable[StateStrikeAction] = (
        StateStrikeAction(endpoint=EndpointChoice.POST_USERS, payload_strategy=PayloadStrategy.VALID),
        StateStrikeAction(endpoint=EndpointChoice.POST_USERS, payload_strategy=PayloadStrategy.REDOS_ATTACK),
        StateStrikeAction(endpoint=EndpointChoice.POST_ORDERS, payload_strategy=PayloadStrategy.VALID, target_user_id=1),
        StateStrikeAction(endpoint=EndpointChoice.GET_ORDERS, payload_strategy=PayloadStrategy.VALID, target_user_id=1),
        StateStrikeAction(endpoint=EndpointChoice.GET_USER, payload_strategy=PayloadStrategy.VALID, target_user_id=1),
        StateStrikeAction(endpoint=EndpointChoice.HEALTH, payload_strategy=PayloadStrategy.VALID),
    )
    return random.choice(tuple(choices))


def _print_progress(step: int, cumulative_reward: float, action: StateStrikeAction, latency_ms: float, vulns: list[str]) -> None:
    table = Table(title="StateStrike Baseline Agent")
    table.add_column("Step")
    table.add_column("Cumulative Reward")
    table.add_column("Last Action")
    table.add_column("Last Latency (ms)")
    table.add_column("Vulns Found")
    endpoint = action.endpoint if isinstance(action.endpoint, str) else action.endpoint.value
    table.add_row(
        str(step),
        f"{cumulative_reward:.2f}",
        endpoint,
        f"{latency_ms:.2f}",
        ", ".join(vulns) if vulns else "none",
    )
    CONSOLE.print(table)


async def run(steps: int = 200) -> None:
    telemetry = TelemetryWriter(file_path=TELEMETRY_FILE)
    env = StateStrikeEnv()

    cumulative_reward = 0.0
    result = await env.reset(task_name="exploit_chain")

    for idx in range(1, steps + 1):
        action = random_policy()
        result = await env.step(action)

        cumulative_reward += result.reward
        telemetry.record(
            result.observation,
            action=action,
            reward=result.reward,
            cumulative_reward=cumulative_reward,
            info=result.info,
        )

        if idx % 10 == 0:
            _print_progress(
                step=result.observation.step,
                cumulative_reward=cumulative_reward,
                action=action,
                latency_ms=result.observation.latency_ms,
                vulns=result.observation.vulnerabilities_found,
            )

        if result.done:
            LOGGER.info(
                "Episode ended at step=%s reward=%.3f",
                result.observation.step,
                cumulative_reward,
            )
            break

    await env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run StateStrike baseline agent")
    parser.add_argument("--steps", type=int, default=200, help="Total steps to execute")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(steps=args.steps))
