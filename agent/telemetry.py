from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import portalocker

from statestrike_env.models import StateStrikeAction, StateStrikeObservation


class TelemetryWriter:
    def __init__(self, file_path: str = "telemetry.json") -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            self.file_path.touch()

    def record(
        self,
        obs: StateStrikeObservation,
        *,
        action: StateStrikeAction,
        reward: float,
        cumulative_reward: float,
        info: dict[str, Any] | None = None,
    ) -> None:
        details = info or {}
        row = {
            "step": obs.step,
            "action_type": action.endpoint if isinstance(action.endpoint, str) else action.endpoint.value,
            "payload_strategy": (
                action.payload_strategy
                if isinstance(action.payload_strategy, str)
                else action.payload_strategy.value
            ),
            "http_status": obs.http_status,
            "latency_ms": obs.latency_ms,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "triggered_vulns": obs.vulnerabilities_found,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reward_breakdown": details.get("reward_breakdown", {}),
        }

        with self.file_path.open("a", encoding="utf-8") as handle:
            portalocker.lock(handle, portalocker.LOCK_EX)
            handle.write(json.dumps(row) + os.linesep)
            handle.flush()
            portalocker.unlock(handle)

    @classmethod
    def read_recent(cls, n: int = 100, file_path: str = "telemetry.json") -> list[dict[str, Any]]:
        path = Path(file_path)
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as handle:
            portalocker.lock(handle, portalocker.LOCK_SH)
            lines = handle.readlines()
            portalocker.unlock(handle)

        records = [json.loads(line) for line in lines if line.strip()]
        return records[-n:]
