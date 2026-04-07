from __future__ import annotations

"""Telemetry persistence helpers for StateStrike agent runs."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import portalocker

from statestrike_env.models import StateStrikeObservation


class TelemetryWriter:
    """Write and read JSON Lines telemetry with file locking.

    Args:
        file_path: JSONL file path.
    """

    def __init__(self, file_path: str = "telemetry.json") -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            self.file_path.touch()

    def record(self, obs: StateStrikeObservation) -> None:
        """Append one observation record to telemetry JSONL file.

        Args:
            obs: Observation frame to persist.
        """

        row = {
            "step": obs.step,
            "action_type": obs.action_taken.action_type.value,
            "payload_strategy": obs.action_taken.payload_strategy.value,
            "http_status": obs.http_status,
            "latency_ms": obs.latency_ms,
            "reward": obs.reward,
            "cumulative_reward": obs.cumulative_reward,
            "triggered_vulns": obs.triggered_vulns,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reward_breakdown": obs.info.get("reward_breakdown", {}),
        }

        with self.file_path.open("a", encoding="utf-8") as handle:
            portalocker.lock(handle, portalocker.LOCK_EX)
            handle.write(json.dumps(row) + os.linesep)
            handle.flush()
            portalocker.unlock(handle)

    @classmethod
    def read_recent(cls, n: int = 100, file_path: str = "telemetry.json") -> list[dict[str, Any]]:
        """Read the most recent telemetry rows.

        Args:
            n: Number of recent rows to return.
            file_path: JSONL file path.

        Returns:
            List of telemetry dicts in chronological order.
        """

        path = Path(file_path)
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as handle:
            portalocker.lock(handle, portalocker.LOCK_SH)
            lines = handle.readlines()
            portalocker.unlock(handle)

        records = [json.loads(line) for line in lines if line.strip()]
        return records[-n:]
