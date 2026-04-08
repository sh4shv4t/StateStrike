from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TaskConfig:
    task_id: str
    description: str
    max_steps: int
    difficulty: str
    success_threshold: float


TASK_ENDPOINT_DISCOVERY = TaskConfig(
    task_id="endpoint_discovery",
    description="Discover all 5 reachable API endpoints by probing them.",
    max_steps=20,
    difficulty="easy",
    success_threshold=0.6,
)

ALL_ENDPOINTS = {
    "POST /users",
    "GET /users/{id}",
    "POST /orders",
    "GET /orders",
    "GET /health",
}


class EndpointDiscoveryGrader:
    def score(self, session_state: dict[str, Any]) -> float:
        found = set(session_state.get("endpoints_discovered", []))
        return round(len(found & ALL_ENDPOINTS) / len(ALL_ENDPOINTS), 4)


TASK_VULNERABILITY_PROBE = TaskConfig(
    task_id="vulnerability_probe",
    description=(
        "Identify and correctly classify at least one vulnerability. "
        "Vulnerabilities: 'redos' (POST /users with attack payload, latency > 1500ms), "
        "'db_degradation' (GET /orders after 20+ orders, latency > 800ms)."
    ),
    max_steps=30,
    difficulty="medium",
    success_threshold=0.5,
)


class VulnerabilityProbeGrader:
    REDOS_LATENCY_THRESHOLD_MS = 500.0
    DB_LATENCY_THRESHOLD_MS = 800.0

    def score(self, session_state: dict[str, Any]) -> float:
        vulns = set(session_state.get("vulnerabilities_found", []))
        steps_history = session_state.get("steps_history", [])

        score = 0.0

        if any(s.get("endpoint") != "GET /health" for s in steps_history):
            score = max(score, 0.2)

        if any(s.get("payload_strategy") != "valid" for s in steps_history):
            score = max(score, 0.4)

        if "redos" in vulns:
            score = max(score, 0.6)

        if "db_degradation" in vulns:
            score = max(score, 0.8)

        if "redos" in vulns and "db_degradation" in vulns:
            score = 1.0

        return round(score, 4)


TASK_EXPLOIT_CHAIN = TaskConfig(
    task_id="exploit_chain",
    description=(
        "Execute the full stateful exploit chain in order: "
        "1. Create a user (POST /users, valid payload). "
        "2. Build order history (POST /orders x 21 for same user). "
        "3. Trigger DB degradation (GET /orders with 21 orders, latency > 800ms). "
        "4. Trigger ReDoS (POST /users, redos payload, latency > 1500ms). "
        "All four sub-objectives must be completed for full score."
    ),
    max_steps=60,
    difficulty="hard",
    success_threshold=0.75,
)


class ExploitChainGrader:
    def score(self, session_state: dict[str, Any]) -> float:
        score = 0.0

        if session_state.get("user_created", False):
            score += 0.25
        if session_state.get("order_count", 0) >= 21:
            score += 0.25
        if "db_degradation" in session_state.get("vulnerabilities_found", []):
            score += 0.25
        if "redos" in session_state.get("vulnerabilities_found", []):
            score += 0.25

        return round(score, 4)


TASK_REGISTRY = {
    "endpoint_discovery": (TASK_ENDPOINT_DISCOVERY, EndpointDiscoveryGrader()),
    "vulnerability_probe": (TASK_VULNERABILITY_PROBE, VulnerabilityProbeGrader()),
    "exploit_chain": (TASK_EXPLOIT_CHAIN, ExploitChainGrader()),
}
