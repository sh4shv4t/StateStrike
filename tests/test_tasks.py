from __future__ import annotations

from statestrike_env.tasks import (
    ALL_ENDPOINTS,
    EndpointDiscoveryGrader,
    ExploitChainGrader,
    VulnerabilityProbeGrader,
)


def test_endpoint_discovery_partial_credit() -> None:
    grader = EndpointDiscoveryGrader()
    state = {"endpoints_discovered": ["GET /health", "POST /users"]}
    score = grader.score(state)
    assert score == 0.4


def test_endpoint_discovery_full_credit() -> None:
    grader = EndpointDiscoveryGrader()
    score = grader.score({"endpoints_discovered": list(ALL_ENDPOINTS)})
    assert score == 1.0


def test_vulnerability_probe_progressive_scoring() -> None:
    grader = VulnerabilityProbeGrader()

    score_none = grader.score({"steps_history": [], "vulnerabilities_found": []})
    assert score_none == 0.0

    score_attempt = grader.score(
        {
            "steps_history": [{"endpoint": "POST /users", "payload_strategy": "valid"}],
            "vulnerabilities_found": [],
        }
    )
    assert score_attempt >= 0.2

    score_payload = grader.score(
        {
            "steps_history": [{"endpoint": "POST /users", "payload_strategy": "redos"}],
            "vulnerabilities_found": [],
        }
    )
    assert score_payload >= 0.4

    score_both = grader.score(
        {
            "steps_history": [{"endpoint": "POST /users", "payload_strategy": "redos"}],
            "vulnerabilities_found": ["redos", "db_degradation"],
        }
    )
    assert score_both == 1.0


def test_exploit_chain_partial_and_full() -> None:
    grader = ExploitChainGrader()

    partial = grader.score(
        {
            "user_created": True,
            "order_count": 21,
            "vulnerabilities_found": [],
        }
    )
    assert partial == 0.5

    full = grader.score(
        {
            "user_created": True,
            "order_count": 21,
            "vulnerabilities_found": ["redos", "db_degradation"],
        }
    )
    assert full == 1.0
