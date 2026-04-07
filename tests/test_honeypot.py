from __future__ import annotations

"""API contract tests for the vulnerable honeypot."""

from fastapi.testclient import TestClient

from honeypot.app import app


def test_health_contract() -> None:
    """Validate health endpoint shape for orchestration and monitoring checks.

    Why it matters:
        Reliable liveness semantics are required for docker-compose healthchecks
        and reproducible OpenEnv training orchestration.
    """

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["ts"], int)


def test_user_and_order_contract_flow() -> None:
    """Validate expected CRUD contract for user creation and order lifecycle.

    Why it matters:
        Ensures stateful action chains are possible, which is central to the
        challenge objective of sequence-based vulnerability discovery.
    """

    client = TestClient(app)
    create_user = client.post("/users", json={"email": "validuser123"})
    assert create_user.status_code == 200
    user_id = create_user.json()["id"]

    create_order = client.post("/orders", json={"user_id": user_id, "item": "widget"})
    assert create_order.status_code == 200

    list_orders = client.get("/orders", params={"user_id": user_id})
    assert list_orders.status_code == 200
    assert list_orders.json()["count"] >= 1


def test_redos_style_payload_rejected() -> None:
    """Reject invalid email formats with a fast, safe contract payload.

    Why it matters:
        Unit tests must verify API contract deterministically without using
        catastrophic backtracking payloads that can deadlock test runners.
    """

    client = TestClient(app)
    payload = {"email": "invalid_email_format@@"}
    response = client.post("/users", json=payload)

    assert response.status_code == 400
    assert "detail" in response.json()


def test_stateful_degradation_path_increases_latency() -> None:
    """Trigger >20 order state and confirm synthetic degradation latency exists.

    Why it matters:
        Validates that the environment contains a sequence-dependent bottleneck
        discoverable by an RL policy but hard for stateless fuzzers.
    """

    client = TestClient(app)
    create_user = client.post("/users", json={"email": "loadtester123"})
    user_id = create_user.json()["id"]

    for idx in range(21):
        response = client.post("/orders", json={"user_id": user_id, "item": f"item-{idx}"})
        assert response.status_code == 200

    response = client.get("/orders", params={"user_id": user_id})
    assert response.status_code == 200

    latency_header = response.headers.get("X-Process-Time-Ms")
    assert latency_header is not None
    assert float(latency_header) >= 700.0
