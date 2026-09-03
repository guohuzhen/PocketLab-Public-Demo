from __future__ import annotations

import os


def _draft() -> dict[str, object]:
    return {
        "title": "Public demo vibration comparison",
        "question": "Does the comparison condition have higher vibration than the reference?",
        "objective": "compare_conditions",
        "requested_claim": "relative_comparison",
        "independent_variable": "machine operating state",
        "conditions": [
            {
                "condition_id": "reference",
                "label": "machine stopped",
                "factor_level": "reference",
                "instruction": "Keep the phone fixed and record the stopped condition.",
                "activation": "required",
            },
            {
                "condition_id": "comparison",
                "label": "machine running steadily",
                "factor_level": "comparison",
                "instruction": "Change only the operating state and record again.",
                "activation": "required",
            },
        ],
        "sensor_intents": [
            {
                "sensor": "accelerometer",
                "role": "primary",
                "activation": "required",
                "metric_key": "selected_axis_rms_m_s2",
                "metric_unit": "m/s^2",
                "measurement_purpose": "Compare bounded vibration magnitude between conditions.",
            }
        ],
        "alignment": "sequential",
        "controls": ["Keep the same phone and placement.", "Change only the target condition."],
        "expected_pattern": "The comparison may exceed repeat variation.",
        "safety_notes": ["This smoke test creates protocol state only."],
        "privacy_notes": ["No phone, provider, or user dataset is read."],
        "claim_boundaries": [
            "Protocol creation alone is not physical evidence.",
            "A simulated protocol cannot establish a real-world causal claim.",
        ],
    }


def main() -> int:
    os.environ["POCKETLAB_DB_PATH"] = ":memory:"
    os.environ["LLM_API_KEY"] = ""
    os.environ["LLM_BASE_URL"] = ""
    os.environ["LLM_MODEL"] = ""

    from fastapi.testclient import TestClient

    from pocketlab.main import app

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200, health.text
    assert health.json()["status"] == "ok"

    registration = client.post(
        "/api/v1/auth/register",
        json={
            "username": "public_demo_smoke",
            "password": "isolated-public-demo-password",
            "display_name": "Public Demo Smoke",
            "claim_local_data": False,
        },
    )
    assert registration.status_code == 200, registration.text

    replays = client.get("/api/v2/public-replays")
    assert replays.status_code == 200, replays.text
    assert replays.json() == [], "The public snapshot must not silently bundle replay datasets."

    diagnostic_start = client.post("/api/v2/showcase-replays/diagnostic")
    assert diagnostic_start.status_code == 200, diagnostic_start.text
    diagnostic_case = diagnostic_start.json()["case"]
    diagnostic_clicks = 0
    while diagnostic_case["current_task"] is not None:
        task_id = diagnostic_case["current_task"]["task_id"]
        diagnostic_step = client.post(
            f"/api/v2/showcase-replays/diagnostic/{diagnostic_case['case_id']}/tasks/{task_id}"
        )
        assert diagnostic_step.status_code == 200, diagnostic_step.text
        diagnostic_case = diagnostic_step.json()["case"]
        diagnostic_clicks += 1
        assert diagnostic_clicks <= 2
    assert diagnostic_clicks == 2
    assert diagnostic_case["final_report"] is not None

    exploration_start = client.post("/api/v2/showcase-replays/exploration")
    assert exploration_start.status_code == 200, exploration_start.text
    exploration_case = exploration_start.json()
    exploration_clicks = 0
    while exploration_case["current_task"] is not None:
        task_id = exploration_case["current_task"]["task_id"]
        exploration_step = client.post(
            f"/api/v2/showcase-replays/exploration/{exploration_case['case_id']}/tasks/{task_id}",
            json={"expected_revision": exploration_case["revision"]},
        )
        assert exploration_step.status_code == 200, exploration_step.text
        exploration_case = exploration_step.json()["case"]
        exploration_clicks += 1
        assert exploration_clicks <= 4
    assert exploration_clicks == 4
    assert exploration_case["report"] is not None

    created = client.post(
        "/api/v2/general-explorations",
        json={
            "draft": _draft(),
            "source": "protocol_emulator",
            "privacy_acknowledged_sensors": [],
        },
    )
    assert created.status_code == 200, created.text
    case = created.json()
    assert case["status"] == "collecting"
    assert case["protocol"]["selected_sources"] == ["protocol_emulator"]
    assert case["current_task"]["sensors"] == ["accelerometer"]

    print(
        "Public demo smoke passed: health, auth, empty third-party replay catalog, "
        "two-step diagnosis, four-step light exploration, and protocol creation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
