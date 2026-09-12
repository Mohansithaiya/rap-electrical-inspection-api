"""Tests for the /ask endpoint, using mock detection payloads only."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def _payload(question, detections, image_id="img1", width=640, height=480):
    return {
        "question": question,
        "detections": {
            "image_id": image_id,
            "image_width": width,
            "image_height": height,
            "detections": detections,
        },
    }


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_confident_presence():
    payload = _payload(
        "Is there an insulator in this image?",
        [{"class": "insulator", "confidence": 0.91, "bbox": [10, 10, 50, 50]}],
    )
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer_state"] == "ANSWERED"
    assert body["intent"] == "PRESENCE"
    assert "insulator" in body["answer"].lower()


def test_ask_insufficient_info_absent_class():
    payload = _payload(
        "Is there a spark visible?",
        [{"class": "transformer", "confidence": 0.9, "bbox": [0, 0, 10, 10]}],
    )
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer_state"] == "INSUFFICIENT_INFO"
    assert "not detected" in body["answer"].lower()


def test_ask_risk_assessment_unsafe():
    payload = _payload(
        "Is this scene safe?",
        [{"class": "arc", "confidence": 0.8, "bbox": [0, 0, 5, 5]}],
    )
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "RISK_ASSESSMENT"
    assert "unsafe" in body["answer"].lower()


def test_ask_empty_detections_insufficient_info():
    payload = _payload("Is this scene safe?", [])
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200
    assert resp.json()["answer_state"] == "INSUFFICIENT_INFO"


def test_ask_missing_question_returns_422():
    payload = _payload("", [])
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 422


def test_ask_invalid_confidence_returns_422():
    payload = _payload(
        "Is there an arc?",
        [{"class": "arc", "confidence": 1.5, "bbox": [0, 0, 5, 5]}],
    )
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 422


def test_ask_invalid_bbox_length_returns_422():
    payload = _payload(
        "Is there an arc?",
        [{"class": "arc", "confidence": 0.9, "bbox": [0, 0, 5]}],
    )
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 422


def test_ask_missing_detections_field_returns_422():
    resp = client.post("/ask", json={"question": "Is there an arc?"})
    assert resp.status_code == 422


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
