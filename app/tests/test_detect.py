"""Tests for the /detect endpoint, with RT-DETR inference mocked out."""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def _upload():
    return {"file": ("test.jpg", b"fake-image-bytes", "image/jpeg")}


def test_detect_returns_mocked_detections():
    mock_detections = [{"class": "insulator", "confidence": 0.9, "bbox": [1.0, 2.0, 3.0, 4.0]}]
    with patch("app.main.run_inference", return_value=(mock_detections, 640, 480)):
        resp = client.post("/detect", files=_upload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["detections"][0]["class"] == "insulator"
    assert body["image_width"] == 640
    assert body["image_height"] == 480


def test_detect_returns_empty_detections():
    with patch("app.main.run_inference", return_value=([], 0, 0)):
        resp = client.post("/detect", files=_upload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["detections"] == []
    assert body["image_width"] == 0
    assert body["image_height"] == 0


def test_detect_invalid_image_returns_400():
    # Not mocked: exercises the real detector.run_inference decode path,
    # which raises before the model is ever loaded.
    resp = client.post("/detect", files=_upload())
    assert resp.status_code == 400
    assert resp.json() == {"detail": "Uploaded file is not a valid image"}


def test_detect_unexpected_error_returns_clean_500():
    # raise_server_exceptions=False: we're verifying the global handler's
    # HTTP response, not letting the test client re-raise for debugging.
    non_raising_client = TestClient(app, raise_server_exceptions=False)
    with patch("app.main.run_inference", side_effect=RuntimeError("boom")):
        resp = non_raising_client.post("/detect", files=_upload())
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Internal server error"}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
