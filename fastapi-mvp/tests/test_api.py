from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # Import lazily so patches in tests can intercept NudeNet before lifespan runs
    with patch("src.inference.NudeDetector") as mock_detector:
        mock_detector.return_value.detect.return_value = []
        from src.main import app

        with TestClient(app) as c:
            yield c


def test_health_endpoint_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "moderation"


def test_ready_endpoint_returns_ready_after_startup(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert "threshold" in body


def test_root_endpoint_returns_metadata(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "whispr-moderation"
    assert "threshold" in body


def test_moderate_image_rejects_non_image_content_type(client):
    r = client.post(
        "/moderate/image",
        files={"file": ("a.txt", BytesIO(b"not an image"), "text/plain")},
    )
    assert r.status_code == 400


def test_moderate_image_accepts_image_and_returns_decision(client):
    fake_image = BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    r = client.post(
        "/moderate/image",
        files={"file": ("a.png", fake_image, "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"approved", "rejected"}
    assert "confidence" in body
