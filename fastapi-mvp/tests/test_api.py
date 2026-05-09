import importlib
from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _make_png_bytes(size=(8, 8), color=(255, 0, 0)) -> BytesIO:
    """Genere un vrai PNG (passera la verif PIL)."""
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def client():
    # Import lazily so patches in tests can intercept NudeNet before lifespan runs
    with patch("src.inference.NudeDetector") as mock_detector:
        mock_detector.return_value.detect.return_value = []
        from src import main as main_module

        importlib.reload(main_module)
        with TestClient(main_module.app) as c:
            yield c


@pytest.fixture
def auth_client(monkeypatch):
    """Client avec MODERATION_REQUIRE_AUTH=true et un secret connu."""
    monkeypatch.setenv("MODERATION_REQUIRE_AUTH", "true")
    monkeypatch.setenv("MODERATION_INTERNAL_SECRET", "s3cret-test")
    with patch("src.inference.NudeDetector") as mock_detector:
        mock_detector.return_value.detect.return_value = []
        from src import main as main_module

        importlib.reload(main_module)
        with TestClient(main_module.app) as c:
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
    fake_image = _make_png_bytes()
    r = client.post(
        "/moderate/image",
        files={"file": ("a.png", fake_image, "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"approved", "rejected"}
    assert "confidence" in body


def test_moderate_image_rejects_when_auth_required_without_header(auth_client):
    fake_image = _make_png_bytes()
    r = auth_client.post(
        "/moderate/image",
        files={"file": ("a.png", fake_image, "image/png")},
    )
    assert r.status_code == 401


def test_moderate_image_accepts_when_auth_header_valid(auth_client):
    fake_image = _make_png_bytes()
    r = auth_client.post(
        "/moderate/image",
        files={"file": ("a.png", fake_image, "image/png")},
        headers={"X-Internal-Secret": "s3cret-test"},
    )
    assert r.status_code == 200


def test_moderate_image_rejects_malformed_image(client):
    """Content-type image/png mais payload n'est pas une vraie image -> 415."""
    r = client.post(
        "/moderate/image",
        files={"file": ("a.png", BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 32), "image/png")},
    )
    assert r.status_code == 415


def test_moderate_image_rejects_svg(client):
    """SVG n'est pas dans la whitelist -> 415."""
    svg = BytesIO(b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"/>')
    r = client.post(
        "/moderate/image",
        files={"file": ("a.svg", svg, "image/svg+xml")},
    )
    assert r.status_code == 415


def test_moderate_image_rejects_when_auth_header_wrong(auth_client):
    fake_image = _make_png_bytes()
    r = auth_client.post(
        "/moderate/image",
        files={"file": ("a.png", fake_image, "image/png")},
        headers={"X-Internal-Secret": "wrong"},
    )
    assert r.status_code == 401
