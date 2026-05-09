import threading
from unittest.mock import patch

from src import inference


def _mk_det(label: str, score: float) -> dict:
    return {"class": label, "score": score}


def test_classify_image_approves_when_no_unsafe_detections():
    with patch.object(inference, "load_model") as lm:
        lm.return_value.detect.return_value = [
            _mk_det("FACE_FEMALE", 0.99),
            _mk_det("FEET_EXPOSED", 0.8),
        ]
        result = inference.classify_image("ignored")

    assert result["decision"] == "approved"
    assert result["category"] is None
    assert result["all_detections"] == 2


def test_classify_image_rejects_when_unsafe_score_above_threshold():
    with patch.object(inference, "load_model") as lm:
        lm.return_value.detect.return_value = [
            _mk_det("FEMALE_BREAST_EXPOSED", 0.92),
            _mk_det("FACE_FEMALE", 0.5),
        ]
        result = inference.classify_image("ignored")

    assert result["decision"] == "rejected"
    assert result["category"] == "FEMALE_BREAST_EXPOSED"
    assert result["confidence"] == 0.92


def test_classify_image_approves_when_unsafe_below_threshold():
    with patch.object(inference, "load_model") as lm:
        lm.return_value.detect.return_value = [
            _mk_det("FEMALE_BREAST_EXPOSED", 0.4),
        ]
        result = inference.classify_image("ignored")

    assert result["decision"] == "approved"
    assert result["category"] is None


def test_classify_image_picks_highest_unsafe_score():
    with patch.object(inference, "load_model") as lm:
        lm.return_value.detect.return_value = [
            _mk_det("BUTTOCKS_EXPOSED", 0.7),
            _mk_det("FEMALE_GENITALIA_EXPOSED", 0.9),
            _mk_det("ANUS_EXPOSED", 0.65),
        ]
        result = inference.classify_image("ignored")

    assert result["decision"] == "rejected"
    assert result["category"] == "FEMALE_GENITALIA_EXPOSED"
    assert result["confidence"] == 0.9


def test_load_model_loads_only_once_under_concurrent_calls():
    """N threads qui appellent load_model en parallele ne doivent
    instancier NudeDetector qu'une seule fois (pas de double-load)."""
    inference.detector = None
    call_count = 0

    class FakeDetector:
        def __init__(self):
            nonlocal call_count
            call_count += 1
            # petite pause pour favoriser la collision entre threads
            import time as _t
            _t.sleep(0.05)

    with patch.object(inference, "NudeDetector", FakeDetector):
        threads = [threading.Thread(target=inference.load_model) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert call_count == 1
    inference.detector = None
