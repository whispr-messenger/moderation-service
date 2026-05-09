import logging
import os
import threading
import time

from nudenet import NudeDetector

logger = logging.getLogger("moderation")

# Categories considered unsafe
UNSAFE_CATEGORIES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_BREAST_EXPOSED",
}

# Threshold for blocking, env-driven (0.0 - 1.0)
BLOCK_THRESHOLD = float(os.getenv("MOD_BLOCK_THRESHOLD", "0.6"))

detector = None
# Lock pour eviter le double-load si deux requetes arrivent au cold-start
# avant que le lifespan ait fini d'instancier NudeDetector. Sans ca, on
# peut charger le modele 2x (waste memoire) ou laisser un detector partiel.
_load_lock = threading.Lock()


def load_model():
    global detector
    # double-checked locking : verif rapide sans lock pour le hot path
    if detector is not None:
        return detector
    with _load_lock:
        # re-verifier sous lock au cas ou un autre thread a charge entre temps
        if detector is None:
            logger.info("Loading NudeNet model...")
            t0 = time.perf_counter()
            detector = NudeDetector()
            logger.info("NudeNet model loaded in %.2fs", time.perf_counter() - t0)
    return detector


def classify_image(image_path: str) -> dict:
    """Classify an image and return moderation decision."""
    model = load_model()
    t0 = time.perf_counter()
    detections = model.detect(image_path)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Find the most unsafe detection
    max_unsafe_score = 0.0
    max_category = None

    for det in detections:
        label = det.get("class", "")
        score = det.get("score", 0.0)
        if label in UNSAFE_CATEGORIES and score > max_unsafe_score:
            max_unsafe_score = score
            max_category = label

    logger.info(
        "inference done path=%s latency_ms=%.1f detections=%d top=%s score=%.3f",
        image_path, latency_ms, len(detections), max_category, max_unsafe_score,
    )

    if max_unsafe_score >= BLOCK_THRESHOLD:
        return {
            "decision": "rejected",
            "confidence": round(max_unsafe_score, 4),
            "category": max_category,
            "all_detections": len(detections),
            "latency_ms": round(latency_ms, 1),
        }

    return {
        "decision": "approved",
        "confidence": round(1.0 - max_unsafe_score, 4),
        "category": None,
        "all_detections": len(detections),
        "latency_ms": round(latency_ms, 1),
    }
