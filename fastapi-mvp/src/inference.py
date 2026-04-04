from nudenet import NudeDetector
import logging

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

# Threshold for blocking
BLOCK_THRESHOLD = 0.6

detector = None

def load_model():
    global detector
    if detector is None:
        logger.info("Loading NudeNet model...")
        detector = NudeDetector()
        logger.info("NudeNet model loaded")
    return detector

def classify_image(image_path: str) -> dict:
    """Classify an image and return moderation decision."""
    model = load_model()
    detections = model.detect(image_path)

    # Find the most unsafe detection
    max_unsafe_score = 0.0
    max_category = None

    for det in detections:
        label = det.get("class", "")
        score = det.get("score", 0.0)
        if label in UNSAFE_CATEGORIES and score > max_unsafe_score:
            max_unsafe_score = score
            max_category = label

    if max_unsafe_score >= BLOCK_THRESHOLD:
        return {
            "decision": "rejected",
            "confidence": round(max_unsafe_score, 4),
            "category": max_category,
            "all_detections": len(detections),
        }

    return {
        "decision": "approved",
        "confidence": round(1.0 - max_unsafe_score, 4),
        "category": None,
        "all_detections": len(detections),
    }
