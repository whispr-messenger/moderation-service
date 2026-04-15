import logging
import sys
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Import the canonical health-label mapping (src/common/health_labels.py).
try:
    from common.health_labels import HEALTH_LABELS, UNHEALTHY_CLASSES
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from common.health_labels import HEALTH_LABELS, UNHEALTHY_CLASSES


def _load_model_and_classes(cfg: dict):
    """Load the trained model and discover class names from the training directory."""
    train_cfg = cfg["train_config"]
    comp_cfg = cfg["compilation_config"]

    dataset_root = Path.cwd() / train_cfg["dataset_dir"]
    train_dir = dataset_root / train_cfg["train_dir"]
    model_path = Path.cwd() / comp_cfg["model_name"]

    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(
        str(train_dir),
        target_size=(224, 224),
        batch_size=8,
        class_mode="categorical"
    )
    class_names = list(train_gen.class_indices.keys())
    logger.info("Classes: %s", class_names)

    model = load_model(str(model_path))
    logger.info("Model loaded: %s", model_path)

    return model, class_names


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(model, class_names, img_path, threshold=0.9):
    img = preprocess_image(img_path)
    preds = model.predict(img)[0]
    best_idx = np.argmax(preds)
    best_prob = float(preds[best_idx])
    best_class = class_names[best_idx]

    for name, p in zip(class_names, preds):
        logger.info("  %s: %.2f%%", name, p * 100)

    is_unhealthy = best_class in UNHEALTHY_CLASSES

    if best_prob >= threshold:
        health = HEALTH_LABELS.get(best_class, "unknown")
        result = (
            f"Predicted: {best_class} ({best_prob:.2%})\n"
            f"Health: {health} | Unhealthy: {'YES' if is_unhealthy else 'NO'}"
        )
    else:
        result = (
            f"Uncertain (best: {best_class} "
            f"at {best_prob:.2%} < {threshold:.0%})"
        )

    return result


def show_prediction(model, class_names, img_path, threshold=0.9):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = predict_image(model, class_names, img_path, threshold=threshold)
    plt.imshow(img)
    plt.axis("off")
    plt.title(result, fontsize=10, color="red")
    plt.show()


def run(cfg: dict):
    model, class_names = _load_model_and_classes(cfg)

    eval_dir = Path.cwd() / "validation" / "images"
    if not eval_dir.exists():
        logger.warning("No validation images directory found at %s", eval_dir)
        return

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in eval_dir.iterdir() if p.suffix.lower() in image_exts]
    if not images:
        logger.warning("No images found in validation/images/")
        return

    for img_path in images:
        logger.info("--- %s ---", img_path.name)
        show_prediction(model, class_names, str(img_path), threshold=0.4)
