import sys
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Import the canonical health-label mapping (src/common/health_labels.py).
try:
    from common.health_labels import HEALTH_LABELS, UNHEALTHY_CLASSES
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from common.health_labels import HEALTH_LABELS, UNHEALTHY_CLASSES


def _resolve_path(base_dir: Path, value, fallback: Path) -> Path:
    if not value:
        return fallback.resolve()
    p = Path(value)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _load_class_names(dataset_dir: Path):
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found for validation: {dataset_dir}"
        )
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_gen = datagen.flow_from_directory(
        str(dataset_dir),
        target_size=(224, 224),
        batch_size=8,
        class_mode="categorical",
    )
    class_names = list(train_gen.class_indices.keys())
    print("class_indices:", train_gen.class_indices)
    return class_names


def preprocess_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(
            f"Impossible de lire l'image : {img_path}. Vérifiez que le fichier existe et que le chemin est correct."
        )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(model, class_names, img_path, threshold=0.9):
    img = preprocess_image(img_path)
    preds = model.predict(img, verbose=0)[0]
    best_idx = int(np.argmax(preds))
    best_prob = float(preds[best_idx])
    best_class = class_names[best_idx]

    for name, p in zip(class_names, preds):
        print(f"  {name}: {float(p):.2%}")

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


def show_prediction(model, class_names, img_path, threshold=0.9, show_plot=True):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(
            f"Impossible de lire l'image : {img_path}. Vérifiez que le fichier existe et que le chemin est correct."
        )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = predict_image(model, class_names, img_path, threshold=threshold)
    if show_plot:
        plt.imshow(img)
        plt.axis("off")
        plt.title(result, fontsize=10, color="red")
        plt.show()
    return result


def run(cfg: dict):
    module_root = Path(__file__).resolve().parents[1]
    train_cfg = cfg.get("train_config", {})
    comp_cfg = cfg.get("compilation_config", {})
    val_cfg = cfg.get("validation_config", {})

    dataset_root = train_cfg.get("dataset_dir", "train/dataset")
    train_dir_name = train_cfg.get("train_dir", "Train")
    default_dataset_dir = (module_root / dataset_root / train_dir_name).resolve()

    default_model = (module_root / comp_cfg.get("model_name", "BestModelEfficientNetLite.h5")).resolve()
    default_image = (module_root / "validation/images/junk_food.jpg").resolve()

    dataset_dir = _resolve_path(module_root, val_cfg.get("dataset_dir"), default_dataset_dir)
    model_path = _resolve_path(module_root, val_cfg.get("model_path"), default_model)
    img_path = _resolve_path(module_root, val_cfg.get("image_path"), default_image)
    threshold = float(val_cfg.get("threshold", 0.4))
    show_plot = bool(val_cfg.get("show_plot", True))

    print(f"Using validation dataset dir: {dataset_dir}")
    print(f"Using validation model path: {model_path}")
    print(f"Using validation image path: {img_path}")
    print(f"Using validation threshold: {threshold}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Please train first or pass --validation-model <path_to_h5_or_keras>."
        )

    class_names = _load_class_names(dataset_dir)
    model = load_model(str(model_path))
    print("model loaded")

    result = show_prediction(model, class_names, img_path, threshold=threshold, show_plot=show_plot)
    print(result)
