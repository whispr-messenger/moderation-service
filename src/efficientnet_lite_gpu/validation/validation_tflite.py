"""Validation using the TFLite model for inference.

Usage from CLI:
    python -m validation.validation_tflite --model path/to/model.tflite --image path/to/image.jpg

Or from main.py via the 'eval_tflite' action.
"""

import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Import the canonical health-label mapping (src/common/health_labels.py).
try:
    from common.health_labels import HEALTH_LABELS, UNHEALTHY_CLASSES
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from common.health_labels import HEALTH_LABELS, UNHEALTHY_CLASSES


def load_tflite_model(model_path: str):
    """Load a TFLite model and return the interpreter with allocated tensors."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded:", model_path)
    return interpreter, input_details, output_details


def get_class_names(train_dir: Path) -> list[str]:
    """Discover class names from the training directory."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(
        str(train_dir),
        target_size=(224, 224),
        batch_size=1,
        class_mode="categorical"
    )
    return list(train_gen.class_indices.keys())


def preprocess_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)
    return img


def tflite_predict(interpreter, input_details, output_details, img_path: str) -> np.ndarray:
    img = preprocess_image(img_path)
    input_index = input_details[0]["index"]
    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.uint8:
        img_in = (img * 255).astype(np.uint8)
    else:
        img_in = img.astype(input_dtype)

    interpreter.set_tensor(input_index, img_in)
    interpreter.invoke()

    output_index = output_details[0]["index"]
    return interpreter.get_tensor(output_index)[0]


def predict_image(interpreter, input_details, output_details, class_names, img_path: str, threshold=0.9) -> str:
    preds = tflite_predict(interpreter, input_details, output_details, img_path)
    best_idx = np.argmax(preds)
    best_prob = float(preds[best_idx])
    best_class = class_names[best_idx]

    for name, p in zip(class_names, preds):
        print(f"  {name}: {p:.2%}")

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


def show_prediction(interpreter, input_details, output_details, class_names, img_path: str, threshold=0.9):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = predict_image(interpreter, input_details, output_details, class_names, img_path, threshold)
    plt.imshow(img)
    plt.axis("off")
    plt.title(result, fontsize=10, color="red")
    plt.show()


def run(cfg: dict):
    """Entry point called from main.py."""
    train_cfg = cfg["train_config"]
    comp_cfg = cfg["compilation_config"]

    dataset_root = Path.cwd() / train_cfg["dataset_dir"]
    train_dir = dataset_root / train_cfg["train_dir"]

    model_name = comp_cfg["model_name"]
    tflite_path = Path.cwd() / (Path(model_name).stem + "_inference.tflite")
    if not tflite_path.exists():
        tflite_path = Path.cwd() / "BestModelEfficientNetLite_inference.tflite"

    if not tflite_path.exists():
        print(f"TFLite model not found. Run convert_to_tflite.py first.")
        return

    class_names = get_class_names(train_dir)
    interpreter, input_details, output_details = load_tflite_model(str(tflite_path))

    eval_dir = Path.cwd() / "validation" / "images"
    if not eval_dir.exists():
        print(f"No validation images directory found at {eval_dir}")
        return

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in eval_dir.iterdir() if p.suffix.lower() in image_exts]
    if not images:
        print("No images found in validation/images/")
        return

    for img_path in images:
        print(f"\n--- {img_path.name} ---")
        show_prediction(interpreter, input_details, output_details, class_names, str(img_path), threshold=0.4)
