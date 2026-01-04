import numpy as np
import pytest
from validation.validation import preprocess_image, predict_image
import cv2


def test_preprocess_image_invalid_path():
    with pytest.raises(FileNotFoundError):
        preprocess_image("tests_images/not_found.jpg")

def test_preprocess_image_shape_dtype(tmp_path):
    # crée une petite image factice 224x224x3
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img_path = tmp_path / "img.jpg"
    cv2.imwrite(str(img_path), img)

    x = preprocess_image(str(img_path))
    assert x.shape == (1, 224, 224, 3)
    assert x.dtype == np.float32

def test_predict_image_returns_string():
    result = predict_image("test/test_images/fries.jpg", threshold=0.9)
    assert isinstance(result, str)
    assert "Catégorie prédite" in result or "Prédiction incertaine" in result

def test_predict_image_junk_flag():
    result = predict_image("test/test_images/fries.jpg", threshold=0.1)
    assert "Junk food" in result  # raffiner si tu as une image bien étiquetée
