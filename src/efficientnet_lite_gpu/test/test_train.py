import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import pytest

from train import (
    _get_img_size,
    _build_paths,
    _get_efficientnet_class,
    _build_and_train_model,
    _build_datasets,
)

def test_get_img_size_int():
    train_cfg = {"image_size": 224}
    assert _get_img_size(train_cfg) == (224, 224)


def test_get_img_size_tuple():
    train_cfg = {"image_size": (128, 256)}
    assert _get_img_size(train_cfg) == (128, 256)


def test_get_img_size_invalid():
    train_cfg = {"image_size": "wrong"}
    with pytest.raises(ValueError):
        _get_img_size(train_cfg)

def test_build_paths_creates_dirs(tmp_path, monkeypatch):
    # On force le cwd sur un dossier temporaire pour ne rien polluer
    monkeypatch.chdir(tmp_path)

    train_cfg = {
        "dataset_dir": "train/dataset",
        "train_dir": "Train",
        "val_dir": "Train",
        "test_dir": "Test",
        "results_dir": "train/results",
        "data_exploration_dir": "data_exp",
        "evaluation_results_dir": "eval",
        "training_logs_dir": "logs",
        "training_results_dir": "train_results",
    }

    paths = _build_paths(train_cfg)

    # Chemins de base
    assert paths["train_dir"] == tmp_path / "train/dataset" / "Train"
    assert paths["test_dir"] == tmp_path / "train/dataset" / "Test"

    # Dossiers de résultats doivent exister
    assert paths["data_exploration_dir"].exists()
    assert paths["evaluation_results_dir"].exists()
    assert paths["training_logs_dir"].exists()
    assert paths["training_results_dir"].exists()


def _create_dummy_image(path: Path, size=(32, 32, 3)):
    """Crée une petite image noire pour les tests."""
    import cv2

    arr = np.zeros(size, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr)

def test_build_datasets_minimal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    dataset_root = tmp_path / "train" / "dataset"
    train_dir = dataset_root / "Train"
    test_dir = dataset_root / "Test"

    # Train : 5 images par classe (10 au total) -> 80% train / 20% val ok
    _create_dummy_image(train_dir / "classA" / "a1.jpg")
    _create_dummy_image(train_dir / "classA" / "a2.jpg")
    _create_dummy_image(train_dir / "classA" / "a3.jpg")
    _create_dummy_image(train_dir / "classA" / "a4.jpg")
    _create_dummy_image(train_dir / "classA" / "a5.jpg")

    _create_dummy_image(train_dir / "classB" / "b1.jpg")
    _create_dummy_image(train_dir / "classB" / "b2.jpg")
    _create_dummy_image(train_dir / "classB" / "b3.jpg")
    _create_dummy_image(train_dir / "classB" / "b4.jpg")
    _create_dummy_image(train_dir / "classB" / "b5.jpg")

    # Test : quelques images par classe (par exemple 2 chacune)
    _create_dummy_image(test_dir / "classA" / "a6.jpg")
    _create_dummy_image(test_dir / "classA" / "a7.jpg")
    _create_dummy_image(test_dir / "classB" / "b6.jpg")
    _create_dummy_image(test_dir / "classB" / "b7.jpg")

    train_cfg = {
        "batch_size": 10,
        "image_size": 32,
        "dataset_dir": "train/dataset",
        "train_dir": "Train",
        "val_dir": "Train",
        "test_dir": "Test",
        "results_dir": "train/results",
        "data_exploration_dir": "data_exp",
        "evaluation_results_dir": "eval",
        "training_logs_dir": "logs",
        "training_results_dir": "train_results",
    }

    paths = _build_paths(train_cfg)
    img_size = _get_img_size(train_cfg)

    train_ds, val_ds, test_ds, class_names, num_classes = _build_datasets(
        train_cfg, paths, img_size
    )

    assert num_classes == 2
    assert set(class_names) == {"classA", "classB"}

    next(iter(train_ds))
    next(iter(val_ds))
    next(iter(test_ds))

def test_build_and_train_model_tiny(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    dataset_root = tmp_path / "train" / "dataset"
    train_dir = dataset_root / "Train"
    test_dir = dataset_root / "Test"

    # Train : 5 images par classe (10 au total) -> 80% train / 20% val ok
    _create_dummy_image(train_dir / "classA" / "a1.jpg")
    _create_dummy_image(train_dir / "classA" / "a2.jpg")
    _create_dummy_image(train_dir / "classA" / "a3.jpg")
    _create_dummy_image(train_dir / "classA" / "a4.jpg")
    _create_dummy_image(train_dir / "classA" / "a5.jpg")

    _create_dummy_image(train_dir / "classB" / "b1.jpg")
    _create_dummy_image(train_dir / "classB" / "b2.jpg")
    _create_dummy_image(train_dir / "classB" / "b3.jpg")
    _create_dummy_image(train_dir / "classB" / "b4.jpg")
    _create_dummy_image(train_dir / "classB" / "b5.jpg")

    # Test : quelques images par classe (par exemple 2 chacune)
    _create_dummy_image(test_dir / "classA" / "a6.jpg")
    _create_dummy_image(test_dir / "classA" / "a7.jpg")
    _create_dummy_image(test_dir / "classB" / "b6.jpg")
    _create_dummy_image(test_dir / "classB" / "b7.jpg")


    train_cfg = {
        "batch_size": 10,
        "image_size": 32,
        "dataset_dir": "train/dataset",
        "train_dir": "Train",
        "val_dir": "Train",
        "test_dir": "Test",
        "results_dir": "train/results",
        "data_exploration_dir": "data_exp",
        "evaluation_results_dir": "eval",
        "training_logs_dir": "logs",
        "training_results_dir": "train_results",
        "initial_epochs": 1,
        "fine_tune_epochs": 1,
        "fine_tune": False,
        "data_augmentation": {},
        "model_config": {},
    }

    paths = _build_paths(train_cfg)
    img_size = _get_img_size(train_cfg)
    train_ds, val_ds, test_ds, class_names, num_classes = _build_datasets(
        train_cfg, paths, img_size
    )

    model_cfg = {
        "model_name": "efficientnet-b0",
        "include_top": False,
        "weights": None,  # pas de download
        "trainable": False,
        "output_activation": "softmax",
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"],
        "EarlyStopping": {
            "monitor": "val_accuracy",
            "patience": 1,
            "restore_best_weights": True,
        },
        "ReduceLROnPlateau": {
            "monitor": "val_loss",
            "factor": 0.5,
            "patience": 1,
            "min_lr": 1e-6,
        },
    }

    model, history_stage1, history_stage2, best_val_acc_stage1 = _build_and_train_model(
        train_cfg, model_cfg, img_size, num_classes, train_ds, val_ds
    )

    assert model.output_shape[-1] == num_classes
    assert "loss" in history_stage1.history
    assert "val_accuracy" in history_stage1.history
    assert history_stage2 is None
    assert isinstance(best_val_acc_stage1, float)
