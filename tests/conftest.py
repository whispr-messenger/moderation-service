"""Fixtures partagées."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml


def pytest_configure(config):
    """validation.validation exécute des chemins locaux et flow_from_directory à l'import ; module de remplacement pour les tests."""
    key = "validation.validation"
    if key in sys.modules:
        return
    pkg = types.ModuleType("validation")
    mod = types.ModuleType(key)

    def _eval_run(cfg=None):
        return None

    mod.run = _eval_run
    sys.modules["validation"] = pkg
    sys.modules[key] = mod


@pytest.fixture
def valid_train_config_dict() -> dict:
    return {
        "training_divice": "cpu",
        "mixed_precision": False,
        "dataset_dir": "train/dataset",
        "results_dir": "train/results",
        "data_exploration_dir": "data_exploration",
        "evaluation_results_dir": "evaluation_results",
        "training_logs_dir": "training_logs",
        "training_results_dir": "training_results",
        "train_dir": "Train",
        "val_dir": "Val",
        "test_dir": "Test",
        "batch_size": 8,
        "image_size": 224,
        "initial_epochs": 1,
        "fine_tune": False,
        "fine_tune_epochs": 0,
        "data_augmentation": {"randomFlip": "horizontal"},
        "model_config": {
            "model_name": "efficientnet-b0",
            "include_top": False,
            "weights": "imagenet",
            "trainable": False,
            "optimizer": "adam",
            "output_activation": "softmax",
            "learning_rate": 0.01,
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"],
            "EarlyStopping": {
                "monitor": "val_loss",
                "patience": 2,
                "restore_best_weights": True,
            },
            "ReduceLROnPlateau": {
                "monitor": "val_loss",
                "factor": 0.5,
                "patience": 1,
                "min_lr": 1e-6,
            },
        },
    }


@pytest.fixture
def valid_full_config_dict(valid_train_config_dict) -> dict:
    return {
        "train_config": valid_train_config_dict,
        "sys_config": {
            "disable_XLA_logs": True,
            "tf_fore_gpu_allow_growth": True,
        },
        "compilation_config": {
            "compiler": "gcc",
            "compiler_args": [],
            "model_name": "out.h5",
        },
    }


def write_config_yaml(path: Path, cfg: dict) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


@pytest.fixture
def project_tree_with_dataset(tmp_path: Path, valid_full_config_dict: dict):
    """Arborescence minimale valide : dataset_dir existe avec Train/Test et des images."""
    ds = tmp_path / "train" / "dataset"
    for split in ("Train", "Test"):
        d = ds / split / "c0"
        d.mkdir(parents=True)
        (d / "a.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")  # en-tête JPEG minimal
    valid_full_config_dict["train_config"]["dataset_dir"] = "train/dataset"
    write_config_yaml(tmp_path / "config.yaml", valid_full_config_dict)
    return tmp_path
