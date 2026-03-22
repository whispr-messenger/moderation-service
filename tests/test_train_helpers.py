from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

import train.train as tr


def test_get_img_size_int():
    assert tr._get_img_size({"image_size": 224}) == (224, 224)


def test_get_img_size_hw():
    assert tr._get_img_size({"image_size": [128, 256]}) == (128, 256)


def test_get_img_size_invalid():
    with pytest.raises(ValueError):
        tr._get_img_size({"image_size": {}})


def test_get_efficientnet_class_variants():
    assert tr._get_efficientnet_class("EfficientNet-B0") is not None
    assert tr._get_efficientnet_class("efficientnet_b1") is not None


def test_get_efficientnet_class_unsupported():
    with pytest.raises(ValueError):
        tr._get_efficientnet_class("efficientnet-z9")


def test_to_json_safe():
    assert tr._to_json_safe({"a": np.float32(1.5)}) == {"a": 1.5}
    assert tr._to_json_safe([np.int64(3)]) == [3]
    assert tr._to_json_safe("x") == "x"


def test_build_paths(tmp_path, monkeypatch, valid_train_config_dict):
    monkeypatch.chdir(tmp_path)
    cfg = {**valid_train_config_dict, "dataset_dir": "train/dataset", "results_dir": "train/results"}
    paths = tr._build_paths(cfg)
    assert paths["train_dir"] == tmp_path / "train" / "dataset" / "Train"
    assert paths["data_exploration_dir"].is_dir()


def test_apply_sys_config_mixed_precision(monkeypatch, valid_train_config_dict):
    sc = {"disable_XLA_logs": True, "tf_fore_gpu_allow_growth": True}
    tc = {**valid_train_config_dict, "mixed_precision": True}
    mp_mod = mock.Mock()
    mp_mod.set_global_policy = mock.Mock()
    monkeypatch.setattr(tr.tf.keras, "mixed_precision", mp_mod)
    monkeypatch.setattr(tr.tf.config.optimizer, "set_jit", mock.Mock())
    tr._apply_sys_config(sc, tc)
    mp_mod.set_global_policy.assert_called_once_with("mixed_float16")
