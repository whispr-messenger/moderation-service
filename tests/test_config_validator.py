from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

import tools.configuration_generator as cg
import tools.config_validator as cv


def _write_cfg(tmp: Path, data: dict) -> None:
    (tmp / "config.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_validate_config_passes(project_tree_with_dataset, monkeypatch, valid_full_config_dict):
    root = project_tree_with_dataset
    monkeypatch.chdir(root)
    monkeypatch.setattr(cg, "CONFIG_PATH", root / "config.yaml")
    cfg = cv.validate_config()
    assert cfg["train_config"]["batch_size"] == 8


@pytest.mark.parametrize(
    "mutator,expect_substr",
    [
        (lambda c: c.pop("train_config"), "train_config"),
        (lambda c: c["train_config"].pop("batch_size"), "batch_size"),
        (lambda c: c["train_config"].update({"batch_size": "x"}), "int"),
        (lambda c: c["train_config"].update({"batch_size": 0}), ">="),
        (lambda c: c["train_config"].update({"image_size": 0}), "image_size"),
        (lambda c: c["train_config"].update({"image_size": [1]}), "image_size"),
        (lambda c: c["train_config"].update({"image_size": "nope"}), "image_size"),
        (lambda c: c["train_config"].update({"mixed_precision": "yes"}), "bool"),
        (lambda c: c["train_config"]["model_config"].update({"metrics": []}), "metrics"),
        (lambda c: c["train_config"]["model_config"].update({"learning_rate": 0}), "learning_rate"),
        (lambda c: c["compilation_config"].update({"compiler_args": {}}), "compiler_args"),
        (lambda c: c.pop("sys_config"), "sys_config"),
        (lambda c: c.pop("compilation_config"), "compilation_config"),
        (lambda c: c["train_config"]["model_config"].update({"optimizer": ""}), "optimizer"),
        (lambda c: c["train_config"]["model_config"].update({"metrics": "x"}), "metrics"),
        (lambda c: c["train_config"].update({"dataset_dir": 123}), "dataset_dir"),
    ],
)
def test_validate_mutations_fail(
    project_tree_with_dataset, monkeypatch, valid_full_config_dict, mutator, expect_substr, capsys
):
    root = project_tree_with_dataset
    cfg = copy.deepcopy(valid_full_config_dict)
    mutator(cfg)
    _write_cfg(root, cfg)
    monkeypatch.chdir(root)
    monkeypatch.setattr(cg, "CONFIG_PATH", root / "config.yaml")
    with pytest.raises(SystemExit):
        cv.validate_config()
    err = capsys.readouterr().out
    assert expect_substr in err


def test_validate_file_not_found(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cg, "CONFIG_PATH", tmp_path / "nope.yaml")
    with pytest.raises(SystemExit):
        cv.validate_config()


def test_validate_load_exception(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "config.yaml"
    p.write_text("::: not yaml", encoding="utf-8")
    monkeypatch.setattr(cg, "CONFIG_PATH", p)
    with pytest.raises(SystemExit):
        cv.validate_config()


def test_validate_dataset_dir_missing(tmp_path, monkeypatch, valid_full_config_dict):
    monkeypatch.chdir(tmp_path)
    cfg = copy.deepcopy(valid_full_config_dict)
    cfg["train_config"]["dataset_dir"] = "missing"
    _write_cfg(tmp_path, cfg)
    monkeypatch.setattr(cg, "CONFIG_PATH", tmp_path / "config.yaml")
    with pytest.raises(SystemExit):
        cv.validate_config()


def test_validate_dataset_not_dir(tmp_path, monkeypatch, valid_full_config_dict):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "notdir"
    f.write_text("x", encoding="utf-8")
    cfg = copy.deepcopy(valid_full_config_dict)
    cfg["train_config"]["dataset_dir"] = "notdir"
    _write_cfg(tmp_path, cfg)
    monkeypatch.setattr(cg, "CONFIG_PATH", tmp_path / "config.yaml")
    with pytest.raises(SystemExit):
        cv.validate_config()


def test_validate_image_size_hw_list(project_tree_with_dataset, monkeypatch, valid_full_config_dict):
    root = project_tree_with_dataset
    cfg = copy.deepcopy(valid_full_config_dict)
    cfg["train_config"]["image_size"] = [224, 224]
    _write_cfg(root, cfg)
    monkeypatch.chdir(root)
    monkeypatch.setattr(cg, "CONFIG_PATH", root / "config.yaml")
    cv.validate_config()


def test_validate_train_cfg_not_dict(tmp_path, monkeypatch, valid_full_config_dict):
    cfg = copy.deepcopy(valid_full_config_dict)
    cfg["train_config"] = []
    _write_cfg(tmp_path, cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cg, "CONFIG_PATH", tmp_path / "config.yaml")
    with pytest.raises(SystemExit):
        cv.validate_config()


def test_validate_model_config_not_dict(project_tree_with_dataset, monkeypatch, valid_full_config_dict):
    root = project_tree_with_dataset
    cfg = copy.deepcopy(valid_full_config_dict)
    cfg["train_config"]["model_config"] = []
    _write_cfg(root, cfg)
    monkeypatch.chdir(root)
    monkeypatch.setattr(cg, "CONFIG_PATH", root / "config.yaml")
    with pytest.raises(SystemExit):
        cv.validate_config()


def test_validate_data_augmentation_not_dict(project_tree_with_dataset, monkeypatch, valid_full_config_dict):
    root = project_tree_with_dataset
    cfg = copy.deepcopy(valid_full_config_dict)
    cfg["train_config"]["data_augmentation"] = []
    _write_cfg(root, cfg)
    monkeypatch.chdir(root)
    monkeypatch.setattr(cg, "CONFIG_PATH", root / "config.yaml")
    with pytest.raises(SystemExit):
        cv.validate_config()


def test_validate_es_rlr_not_dict(project_tree_with_dataset, monkeypatch, valid_full_config_dict):
    root = project_tree_with_dataset
    cfg = copy.deepcopy(valid_full_config_dict)
    cfg["train_config"]["model_config"]["EarlyStopping"] = []
    _write_cfg(root, cfg)
    monkeypatch.chdir(root)
    monkeypatch.setattr(cg, "CONFIG_PATH", root / "config.yaml")
    with pytest.raises(SystemExit):
        cv.validate_config()


def test_warning_training_device_and_slash_paths(project_tree_with_dataset, monkeypatch, valid_full_config_dict, capsys):
    root = project_tree_with_dataset
    cfg = copy.deepcopy(valid_full_config_dict)
    cfg["train_config"]["training_divice"] = "tpu"
    cfg["train_config"]["train_dir"] = "/abs"
    _write_cfg(root, cfg)
    monkeypatch.chdir(root)
    monkeypatch.setattr(cg, "CONFIG_PATH", root / "config.yaml")
    cv.validate_config()
    out = capsys.readouterr().out
    assert "training_divice" in out or "⚠️" in out
