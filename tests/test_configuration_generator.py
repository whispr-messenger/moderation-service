from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml

import tools.configuration_generator as cg


def test_load_config_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cg, "CONFIG_PATH", tmp_path / "missing.yaml")
    with pytest.raises(FileNotFoundError):
        cg.load_config()


def test_config_writes_and_loads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "config.yaml"
    monkeypatch.setattr(cg, "CONFIG_PATH", path)
    with mock.patch("tools.configuration_generator.check_gpus", return_value="cpu"):
        cfg = cg.config("efficientnet-b0")
    assert path.exists()
    assert cfg["train_config"]["model_config"]["model_name"] == "efficientnet-b0"
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["train_config"]["training_divice"] == "cpu"


def test_config_write_failure_exits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_path = mock.MagicMock()
    mock_path.write_text.side_effect = OSError("disk full")
    monkeypatch.setattr(cg, "CONFIG_PATH", mock_path)
    with mock.patch("tools.configuration_generator.check_gpus", return_value="cpu"):
        with pytest.raises(SystemExit):
            cg.config("efficientnet-b0")
