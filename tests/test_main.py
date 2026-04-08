from __future__ import annotations

from unittest import mock

import pytest

import main as cli


def test_parse_args_requires_action(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py"])
    with pytest.raises(SystemExit):
        cli.parse_args()


def test_parse_args_ok(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--action", "train", "--model", "efficientnet-b0"])
    args = cli.parse_args()
    assert args.action == "train"
    assert args.model == "efficientnet-b0"


def test_init_unknown_action(monkeypatch):
    monkeypatch.setattr(cli, "config", mock.Mock())
    monkeypatch.setattr(cli, "validate_config", mock.Mock(return_value={}))
    with pytest.raises(SystemExit):
        cli.init("unknown", "efficientnet-b0")


def test_init_train_dispatches(project_tree_with_dataset, monkeypatch, valid_full_config_dict):
    root = project_tree_with_dataset
    monkeypatch.chdir(root)
    import tools.configuration_generator as cg

    monkeypatch.setattr(cg, "CONFIG_PATH", root / "config.yaml")
    stub = mock.Mock()
    monkeypatch.setitem(cli.ACTIONS, "train", stub)
    monkeypatch.setattr(cli, "config", mock.Mock())
    monkeypatch.setattr(cli, "validate_config", mock.Mock(return_value=valid_full_config_dict))
    cli.init("train", "efficientnet-b0")
    stub.assert_called_once_with(valid_full_config_dict)
