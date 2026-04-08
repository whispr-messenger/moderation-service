from __future__ import annotations

import builtins
import copy
from pathlib import Path
from unittest import mock

import pytest

import tools.hardware_test as ht


def test_ensure_dir_writable_ok(tmp_path):
    ht.ensure_dir_writable(tmp_path / "out")


def test_ensure_dir_writable_open_fails(tmp_path, monkeypatch):
    p = tmp_path / "out"
    real_open = open

    def fake_open(file, *args, **kwargs):
        s = str(file)
        if ".write_test" in s:
            raise OSError("simulated read-only")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    with pytest.raises(SystemExit):
        ht.ensure_dir_writable(p)


def test_check_dataset_dir_missing(tmp_path):
    with pytest.raises(SystemExit):
        ht.check_dataset_dir(tmp_path / "missing")


def test_check_dataset_dir_empty(tmp_path):
    d = tmp_path / "e"
    d.mkdir()
    with pytest.raises(SystemExit):
        ht.check_dataset_dir(d)


def test_check_dataset_dir_ok(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "f.txt").write_text("x")
    assert ht.check_dataset_dir(d) == d


def test_check_dataset_dir_with_sub(tmp_path):
    root = tmp_path / "r"
    sub = root / "s"
    sub.mkdir(parents=True)
    (sub / "a").write_text("1")
    assert ht.check_dataset_dir(root, "s") == sub


def test_count_images_in_folder_ok(tmp_path):
    d = tmp_path / "i"
    d.mkdir()
    (d / "a.jpg").write_text("x")
    assert ht.count_images_in_folder(d) == 1


def test_count_images_recursive(tmp_path):
    d = tmp_path / "i"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "b.png").write_text("x")
    assert ht.count_images_in_folder(d, recursive=True) == 1


def test_count_images_missing(tmp_path):
    with pytest.raises(SystemExit):
        ht.count_images_in_folder(tmp_path / "nope")


def test_count_images_not_dir(tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    with pytest.raises(SystemExit):
        ht.count_images_in_folder(f)


def test_count_images_none(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(SystemExit):
        ht.count_images_in_folder(d)


def test_check_requirement_file_missing(tmp_path, capsys):
    ht.check_requirement(str(tmp_path / "req.txt"))
    assert "not found" in capsys.readouterr().out.lower()


def test_check_requirement_ok(tmp_path):
    p = tmp_path / "req.txt"
    p.write_text("# c\n", encoding="utf-8")
    ht.check_requirement(str(p))


def test_check_requirement_missing_pkg(tmp_path, monkeypatch):
    p = tmp_path / "req.txt"
    p.write_text("nonexistent-pkg-zzz==999\n", encoding="utf-8")
    monkeypatch.setattr(ht.pkg_resources, "require", mock.Mock(side_effect=ht.pkg_resources.DistributionNotFound))
    with pytest.raises(SystemExit):
        ht.check_requirement(str(p))


def test_check_requirement_version_conflict(tmp_path, monkeypatch, capsys):
    p = tmp_path / "req.txt"
    p.write_text("numpy\n", encoding="utf-8")

    def conflict(req):
        raise ht.pkg_resources.VersionConflict(None, None)

    monkeypatch.setattr(ht.pkg_resources, "require", conflict)
    ht.check_requirement(str(p))
    out = capsys.readouterr().out.lower()
    assert "mismatch" in out or "version" in out


def test_check_gpus_cpu(monkeypatch, capsys):
    monkeypatch.setattr(ht.tf.config, "list_physical_devices", lambda *_a, **_k: [])
    monkeypatch.setattr(ht, "check_nvidia_driver_and_cuda", lambda: None)
    monkeypatch.setattr(ht, "check_nvcc", lambda: None)
    monkeypatch.setattr(ht, "check_tf_cuda_build_info", lambda: None)
    assert ht.check_gpus() == "cpu"


def test_check_gpus_gpu(monkeypatch, capsys):
    dev = mock.Mock()

    monkeypatch.setattr(ht.tf.config, "list_physical_devices", lambda *_a, **_k: [dev])
    monkeypatch.setattr(ht, "check_nvidia_driver_and_cuda", lambda: None)
    monkeypatch.setattr(ht, "check_nvcc", lambda: None)
    monkeypatch.setattr(ht, "check_tf_cuda_build_info", lambda: None)
    assert ht.check_gpus() == "gpu"


def test_scan_bad_images(tmp_path, monkeypatch):
    d = tmp_path / "scan"
    d.mkdir()
    (d / "ok.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00")
    (d / "bad.xyz").write_text("nope")
    monkeypatch.setattr(ht.tf.io, "read_file", lambda p: b"fake")
    monkeypatch.setattr(ht.tf.image, "decode_image", lambda b: object())
    bad = ht._scan_bad_images(d)
    assert isinstance(bad, list)
    assert any(p.name == "bad.xyz" for p in bad)


def test_check_python_version_ok(capsys):
    ht.check_python_version()
    assert "Python" in capsys.readouterr().out


def test_check_python_version_fail(monkeypatch):
    # Comme le code source : comparaison lexicographique des versions ; 3.0.x < "3.10"
    monkeypatch.setattr(ht.sys, "version", "3.0.0 (fake)")
    with pytest.raises(SystemExit):
        ht.check_python_version()


def test_scan_bad_images_decode_error(tmp_path, monkeypatch):
    d = tmp_path / "scan2"
    d.mkdir()
    (d / "x.jpg").write_bytes(b"\xff\xd8\xff")
    monkeypatch.setattr(ht.tf.io, "read_file", lambda p: b"data")
    monkeypatch.setattr(ht.tf.image, "decode_image", mock.Mock(side_effect=ValueError("bad")))
    bad = ht._scan_bad_images(d)
    assert any(p.name == "x.jpg" for p in bad)


def test_check_train_dataset_dir_patched(tmp_path, monkeypatch, valid_full_config_dict):
    monkeypatch.chdir(tmp_path)
    cfg = copy.deepcopy(valid_full_config_dict)
    ds = tmp_path / "train" / "dataset"
    (ds / "Train" / "c").mkdir(parents=True)
    (ds / "Test" / "c").mkdir(parents=True)
    (ds / "Train" / "c" / "i.jpg").write_bytes(b"\xff\xd8\xff")
    (ds / "Test" / "c" / "j.jpg").write_bytes(b"\xff\xd8\xff")
    cfg["train_config"]["dataset_dir"] = "train/dataset"
    monkeypatch.setattr(ht, "ensure_dir_writable", lambda p: Path(p).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(ht, "check_dataset_dir", lambda *a, **k: Path(a[0]))
    monkeypatch.setattr(ht, "count_images_in_folder", lambda *a, **k: 2)
    monkeypatch.setattr(ht, "_scan_bad_images", lambda p: [])
    ht.check_train_dataset_dir(cfg)


def test_check_train_results_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {
        "train_config": {
            "results_dir": "res",
            "data_exploration_dir": "de",
            "evaluation_results_dir": "ev",
            "training_logs_dir": "tl",
            "training_results_dir": "tr",
        }
    }
    ht.check_train_results_dir(cfg)
    assert (tmp_path / "res" / "de").is_dir()
