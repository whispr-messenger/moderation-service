from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import yaml

from tools import fetch_google_dataset as fgd


def test_image_extensions_constant():
    assert ".jpg" in fgd.IMAGE_EXTENSIONS


def test_load_main_config_missing_uses_defaults(tmp_path):
    root = tmp_path
    cfg = fgd._load_main_config(root)
    assert cfg["dataset_dir"] == "train/dataset"
    assert cfg["train_dir"] == "Train"


def test_load_main_config_from_yaml(tmp_path):
    data = {
        "train_config": {
            "dataset_dir": "data/x",
            "train_dir": "T",
            "test_dir": "Te",
        }
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = fgd._load_main_config(tmp_path)
    assert cfg["dataset_dir"] == "data/x"
    assert cfg["train_dir"] == "T"


def test_load_download_config_missing_raises(tmp_path):
    p = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError):
        fgd._load_download_config(p)


def test_load_download_config_ok(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text(yaml.safe_dump({"a": 1}), encoding="utf-8")
    assert fgd._load_download_config(p) == {"a": 1}


def test_get_classes_from_existing_train(tmp_path):
    td = tmp_path / "Train"
    (td / "A").mkdir(parents=True)
    (td / "B").mkdir(parents=True)
    (td / ".hidden").mkdir()
    assert fgd._get_classes_from_existing_train(td) == ["A", "B"]


def test_get_classes_missing_train(tmp_path):
    assert fgd._get_classes_from_existing_train(tmp_path / "Train") == []


def test_sanitize_filename():
    assert fgd._sanitize_filename('a<b>:"|?*c') == "a_b______c"
    assert fgd._sanitize_filename("   ") == "image"


def test_count_and_list_images(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    (d / "a.jpg").write_text("x")
    (d / "b.txt").write_text("x")
    assert fgd._count_images_in_dir(d) == 1
    assert len(fgd._list_image_files(d)) == 1


def test_count_images_not_dir(tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    assert fgd._count_images_in_dir(f) == 0
    assert fgd._list_image_files(f) == []


def test_download_image_url_success(tmp_path, monkeypatch):
    class Resp:
        content = b"x" * 600
        headers = {"Content-Type": "image/jpeg"}

        def raise_for_status(self):
            return None

    def fake_get(*_a, **_k):
        return Resp()

    monkeypatch.setitem(sys.modules, "requests", mock.Mock(get=fake_get))
    out = tmp_path / "img"
    assert fgd._download_image_url("http://x", out) is True
    assert list(tmp_path.glob("img.*"))


def test_download_image_url_too_small(tmp_path, monkeypatch):
    class Resp:
        content = b"small"
        headers = {}

        def raise_for_status(self):
            return None

    monkeypatch.setitem(sys.modules, "requests", mock.Mock(get=lambda *a, **k: Resp()))
    assert fgd._download_image_url("http://x", tmp_path / "z") is False


def test_download_image_url_exception(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise OSError("nope")

    monkeypatch.setitem(sys.modules, "requests", mock.Mock(get=boom))
    assert fgd._download_image_url("http://x", tmp_path / "z") is False


def test_get_ddgs_from_ddgs_module(monkeypatch):
    class DDGS:
        pass

    fake_ddgs = types.ModuleType("ddgs")
    fake_ddgs.DDGS = DDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_ddgs)
    monkeypatch.delitem(sys.modules, "duckduckgo_search", raising=False)
    got = fgd._get_ddgs()
    assert got is DDGS


def test_get_ddgs_none(monkeypatch):
    import builtins

    for key in ("ddgs", "duckduckgo_search"):
        monkeypatch.delitem(sys.modules, key, raising=False)

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name in ("ddgs", "duckduckgo_search"):
            raise ImportError(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert fgd._get_ddgs() is None


def test_fetch_class_duckduckgo_no_ddgs(monkeypatch, tmp_path):
    monkeypatch.setattr(fgd, "_get_ddgs", lambda: None)
    assert fgd._fetch_class_duckduckgo("kw", tmp_path, 1) == 0


def test_fetch_class_duckduckgo_empty_keywords(monkeypatch):
    monkeypatch.setattr(fgd, "_get_ddgs", lambda: object)
    assert fgd._fetch_class_duckduckgo([], Path("."), 1) == 0
    assert fgd._fetch_class_duckduckgo(["", "  "], Path("."), 1) == 0


def test_fetch_class_duckduckgo_happy_path(tmp_path, monkeypatch):
    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def images(self, kw, max_results=100, page=1):
            return [{"image": "http://a/1.jpg"}, {"url": "http://b/2.png"}]

    monkeypatch.setattr(fgd, "_get_ddgs", lambda: FakeDDGS)
    monkeypatch.setattr(fgd, "_download_image_url", lambda *a, **k: True)
    monkeypatch.setattr(fgd.time, "sleep", lambda *_a, **_k: None)
    d = tmp_path / "cls"
    d.mkdir()
    n = fgd._fetch_class_duckduckgo("cat", d, 2, delay_after_download=0, delay_between_pages=0)
    assert n == 2


def test_fetch_class_typeerror_fallback_batch(tmp_path, monkeypatch):
    """Branche TypeError lorsque ddgs.images n'accepte pas le paramètre page."""

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def images(self, kw, max_results=100, page=1):
            if page != 1:
                raise TypeError("no page")
            return [{"image": "http://x/z.jpg"}]

    monkeypatch.setattr(fgd, "_get_ddgs", lambda: FakeDDGS)
    monkeypatch.setattr(fgd, "_download_image_url", lambda *a, **k: True)
    monkeypatch.setattr(fgd.time, "sleep", lambda *_a, **_k: None)
    d = tmp_path / "c"
    d.mkdir()
    assert fgd._fetch_class_duckduckgo("k", d, 1, delay_after_download=0, delay_between_pages=0) == 1


def test_fetch_class_search_exception(tmp_path, monkeypatch, capsys):
    class Boom:
        def __enter__(self):
            raise RuntimeError("ddg down")

        def __exit__(self, *a):
            return None

    monkeypatch.setattr(fgd, "_get_ddgs", lambda: Boom)
    assert fgd._fetch_class_duckduckgo("k", tmp_path, 1) == 0


def test_balance_classes_already_balanced(tmp_path, capsys):
    td = tmp_path / "Train"
    for name in ("A", "B"):
        (td / name).mkdir(parents=True)
        for i in range(2):
            (td / name / f"{i}.jpg").write_bytes(b"\xff\xd8\xff")
    fgd._balance_classes(td, ["A", "B"], lambda x: x)
    assert fgd._count_images_in_dir(td / "A") == 2


def test_balance_classes_reduces(tmp_path, monkeypatch):
    monkeypatch.setattr(fgd.random, "shuffle", lambda x: x.sort())
    td = tmp_path / "Train"
    (td / "A").mkdir(parents=True)
    (td / "B").mkdir(parents=True)
    for i in range(3):
        (td / "A" / f"{i}.jpg").write_bytes(b"\xff\xd8\xff")
    for i in range(1):
        (td / "B" / f"{i}.jpg").write_bytes(b"\xff\xd8\xff")
    fgd._balance_classes(td, ["A", "B"], lambda x: x)
    assert fgd._count_images_in_dir(td / "A") == 1
    assert fgd._count_images_in_dir(td / "B") == 1


def test_balance_classes_min_zero_skips(tmp_path, capsys):
    td = tmp_path / "Train"
    (td / "A").mkdir(parents=True)
    (td / "A" / "0.jpg").write_bytes(b"\xff\xd8\xff")
    (td / "B").mkdir()
    fgd._balance_classes(td, ["A", "B"], lambda x: x)


def test_balance_classes_empty_names():
    fgd._balance_classes(Path("/tmp"), [], lambda x: x)  # noqa: S108


def _write_fetch_project(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "tools").mkdir(parents=True)
    cfg_main = {"train_config": {"dataset_dir": "train/dataset", "train_dir": "Train", "test_dir": "Test"}}
    (root / "config.yaml").write_text(yaml.safe_dump(cfg_main), encoding="utf-8")
    dl = {
        "search_keywords": {"Zebra": ["z"]},
        "max_num_per_class": 10,
        "search_engine": "duckduckgo",
        "delay_after_download": 0,
        "delay_between_pages": 0,
        "delay_between_categories": 0,
        "max_rounds_per_class": 1,
        "balance": False,
    }
    (root / "tools" / "dataset_download_config.yaml").write_text(yaml.safe_dump(dl), encoding="utf-8")
    train = root / "train" / "dataset" / "Train"
    train.mkdir(parents=True)


def test_run_dry_run(tmp_path, monkeypatch):
    _write_fetch_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    fgd.run(root_dir=tmp_path, dry_run=True)


def test_run_no_categories_exits(tmp_path, monkeypatch):
    root = tmp_path / "p"
    root.mkdir()
    (root / "tools").mkdir()
    (root / "config.yaml").write_text(
        yaml.safe_dump({"train_config": {"dataset_dir": "train/dataset", "train_dir": "Train", "test_dir": "Test"}}),
        encoding="utf-8",
    )
    (root / "tools" / "dataset_download_config.yaml").write_text(
        yaml.safe_dump({"search_keywords": {}, "only_classes": ["none"]}),
        encoding="utf-8",
    )
    (root / "train" / "dataset" / "Train").mkdir(parents=True)
    monkeypatch.chdir(root)
    with pytest.raises(SystemExit) as ei:
        fgd.run(root_dir=root, dry_run=False)
    assert ei.value.code == 0


def test_run_icrawler_missing_exits(tmp_path, monkeypatch):
    _write_fetch_project(tmp_path)
    dl_path = tmp_path / "tools" / "dataset_download_config.yaml"
    data = yaml.safe_load(dl_path.read_text(encoding="utf-8"))
    data["search_engine"] = "google"
    dl_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "icrawler.builtin" or (fromlist and "GoogleImageCrawler" in fromlist):
            raise ImportError("no icrawler")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit) as ei:
        fgd.run(root_dir=tmp_path, dry_run=False)
    assert ei.value.code == 1


def test_run_balance_branch(tmp_path, monkeypatch):
    _write_fetch_project(tmp_path)
    td = tmp_path / "train" / "dataset" / "Train"
    (td / "Zebra").mkdir()
    (td / "Zebra" / "a.jpg").write_bytes(b"\xff\xd8\xff")
    dl_path = tmp_path / "tools" / "dataset_download_config.yaml"
    data = yaml.safe_load(dl_path.read_text(encoding="utf-8"))
    data["balance"] = True
    dl_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(fgd, "_fetch_class_duckduckgo", lambda *a, **k: 0)
    monkeypatch.chdir(tmp_path)
    fgd.run(root_dir=tmp_path, dry_run=False)


def test_run_duckduckgo_delay_between_categories(tmp_path, monkeypatch):
    _write_fetch_project(tmp_path)
    dl_path = tmp_path / "tools" / "dataset_download_config.yaml"
    data = yaml.safe_load(dl_path.read_text(encoding="utf-8"))
    data["search_keywords"] = {"A": "a", "B": "b"}
    (tmp_path / "train" / "dataset" / "Train" / "A").mkdir(exist_ok=True)
    (tmp_path / "train" / "dataset" / "Train" / "B").mkdir(exist_ok=True)
    data["delay_between_categories"] = 2
    data["max_num_per_class"] = 0
    dl_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(fgd.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.chdir(tmp_path)
    fgd.run(root_dir=tmp_path, dry_run=False)


def test_run_min_size_and_only_classes_scalar(tmp_path, monkeypatch):
    _write_fetch_project(tmp_path)
    dl_path = tmp_path / "tools" / "dataset_download_config.yaml"
    data = yaml.safe_load(dl_path.read_text(encoding="utf-8"))
    data["min_size"] = [100, 100]
    data["only_classes"] = "Zebra"
    data["search_engine"] = "invalid_engine"
    dl_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(fgd, "_fetch_class_duckduckgo", lambda *a, **k: 0)
    monkeypatch.chdir(tmp_path)
    fgd.run(root_dir=tmp_path, dry_run=False)


def test_run_bing_engine(tmp_path, monkeypatch):
    _write_fetch_project(tmp_path)
    (tmp_path / "train" / "dataset" / "Train" / "Zebra").mkdir(parents=True)
    dl_path = tmp_path / "tools" / "dataset_download_config.yaml"
    data = yaml.safe_load(dl_path.read_text(encoding="utf-8"))
    data["search_engine"] = "bing"
    data["max_num_per_class"] = 1
    data["balance"] = False
    dl_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    class FakeBing:
        def __init__(self, *a, **k):
            self.storage = k.get("storage")

        def crawl(self, **kwargs):
            return None

    fake_builtin = mock.Mock(
        BingImageCrawler=FakeBing,
        GoogleImageCrawler=mock.Mock(),
    )
    monkeypatch.setitem(sys.modules, "icrawler.builtin", fake_builtin)
    monkeypatch.chdir(tmp_path)
    fgd.run(root_dir=tmp_path, dry_run=False)


def test_run_google_engine(tmp_path, monkeypatch):
    _write_fetch_project(tmp_path)
    (tmp_path / "train" / "dataset" / "Train" / "Zebra").mkdir(parents=True)
    dl_path = tmp_path / "tools" / "dataset_download_config.yaml"
    data = yaml.safe_load(dl_path.read_text(encoding="utf-8"))
    data["search_engine"] = "google"
    data["max_num_per_class"] = 1
    data["min_size"] = [300, 300]
    data["balance"] = False
    dl_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    class FakeGoogle:
        def __init__(self, *a, **k):
            pass

        def crawl(self, **kwargs):
            return None

    fake_builtin = mock.Mock(
        BingImageCrawler=mock.Mock(),
        GoogleImageCrawler=FakeGoogle,
    )
    monkeypatch.setitem(sys.modules, "icrawler.builtin", fake_builtin)
    monkeypatch.chdir(tmp_path)
    fgd.run(root_dir=tmp_path, dry_run=False)


def test_run_class_crawler_raises_logged(tmp_path, monkeypatch):
    _write_fetch_project(tmp_path)
    (tmp_path / "train" / "dataset" / "Train" / "Zebra").mkdir(parents=True)
    dl_path = tmp_path / "tools" / "dataset_download_config.yaml"
    data = yaml.safe_load(dl_path.read_text(encoding="utf-8"))
    data["search_engine"] = "bing"
    data["max_num_per_class"] = 1
    data["balance"] = False
    dl_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    class Boom:
        def __init__(self, *a, **k):
            pass

        def crawl(self, **kwargs):
            raise RuntimeError("crawl failed")

    fake_builtin = mock.Mock(BingImageCrawler=Boom, GoogleImageCrawler=mock.Mock())
    monkeypatch.setitem(sys.modules, "icrawler.builtin", fake_builtin)
    monkeypatch.chdir(tmp_path)
    fgd.run(root_dir=tmp_path, dry_run=False)


def test_main_argv(monkeypatch, tmp_path):
    _write_fetch_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["fetch_google_dataset", "--dry-run"])
    monkeypatch.setattr(fgd, "run", mock.Mock())
    fgd.main()
    fgd.run.assert_called_once()


def test_download_image_url_content_types(tmp_path, monkeypatch):
    def make_resp(ct, raw=None):
        payload = raw or (b"x" * 600)

        class Resp:
            headers = {"Content-Type": ct}
            content = payload

            def raise_for_status(self):
                return None

        return Resp()

    def inject(ct):
        import types

        m = types.ModuleType("requests")

        def get(*_a, **_k):
            return make_resp(ct)

        m.get = get
        monkeypatch.setitem(sys.modules, "requests", m)

    inject("image/png")
    assert fgd._download_image_url("http://x", tmp_path / "z") is True

    inject("image/webp")
    assert fgd._download_image_url("http://x", tmp_path / "w") is True

    inject("image/gif")
    assert fgd._download_image_url("http://x", tmp_path / "g") is True

    inject("application/octet-stream")
    assert fgd._download_image_url("http://u/file.xyz", tmp_path / "u") is True
