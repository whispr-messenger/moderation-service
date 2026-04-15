#!/usr/bin/env python3
"""
Download an image dataset from web search engines into a Keras-style Train/ClassName/ layout.

This is the canonical implementation, shared across all model directories.
Each model exposes a thin shim at `tools/fetch_google_dataset.py` that calls `run()`
with its own project root so existing CLI invocations keep working.

Usage from a model directory:
  python -m tools.fetch_google_dataset [--config tools/dataset_download_config.yaml]

Dependencies: pip install icrawler pyyaml ddgs requests
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path

import yaml

# Image extensions recognized for counting and balancing
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

# No module-level ROOT_DIR: the caller (shim or CLI) passes project root via `run(root_dir=...)`.
# When run directly as `python -m common.fetch_google_dataset`, --root must be supplied.


def _load_main_config(root: Path) -> dict:
    """Load config.yaml to resolve dataset_dir, train_dir, test_dir."""
    cfg_path = root / "config.yaml"
    if not cfg_path.exists():
        return {
            "dataset_dir": "train/dataset",
            "train_dir": "Train",
            "test_dir": "Test",
        }
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    tc = data.get("train_config", {})
    return {
        "dataset_dir": tc.get("dataset_dir", "train/dataset"),
        "train_dir": tc.get("train_dir", "Train"),
        "test_dir": tc.get("test_dir", "Test"),
    }


def _load_download_config(config_path: Path) -> dict:
    """Load the download config (categories and keywords)."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_classes_from_existing_train(train_dir: Path) -> list[str]:
    """Read category names from Train/ subdirectories."""
    if not train_dir.exists():
        return []
    classes = []
    for p in train_dir.iterdir():
        if p.is_dir() and not p.name.startswith("."):
            classes.append(p.name)
    return sorted(classes)


def _sanitize_filename(name: str) -> str:
    """Strip characters that are illegal in filenames."""
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip() or "image"


def _log(msg: str) -> None:
    """Print a message and flush stdout for live output in long-running jobs."""
    print(msg, flush=True)


def _count_images_in_dir(dir_path: Path) -> int:
    """Count image files in a directory."""
    if not dir_path.is_dir():
        return 0
    return sum(
        1
        for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def _list_image_files(dir_path: Path) -> list[Path]:
    """Return the list of image files in a directory."""
    if not dir_path.is_dir():
        return []
    return [
        f
        for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]


# Browser-style headers to reduce 403s
_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Referer": "https://duckduckgo.com/",
}


def _download_image_url(url: str, filepath: Path, timeout: int = 12) -> bool:
    """Download an image from a URL with browser headers. Returns True on success."""
    try:
        import requests
        r = requests.get(url, headers=_DOWNLOAD_HEADERS, timeout=timeout, stream=True)
        r.raise_for_status()
        content = r.content
        if len(content) < 500:
            return False
        # Derive extension from Content-Type to keep filenames consistent.
        ct = (r.headers.get("Content-Type") or "").lower()
        if "jpeg" in ct or "jpg" in ct:
            suf = ".jpg"
        elif "png" in ct:
            suf = ".png"
        elif "webp" in ct:
            suf = ".webp"
        elif "gif" in ct:
            suf = ".gif"
        else:
            suf = Path(url).suffix or ".jpg"
            if suf.lower() not in IMAGE_EXTENSIONS:
                suf = ".jpg"
        out = filepath.with_suffix(suf) if filepath.suffix != suf else filepath
        out.write_bytes(content)
        return True
    except Exception:
        return False


def _get_ddgs():
    """Import DDGS (new `ddgs` package or legacy `duckduckgo_search`)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*renamed to.*ddgs.*", category=RuntimeWarning)
        try:
            from ddgs import DDGS
            return DDGS
        except ImportError:
            pass
        try:
            from duckduckgo_search import DDGS
            return DDGS
        except ImportError:
            pass
    return None


def _fetch_class_duckduckgo(
    keywords: str | list[str],
    class_dir: Path,
    max_num: int,
    delay_after_download: float = 0.5,
    delay_between_pages: float = 2.0,
    start_index: int = 0,
) -> int:
    """
    Fetch images via DuckDuckGo (multiple keywords, multiple pages) into class_dir.

    keywords: one keyword (str) or a list of keywords to diversify URLs.
    start_index: starting index for filenames (prevents overwriting existing images).
    Returns the number of images saved.
    """
    DDGS = _get_ddgs()
    if DDGS is None:
        _log("  Install ddgs: pip install ddgs  (or duckduckgo-search)")
        return 0
    if isinstance(keywords, str):
        keywords = [keywords]
    keywords = [k.strip() for k in keywords if k and isinstance(k, str)]
    if not keywords:
        return 0
    # Collect more URLs than needed to compensate for invalid links.
    urls = []
    seen = set()
    page_size = 100
    target_urls = min(max_num * 4, 5000)
    try:
        with DDGS() as ddgs:
            for ki, kw in enumerate(keywords):
                if len(urls) >= target_urls:
                    break
                _log(f"  [Search {ki+1}/{len(keywords)}] \"{kw}\"...")
                page = 1
                while len(urls) < target_urls:
                    try:
                        kwargs = {"max_results": page_size}
                        if page > 1:
                            kwargs["page"] = page
                        batch = list(ddgs.images(kw, **kwargs))
                    except TypeError:
                        batch = list(ddgs.images(kw, max_results=page_size))
                    if not batch:
                        break
                    for r in batch:
                        u = r.get("image") or r.get("url")
                        if u and isinstance(u, str) and u.startswith("http") and u not in seen:
                            seen.add(u)
                            urls.append(u)
                    _log(f"  -> {len(urls)} URLs (page {page})")
                    if len(batch) < page_size:
                        break
                    page += 1
                    if len(urls) >= target_urls:
                        break
                    if delay_between_pages > 0:
                        time.sleep(delay_between_pages)
                if len(keywords) > 1 and delay_between_pages > 0:
                    time.sleep(delay_between_pages)
    except Exception as e:
        _log(f"  DuckDuckGo search error: {e}")
    if not urls:
        return 0
    _log(f"  Downloading: {len(urls)} URLs -> target {max_num} images...")
    saved = 0
    total_urls = len(urls)
    # Progressive download with periodic logs to track progress.
    for i, url in enumerate(urls):
        if saved >= max_num:
            break
        if (i + 1) % 100 == 0 or i == 0:
            _log(f"  -> URL {i+1}/{total_urls} | saved: {saved}/{max_num}")
        name = f"{start_index + saved + 1:04d}"
        filepath = class_dir / name
        if _download_image_url(url, filepath):
            saved += 1
            if saved % 20 == 0 or saved == max_num:
                _log(f"  OK {saved}/{max_num} images saved")
        if delay_after_download > 0:
            time.sleep(delay_after_download)
    return saved


def _balance_classes(train_dir: Path, class_names: list[str], sanitize_fn) -> None:
    """
    Balance image counts per category: deletes random images from over-filled
    categories so every class ends up with the same count (the minimum).
    No explicit cap -- we keep as many images as possible while equalizing.
    """
    counts = {}
    for name in class_names:
        class_dir = train_dir / sanitize_fn(name)
        counts[name] = _count_images_in_dir(class_dir)

    if not counts:
        return

    _log("--- Before balancing ---")
    for name in class_names:
        _log(f"  {name}: {counts[name]} images")
    min_count = min(counts.values())
    max_count = max(counts.values())
    if min_count == max_count and min_count > 0:
        _log("Already balanced: all categories have the same image count.")
        return
    if min_count <= 0:
        _log("Balancing skipped: at least one category is empty. Fill empty categories and rerun.")
        return

    _log(f"--- Balancing: trimming each category to {min_count} images ---")
    for name in class_names:
        class_dir = train_dir / sanitize_fn(name)
        n = counts[name]
        if n <= min_count:
            _log(f"  {name}: {n} images (unchanged)")
            continue
        to_remove = n - min_count
        files = _list_image_files(class_dir)
        random.shuffle(files)
        for f in files[:to_remove]:
            try:
                f.unlink()
            except OSError:
                pass
        _log(f"  {name}: {n} -> {min_count} images (removed {to_remove})")

    _log(f"Balancing done: every category now has {min_count} images.")


def run(
    root_dir: Path | None = None,
    download_config_path: Path | None = None,
    dry_run: bool = False,
) -> None:
    if root_dir is None:
        raise ValueError("root_dir is required (pass the model directory root, e.g. Path(__file__).resolve().parent.parent)")
    root_dir = Path(root_dir).resolve()
    os.chdir(root_dir)

    _log(f"Project root: {root_dir}")

    main_cfg = _load_main_config(root_dir)
    dataset_root = root_dir / main_cfg["dataset_dir"]
    train_dir = dataset_root / main_cfg["train_dir"]

    # Download config path (defaults to tools/dataset_download_config.yaml)
    if download_config_path is None:
        download_config_path = root_dir / "tools" / "dataset_download_config.yaml"
    else:
        download_config_path = Path(download_config_path).resolve()
    _log(f"Download config: {download_config_path}")
    download_cfg = _load_download_config(download_config_path)

    search_keywords = download_cfg.get("search_keywords") or {}
    max_num_per_class = int(download_cfg.get("max_num_per_class", 1000))
    min_size = download_cfg.get("min_size")
    if isinstance(min_size, list) and len(min_size) >= 2:
        min_size = tuple(int(x) for x in min_size[:2])
    else:
        min_size = (200, 200)
    only_classes = download_cfg.get("only_classes")
    if only_classes is not None and not isinstance(only_classes, list):
        only_classes = [only_classes]

    # Resolve categories: union of existing Train/ folders and search_keywords entries
    existing_train = _get_classes_from_existing_train(train_dir)
    all_class_names = sorted(set(existing_train) | set(search_keywords.keys()))
    for c in all_class_names:
        if c not in search_keywords:
            search_keywords[c] = c
    classes_to_run = all_class_names
    if only_classes:
        classes_to_run = [c for c in classes_to_run if c in only_classes]

    if not classes_to_run:
        _log("No categories to process. Check only_classes or add subfolders under Train/.")
        sys.exit(0)
    _log(f"Categories detected (Train + config): {len(classes_to_run)}")

    search_engine = (download_cfg.get("search_engine") or "duckduckgo").strip().lower()
    if search_engine not in ("duckduckgo", "bing", "google"):
        search_engine = "duckduckgo"
    _log(f"Search engine: {search_engine}")
    _log(f"Categories to process: {len(classes_to_run)} (max {max_num_per_class} images/category)")
    if dry_run:
        _log("--- Dry-run mode (no downloads) ---")

    if search_engine in ("bing", "google"):
        try:
            from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
        except ImportError:
            _log("Install icrawler: pip install icrawler")
            sys.exit(1)
        logging.getLogger("icrawler.downloader").setLevel(logging.CRITICAL)
        _log("(Some images may be skipped if the source site refuses the download. This is expected.)")
    else:
        _log("(DuckDuckGo + browser headers to limit 403s.)")

    train_dir.mkdir(parents=True, exist_ok=True)

    delay_between_categories = max(0, float(download_cfg.get("delay_between_categories", 0)))
    delay_after_download = max(0.0, float(download_cfg.get("delay_after_download", 0.5)))
    delay_between_pages = max(0.0, float(download_cfg.get("delay_between_pages", 2.0)))
    if search_engine == "duckduckgo" and delay_between_categories > 0:
        _log(f"(Delay between categories: {delay_between_categories} s to avoid rate-limiting.)")

    for idx, class_name in enumerate(classes_to_run):
        keywords_raw = search_keywords[class_name]
        if isinstance(keywords_raw, list):
            keywords_list = [str(k).strip() for k in keywords_raw if k]
            keyword_display = " | ".join(keywords_list[:5])
        else:
            keywords_list = [str(keywords_raw).strip() or class_name]
            keyword_display = keywords_list[0]

        class_dir = train_dir / _sanitize_filename(class_name)
        class_dir.mkdir(parents=True, exist_ok=True)

        if dry_run:
            _log(f"  [dry-run] {class_name} -> \"{keyword_display}\" -> {class_dir}")
            continue

        if search_engine == "duckduckgo" and idx > 0 and delay_between_categories > 0:
            d = int(delay_between_categories)
            _log(f"  Pause {d} s before category {idx+1}/{len(classes_to_run)}...")
            for _ in range(d):
                time.sleep(1)
                if (_ + 1) % 5 == 0:
                    _log(f"  ... {d - (_ + 1)} s remaining")

        current_count = _count_images_in_dir(class_dir)
        need_to_fetch = max(0, max_num_per_class - current_count)
        if need_to_fetch == 0:
            _log(f"{class_name}: already has {current_count} images (>= {max_num_per_class}), skipping.")
            continue

        max_rounds = max(1, int(download_cfg.get("max_rounds_per_class", 5)))
        _log(f"  [{idx+1}/{len(classes_to_run)}] {class_name} (current: {current_count}, target {max_num_per_class}, max {max_rounds} rounds)")
        try:
            if search_engine == "duckduckgo":
                round_num = 0
                while round_num < max_rounds:
                    current_count = _count_images_in_dir(class_dir)
                    need_to_fetch = max(0, max_num_per_class - current_count)
                    if need_to_fetch == 0:
                        _log(f"  OK {class_name}: target {max_num_per_class} reached.")
                        break
                    round_num += 1
                    _log(f"  --- Round {round_num}/{max_rounds}: {current_count} images, {need_to_fetch} missing ---")
                    n = _fetch_class_duckduckgo(
                        keywords_list,
                        class_dir,
                        need_to_fetch,
                        delay_after_download,
                        delay_between_pages,
                        start_index=current_count,
                    )
                    new_total = current_count + n
                    _log(f"  Saved: +{n} (total ~{new_total})")
                    if n == 0:
                        _log(f"  No new images this round, moving to next category.")
                        break
                    if new_total >= max_num_per_class:
                        break
                    if round_num < max_rounds:
                        _log(f"  Pause 20 s before round {round_num+1}...")
                        time.sleep(20)
            elif search_engine == "bing":
                crawler = BingImageCrawler(
                    storage={"root_dir": str(class_dir)},
                    log_level=None,
                )
                filters = {"size": "large"} if min_size and min_size[0] >= 200 else None
                crawler.crawl(
                    keyword=keyword_display,
                    max_num=need_to_fetch,
                    filters=filters,
                )
            else:
                crawler = GoogleImageCrawler(
                    storage={"root_dir": str(class_dir)},
                    log_level=None,
                )
                crawler.crawl(
                    keyword=keyword_display,
                    max_num=need_to_fetch,
                    min_size=min_size,
                )
        except Exception as e:
            _log(f"  Warning: error for category {class_name}: {e}")

    # Balancing: equalize every category to the minimum count
    if not dry_run:
        if download_cfg.get("balance", True):
            _log("")
            _log("Balancing categories (all Train/ subfolders to the same image count)...")
            all_train_classes = _get_classes_from_existing_train(train_dir)
            _balance_classes(train_dir, all_train_classes, _sanitize_filename)
        else:
            _log("Balancing disabled (balance: false). Current counts:")
            all_train_classes = _get_classes_from_existing_train(train_dir)
            for name in all_train_classes:
                class_dir = train_dir / _sanitize_filename(name)
                n = _count_images_in_dir(class_dir)
                _log(f"  {name}: {n} images")
        _log("")
        _log("All downloads launched.")


def main() -> None:
    print("fetch_google_dataset: starting...", flush=True)
    parser = argparse.ArgumentParser(description="Fetch web-search images into Train/<class>/ folders")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (default: tools/dataset_download_config.yaml)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Project root (the model directory, e.g. src/mobilenet_v3_small)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print categories and keywords, no downloading",
    )
    args = parser.parse_args()
    run(
        root_dir=args.root,
        download_config_path=args.config,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
