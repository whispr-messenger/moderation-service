#!/usr/bin/env python3
"""Thin shim that delegates to the shared scraper in `src/common/fetch_google_dataset.py`.

Keeps the historical invocation path working:
    python -m tools.fetch_google_dataset [--config tools/dataset_download_config.yaml] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src/ to sys.path so `common.fetch_google_dataset` is importable.
_MODEL_ROOT = Path(__file__).resolve().parent.parent  # src/mobilenet_v3_small
sys.path.insert(0, str(_MODEL_ROOT.parent))  # src/

from common.fetch_google_dataset import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch web-search images into Train/<class>/ folders (mobilenet_v3_small)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (default: tools/dataset_download_config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print categories and keywords, no downloading",
    )
    args = parser.parse_args()
    run(
        root_dir=_MODEL_ROOT,
        download_config_path=args.config,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
