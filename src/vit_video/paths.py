"""Default dataset locations under the ``vit_video`` package directory."""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PACKAGE_ROOT / "food_data"
DEFAULT_FRAMES_DIR = DEFAULT_DATASET_DIR / "frames"
