"""Shared logging setup for moderation-service models.

Usage:
    from common.logging_config import setup_logging
    setup_logging()  # call once at entrypoint

Then in any module:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("...")
"""

from __future__ import annotations

import logging
import os


def setup_logging(level: int | str | None = None) -> None:
    """Configure root logging once. Idempotent.

    Level precedence: explicit `level` arg > `MODERATION_LOG_LEVEL` env var > INFO.
    Suppresses chatty third-party loggers (tensorflow, absl, matplotlib).
    """
    if level is None:
        level = os.environ.get("MODERATION_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        root.setLevel(level)

    # Quiet noisy third-party loggers; callers can re-enable if needed.
    for noisy in ("tensorflow", "absl", "matplotlib", "PIL", "h5py", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
