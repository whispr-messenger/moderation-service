from __future__ import annotations

import contextlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

_js_runtime_notice_shown = False


class _YtdlpLogger:
    __slots__ = ()

    def debug(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        global _js_runtime_notice_shown
        low = msg.lower()
        if "javascript runtime" in low or "js-runtimes" in low:
            if not _js_runtime_notice_shown:
                sys.stderr.write(
                    "[INFO] yt-dlp: no Node/Deno/Bun found — YouTube may warn; "
                    "install Node.js: https://github.com/yt-dlp/yt-dlp/wiki/EJS\n"
                )
                _js_runtime_notice_shown = True
            return
        # Suppress PO Token warnings (android/ios clients not used)
        if "po token" in low or "po_token" in low:
            return
        sys.stderr.write(f"WARNING: {msg}\n")

    def error(self, msg: str) -> None:
        sys.stderr.write(f"ERROR: {msg}\n")


def _resolve_node_exe() -> str | None:
    w = shutil.which("node")
    if w:
        return w
    if sys.platform != "win32":
        return None
    bases = [
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        os.path.expanduser(r"~\AppData\Local\Programs\node"),
    ]
    for base in bases:
        if not base:
            continue
        p = Path(base) / "nodejs" / "node.exe"
        if p.is_file():
            return str(p)
    return None


def match_filter_max_duration_seconds(max_seconds: float) -> Callable[..., str | None]:
    """Callable ``match_filter`` for yt-dlp: reject entries with ``duration >= max_seconds``."""

    def _match(info: dict, *, incomplete: bool = False) -> str | None:
        if incomplete:
            return None
        dur = info.get("duration")
        if dur is None:
            return None
        if float(dur) >= max_seconds:
            return f"duration {dur}s >= {max_seconds}s"
        return None

    return _match


@contextlib.contextmanager
def ytdlp_download_stderr_filtered() -> Iterator[None]:
    """Drop repetitive YouTube JS-runtime stderr spam (yt-dlp prints it outside the logger)."""
    real = sys.stderr
    shown = False

    class _F:
        def write(self, s: str) -> int:
            nonlocal shown
            if not s:
                return 0
            low = s.lower()
            if "javascript runtime" in low or "js-runtimes" in low:
                if not shown:
                    real.write(
                        "[INFO] yt-dlp: install Node.js to reduce YouTube noise — "
                        "https://github.com/yt-dlp/yt-dlp/wiki/EJS\n"
                    )
                    shown = True
                return len(s)
            real.write(s)
            return len(s)

        def flush(self) -> None:
            real.flush()

        def isatty(self) -> bool:
            return getattr(real, "isatty", lambda: False)()

    wrapper = _F()
    prev = sys.stderr
    sys.stderr = wrapper  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stderr = prev


def youtube_dl_base_options(*, ffmpeg_location: str | None = None) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "noplaylist": True,
        "abort_on_error": False,
        "ignoreerrors": True,
        "logger": _YtdlpLogger(),
        # Use web client only — android/ios now require GVS PO tokens.
        "extractor_args": {
            "youtube": {
                "player_client": ["web"],
            },
        },
    }
    if ffmpeg_location:
        opts["ffmpeg_location"] = ffmpeg_location

    node = _resolve_node_exe()
    if node:
        opts["js_runtimes"] = {"node": {"path": node}}
        return opts
    for name, exe in (("deno", "deno"), ("bun", "bun"), ("node", "node")):
        if shutil.which(exe):
            opts["js_runtimes"] = {name: {}}
            return opts

    # Do not pass js_runtimes={}: that disables yt-dlp's default (deno) and increases warnings.
    return opts
