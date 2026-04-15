from __future__ import annotations

from pathlib import Path
import cv2

try:
    # Silence OpenCV's verbose stderr warnings for corrupt/truncated frames.
    if hasattr(cv2, "utils"):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    else:
        cv2.setLogLevel(getattr(cv2, "LOG_LEVEL_ERROR", 0))
except (AttributeError, RuntimeError):
    # OpenCV log-level APIs vary across builds; the silence is a nice-to-have, not required.
    pass


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    max_frames: int = 60,
    frame_size: int = 224,
    stem: str | None = None,
    min_frames: int = 0,
    min_short_side: int = 0,
) -> int:
    """Extract evenly-spaced frames by seeking directly to target positions."""
    stem = stem or video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total <= 0:
        cap.release()
        return 0
    if min_frames > 0 and total < min_frames:
        cap.release()
        return 0
    if min_short_side > 0 and min(w, h) < min_short_side:
        cap.release()
        return 0

    step = max(1, total // max_frames)
    targets = [i for i in range(0, total, step)][:max_frames]

    if not targets:
        cap.release()
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for target_frame in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (frame_size, frame_size))
        cv2.imwrite(str(output_dir / f"{stem}_frame_{saved}.jpg"), frame)
        saved += 1

    cap.release()
    return saved


def extract_frames_job(job: tuple[str, str, int, int, int, int]) -> tuple[str, int]:
    """Worker for parallel extraction (thread pool)."""
    v, o, mf, fs, min_f, min_side = job
    n = extract_frames_from_video(
        Path(v), Path(o), max_frames=mf, frame_size=fs,
        min_frames=min_f, min_short_side=min_side,
    )
    return (Path(v).name, n)
