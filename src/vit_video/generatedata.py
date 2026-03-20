import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
from yt_dlp import YoutubeDL


# Default search categories for healthy vs unhealthy food.
DEFAULT_FOOD_CATEGORIES = {
    "healthy": [
        "banana fruit close up",
        "fresh apple on table",
        "broccoli vegetable cooking",
        "green salad bowl",
        "healthy oatmeal breakfast",
        "grilled salmon with vegetables",
    ],
    "unhealthy": [
        "cheeseburger close up",
        "pepperoni pizza slice",
        "french fries fast food",
        "glazed donut",
        "fried chicken bucket",
        "hot dog street food",
    ],
}


def _find_ffmpeg() -> str | None:
    """Return the path to ffmpeg if available, else None."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    # imageio-ffmpeg bundles a static ffmpeg binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _ensure_ffmpeg() -> str | None:
    """Try to locate ffmpeg; install imageio-ffmpeg as fallback."""
    path = _find_ffmpeg()
    if path:
        return path
    print("[INFO] ffmpeg not found. Installing imageio-ffmpeg (bundled ffmpeg)...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "imageio-ffmpeg"],
        stdout=subprocess.DEVNULL,
    )
    return _find_ffmpeg()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare healthy/unhealthy food video frames."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="food_data",
        help="Root directory to store videos and extracted frames.",
    )
    parser.add_argument(
        "--videos-per-keyword",
        type=int,
        default=5,
        help="Number of YouTube results to download per search keyword.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=60,
        help="Maximum number of frames to extract per video.",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=224,
        help="Output frame size (square, e.g. 224 -> 224x224).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=16,
        help="Skip videos with fewer than this many frames.",
    )
    parser.add_argument(
        "--categories-json",
        type=str,
        default=None,
        help=(
            "Optional path to JSON mapping class name -> list of search keywords. "
            "If provided, overrides the default healthy/unhealthy keywords."
        ),
    )
    return parser.parse_args()


def load_categories(categories_json: str | None) -> dict:
    if not categories_json:
        return DEFAULT_FOOD_CATEGORIES
    path = Path(categories_json)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("categories-json must contain an object mapping class -> [keywords].")
    return data


def setup_folders(dataset_dir: Path, categories: dict) -> None:
    """Create raw video and frame folders per category."""
    for cat in categories.keys():
        (dataset_dir / "raw_videos" / cat).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "frames" / cat).mkdir(parents=True, exist_ok=True)
    print(f"[OK] Folders initialized under {dataset_dir.resolve()}")


def download_web_videos(
    dataset_dir: Path,
    categories: dict,
    num_videos_per_keyword: int,
) -> None:
    """Search and download videos using yt-dlp for each category/keyword."""
    ffmpeg_path = _ensure_ffmpeg()
    has_ffmpeg = ffmpeg_path is not None
    if has_ffmpeg:
        print(f"[OK] ffmpeg found: {ffmpeg_path}")
    else:
        print("[WARN] ffmpeg not available -- will download single-stream mp4 only (lower quality).")

    for category, keywords in categories.items():
        for query in keywords:
            save_path = dataset_dir / "raw_videos" / category / "%(title)s.%(ext)s"

            if has_ffmpeg:
                fmt = "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best"
            else:
                fmt = "best[ext=mp4][height<=720]/best[ext=mp4]/best"

            ydl_opts = {
                "format": fmt,
                "outtmpl": str(save_path),
                "quiet": True,
                "noplaylist": True,
                "abort_on_error": False,
                "ignoreerrors": True,
                "no_warnings": False,
            }

            if has_ffmpeg:
                ydl_opts["ffmpeg_location"] = ffmpeg_path

            print(f"\nSearching the web for: '{query}' (class='{category}')")
            with YoutubeDL(ydl_opts) as ydl:
                try:
                    ydl.download([f"ytsearch{num_videos_per_keyword}:{query}"])
                except Exception as e:
                    print(f"  [FAIL] Could not download results for '{query}': {e}")

    # Report what was downloaded.
    for category in categories.keys():
        vids = list((dataset_dir / "raw_videos" / category).glob("*.mp4"))
        print(f"  [{category}] {len(vids)} videos downloaded.")


def extract_frames(
    dataset_dir: Path,
    categories: dict,
    max_frames_per_video: int,
    frame_size: int,
    min_frames: int,
) -> None:
    """Convert downloaded videos into filtered, resized frames."""
    print("\n--- Processing videos into frames ---")
    frame_shape = (frame_size, frame_size)

    for category in categories.keys():
        video_folder = dataset_dir / "raw_videos" / category
        output_folder = dataset_dir / "frames" / category

        for video_path in video_folder.glob("*.mp4"):
            # Skip if we have already extracted frames for this source video
            existing = list(output_folder.glob(f"{video_path.stem}_frame_*.jpg"))
            if existing:
                print(f"[SKIP] {video_path.name} (frames already exist: {len(existing)})")
                continue

            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if total_frames <= 0:
                print(f"[SKIP] {video_path.name}: unable to read frames.")
                cap.release()
                continue

            if total_frames < min_frames:
                print(
                    f"✖ Skipping {video_path.name}: too short "
                    f"({total_frames} frames < min_frames={min_frames})."
                )
                cap.release()
                continue

            if min(width, height) < frame_size:
                print(
                    f"[SKIP] {video_path.name}: resolution too small "
                    f"({width}x{height} < {frame_size}x{frame_size})."
                )
                cap.release()
                continue

            # Extract frames at even intervals
            step = max(1, total_frames // max_frames_per_video)
            count = 0
            saved_count = 0
            while cap.isOpened() and saved_count < max_frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break

                if count % step == 0:
                    frame = cv2.resize(frame, frame_shape)
                    filename = f"{video_path.stem}_frame_{saved_count}.jpg"
                    cv2.imwrite(str(output_folder / filename), frame)
                    saved_count += 1
                count += 1
            cap.release()
            print(f"[OK] Extracted {saved_count} frames from {video_path.name}")


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    categories = load_categories(args.categories_json)

    setup_folders(dataset_dir, categories)
    download_web_videos(
        dataset_dir=dataset_dir,
        categories=categories,
        num_videos_per_keyword=args.videos_per_keyword,
    )
    extract_frames(
        dataset_dir=dataset_dir,
        categories=categories,
        max_frames_per_video=args.max_frames_per_video,
        frame_size=args.frame_size,
        min_frames=args.min_frames,
    )
    print(f"\nDONE! Your dataset is ready in: {dataset_dir / 'frames'}")


if __name__ == "__main__":
    main()