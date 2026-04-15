from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import _bootstrap; _bootstrap.setup()
except ImportError:
    pass  # sys.path already configured (e.g. Colab notebook)

from vit_video.paths import DEFAULT_DATASET_DIR
from vit_video.utils.video import extract_frames_job
from vit_video.utils.ytdlp_helpers import youtube_dl_base_options, ytdlp_download_stderr_filtered

try:
    from yt_dlp import YoutubeDL
except ImportError:
    YoutubeDL = None

DEFAULT_FOOD_CATEGORIES = {
    "healthy": [
        "banana fruit close up",
        "fresh apple fruit plate",
        "broccoli vegetable cooking",
        "green salad bowl meal",
        "healthy oatmeal breakfast bowl",
        "grilled salmon vegetables plate",
        "quinoa bowl healthy meal",
        "greek yogurt berries breakfast",
        "green smoothie drink healthy",
        "avocado toast breakfast",
        "steamed vegetables plate",
        "fresh orange citrus fruit",
        "vegetable stir fry healthy cooking",
        "tofu vegetable bowl vegan",
        "mediterranean hummus plate",
        "brown rice bowl vegetables",
        "fresh mixed berries bowl",
        "kale spinach salad",
        "baked sweet potato meal",
        "lentil soup healthy",
        "whole grain toast breakfast",
        "cucumber tomato salad fresh",
        "edamame beans healthy snack",
        "grilled chicken breast vegetables",
        "sushi roll fresh fish",
        "chickpea buddha bowl",
        "overnight oats fruit jar",
        "roasted vegetables oven tray",
        "poke bowl fresh tuna",
        "fruit platter party healthy",
    ],
    "unhealthy": [
        # --- Classic fast food ---
        "cheeseburger close up eating",
        "pepperoni pizza slice cheese",
        "french fries fast food",
        "glazed donut dessert",
        "fried chicken bucket crispy",
        "hot dog street food",
        "bacon cheeseburger fast food",
        "pepperoni pizza whole pie",
        "double cheeseburger fast food",
        "fast food breakfast sandwich",
        # --- Fried food ---
        "deep fried chicken wings",
        "fried onion rings basket",
        "mozzarella sticks fried",
        "deep fried corn dog",
        "fried fish and chips",
        "tempura deep fried batter",
        "fried spring rolls plate",
        "crispy fried dumplings",
        # --- Sugary desserts ---
        "milkshake dessert drink",
        "ice cream sundae chocolate",
        "chocolate chip cookies baking",
        "chocolate cake slice frosting",
        "churros fried sugar dessert",
        "glazed donuts box dozen",
        "cotton candy carnival",
        "funnel cake powdered sugar",
        "cinnamon roll frosting",
        "brownie chocolate fudge",
        "cupcake frosting sprinkles",
        "cheesecake slice strawberry",
        # --- Sugary drinks ---
        "soda cola pouring glass",
        "energy drink can pouring",
        "frappuccino whipped cream",
        "bubble tea boba milk",
        # --- Processed snacks ---
        "nachos cheese jalapeno loaded",
        "loaded fries cheese bacon",
        "candy sweets unwrapping",
        "mac and cheese creamy bowl",
        "instant ramen noodles bowl",
        "potato chips bag eating",
        "cheese puffs snack bowl",
        "microwave popcorn butter",
        # --- Street food / takeout ---
        "street food taco greasy",
        "kebab wrap doner takeaway",
        "fried street food vendor",
        "carnival food eating fair",
        # --- Buffet / indulgent ---
        "all you can eat buffet",
        "pizza delivery box opening",
        "fast food drive through meal",
        "mukbang eating show junk food",
    ],
}


def _find_ffmpeg() -> str | None:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _ensure_ffmpeg() -> str | None:
    path = _find_ffmpeg()
    if path:
        return path
    print("[INFO] ffmpeg not found. Installing imageio-ffmpeg...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "imageio-ffmpeg"],
        stdout=subprocess.DEVNULL,
    )
    return _find_ffmpeg()


def load_categories(categories_json: str | None) -> dict:
    if not categories_json:
        return DEFAULT_FOOD_CATEGORIES
    with Path(categories_json).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("categories-json must map class -> [keywords].")
    return data


def setup_folders(dataset_dir: Path, categories: dict) -> None:
    for cat in categories:
        (dataset_dir / "raw_videos" / cat).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "frames" / cat).mkdir(parents=True, exist_ok=True)


def download_web_videos(dataset_dir: Path, categories: dict, num_videos_per_keyword: int) -> None:
    if YoutubeDL is None:
        raise RuntimeError("yt-dlp is required. Install with: pip install yt-dlp")
    ffmpeg_path = _ensure_ffmpeg()
    has_ffmpeg = ffmpeg_path is not None
    n_kw = sum(len(kw) for kw in categories.values())
    if not has_ffmpeg:
        print("[WARN] ffmpeg not available — single-stream mp4 only.")
    print(f"Downloading: {n_kw} queries × {num_videos_per_keyword} hits…")

    for category, keywords in categories.items():
        for query in keywords:
            save_path = dataset_dir / "raw_videos" / category / "%(title)s.%(ext)s"
            fmt = (
                "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best"
                if has_ffmpeg else
                "best[ext=mp4][height<=720]/best[ext=mp4]/best"
            )
            ydl_opts = {
                **youtube_dl_base_options(ffmpeg_location=ffmpeg_path if has_ffmpeg else None),
                "format": fmt,
                "outtmpl": str(save_path),
            }

            with YoutubeDL(ydl_opts) as ydl:
                try:
                    with ytdlp_download_stderr_filtered():
                        ydl.download([f"ytsearch{num_videos_per_keyword}:{query}"])
                except Exception as e:
                    print(f"  [FAIL] {category!r} / {query!r}: {e}")

    for category in categories:
        vids = list((dataset_dir / "raw_videos" / category).glob("*.mp4"))
        print(f"  {category}: {len(vids)} mp4")


def _existing_frame_stems(output_folder: Path) -> set[str]:
    stems: set[str] = set()
    for p in output_folder.glob("*_frame_*.jpg"):
        if "_frame_" in p.name:
            stems.add(p.name.split("_frame_", 1)[0])
    return stems


def extract_frames(
    dataset_dir: Path, categories: dict,
    max_frames_per_video: int, frame_size: int, min_frames: int,
    num_workers: int = 0,
) -> None:
    auto_workers = max(1, min(os.cpu_count() or 4, 8))
    nw = num_workers if num_workers > 0 else auto_workers

    if nw > 1:
        print(f"Frame extraction ({nw} threads)…")
    else:
        print("Frame extraction…")

    for category in categories:
        video_folder = dataset_dir / "raw_videos" / category
        output_folder = dataset_dir / "frames" / category
        output_folder.mkdir(parents=True, exist_ok=True)
        already = _existing_frame_stems(output_folder)

        all_mp4 = list(video_folder.glob("*.mp4"))
        jobs: list[tuple[str, str, int, int, int, int]] = []
        for video_path in all_mp4:
            if video_path.stem in already:
                continue
            jobs.append((
                str(video_path.resolve()), str(output_folder.resolve()),
                max_frames_per_video, frame_size, min_frames, 0,
            ))

        n_skip_stem = len(all_mp4) - len(jobs)
        if not all_mp4:
            print(f"  {category}: no .mp4 under raw_videos/{category}/ (skipped)")
            continue
        if not jobs:
            print(
                f"  {category}: all {len(all_mp4)} mp4 already have frames "
                f"({n_skip_stem} stems, {len(already)} known in frames/)"
            )
            continue

        n_jobs = len(jobs)
        print(
            f"  {category}: extracting {n_jobs} mp4 "
            f"({n_skip_stem} already done, {len(all_mp4)} total in folder)...",
            flush=True,
        )
        if category == "healthy":
            print(
                "    (Unhealthy runs only after this class finishes — no output until then.)",
                flush=True,
            )

        v_ok = skip_empty = 0
        total_frames = 0
        report_every = max(5, min(50, n_jobs // 15))

        if nw == 1:
            for i, job in enumerate(jobs, 1):
                _name, saved = extract_frames_job(job)
                if saved > 0:
                    v_ok += 1
                    total_frames += saved
                else:
                    skip_empty += 1
                if i % report_every == 0 or i == n_jobs:
                    print(f"    {category} progress: {i}/{n_jobs} mp4", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=nw) as pool:
                futures = [pool.submit(extract_frames_job, j) for j in jobs]
                done = 0
                for fut in as_completed(futures):
                    _name, saved = fut.result()
                    if saved > 0:
                        v_ok += 1
                        total_frames += saved
                    else:
                        skip_empty += 1
                    done += 1
                    if done % report_every == 0 or done == n_jobs:
                        print(f"    {category} progress: {done}/{n_jobs} mp4", flush=True)

        print(
            f"  {category}: done — {total_frames} frames from {v_ok} videos"
            + (f" ({skip_empty} empty/skipped)" if skip_empty else ""),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare food video frames.")
    parser.add_argument(
        "--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR),
        help="Dataset root (default: vit_video/food_data)",
    )
    parser.add_argument("--videos-per-keyword", type=int, default=15)
    parser.add_argument("--max-frames-per-video", type=int, default=60)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument(
        "--min-frames", type=int, default=1,
        help="Skip videos with fewer decoded frames (default 1; use 16 to drop very short clips)",
    )
    parser.add_argument("--categories-json", type=str, default=None)
    parser.add_argument("--extract-workers", type=int, default=0)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    categories = load_categories(args.categories_json)

    setup_folders(dataset_dir, categories)
    download_web_videos(dataset_dir, categories, args.videos_per_keyword)
    extract_frames(
        dataset_dir, categories,
        args.max_frames_per_video, args.frame_size, args.min_frames,
        num_workers=args.extract_workers,
    )
    print(f"\nDONE! Dataset ready in: {dataset_dir / 'frames'}")


if __name__ == "__main__":
    main()
