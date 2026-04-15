from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import _bootstrap; _bootstrap.setup()
except ImportError:
    pass  # sys.path already configured (e.g. Colab notebook)

from vit_video.data.splits import (
    ensure_split_manifest,
    frames_directory_has_images,
    manifest_path_for_frames_dir,
)
from vit_video.paths import DEFAULT_DATASET_DIR


def _make_namespace(**kwargs) -> argparse.Namespace:
    """Build an argparse.Namespace from keyword arguments."""
    return argparse.Namespace(**kwargs)


def _require_torch() -> None:
    """Fail fast with actionable hints if PyTorch native DLLs do not load (common on Windows)."""
    try:
        import torch  # noqa: F401
    except OSError as e:
        print("\nERROR: PyTorch could not load native libraries (import torch failed).", file=sys.stderr)
        if sys.platform == "win32":
            print(
                "On Windows, fix this in order:\n"
                "  1. Install Microsoft Visual C++ Redistributable (x64):\n"
                "     https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                "  2. Reinstall PyTorch in this environment (repair corrupted wheels):\n"
                "     pip uninstall torch torchvision -y\n"
                "     pip install --no-cache-dir torch torchvision\n"
                "  3. Confirm 64-bit Python (should print 64):\n"
                "     python -c \"import struct; print(struct.calcsize('P') * 8)\"\n",
                file=sys.stderr,
            )
        print(f"Detail: {e}", file=sys.stderr)
        raise SystemExit(1) from e


def _raw_videos_have_mp4(dataset_dir: Path, categories: dict) -> bool:
    for cat in categories:
        folder = dataset_dir / "raw_videos" / cat
        if folder.is_dir() and any(folder.glob("*.mp4")):
            return True
    return False


def _run_extract_frames(args: argparse.Namespace) -> Path:
    """Extract jpg frames from ``raw_videos/<class>/*.mp4`` into ``frames/<class>/``."""
    from vit_video.generatedata import extract_frames, load_categories, setup_folders

    dataset_dir = Path(args.dataset_dir)
    categories = load_categories(args.categories_json)
    setup_folders(dataset_dir, categories)
    extract_frames(
        dataset_dir, categories,
        max_frames_per_video=args.max_frames_per_video,
        frame_size=args.frame_size,
        min_frames=args.min_frames,
        num_workers=args.extract_workers,
    )
    return dataset_dir / "frames"


def step_download(args: argparse.Namespace) -> Path:
    """Step 1: Download YouTube videos and extract frames."""
    print("\n[1/5] Data download (YouTube + frame extraction)")

    from vit_video.generatedata import load_categories, setup_folders, download_web_videos

    dataset_dir = Path(args.dataset_dir)
    categories = load_categories(args.categories_json)
    setup_folders(dataset_dir, categories)
    download_web_videos(dataset_dir, categories, args.videos_per_keyword)
    return _run_extract_frames(args)


def step_extract_frames_only(args: argparse.Namespace) -> Path:
    """Step 1: Only extract frames from existing ``raw_videos`` (no YouTube)."""
    print("\n[1/5] Frame extraction from raw_videos (skipping YouTube)")
    return _run_extract_frames(args)


def step_train(args: argparse.Namespace, frames_dir: Path, split_manifest_path: Path) -> Path:
    """Step 2: Train the model."""
    print("\n[2/5] Training")

    _require_torch()
    from vit_video.train import main as train_main

    model_path = Path(args.output_model).resolve()

    train_args = _make_namespace(
        dataset_dir=str(frames_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        dropout=args.dropout,
        num_frames=args.num_frames,
        img_size=args.img_size,
        disable_augmentation=False,
        class_weighting=args.class_weighting,
        min_samples_per_class=50,
        temporal_pool=args.temporal_pool,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
        hparam_search_epochs=args.hparam_search_epochs,
        lr_candidates=args.lr_candidates,
        num_workers=0 if sys.platform == "win32" else 4,
        patience=args.patience,
        min_delta=1e-4,
        backbone=args.backbone,
        output_model=str(model_path),
        resume_from="",
        split_manifest=str(split_manifest_path),
        no_auto_split_manifest=True,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )

    train_main(train_args)
    print(f"\nModel saved: {model_path}")
    return model_path


def step_test(args: argparse.Namespace, model_path: Path, frames_dir: Path, split_manifest_path: Path) -> Path:
    """Step 3: Evaluate the model."""
    print("\n[3/5] Test")

    from vit_video.test import main as test_main

    results_dir = Path(args.results_dir)

    test_args = _make_namespace(
        model=str(model_path),
        dataset_dir=str(frames_dir),
        output_dir=str(results_dir),
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_workers=0 if sys.platform == "win32" else 2,
        backbone=args.backbone,
        test_split=0.2,
        seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_classes=None,
        classes=None,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
        split_manifest=str(split_manifest_path),
    )

    test_main(test_args)
    results_file = results_dir / "test_results.json"
    print(f"\nResults saved: {results_file}")
    return results_file


def step_export(args: argparse.Namespace, model_path: Path, results_file: Path) -> None:
    """Step 4: Export to mobile formats."""
    print("\n[4/5] Export")

    from vit_video.export_mobile import main as export_main

    export_dir = Path(args.export_dir)
    metrics_path = model_path.with_name(model_path.stem + "_training_metrics.json")

    export_args = _make_namespace(
        model=str(model_path),
        output_dir=str(export_dir),
        format=args.export_formats,
        num_classes=args.num_classes,
        classes=args.classes,
        backbone=args.backbone,
        num_frames=args.num_frames,
        img_size=args.img_size,
        quantize=False,
        eval_results=str(results_file) if results_file.exists() else "",
        training_metrics=str(metrics_path) if metrics_path.exists() else "",
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
    )

    export_main(export_args)
    print(f"\nExported models: {export_dir}")


def step_upload_hf(args: argparse.Namespace) -> None:
    """Step 5: Upload exported models to Hugging Face Hub."""
    print("\n[5/5] Hugging Face upload")

    from vit_video.upload_hf import main as upload_main

    upload_args = _make_namespace(
        repo_id=args.hf_repo_id,
        export_dir=str(args.export_dir),
        private=args.hf_private,
        commit_message=f"Upload {args.backbone} food classifier",
    )

    upload_main(upload_args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full pipeline: download > train > test > export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Pipeline control
    skip = parser.add_argument_group("pipeline steps")
    skip.add_argument("--skip-download", action="store_true", help="Skip data download (use existing data)")
    skip.add_argument("--skip-export", action="store_true", help="Skip mobile export")
    skip.add_argument("--skip-upload", action="store_true", help="Skip Hugging Face upload")

    # Data
    data = parser.add_argument_group("data")
    data.add_argument(
        "--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR),
        help="Root dir for videos and frames (default: vit_video/food_data)",
    )
    data.add_argument("--videos-per-keyword", type=int, default=15,
                      help="yt-dlp search hits per query (more => larger dataset)")
    data.add_argument("--max-frames-per-video", type=int, default=60)
    data.add_argument("--frame-size", type=int, default=224)
    data.add_argument(
        "--min-frames", type=int, default=1,
        help="Skip raw mp4 with fewer decoded frames (default 1)",
    )
    data.add_argument(
        "--extract-workers", type=int, default=0,
        help="Parallel frame extraction threads (0=auto up to 8, 1=sequential)",
    )
    data.add_argument("--categories-json", type=str, default=None)

    spl = parser.add_argument_group("train/val/test split (video-level, one manifest)")
    spl.add_argument("--split-manifest", type=str, default="",
                     help="Path to video_split_manifest.json (default: <dataset-dir>/video_split_manifest.json)")
    spl.add_argument("--regenerate-splits", action="store_true",
                     help="Rebuild manifest from all videos on disk (e.g. after new downloads)")
    spl.add_argument("--train-ratio", type=float, default=0.7)
    spl.add_argument("--val-ratio", type=float, default=0.15)
    spl.add_argument("--test-ratio", type=float, default=0.15)
    spl.add_argument("--split-seed", type=int, default=42)

    # Training
    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--lr", type=float, default=3e-5)
    train.add_argument("--weight-decay", type=float, default=1e-3)
    train.add_argument("--dropout", type=float, default=0.4)
    train.add_argument("--patience", type=int, default=7)
    train.add_argument("--backbone", type=str, default="auto")
    train.add_argument("--temporal-pool", type=str, default="avg", choices=["avg", "max", "conv1d"])
    train.add_argument("--class-weighting", action="store_true", default=True,
                        help="Enable class weighting for imbalanced data (default: on)")
    train.add_argument("--no-class-weighting", dest="class_weighting", action="store_false")
    train.add_argument("--num-frames", type=int, default=8)
    train.add_argument("--img-size", type=int, default=224)
    train.add_argument("--hparam-search-epochs", type=int, default=0)
    train.add_argument("--lr-candidates", type=str, default="5e-6,1e-5,3e-5,5e-5,1e-4")
    train.add_argument("--norm-mean", type=str, default="0.485,0.456,0.406")
    train.add_argument("--norm-std", type=str, default="0.229,0.224,0.225")

    # Output
    out = parser.add_argument_group("output")
    out.add_argument("--output-model", type=str, default="models/best_food_classifier.pth")
    out.add_argument("--results-dir", type=str, default="results")
    out.add_argument("--export-dir", type=str, default="exported_models")
    out.add_argument("--export-formats", type=str, nargs="+", default=["torchscript", "onnx"],
                     choices=["torchscript", "onnx", "coreml", "tflite", "all"])
    out.add_argument("--num-classes", type=int, default=2)
    out.add_argument("--classes", type=str, default="healthy,unhealthy")

    # Hugging Face
    hf = parser.add_argument_group("hugging face upload")
    hf.add_argument("--hf-repo-id", type=str, default="",
                    help="HF repo ID, e.g. your-username/food-classifier (required for upload)")
    hf.add_argument("--hf-private", action="store_true",
                    help="Make the HF repository private")

    args = parser.parse_args()

    # Resolve paths
    dataset_dir = Path(args.dataset_dir)
    frames_dir = dataset_dir / "frames"
    model_path = Path(args.output_model).resolve()

    print("vit_video pipeline")
    print(f"Dataset:  {dataset_dir.resolve()}")
    print(f"Model:    {model_path}")
    print(f"Backbone: {args.backbone}")
    print(f"Steps:    {'download > ' if not args.skip_download else ''}train > test"
          f"{' > export' if not args.skip_export else ''}"
          f"{' > upload' if not args.skip_upload and args.hf_repo_id else ''}")

    # Step 1: Download / ensure frame images exist (empty class folders are not "existing data")
    frames_dir.mkdir(parents=True, exist_ok=True)
    has_frames = frames_directory_has_images(frames_dir)
    extracted_from_raw_only = False

    if not args.skip_download:
        from vit_video.generatedata import load_categories

        categories = load_categories(args.categories_json)
        raw_has_mp4 = _raw_videos_have_mp4(dataset_dir, categories)

        if not has_frames:
            if raw_has_mp4:
                print("\n[INFO] Found mp4 under raw_videos; skipping YouTube download.")
                frames_dir = step_extract_frames_only(args)
                extracted_from_raw_only = True
            else:
                frames_dir = step_download(args)
        else:
            if raw_has_mp4:
                print(
                    f"\n[INFO] Frame images exist; extracting any new raw_videos → "
                    f"{frames_dir.resolve()}"
                )
                frames_dir = step_extract_frames_only(args)
                extracted_from_raw_only = True
            else:
                print(f"\n[INFO] Frame images already present: {frames_dir.resolve()}")
    else:
        if not has_frames:
            print(f"\nERROR: No frame images under {frames_dir.resolve()}")
            print(
                "Remove --skip-download to download/extract, or copy frame images into "
                "frames/healthy|unhealthy/, or pass --dataset-dir to a dataset root that has them."
            )
            return
        print(f"\n[SKIP] Download skipped; using {frames_dir.resolve()}")

    if not frames_directory_has_images(frames_dir):
        print(f"\nERROR: Still no frame images under {frames_dir.resolve()}")
        print("Check downloads and frame extraction, or fix --dataset-dir.")
        return

    regenerate_manifest = args.regenerate_splits or extracted_from_raw_only
    if extracted_from_raw_only:
        print("\n[INFO] Rebuilding video split manifest (includes new extractions).")

    split_manifest_path = (
        Path(args.split_manifest).resolve()
        if args.split_manifest
        else manifest_path_for_frames_dir(frames_dir)
    )
    tr, va, te = args.train_ratio, args.val_ratio, args.test_ratio
    if abs(tr + va + te - 1.0) > 1e-3:
        print("ERROR: --train-ratio, --val-ratio, --test-ratio must sum to 1.0")
        return
    ensure_split_manifest(
        frames_dir.resolve(),
        train_ratio=tr,
        val_ratio=va,
        test_ratio=te,
        seed=args.split_seed,
        manifest_path=split_manifest_path,
        regenerate=regenerate_manifest,
    )

    # Step 2: Train
    model_path = step_train(args, frames_dir, split_manifest_path)

    # Step 3: Test
    results_file = step_test(args, model_path, frames_dir, split_manifest_path)

    # Step 4: Export
    if not args.skip_export:
        step_export(args, model_path, results_file)
    else:
        print("\n[SKIP] Export")

    # Step 5: Upload to Hugging Face
    if not args.skip_upload and args.hf_repo_id:
        step_upload_hf(args)
    elif not args.skip_upload and not args.hf_repo_id:
        print("\n[SKIP] HF upload (no --hf-repo-id provided)")
    else:
        print("\n[SKIP] HF upload")

    print("\nDone.")
    print(f"  Dataset:  {frames_dir}")
    print(f"  Model:    {model_path}")
    print(f"  Results:  {Path(args.results_dir) / 'test_results.json'}")
    if not args.skip_export:
        print(f"  Exports:  {args.export_dir}/")


if __name__ == "__main__":
    main()
