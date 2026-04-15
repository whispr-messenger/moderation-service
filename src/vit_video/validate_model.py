from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset

try:
    import _bootstrap; _bootstrap.setup()
except ImportError:
    pass  # sys.path already configured (e.g. Colab notebook)

from vit_video.data import VideoDataset
from vit_video.data.splits import frames_directory_has_images, video_stem_from_path
from vit_video.engine import Trainer, compute_class_weights_from_dataset
from vit_video.models import MobileViTModel
from vit_video.test import evaluate
from vit_video.paths import DEFAULT_FRAMES_DIR, PACKAGE_ROOT
from vit_video.utils import get_device, load_model_from_checkpoint, print_device_info
from vit_video.utils.ytdlp_helpers import (
    match_filter_max_duration_seconds,
    youtube_dl_base_options,
    ytdlp_download_stderr_filtered,
)


def audit_data_leakage(dataset_root: Path, val_split: float = 0.15, seed: int = 42) -> Dict:
    """Detect frame-level split video overlap and class imbalance."""
    print("\n--- Data-leakage audit ---")

    ds = VideoDataset(root=dataset_root, frames_per_video=8, augment=False)
    if not ds.items:
        print(f"\nNo frame images found under {dataset_root.resolve()}")
        print("  Point --dataset-dir at .../food_data/frames (with healthy/ and unhealthy/).")
        return {"error": "no_data", "n_frames": 0, "n_videos": 0}

    # Group frames by video
    video_groups: dict[Tuple[int, str], List[int]] = defaultdict(list)
    for idx, (path, label) in enumerate(ds.items):
        video_groups[(label, video_stem_from_path(path))].append(idx)

    n_frames = len(ds.items)
    n_videos = len(video_groups)

    class_counts: Dict[str, int] = Counter()
    class_video_counts: Dict[str, int] = Counter()
    for (label, _), idxs in video_groups.items():
        cls = ds.classes[label]
        class_counts[cls] += len(idxs)
        class_video_counts[cls] += 1

    print(f"\nDataset: {dataset_root}")
    print(f"Total frames: {n_frames}, Unique videos: {n_videos}")
    for cls in ds.classes:
        print(f"  {cls}: {class_counts[cls]} frames from {class_video_counts[cls]} videos")

    # Simulate old frame-level split to measure leakage
    indices = list(range(n_frames))
    labels = [ds.items[i][1] for i in indices]
    cnt = Counter(labels)
    do_stratify = all(v >= 2 for v in cnt.values())
    if do_stratify:
        train_idx, val_idx = train_test_split(indices, test_size=val_split, stratify=labels, random_state=seed)
    else:
        train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=seed)

    train_stems = {video_stem_from_path(ds.items[i][0]) for i in train_idx}
    val_stems = {video_stem_from_path(ds.items[i][0]) for i in val_idx}
    overlap = train_stems & val_stems
    overlap_pct = 100.0 * len(overlap) / len(train_stems | val_stems) if (train_stems | val_stems) else 0
    leaked_val = sum(1 for i in val_idx if video_stem_from_path(ds.items[i][0]) in overlap)
    leaked_pct = 100.0 * leaked_val / len(val_idx) if val_idx else 0

    print("\n--- Frame-level split (OLD) ---")
    print(f"  Video overlap: {len(overlap)}/{n_videos} ({overlap_pct:.1f}%)")
    print(f"  Leaked val frames: {leaked_val}/{len(val_idx)} ({leaked_pct:.1f}%)")
    if overlap_pct > 50:
        print("  ** SEVERE DATA LEAKAGE **")
    elif overlap_pct > 0:
        print("  ** DATA LEAKAGE DETECTED **")

    counts = list(class_counts.values())
    min_count = min(counts) if counts else 0
    max_count = max(counts) if counts else 0
    if min_count > 0 and max_count / min_count > 3:
        majority = max(class_counts, key=class_counts.get)
        baseline = 100.0 * max_count / n_frames
        print(f"\n  ** CLASS IMBALANCE ** {max_count}:{min_count} ({max_count / min_count:.1f}x)")
        print(f"  Majority-class baseline: {baseline:.1f}% (always predict '{majority}')")
    elif min_count == 0:
        print("\n  ** WARNING: At least one class has zero frames **")

    return {
        "n_frames": n_frames, "n_videos": n_videos,
        "class_frame_counts": dict(class_counts),
        "class_video_counts": dict(class_video_counts),
        "frame_split_video_overlap": len(overlap),
        "frame_split_leaked_val_pct": round(leaked_pct, 2),
        "classes": ds.classes,
    }



def _cv_skipped(
    reason: str, *, n_videos: int = 0, n_folds_requested: int = 5,
) -> Dict:
    return {
        "skipped": True,
        "reason": reason,
        "n_videos": n_videos,
        "n_folds_requested": n_folds_requested,
        "n_folds": 0,
        "mean_accuracy": 0.0,
        "std_accuracy": 0.0,
        "mean_f1": 0.0,
        "std_f1": 0.0,
        "folds": [],
    }


def run_kfold_cv(
    dataset_root: Path, backbone: str, n_folds: int = 5, epochs: int = 10,
    batch_size: int = 8, frames_per_video: int = 8, img_size: int = 224,
    dropout: float = 0.7, weight_decay: float = 1e-3, lr: float = 1e-4,
    patience: int = 3, device: torch.device | None = None,
) -> Dict:
    """K-fold CV with video-level grouping (stratified when each class has enough videos)."""
    print(f"\n{n_folds}-fold CV (video-level groups)…")

    device = device or get_device()
    root = dataset_root.resolve()
    if not frames_directory_has_images(dataset_root):
        print(f"No frame images under {root} — skipping CV.")
        return _cv_skipped("no_frames", n_videos=0, n_folds_requested=n_folds)

    base_ds = VideoDataset(root=dataset_root, frames_per_video=frames_per_video,
                           img_size=img_size, augment=False)
    classes = base_ds.classes
    n_classes = len(classes)

    if not base_ds.items:
        print(f"No samples in VideoDataset for {root} — skipping CV.")
        return _cv_skipped("no_dataset_items", n_videos=0, n_folds_requested=n_folds)

    video_groups: dict[Tuple[int, str], List[int]] = defaultdict(list)
    for idx, (path, label) in enumerate(base_ds.items):
        video_groups[(label, video_stem_from_path(path))].append(idx)

    group_keys = list(video_groups.keys())
    group_labels = np.array([k[0] for k in group_keys], dtype=int)

    if len(group_keys) < 2:
        print(f"Need at least 2 distinct videos for CV; found {len(group_keys)} — skipping CV.")
        return _cv_skipped("insufficient_videos", n_videos=len(group_keys), n_folds_requested=n_folds)

    n_folds_eff = min(max(2, n_folds), len(group_keys))
    if n_folds_eff < n_folds:
        print(f"Using {n_folds_eff} folds ({len(group_keys)} videos available).")

    label_counts = Counter(group_labels.tolist())
    use_stratify = len(label_counts) >= 2 and min(label_counts.values()) >= n_folds_eff
    if not use_stratify:
        print(
            f"[CV] Non-stratified KFold (need ≥{n_folds_eff} videos per class for stratified; "
            f"counts={dict(label_counts)})."
        )

    if use_stratify:
        splitter = StratifiedKFold(n_splits=n_folds_eff, shuffle=True, random_state=42)
        split_iter = splitter.split(np.zeros(len(group_keys)), group_labels)
    else:
        splitter = KFold(n_splits=n_folds_eff, shuffle=True, random_state=42)
        split_iter = splitter.split(np.arange(len(group_keys)))

    fold_results = []
    num_workers = 0 if sys.platform == "win32" else 2

    for fold_i, (train_gi, val_gi) in enumerate(split_iter, 1):
        train_idx = [i for gi in train_gi for i in video_groups[group_keys[gi]]]
        val_idx = [i for gi in val_gi for i in video_groups[group_keys[gi]]]

        train_ds = VideoDataset(root=dataset_root, classes=classes,
                                frames_per_video=frames_per_video, img_size=img_size, augment=True)
        val_ds = VideoDataset(root=dataset_root, classes=classes,
                              frames_per_video=frames_per_video, img_size=img_size, augment=False)

        train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)

        model = MobileViTModel(num_classes=n_classes, model_name=backbone,
                                pretrained=True, dropout=dropout)
        cw = compute_class_weights_from_dataset(Subset(train_ds, train_idx), n_classes)
        trainer = Trainer(model=model, device=device, train_loader=train_loader,
                          val_loader=val_loader, lr=lr, weight_decay=weight_decay,
                          output_path=PACKAGE_ROOT / "models", class_weights=cw)
        history = trainer.fit(epochs=epochs, early_stopping_patience=patience,
                              checkpoint_name=f"_cv_fold_{fold_i}.pth")

        results = evaluate(model, val_loader, device, classes)
        print(
            f"  Fold {fold_i}/{n_folds_eff}: train {len(train_idx)} / val {len(val_idx)} frames — "
            f"Acc {results['accuracy']*100:.2f}%, F1 {results['f1_macro']*100:.2f}%"
        )

        fold_results.append({
            "fold": fold_i,
            "accuracy": results["accuracy"],
            "f1_macro": results["f1_macro"],
            "precision_macro": results["precision_macro"],
            "recall_macro": results["recall_macro"],
            "train_frames": len(train_idx),
            "val_frames": len(val_idx),
            "epochs_completed": len(history.get("train_loss", [])),
            "best_val_loss": min(history.get("val_loss", [float("inf")])),
        })

    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1_macro"] for r in fold_results]
    summary = {
        "skipped": False,
        "n_folds": n_folds_eff,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "folds": fold_results,
    }
    print(
        f"CV: Acc {summary['mean_accuracy']*100:.2f}% ± {summary['std_accuracy']*100:.2f}%, "
        f"F1 {summary['mean_f1']*100:.2f}% ± {summary['std_f1']*100:.2f}%"
    )
    return summary



EXTERNAL_TEST_QUERIES = {
    "healthy": [
        "quinoa bowl meal prep",
        "fresh fruit smoothie recipe",
        "steamed vegetables dinner plate",
        "grilled fish lemon herbs",
    ],
    "other": [
        "city walking tour vlog",
        "pet cat playing toys",
        "car review driving test",
        "landscape photography timelapse",
    ],
    "unhealthy": [
        "deep fried mozzarella sticks",
        "loaded nachos cheese",
        "chocolate cake frosting slice",
        "fast food mukbang eating",
        "candy haul unboxing sweets",
        "fried chicken wings sauce",
    ],
}


def download_external_test_videos(output_dir: Path, videos_per_query: int = 2) -> Dict[str, List[Path]]:
    n_queries = sum(len(q) for q in EXTERNAL_TEST_QUERIES.values())
    print(f"External test videos: {n_queries} ytsearch queries ({videos_per_query} hits each)…")
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        print("[SKIP] yt-dlp not installed.")
        return {}

    import shutil
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            # ffmpeg is genuinely optional here; callers degrade gracefully.
            ffmpeg_path = None

    downloaded: Dict[str, List[Path]] = {}
    consecutive_fails = 0
    for cls, queries in EXTERNAL_TEST_QUERIES.items():
        cls_dir = output_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for query in queries:
            if consecutive_fails >= 3:
                print("  [SKIP] 3 consecutive failures — yt-dlp likely broken (install Node.js)")
                break
            opts = {
                **youtube_dl_base_options(ffmpeg_location=ffmpeg_path),
                "format": "best[ext=mp4][height<=480]/best[ext=mp4]/best",
                "outtmpl": str(cls_dir / "%(title)s.%(ext)s"),
                "match_filter": match_filter_max_duration_seconds(60),
            }
            before = set(cls_dir.glob("*.mp4"))
            try:
                with YoutubeDL(opts) as ydl:
                    with ytdlp_download_stderr_filtered():
                        ydl.download([f"ytsearch{videos_per_query}:{query}"])
            except Exception as e:
                print(f"  [FAIL] {cls!r} {query!r}: {e}")
            after = set(cls_dir.glob("*.mp4"))
            if after - before:
                consecutive_fails = 0
            else:
                consecutive_fails += 1
        else:
            continue
        break  # bail out of outer loop too if inner broke
    for cls in EXTERNAL_TEST_QUERIES:
        cls_dir = output_dir / cls
        downloaded[cls] = sorted(cls_dir.glob("*.mp4"))
        print(f"  [{cls}] {len(downloaded[cls])} videos")
    return downloaded


def extract_external_frames(video_dir: Path, output_dir: Path, max_frames: int = 30, frame_size: int = 224):
    from vit_video.utils.video import extract_frames_from_video
    for cls_dir in sorted(video_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        out_cls = output_dir / cls_dir.name
        v_ok = 0
        frames = 0
        for vp in cls_dir.glob("*.mp4"):
            saved = extract_frames_from_video(vp, out_cls, max_frames=max_frames, frame_size=frame_size)
            if saved:
                v_ok += 1
                frames += saved
        if v_ok:
            print(f"  {cls_dir.name}: {v_ok} videos → {frames} frames")


def test_on_external_data(
    model_path: Path, external_frames_dir: Path, backbone: str = "auto",
    frames_per_video: int = 8, batch_size: int = 4, device: torch.device | None = None,
) -> Dict:
    print("\n--- External evaluation ---")
    device = device or get_device()

    if not external_frames_dir.exists() or not any(external_frames_dir.iterdir()):
        print("[SKIP] No external frames found.")
        return {}

    ds = VideoDataset(root=external_frames_dir, frames_per_video=frames_per_video, augment=False)
    if len(ds) == 0:
        print("[SKIP] External dataset is empty.")
        return {}

    num_workers = 0 if sys.platform == "win32" else 2
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = load_model_from_checkpoint(model_path, len(ds.classes), backbone, device)

    results = evaluate(model, loader, device, ds.classes)

    acc = results["accuracy"]
    print(f"\nExternal: {results['num_samples']} samples, Acc: {acc*100:.2f}%, F1: {results['f1_macro']*100:.2f}%")
    if acc >= 0.95:
        print("  ** WARNING: 95%+ on external data — still suspicious")
    elif acc >= 0.85:
        print("  ** REALISTIC: 85-95% suggests genuine learning")
    elif acc >= 0.70:
        print("  ** MODERATE: Struggles with novel data")
    else:
        print("  ** POOR: Below 70% — likely memorized training data")
    return results



def generate_diagnostic_report(leakage: Dict, cv: Dict, external: Dict, output_path: Path) -> Dict:
    print("\n--- Diagnostic report ---")

    report: Dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if leakage:
        report["data_leakage"] = {
            "leaked_val_pct": leakage.get("frame_split_leaked_val_pct", 0),
            "total_videos": leakage.get("n_videos", 0),
            "class_frame_counts": leakage.get("class_frame_counts", {}),
            "class_video_counts": leakage.get("class_video_counts", {}),
        }

    if cv:
        if cv.get("skipped"):
            report["cross_validation"] = {
                "skipped": True,
                "reason": cv.get("reason", ""),
                "n_videos": cv.get("n_videos", 0),
            }
        else:
            report["cross_validation"] = {
                "mean_accuracy": cv["mean_accuracy"],
                "std_accuracy": cv["std_accuracy"],
                "mean_f1": cv["mean_f1"],
                "n_folds": cv.get("n_folds", 0),
            }

    if external:
        report["external_test"] = {
            "accuracy": external.get("accuracy", 0),
            "f1_macro": external.get("f1_macro", 0),
            "n_samples": external.get("num_samples", 0),
        }

    issues = []
    if leakage and leakage.get("frame_split_leaked_val_pct", 0) > 50:
        issues.append("Frame-level splitting causes video leakage across train/val")
    if leakage:
        counts = list(leakage.get("class_frame_counts", {}).values())
        if counts and min(counts) > 0 and max(counts) / min(counts) > 3:
            issues.append(f"Severe class imbalance: {max(counts)}:{min(counts)} frames")
        elif counts and min(counts) == 0:
            issues.append("At least one class has zero frames")
        vid_counts = list(leakage.get("class_video_counts", {}).values())
        if vid_counts and min(vid_counts) < 10:
            issues.append(f"Too few videos in minority class ({min(vid_counts)})")

    report["issues_found"] = issues
    report["recommendations"] = [
        "Use video-level splitting (build_dataloaders does this by default)",
        "Collect more videos per class (20-30 unique)",
        "Enable --class-weighting for imbalanced data",
        "Always validate on external data before reporting accuracy",
    ]

    if cv and not cv.get("skipped"):
        print(f"Video-level CV:    {cv['mean_accuracy']*100:.1f}% +/- {cv['std_accuracy']*100:.1f}%")
    elif cv and cv.get("skipped"):
        print(f"Video-level CV:    skipped ({cv.get('reason', 'unknown')})")
    if external:
        print(f"External test:     {external['accuracy']*100:.1f}%")
    if issues:
        print(f"\nIssues ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\nIssues: none")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {output_path}")
    return report



def main():
    parser = argparse.ArgumentParser(description="Validate food classification model")
    parser.add_argument(
        "--model", type=str,
        default=str(PACKAGE_ROOT / "models" / "best_food_classifier.pth"),
        help="Checkpoint under vit_video/models by default",
    )
    parser.add_argument(
        "--dataset-dir", type=str, default=str(DEFAULT_FRAMES_DIR),
        help="Frames root (default: vit_video/food_data/frames)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PACKAGE_ROOT / "results" / "validation"),
    )
    parser.add_argument("--backbone", type=str, default="auto")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--videos-per-query", type=int, default=2)
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--only-external", action="store_true")
    parser.add_argument("--skip-cv", action="store_true")

    args = parser.parse_args()
    device = get_device()
    print_device_info()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    leakage_report: Dict = {}
    cv_report: Dict = {}
    external_report: Dict = {}

    if not args.only_external:
        leakage_report = audit_data_leakage(dataset_dir)
        if not args.skip_cv:
            cv_report = run_kfold_cv(
                dataset_root=dataset_dir,
                backbone=args.backbone if args.backbone != "auto" else "mobilevit_xxs",
                n_folds=args.n_folds, epochs=args.epochs, batch_size=args.batch_size,
                frames_per_video=args.num_frames, img_size=args.img_size,
                dropout=args.dropout, weight_decay=args.weight_decay,
                lr=args.lr, patience=args.patience, device=device,
            )

    if not args.skip_external:
        ext_video_dir = output_dir / "external_videos"
        ext_frames_dir = output_dir / "external_frames"
        download_external_test_videos(ext_video_dir, args.videos_per_query)
        extract_external_frames(ext_video_dir, ext_frames_dir)

        model_path = Path(args.model)
        if model_path.exists():
            external_report = test_on_external_data(
                model_path, ext_frames_dir, args.backbone,
                args.num_frames, args.batch_size, device,
            )
        else:
            print(f"[SKIP] Model not found at {model_path}")

    generate_diagnostic_report(leakage_report, cv_report, external_report,
                               output_dir / "diagnostic_report.json")

    for f in (PACKAGE_ROOT / "models").glob("_cv_fold_*.pth"):
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
