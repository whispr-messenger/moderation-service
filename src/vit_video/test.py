from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from tqdm import tqdm

try:
    import _bootstrap; _bootstrap.setup()
except ImportError:
    pass  # sys.path already configured (e.g. Colab notebook)

from vit_video.paths import DEFAULT_FRAMES_DIR, PACKAGE_ROOT
from vit_video.utils import print_device_info, parse_normalization_values, get_device, extract_state_dict, load_model_from_checkpoint
from vit_video.data import VideoDataset
from vit_video.data.dataset import _indices_for_video_keys, _video_level_split
from vit_video.data.splits import (
    keys_from_manifest_split, load_split_manifest,
    manifest_path_for_frames_dir, sync_manifest_with_frames_dir,
    warn_if_new_videos_not_in_manifest,
)


def build_test_loader(
    dataset_root: Path,
    frames_per_video: int = 8,
    batch_size: int = 4,
    num_workers: int = 0,
    test_split: float = 0.2,
    seed: int = 42,
    filter_classes: Optional[List[str]] = None,
    norm_mean: Optional[List[float]] = None,
    norm_std: Optional[List[float]] = None,
    split_manifest: Optional[Path] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[DataLoader, List[str], List[int]]:
    dataset_root = Path(dataset_root)

    test_dir = dataset_root / "test"
    if test_dir.exists() and any(test_dir.iterdir()):
        ds = VideoDataset(root=test_dir, frames_per_video=frames_per_video,
                          classes=filter_classes, mean=norm_mean, std=norm_std)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return loader, ds.classes, list(range(len(ds)))

    ds = VideoDataset(root=dataset_root, frames_per_video=frames_per_video,
                      classes=filter_classes, mean=norm_mean, std=norm_std)
    if len(ds) == 0:
        raise RuntimeError(f"No data found in {dataset_root}")

    mpath = Path(split_manifest) if split_manifest else manifest_path_for_frames_dir(dataset_root)
    if mpath.exists():
        sync_manifest_with_frames_dir(
            dataset_root, mpath, train_ratio, val_ratio, test_ratio, seed,
        )
        manifest = load_split_manifest(mpath)
        warn_if_new_videos_not_in_manifest(dataset_root, manifest)
        test_keys = keys_from_manifest_split(manifest, "test")
        test_idx = _indices_for_video_keys(ds.items, ds.classes, test_keys)
        print(f"[Split] Test: {len(test_keys)} videos, {len(test_idx)} frame-rows")
        if not test_idx:
            raise RuntimeError("Manifest test split is empty — regenerate the manifest.")
        return DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers), ds.classes, test_idx

    print(f"[WARN] No manifest found — using random {test_split:.0%} video holdout.")
    _, test_idx = _video_level_split(ds.items, val_split=test_split, seed=seed)
    return DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers), ds.classes, test_idx


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, classes: List[str]) -> Dict:
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    for videos, labels in tqdm(dataloader, desc="Evaluating", leave=True):
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        probs = torch.softmax(outputs, dim=1)
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    return {
        "accuracy": float(accuracy_score(all_targets, all_preds)),
        "precision_macro": float(precision_score(all_targets, all_preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(all_targets, all_preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(all_targets, all_preds, average="macro", zero_division=0)),
        "per_class": {
            classes[i]: {
                "precision": float(precision_score(all_targets, all_preds, average=None, zero_division=0)[i]),
                "recall": float(recall_score(all_targets, all_preds, average=None, zero_division=0)[i]),
                "f1": float(f1_score(all_targets, all_preds, average=None, zero_division=0)[i]),
            }
            for i in range(len(classes))
        },
        "confusion_matrix": confusion_matrix(all_targets, all_preds).tolist(),
        "classification_report": classification_report(all_targets, all_preds, target_names=classes, output_dict=True, zero_division=0),
        "predictions": all_preds.tolist(),
        "ground_truth": all_targets.tolist(),
        "probabilities": np.array(all_probs).tolist(),
        "classes": classes,
        "num_samples": len(all_targets),
    }


def print_results(results: Dict) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTotal samples: {results['num_samples']}")
    print(f"Classes: {results['classes']}")
    print("\n--- Overall Metrics ---")
    print(f"  Accuracy:  {results['accuracy'] * 100:.2f}%")
    print(f"  Precision: {results['precision_macro'] * 100:.2f}% (macro)")
    print(f"  Recall:    {results['recall_macro'] * 100:.2f}% (macro)")
    print(f"  F1 Score:  {results['f1_macro'] * 100:.2f}% (macro)")

    print("\n--- Per-Class Metrics ---")
    for cls, metrics in results["per_class"].items():
        print(f"  {cls}:")
        print(f"    Precision: {metrics['precision'] * 100:.2f}%")
        print(f"    Recall:    {metrics['recall'] * 100:.2f}%")
        print(f"    F1:        {metrics['f1'] * 100:.2f}%")

    print("\n--- Confusion Matrix ---")
    cm = np.array(results["confusion_matrix"])
    header = "".join(f"{c[:8]:>10}" for c in results["classes"])
    print(f"{'Pred->':>10}{header}")
    for i, row in enumerate(cm):
        print(f"{results['classes'][i][:8]:>10}{''.join(f'{v:>10}' for v in row)}")
    print("\n" + "=" * 60)


def save_results(results: Dict, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def get_num_classes_from_checkpoint(model_path: Path) -> int:
    checkpoint = torch.load(model_path, map_location="cpu")
    sd = extract_state_dict(checkpoint)
    for k, v in sd.items():
        if "classifier" in k and "weight" in k:
            return v.shape[0]
    return 2


def main(args) -> Dict:
    device = get_device()
    print_device_info()

    dataset_dir = Path(args.dataset_dir)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    num_classes = args.num_classes or get_num_classes_from_checkpoint(model_path)
    print(f"Number of classes (from checkpoint): {num_classes}")

    filter_classes = args.classes.split(",") if args.classes else None
    norm_mean, norm_std = parse_normalization_values(args.norm_mean, args.norm_std)
    num_workers = 0 if sys.platform == "win32" else args.num_workers

    print(f"\nLoading dataset from: {dataset_dir}")
    sm = Path(args.split_manifest) if args.split_manifest else None
    test_loader, classes, _ = build_test_loader(
        dataset_root=dataset_dir, frames_per_video=args.num_frames,
        batch_size=args.batch_size, num_workers=num_workers,
        test_split=args.test_split, seed=args.seed,
        filter_classes=filter_classes, norm_mean=norm_mean, norm_std=norm_std,
        split_manifest=sm,
        train_ratio=getattr(args, "train_ratio", 0.7),
        val_ratio=getattr(args, "val_ratio", 0.15),
        test_ratio=getattr(args, "test_ratio", 0.15),
    )
    print(f"Classes: {classes}")
    print(f"Test samples: {len(test_loader.dataset)}")

    print(f"\nLoading model from: {model_path}")
    model = load_model_from_checkpoint(model_path, num_classes, args.backbone, device)

    print("\nRunning evaluation...")
    results = evaluate(model, test_loader, device, classes)
    print_results(results)

    if args.output_dir:
        save_results(results, args.output_dir)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Video Food Classifier")
    parser.add_argument(
        "--model", type=str,
        default=str(PACKAGE_ROOT / "models" / "best_food_classifier.pth"),
    )
    parser.add_argument(
        "--dataset-dir", type=str, default=str(DEFAULT_FRAMES_DIR),
        help="Frames root (default: vit_video/food_data/frames)",
    )
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backbone", type=str, default="auto")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--split-manifest", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--classes", type=str, default=None)
    parser.add_argument("--norm-mean", type=str, default="0.485,0.456,0.406")
    parser.add_argument("--norm-std", type=str, default="0.229,0.224,0.225")
    args = parser.parse_args()
    main(args)
