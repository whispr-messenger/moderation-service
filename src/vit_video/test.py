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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Ensure vit_video is importable from repo root, src/, or src/vit_video/
_script_dir = Path(__file__).resolve().parent
_src = _script_dir.parent
if _src.name == "src" and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
elif _script_dir.name == "vit_video" and str(_script_dir.parent) not in sys.path:
    sys.path.insert(0, str(_script_dir.parent))

from vit_video.utils import print_device_info, parse_normalization_values, get_device, extract_state_dict, load_model_from_checkpoint
from vit_video.data import VideoDataset


def build_test_loader(
    dataset_root: Path,
    frames_per_video: int = 8,
    batch_size: int = 4,
    num_workers: int = 0,
    test_split: float = 0.2,
    seed: int = 42,
    filter_classes: List[str] = None,
    norm_mean: Optional[List[float]] = None,
    norm_std: Optional[List[float]] = None,
) -> Tuple[DataLoader, List[str], List[int]]:
    """
    Build a test dataloader from the dataset.
    
    If a dedicated test folder exists, use that.
    Otherwise, split the dataset and return only the test portion.
    
    Args:
        filter_classes: If provided, only include these class folders.
        norm_mean: Normalization mean values (ImageNet default if not specified).
        norm_std: Normalization std values (ImageNet default if not specified).
    """
    dataset_root = Path(dataset_root)
    
    # Check for dedicated test folder
    test_dir = dataset_root / "test"
    if test_dir.exists() and any(test_dir.iterdir()):
        ds = VideoDataset(root=test_dir, frames_per_video=frames_per_video, classes=filter_classes, mean=norm_mean, std=norm_std)
        test_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return test_loader, ds.classes, list(range(len(ds)))
    
    # Otherwise, create test split from main dataset
    ds = VideoDataset(root=dataset_root, frames_per_video=frames_per_video, classes=filter_classes, mean=norm_mean, std=norm_std)
    n = len(ds)
    if n == 0:
        raise RuntimeError(f"No data found in {dataset_root}")
    
    indices = list(range(n))
    labels = [ds.items[i][1] for i in indices]
    
    # Stratified split
    from collections import Counter
    cnt = Counter(labels)
    do_stratify = all(v >= 2 for v in cnt.values()) and n >= 2
    
    if do_stratify:
        _, test_idx = train_test_split(
            indices, test_size=test_split, stratify=labels, random_state=seed
        )
    else:
        _, test_idx = train_test_split(indices, test_size=test_split, random_state=seed)
    
    test_subset = Subset(ds, test_idx)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return test_loader, ds.classes, test_idx


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    classes: List[str],
) -> Dict:
    """
    Run evaluation and compute metrics.
    
    Returns dict with:
        - accuracy, precision, recall, f1 (macro and per-class)
        - confusion_matrix
        - predictions and ground truth
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    for videos, labels in tqdm(dataloader, desc="Evaluating", leave=True):
        videos = videos.to(device)
        labels = labels.to(device)
        
        outputs = model(videos)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision_macro = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    
    precision_per_class = precision_score(all_targets, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_targets, all_preds, average=None, zero_division=0)
    
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    # Classification report as dict
    report = classification_report(
        all_targets, all_preds, target_names=classes, output_dict=True, zero_division=0
    )
    
    results = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "per_class": {
            classes[i]: {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i]),
            }
            for i in range(len(classes))
        },
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": report,
        "predictions": all_preds.tolist(),
        "ground_truth": all_targets.tolist(),
        "probabilities": all_probs.tolist(),
        "classes": classes,
        "num_samples": len(all_targets),
    }
    
    return results


def print_results(results: Dict) -> None:
    """Pretty print evaluation results."""
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
        row_str = "".join(f"{v:>10}" for v in row)
        print(f"{results['classes'][i][:8]:>10}{row_str}")
    
    print("\n" + "=" * 60)


def save_results(results: Dict, output_dir: Path) -> None:
    """Save evaluation results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "test_results.json"
    
    # Save results (convert numpy arrays to lists for JSON serialization)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


def get_num_classes_from_checkpoint(model_path: Path) -> int:
    """Extract number of classes from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
    sd = extract_state_dict(checkpoint)
    
    for k, v in sd.items():
        if "classifier" in k and "weight" in k:
            return v.shape[0]
    return 2


def main(args: argparse.Namespace) -> Dict:
    """Main evaluation entry point."""
    device = get_device()
    print_device_info()

    dataset_dir = Path(args.dataset_dir)
    model_path = Path(args.model)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Auto-detect num_classes from checkpoint
    num_classes = args.num_classes if args.num_classes else get_num_classes_from_checkpoint(model_path)
    print(f"Number of classes (from checkpoint): {num_classes}")
    
    # Filter classes to match model if specified
    filter_classes = args.classes.split(",") if args.classes else None

    # Parse normalization parameters
    norm_mean, norm_std = parse_normalization_values(args.norm_mean, args.norm_std)

    # On Windows, num_workers > 0 can cause DataLoader pickle errors
    import sys
    num_workers = 0 if sys.platform == "win32" else args.num_workers

    # Build test dataloader
    print(f"\nLoading dataset from: {dataset_dir}")
    test_loader, classes, _ = build_test_loader(
        dataset_root=dataset_dir,
        frames_per_video=args.num_frames,
        batch_size=args.batch_size,
        num_workers=num_workers,
        test_split=args.test_split,
        seed=args.seed,
        filter_classes=filter_classes,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    print(f"Classes: {classes}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = load_model_from_checkpoint(
        model_path=model_path,
        num_classes=num_classes,
        model_name=args.backbone,
        device=device,
    )
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluate(model, test_loader, device, classes)
    
    # Print results
    print_results(results)
    
    # Save results
    if args.output_dir:
        save_results(results, args.output_dir)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Video Food Classifier")
    parser.add_argument(
        "--model", type=str, default="best_food_classifier.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="food_video_dataset",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num-frames", type=int, default=8,
        help="Number of frames per video"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--backbone", type=str, default="auto",
        help="Backbone model name (e.g., mobilevit_s, vit_b_16, etc.). Use \"auto\" for auto-detection from checkpoint."
    )
    parser.add_argument(
        "--test-split", type=float, default=0.2,
        help="Test set split ratio (if no dedicated test folder)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for test split"
    )
    parser.add_argument(
        "--num-classes", type=int, default=None,
        help="Number of classes (auto-detected from checkpoint if not specified)"
    )
    parser.add_argument(
        "--classes", type=str, default=None,
        help="Comma-separated class names to filter (e.g., 'healthy,unhealthy')"
    )
    parser.add_argument(
        "--norm-mean", type=str, default="0.485,0.456,0.406",
        help="Comma-separated normalization mean values (must be exactly 3 values)"
    )
    parser.add_argument(
        "--norm-std", type=str, default="0.229,0.224,0.225",
        help="Comma-separated normalization std values (must be exactly 3 values)"
    )
    
    args = parser.parse_args()
    main(args)
