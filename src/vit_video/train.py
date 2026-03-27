import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

# Ensure vit_video package is importable whether run from repo root, src/, or src/vit_video/
_script_dir = Path(__file__).resolve().parent
_src = _script_dir.parent
if _src.name == "src" and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
elif _script_dir.name == "vit_video" and str(_script_dir.parent) not in sys.path:
    sys.path.insert(0, str(_script_dir.parent))

import torch

from vit_video.utils import print_device_info, parse_normalization_values, get_device
from vit_video.data import build_dataloaders
from vit_video.models import MobileViTModel
from vit_video.engine import Trainer, compute_class_weights_from_dataset


def _auto_select_backbone() -> str:
    """Pick a good default backbone depending on hardware."""
    if torch.cuda.is_available():
        # ViT-B/16 works well on modern GPUs for 224x224.
        return "vit_b_16"
    return "mobilevit_xxs"


def _select_learning_rate(
    lr_candidates: list[float],
    search_epochs: int,
    classes: list[str],
    backbone: str,
    args,
    device: torch.device,
    train_loader,
    val_loader,
    class_weights,
    out_dir: Path,
) -> float:
    """Run a short LR sweep and pick the candidate with lowest validation loss."""
    if not lr_candidates or search_epochs <= 0:
        return args.lr

    print("\nRunning lightweight LR search...")
    best_lr = args.lr
    best_loss = float("inf")

    for lr in lr_candidates:
        print(f"\n[LR Search] Testing lr={lr}")
        candidate_model = MobileViTModel(
            num_classes=len(classes),
            model_name=backbone,
            pretrained=True,
            temporal_pool=args.temporal_pool,
            dropout=args.dropout,
        )
        candidate_trainer = Trainer(
            model=candidate_model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=lr,
            weight_decay=args.weight_decay,
            output_path=out_dir,
            max_grad_norm=args.max_grad_norm,
            class_weights=class_weights,
        )
        history = candidate_trainer.fit(
            epochs=search_epochs,
            early_stopping_patience=max(1, min(args.patience, search_epochs)),
            min_delta=args.min_delta,
            checkpoint_name=f"_lr_search_{lr:.0e}.pth",
            resume_from=None,
        )
        candidate_val_loss = min(history.get("val_loss", [float("inf")]))
        print(f"[LR Search] lr={lr} -> best val_loss={candidate_val_loss:.4f}")
        if candidate_val_loss < best_loss:
            best_loss = candidate_val_loss
            best_lr = lr

    print(f"\nSelected learning rate: {best_lr} (search best val_loss={best_loss:.4f})")
    return best_lr


def main(args):
    print_device_info()
    device = get_device()

    dataset_dir = Path(args.dataset_dir)

    norm_mean, norm_std = parse_normalization_values(args.norm_mean, args.norm_std)

    # On Windows, num_workers > 0 can cause DataLoader pickle errors; use 0.
    import sys
    num_workers = 0 if sys.platform == "win32" else args.num_workers

    # Use pipeline helper to build GPU-optimized dataloaders with augmentation on train.
    train_loader, val_loader, classes = build_dataloaders(
        dataset_root=dataset_dir,
        frames_per_video=args.num_frames,
        batch_size=args.batch_size,
        num_workers=num_workers,
        img_size=args.img_size,
        train_augment=not args.disable_augmentation,
        norm_mean=norm_mean,
        norm_std=norm_std,
        seed=42,
    )

    print(f"Classes: {classes}")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # Warn when class coverage is likely too small for robust generalization.
    class_counts = {cls: 0 for cls in classes}
    for idx in train_loader.dataset.indices:
        _, label = train_loader.dataset.dataset.items[idx]
        class_counts[classes[label]] += 1

    low_data_classes = {k: v for k, v in class_counts.items() if v < args.min_samples_per_class}
    if low_data_classes:
        print("\n[WARN] Low sample count per class detected:")
        for cls_name, count in low_data_classes.items():
            print(f"  - {cls_name}: {count} training samples (< {args.min_samples_per_class})")
        print("       Consider collecting more videos for better recall and stability.")

    # Backbone selection: allow explicit name or 'auto'.
    backbone = args.backbone
    if backbone == "auto":
        backbone = _auto_select_backbone()
    print(f"Backbone: {backbone}")

    class_weights = None
    if args.class_weighting:
        class_weights = compute_class_weights_from_dataset(train_loader.dataset, len(classes))
        print(f"Class weights enabled: {class_weights.tolist()}")

    # High-quality training loop via Trainer (AMP, AdamW, grad clipping).
    out_path = Path(args.output_model).resolve()
    lr_candidates = [
        float(x.strip()) for x in args.lr_candidates.split(",") if x.strip()
    ] if args.lr_candidates else []
    selected_lr = _select_learning_rate(
        lr_candidates=lr_candidates,
        search_epochs=args.hparam_search_epochs,
        classes=classes,
        backbone=backbone,
        args=args,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        out_dir=out_path.parent,
    )

    model = MobileViTModel(
        num_classes=len(classes),
        model_name=backbone,
        pretrained=True,
        temporal_pool=args.temporal_pool,
        dropout=args.dropout,
    )

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=selected_lr,
        weight_decay=args.weight_decay,
        output_path=out_path.parent,
        max_grad_norm=args.max_grad_norm,
        class_weights=class_weights,
    )

    resume_from = Path(args.resume_from) if args.resume_from else None

    print(f"\nStarting training for up to {args.epochs} epochs...")
    print("=" * 60)

    history = trainer.fit(
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        min_delta=args.min_delta,
        checkpoint_name=out_path.name,
        resume_from=resume_from,
    )

    # Compute best validation metrics from history.
    best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
    best_idx = history["val_loss"].index(best_val_loss) if history["val_loss"] else -1
    best_val_acc = history["val_acc"][best_idx] if best_idx >= 0 else 0.0

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save history next to the model checkpoint for later analysis.
    history_path = out_path.with_name(out_path.stem + "_history.json")
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    metrics_path = out_path.with_name(out_path.stem + "_training_metrics.json")
    training_metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(out_path),
        "history_path": str(history_path),
        "dataset_dir": str(dataset_dir.resolve()),
        "backbone": backbone,
        "classes": classes,
        "num_classes": len(classes),
        "num_frames": args.num_frames,
        "img_size": args.img_size,
        "temporal_pool": args.temporal_pool,
        "class_weighting": args.class_weighting,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "normalization": {"mean": norm_mean, "std": norm_std},
        "lr": selected_lr,
        "lr_search": {
            "enabled": args.hparam_search_epochs > 0,
            "epochs": args.hparam_search_epochs,
            "candidates": lr_candidates,
        },
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.get("train_loss", [])),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(training_metrics, f, indent=2)
    print(f"Training metrics saved to {metrics_path}")


if __name__ == "__main__":
    _default_dataset = Path(__file__).resolve().parent / "food_data" / "frames"

    parser = argparse.ArgumentParser(description="Train Video Food Classifier.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(_default_dataset),
        help="Path to dataset directory (class subfolders with videos or frame folders).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout applied on pooled video features before classifier.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of frames to sample from each video.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image size (height and width).",
    )
    parser.add_argument(
        "--disable-augmentation",
        action="store_true",
        help="Disable training-time data augmentation.",
    )
    parser.add_argument(
        "--class-weighting",
        action="store_true",
        help="Enable inverse-frequency class weighting for imbalanced datasets.",
    )
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=50,
        help="Warn if any class has fewer than this many training samples.",
    )
    parser.add_argument(
        "--temporal-pool",
        type=str,
        default="avg",
        choices=["avg", "max", "conv1d"],
        help="Temporal aggregation strategy over frame embeddings.",
    )
    parser.add_argument(
        "--norm-mean",
        type=str,
        default="0.485,0.456,0.406",
        help="Comma-separated normalization mean (3 values).",
    )
    parser.add_argument(
        "--norm-std",
        type=str,
        default="0.229,0.224,0.225",
        help="Comma-separated normalization std (3 values).",
    )
    parser.add_argument(
        "--hparam-search-epochs",
        type=int,
        default=0,
        help="If > 0, run a quick LR search for this many epochs per candidate.",
    )
    parser.add_argument(
        "--lr-candidates",
        type=str,
        default="1e-5,3e-5,1e-4,3e-4",
        help="Comma-separated learning-rate candidates for optional quick search.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs without val loss improvement).",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum decrease in val loss to count as improvement.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="auto",
        help="Backbone architecture (e.g., auto, mobilevit_s, mobilevit_xs, mobilevit_xxs, vit_b_16).",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="best_food_classifier.pth",
        help="Output path for trained model checkpoint.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Optional checkpoint path to resume fine-tuning from.",
    )

    args = parser.parse_args()
    main(args)
