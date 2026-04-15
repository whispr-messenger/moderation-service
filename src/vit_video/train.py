from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

try:
    import _bootstrap; _bootstrap.setup()
except ImportError:
    pass  # sys.path already configured (e.g. Colab notebook)

from vit_video.paths import DEFAULT_FRAMES_DIR, PACKAGE_ROOT
from vit_video.utils import print_device_info, parse_normalization_values, get_device
from vit_video.data import build_dataloaders
from vit_video.models import MobileViTModel
from vit_video.engine import Trainer, compute_class_weights_from_dataset


def _auto_select_backbone() -> str:
    return "vit_b_16" if torch.cuda.is_available() else "mobilevit_xxs"


def _select_learning_rate(
    lr_candidates: list[float], search_epochs: int,
    classes: list[str], backbone: str, args,
    device: torch.device, train_loader, val_loader, class_weights, out_dir: Path,
) -> float:
    if not lr_candidates or search_epochs <= 0:
        return args.lr

    print("\nRunning lightweight LR search...")
    best_lr = args.lr
    best_loss = float("inf")

    for lr in lr_candidates:
        print(f"\n[LR Search] Testing lr={lr}")
        model = MobileViTModel(
            num_classes=len(classes), model_name=backbone,
            pretrained=True, temporal_pool=args.temporal_pool, dropout=args.dropout,
        )
        trainer = Trainer(
            model=model, device=device, train_loader=train_loader, val_loader=val_loader,
            lr=lr, weight_decay=args.weight_decay, output_path=out_dir,
            max_grad_norm=args.max_grad_norm, class_weights=class_weights,
        )
        history = trainer.fit(
            epochs=search_epochs,
            early_stopping_patience=max(1, min(args.patience, search_epochs)),
            min_delta=args.min_delta,
            checkpoint_name=f"_lr_search_{lr:.0e}.pth",
        )
        val_loss = min(history.get("val_loss", [float("inf")]))
        print(f"[LR Search] lr={lr} -> best val_loss={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_lr = lr

    print(f"\nSelected learning rate: {best_lr} (val_loss={best_loss:.4f})")
    return best_lr


def main(args):
    print_device_info()
    device = get_device()

    dataset_dir = Path(args.dataset_dir)
    norm_mean, norm_std = parse_normalization_values(args.norm_mean, args.norm_std)
    num_workers = min(args.num_workers, 2) if sys.platform == "win32" else args.num_workers

    train_loader, val_loader, classes = build_dataloaders(
        dataset_root=dataset_dir,
        frames_per_video=args.num_frames,
        batch_size=args.batch_size,
        num_workers=num_workers,
        img_size=args.img_size,
        train_augment=not args.disable_augmentation,
        norm_mean=norm_mean, norm_std=norm_std,
        seed=args.split_seed,
        split_manifest=args.split_manifest or None,
        auto_write_manifest=not args.no_auto_split_manifest,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    print(f"Classes: {classes}")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    class_counts = {cls: 0 for cls in classes}
    for idx in train_loader.dataset.indices:
        _, label = train_loader.dataset.dataset.items[idx]
        class_counts[classes[label]] += 1

    low_data = {k: v for k, v in class_counts.items() if v < args.min_samples_per_class}
    if low_data:
        print("\n[WARN] Low sample count per class:")
        for cls_name, count in low_data.items():
            print(f"  - {cls_name}: {count} (< {args.min_samples_per_class})")

    backbone = args.backbone
    if backbone == "auto":
        backbone = _auto_select_backbone()
    print(f"Backbone: {backbone}")

    class_weights = None
    if args.class_weighting:
        class_weights = compute_class_weights_from_dataset(train_loader.dataset, len(classes))
        print(f"Class weights: {class_weights.tolist()}")

    out_path = Path(args.output_model).resolve()
    lr_candidates = [
        float(x.strip()) for x in args.lr_candidates.split(",") if x.strip()
    ] if args.lr_candidates else []

    selected_lr = _select_learning_rate(
        lr_candidates=lr_candidates,
        search_epochs=args.hparam_search_epochs,
        classes=classes, backbone=backbone, args=args,
        device=device, train_loader=train_loader, val_loader=val_loader,
        class_weights=class_weights, out_dir=out_path.parent,
    )

    model = MobileViTModel(
        num_classes=len(classes), model_name=backbone,
        pretrained=True, temporal_pool=args.temporal_pool, dropout=args.dropout,
        freeze_backbone=getattr(args, "freeze_backbone", False),
    )
    trainer = Trainer(
        model=model, device=device, train_loader=train_loader, val_loader=val_loader,
        lr=selected_lr, weight_decay=args.weight_decay, output_path=out_path.parent,
        max_grad_norm=args.max_grad_norm, class_weights=class_weights,
    )

    resume_from = Path(args.resume_from) if args.resume_from else None
    print(f"\nStarting training for up to {args.epochs} epochs...")
    print("=" * 60)
    if device.type == "cpu":
        print(
            "[INFO] On CPU the first training steps can take several minutes "
            "(model + data load); the progress bar may stay at 0% briefly.\n"
        )

    drive_checkpoint_dir = getattr(args, "drive_checkpoint_dir", None)
    history = trainer.fit(
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        min_delta=args.min_delta,
        checkpoint_name=out_path.name,
        resume_from=resume_from,
        drive_checkpoint_dir=drive_checkpoint_dir,
    )

    best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
    best_idx = history["val_loss"].index(best_val_loss) if history["val_loss"] else -1
    best_val_acc = history["val_acc"][best_idx] if 0 <= best_idx < len(history["val_acc"]) else 0.0

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")

    history_path = out_path.with_name(out_path.stem + "_history.json")
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    training_metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(out_path),
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
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.get("train_loss", [])),
    }
    metrics_path = out_path.with_name(out_path.stem + "_training_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(training_metrics, f, indent=2)
    print(f"Training metrics saved to {metrics_path}")

    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Video Food Classifier.")
    parser.add_argument(
        "--dataset-dir", type=str, default=str(DEFAULT_FRAMES_DIR),
        help="Frames root (default: vit_video/food_data/frames)",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--class-weighting", action="store_true", default=True,
                        help="Enable class weighting to handle imbalanced data (default: on)")
    parser.add_argument("--no-class-weighting", dest="class_weighting", action="store_false",
                        help="Disable class weighting")
    parser.add_argument("--min-samples-per-class", type=int, default=50)
    parser.add_argument("--temporal-pool", type=str, default="lstm", choices=["avg", "max", "conv1d", "lstm"])
    parser.add_argument("--norm-mean", type=str, default="0.485,0.456,0.406")
    parser.add_argument("--norm-std", type=str, default="0.229,0.224,0.225")
    parser.add_argument("--hparam-search-epochs", type=int, default=0)
    parser.add_argument("--lr-candidates", type=str, default="5e-6,1e-5,3e-5,5e-5,1e-4")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=5e-5)
    parser.add_argument("--backbone", type=str, default="auto")
    parser.add_argument(
        "--output-model", type=str,
        default=str(PACKAGE_ROOT / "models" / "best_food_classifier.pth"),
    )
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--split-manifest", type=str, default="")
    parser.add_argument("--no-auto-split-manifest", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
