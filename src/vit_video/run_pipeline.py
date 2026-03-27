"""CLI runner for the MobileViT synthetic-data 
  python run_pipeline.py --data-dir ./data/synthetic_food_videos --generate 0 --epochs 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure vit_video is importable from repo root, src/, or src/vit_video/
_script_dir = Path(__file__).resolve().parent
_src = _script_dir.parent
if _src.name == "src" and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
elif _script_dir.name == "vit_video" and str(_script_dir.parent) not in sys.path:
    sys.path.insert(0, str(_script_dir.parent))

from vit_video.utils import get_device
from vit_video.data import DataGenerator, VideoProcessor, build_dataloaders
from vit_video.models import MobileViTModel
from vit_video.engine import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MobileViT synthetic-data pipeline")
    p.add_argument("--data-dir", type=str, default="./data/synthetic_food_videos", help="Path to store synthetic images/videos")
    p.add_argument("--generate", type=int, default=0, choices=[0, 1], help="Whether to run image generation via diffusers (1) or skip (0)")
    p.add_argument("--num-images", type=int, default=20, help="Number of images per class to generate when using diffusers")
    p.add_argument("--frames", type=int, default=16, help="Frames per synthetic video")
    p.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    p.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default 10)")
    p.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    p.add_argument("--out", type=str, default="./models", help="Output path for checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print("Device:", device)

    data_root = Path(args.data_dir)
    data_root.mkdir(parents=True, exist_ok=True)

    prompts = {
        "healthy": "A high-quality photo of a healthy food plate, fresh vegetables and balanced meal",
        "unhealthy": "A high-quality photo of an unhealthy fast food meal, greasy burger and fries",
    }

    generator = DataGenerator(output_dir=data_root)
    if args.generate == 1:
        try:
            print("Running Stable Diffusion generation (may be slow)...")
            generator.generate_with_diffusers(prompts, num_images_per_class=args.num_images, device=device)
        except Exception as exc:
            print("Image generation failed or not available:", exc)
            print("Proceeding with existing images if present.")

    # Convert existing images in data_root into frame folders using VideoProcessor
    vp = VideoProcessor(num_frames=args.frames, out_frame_size=(224, 224))
    for label in prompts.keys():
        label_dir = data_root / label
        if not label_dir.exists():
            continue
        image_files = sorted([p for p in label_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
        for i, img_path in enumerate(image_files):
            frames = vp.image_to_video_frames(img_path)
            out_folder = label_dir / f"video_{i:04d}"
            vp.save_video_frames(frames, out_folder, prefix=f"{label}_{i:04d}")

    # Build dataloaders
    train_loader, val_loader, classes = build_dataloaders(
        data_root, frames_per_video=args.frames, batch_size=args.batch_size
    )

    # Create model
    model = MobileViTModel(num_classes=len(classes), model_name="mobilevit_xxs", pretrained=True)

    trainer = Trainer(model=model, device=device, train_loader=train_loader, val_loader=val_loader, output_path=Path(args.out))
    history = trainer.fit(epochs=min(args.epochs, 10), early_stopping_patience=args.patience)
    print("Done. Training history keys:", list(history.keys()))


if __name__ == "__main__":
    main()
