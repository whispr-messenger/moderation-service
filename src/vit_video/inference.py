from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import _bootstrap; _bootstrap.setup()

from vit_video.paths import PACKAGE_ROOT
from vit_video.utils import (
    print_device_info, parse_normalization_values, get_device,
    build_transform, load_model_from_checkpoint,
)


def _sample_video_frames(cap: cv2.VideoCapture, num_frames: int, img_size: int):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return None, 0

    if total_frames < num_frames:
        frame_indices = np.random.choice(total_frames, num_frames, replace=True)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames, missing = [], 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frames.append(frame)
        else:
            missing += 1
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
    return frames, missing


def predict_video(model, video_path, device, transform, num_frames=8, img_size=224, num_classes=2):
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    frames, missing = _sample_video_frames(cap, num_frames=num_frames, img_size=img_size)
    cap.release()

    if frames is None:
        raise ValueError(f"Could not read video: {video_path}")
    if missing > 0:
        print(f"Warning: {missing}/{num_frames} frames could not be decoded and were backfilled.")

    video_tensor = torch.stack([transform(f) for f in frames]).unsqueeze(0).to(device)

    with torch.no_grad():
        start = time.perf_counter()
        outputs = model(video_tensor)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    label_names = [f"Class {i}" for i in range(num_classes)]
    return label_names[predicted_class], confidence, elapsed_ms


def webcam_inference(model, device, transform, num_frames=8, img_size=224,
                     num_classes=2, max_read_failures=30):
    model.eval()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    frame_buffer: list = []
    label_names = [f"Class {i}" for i in range(num_classes)]
    latency_samples_ms: list[float] = []
    read_failures = 0

    print("Starting webcam... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            read_failures += 1
            print("Warning: Webcam frame read failed. Retrying...")
            if read_failures >= max_read_failures:
                print("Error: Webcam appears disconnected. Stopping inference.")
                break
            time.sleep(0.1)
            continue
        read_failures = 0

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (img_size, img_size))
        frame_buffer.append(frame_resized)
        if len(frame_buffer) > num_frames:
            frame_buffer.pop(0)

        if len(frame_buffer) == num_frames:
            video_tensor = torch.stack([transform(f) for f in frame_buffer]).unsqueeze(0).to(device)

            with torch.no_grad():
                start = time.perf_counter()
                outputs = model(video_tensor)
                latency_ms = (time.perf_counter() - start) * 1000.0
                latency_samples_ms.append(latency_ms)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
            text = f"{label_names[predicted_class]}: {confidence*100:.1f}% | {latency_ms:.1f}ms ({fps:.1f} FPS)"
            color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(frame, "Collecting frames...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Food Classifier (Press Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if latency_samples_ms:
        avg_ms = float(np.mean(latency_samples_ms))
        p95_ms = float(np.percentile(latency_samples_ms, 95))
        print(f"\nInference performance: {len(latency_samples_ms)} frames, "
              f"avg {avg_ms:.2f}ms, p95 {p95_ms:.2f}ms, "
              f"avg FPS {1000.0/avg_ms:.2f}")


def main(args):
    device = get_device()
    print_device_info()

    norm_mean, norm_std = parse_normalization_values(args.norm_mean, args.norm_std)
    transform = build_transform(mean=norm_mean, std=norm_std)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return

    model = load_model_from_checkpoint(model_path, args.num_classes, args.backbone, device)

    if args.webcam:
        webcam_inference(model, device, transform=transform, num_frames=args.num_frames,
                         img_size=args.img_size, num_classes=args.num_classes,
                         max_read_failures=args.max_webcam_read_failures)
    else:
        if not args.video:
            print("Error: Please provide --video path or use --webcam flag")
            return

        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            return

        print(f"\nProcessing video: {video_path}")
        prediction, confidence, latency_ms = predict_video(
            model, video_path, device, transform=transform,
            num_frames=args.num_frames, img_size=args.img_size,
            num_classes=args.num_classes,
        )

        print(f"\n{'='*60}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence*100:.2f}%")
        print(f"Inference latency: {latency_ms:.2f} ms")
        if latency_ms > 0:
            print(f"Approx FPS: {1000.0/latency_ms:.2f}")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Video Food Classifier")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument(
        "--model", type=str,
        default=str(PACKAGE_ROOT / "models" / "best_food_classifier.pth"),
    )
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--backbone", type=str, default="auto")
    parser.add_argument("--norm-mean", type=str, default="0.485,0.456,0.406")
    parser.add_argument("--norm-std", type=str, default="0.229,0.224,0.225")
    parser.add_argument("--max-webcam-read-failures", type=int, default=30)

    args = parser.parse_args()
    main(args)
