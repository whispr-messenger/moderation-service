# Video Classification with Vision Transformer

A lightweight video classifier using Vision Transformers (ViT) for multi-class classification.

## Features

- **ViT Backbone**: torchvision ViT (vit_b_16, vit_l_16) or timm MobileViT
- **Video Processing**: Automatic frame extraction and temporal pooling
- **Temporal Modeling Option**: `avg`, `max`, or lightweight `conv1d` temporal aggregation
- **Multi-format Export**: TorchScript, ONNX, CoreML, TFLite
- **Real-time Inference**: Video file or webcam
- **GPU Optimized**: CUDA-ready with automatic device detection
- **Imbalance Handling**: Optional class-weighted loss
- **Configurable Preprocessing**: Custom normalization mean/std

## Running scripts

All commands below assume you are in the **project root** (`moderation-service`). Launcher scripts in the root (`run_train.py`, `run_test.py`, etc.) add the `vit_video` package to the path automatically. You can also run from `src` with `python -m vit_video.train` or from `src/vit_video` with `python train.py` (same for test, inference, export_mobile, generatedata).

## Quick Start

### 1. Setup Dataset

Create a folder structure:
```
food_video_dataset/
├── healthy/
│   ├── video_1.mp4
│   └── video_2.mp4
├── unhealthy/
│   ├── video_3.mp4
│   └── video_4.mp4
```

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

For good accuracy/recall, target **at least 50–100 videos per class**.

```powershell
# From project root: download healthy/unhealthy food videos and extract frames
python run_generatedata.py --dataset-dir food_data --videos-per-keyword 10 --max-frames-per-video 80

# This will create: food_data/raw_videos/* and food_data/frames/*
# Train using the frames directory as dataset root:
python run_train.py --dataset-dir food_data/frames --epochs 15 --batch-size 16
```

### 2. Train

```powershell
# From project root (recommended)
python run_train.py --dataset-dir food_video_dataset --epochs 10 --batch-size 16

# Custom backbone
python run_train.py --dataset-dir food_video_dataset --epochs 10 --backbone vit_b_16

# MobileViT for mobile deployment
python run_train.py --dataset-dir food_video_dataset --epochs 10 --backbone mobilevit_s

# Fine-tune from checkpoint
python run_train.py --dataset-dir food_video_dataset --epochs 5 --resume-from best_food_classifier.pth
```

### 3. Test

```powershell
python run_test.py --model best_food_classifier.pth --dataset-dir food_video_dataset
```

### 4. Inference

**Single video:**
```powershell
python run_inference.py --video path/to/video.mp4 --model best_food_classifier.pth --num-classes 2
```

**Webcam (real-time):**
```powershell
python run_inference.py --webcam --model best_food_classifier.pth --num-classes 2
```

### 5. Export to Mobile

```powershell
# TorchScript
python run_export_mobile.py --model best_food_classifier.pth --num-classes 2 --format torchscript

# ONNX
python run_export_mobile.py --model best_food_classifier.pth --num-classes 2 --format onnx

# All formats
python run_export_mobile.py --model best_food_classifier.pth --num-classes 2 --format all

# Include evaluation/training metadata in model card
python run_export_mobile.py --model best_food_classifier.pth --num-classes 2 --format all `
    --eval-results src/vit_video/results/test_results.json `
    --training-metrics best_food_classifier_training_metrics.json
```

## Models

**Torchvision ViT (ImageNet pretrained):**
- `vit_b_16` (768 features) - Most common
- `vit_b_32` (768 features) - Faster
- `vit_l_16` (1024 features) - Higher capacity
- `vit_l_32` (1024 features)
- `vit_h_14` (1280 features)

**Timm MobileViT (mobile-friendly):**
- `mobilevit_s` (640 features) - Balanced
- `mobilevit_xs` (384 features) - Lightweight
- `mobilevit_xxs` (320 features) - Very lightweight

## Key Arguments

### Train
```
--epochs           Number of epochs (default: 10)
--batch-size       Batch size (default: 2)
--lr               Learning rate (default: 1e-4)
--num-frames       Frames per video (default: 8)
--backbone         Model (default: mobilevit_s)
--temporal-pool    avg, max, conv1d (default: avg)
--class-weighting  Enable inverse-frequency class weighting
--disable-augmentation Disable train-time augmentation
--norm-mean        Comma-separated normalization mean
--norm-std         Comma-separated normalization std
--hparam-search-epochs Quick LR search epochs per candidate (0 disables)
--lr-candidates    LR candidates for quick search
```

### Test
```
--model            Checkpoint (default: best_food_classifier.pth)
--batch-size       Batch size (default: 4)
```

### Inference
```
--video            Path to video file
--webcam           Use webcam (flag)
--model            Checkpoint (default: best_food_classifier.pth)
--num-classes      Number of classes (default: 2)
--num-frames       Frames per video (default: 8)
--norm-mean        Normalization mean used during inference
--norm-std         Normalization std used during inference
--max-webcam-read-failures Consecutive webcam read failures before stop
```

### Export
```
--model            Checkpoint
--format           torchscript, onnx, coreml, tflite, all
--num-classes      Number of classes
--eval-results     Optional path to test_results.json for model card
--training-metrics Optional path to training metrics json for model card
--norm-mean        Normalization mean for model card/preprocessing metadata
--norm-std         Normalization std for model card/preprocessing metadata
```

## Example: Food Classification

From project root:

```powershell
# Train
python run_train.py --dataset-dir food_dataset --epochs 20 --batch-size 8

# Test
python run_test.py --model best_food_classifier.pth --dataset-dir food_dataset

# Export
python run_export_mobile.py --model best_food_classifier.pth --num-classes 2

# Inference
python run_inference.py --video sample.mp4 --model best_food_classifier.pth --num-classes 2
```

## Architecture

```
Video (8 frames, 224x224)
    ↓
ViT/MobileViT Backbone
    ↓
Temporal Average Pooling
    ↓
Linear Classifier
    ↓
Output Logits
```

