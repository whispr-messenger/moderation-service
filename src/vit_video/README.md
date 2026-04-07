# Video Food Classifier (ViT)

Video classification using Vision Transformers with temporal pooling. Classifies food as **healthy** or **unhealthy** from video frames.

Supports 9 backbones (ViT-B/16, ViT-B/32, ViT-L/16, ViT-L/32, ViT-H/14, MobileViT-XXS, and more via timm), 3 temporal pooling modes (avg, max, conv1d), multi-format mobile export, Hugging Face upload, and real-time webcam inference.

## Pipeline

```
generatedata.py > train.py > test.py > export_mobile.py > upload_hf.py
```

Or run everything at once:

```powershell
python run_pipeline.py --class-weighting --epochs 10
```

The pipeline automatically skips data download if frames already exist in the dataset directory.

## Setup

```powershell
pip install -r requirements.txt

# GPU (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Key dependencies

- `torch`, `torchvision`, `timm` — model training
- `yt-dlp`, `imageio-ffmpeg` — YouTube video download + frame extraction
- `opencv-python` — frame processing and webcam inference
- `scikit-learn` — evaluation metrics and k-fold CV
- `huggingface-hub` — model upload (optional)
- `onnx`, `onnxruntime` — ONNX export (optional)

## Usage

All commands from `src/vit_video/` directory.

By default, videos and frames are stored under **`food_data/` inside this folder** (`food_data/raw_videos/`, `food_data/frames/`, and `food_data/video_split_manifest.json`). Override with `--dataset-dir` if needed.

### Full pipeline (one command)

```powershell
# Download data + train + test + export (skips download if data exists)
python run_pipeline.py --class-weighting

# Skip export step
python run_pipeline.py --skip-export

# Custom dataset location
python run_pipeline.py --dataset-dir D:\datasets\my_food_data --class-weighting

# Upload to Hugging Face after export
python run_pipeline.py --class-weighting --hf-repo-id your-username/food-classifier

# Regenerate train/val/test splits (e.g. after adding new videos)
python run_pipeline.py --regenerate-splits --skip-download
```

### Individual steps

```powershell
# 1. Download data (defaults to ./food_data)
python generatedata.py --videos-per-keyword 10

# 2. Train (defaults to ./food_data/frames)
python train.py --epochs 10 --batch-size 8 --class-weighting

# 3. Test
python test.py --model models/best_food_classifier.pth

# 4. Validate (leakage audit + k-fold CV + external test)
python validate_model.py --model models/best_food_classifier.pth

# 5. Export to mobile
python export_mobile.py --model models/best_food_classifier.pth --format torchscript onnx

# 6. Upload to Hugging Face
python upload_hf.py --repo-id your-username/food-classifier --export-dir exported_models

# 7. Inference
python inference.py --video path/to/video.mp4 --model models/best_food_classifier.pth
python inference.py --webcam --model models/best_food_classifier.pth
```

If you previously used a repo-root `food_data` folder, move its contents into `src/vit_video/food_data/` (same `frames/` and `raw_videos/` layout).

**`food_video_dataset_images/` (or any `*_images/` tree):** These are flat image folders, not the training layout. Training and splits expect **`food_data/frames/<class>/`** with frame files named like `*_frame_*.jpg` (see `VideoDataset`). You can keep `*_images/` for reference (they match `.gitignore`), copy or symlink images into `food_data/frames/<class>/` if you want them in the model, or ignore them.

**Empty `food_data/frames`:** The pipeline only skips download when there is at least one `.jpg`/`.png`/`.webp` under `frames/<class>/`. Empty folders trigger a fresh download (unless `--skip-download`). For `validate_model.py`, pass `--dataset-dir` pointing at `food_data/frames`.

## Training Defaults

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--backbone` | auto | ViT-B/16 on GPU, MobileViT-XXS on CPU |
| `--epochs` | 10 | Training epochs |
| `--batch-size` | 8 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--dropout` | 0.3 | Regularization |
| `--weight-decay` | 1e-3 | L2 regularization |
| `--patience` | 5 | Early stopping patience |
| `--temporal-pool` | avg | Temporal pooling (avg, max, conv1d) |
| `--class-weighting` | off | Enable for imbalanced datasets |
| `--num-frames` | 8 | Frames sampled per video |
| `--img-size` | 224 | Input image size |

### Train/val/test split

Video-level splitting is used by default (70/15/15) to prevent data leakage. A split manifest (`video_split_manifest.json`) is created automatically and reused across train/test/export steps.

| Parameter | Default |
|-----------|---------|
| `--train-ratio` | 0.7 |
| `--val-ratio` | 0.15 |
| `--test-ratio` | 0.15 |
| `--split-seed` | 42 |

## Export Formats

| Format | Flag | Output |
|--------|------|--------|
| TorchScript | `torchscript` | `.pt` (mobile-optimized) |
| ONNX | `onnx` | `.onnx` (dynamic batch) |
| CoreML | `coreml` | `.mlpackage` (iOS 15+, requires `coremltools`) |
| TFLite | `tflite` | `.tflite` (via `ai-edge-torch` or ONNX->TF->TFLite) |

Export also generates a `model_card.json` with model metadata, evaluation results, and training metrics.

## Project Structure

```
src/vit_video/
├── _bootstrap.py           # Shared path setup (adds src/ to sys.path)
├── _build_colab_notebook.py # Generate self-contained Colab notebook
├── run_pipeline.py          # Full pipeline: download > train > test > export > upload
├── generatedata.py          # YouTube download + frame extraction (yt-dlp)
├── train.py                 # Training with optional LR search
├── test.py                  # Evaluation on held-out test split
├── validate_model.py        # Leakage audit, k-fold CV, external data testing
├── inference.py             # Video file / webcam inference
├── export_mobile.py         # TorchScript / ONNX / CoreML / TFLite export
├── upload_hf.py             # Upload exported models to Hugging Face Hub
├── data/
│   ├── dataset.py           # VideoDataset + video-level dataloaders
│   └── splits.py            # Train/val/test manifest (video-level splitting)
├── engine/
│   └── trainer.py           # Trainer (AMP, grad clipping, early stopping, class weights)
├── models/
│   └── vit.py               # MobileViTModel (torchvision + timm backbones, 3 temporal pools)
└── utils/
    ├── hardware.py          # Device detection (CUDA/MPS/CPU)
    ├── data_utils.py        # Transforms and normalization
    ├── model_utils.py       # Checkpoint loading + backbone detection
    ├── video.py             # Shared frame extraction from video files
    └── ytdlp_helpers.py     # yt-dlp download options
```
