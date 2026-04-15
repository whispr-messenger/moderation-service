# Moderation Service

This repository contains three complementary food-moderation models, each with its own training pipeline:

| Model | Framework | Role |
|---|---|---|
| [EfficientNet-Lite](src/efficientnet_lite_gpu/) | TensorFlow / Keras | Higher-accuracy image classifier (mobile-friendly) |
| [MobileNetV3-Small](src/mobilenet_v3_small/) | TensorFlow / Keras | Lightweight image classifier (sub-10 ms on mobile) |
| [ViT-Video](src/vit_video/) | PyTorch | Video / multi-frame classifier (ViT-B/16 or MobileViT-XXS + BiLSTM) |

All three are trained on the same canonical HF dataset (`maia2000/food-classifier-dataset`) — see [`DATASET.md`](DATASET.md) for the data pipelines.

## Quick Start

1. Clone the repository and enter the project:

```bash
git clone https://github.com/whispr-messenger/moderation-service.git
cd moderation-service
```

2. Switch to the active module directory:

```bash
cd src/efficientnet_lite_gpu
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run pipeline actions:

```bash
python main.py --action train
python main.py --action eval
python main.py --action test
```

### MobileNetV3-Small (image classifier, lightweight)

```bash
cd src/mobilenet_v3_small
pip install -r requirements.txt

# Train (downloads/uses HF dataset, runs two-stage training: frozen backbone + fine-tune)
python main.py --action train

# Evaluate the trained Keras model on validation/images/
python main.py --action eval

# Evaluate the TFLite build (after running convert_to_tflite.py)
python convert_to_tflite.py --quantize
python main.py --action eval_tflite

# Smoke test (hardware + dataset sanity checks)
python main.py --action test
```

Full Colab training with HF push + TFLite + TFJS exports: open [`src/mobilenet_v3_small/mobilenet_colab.ipynb`](src/mobilenet_v3_small/mobilenet_colab.ipynb). See [`src/mobilenet_v3_small/TECHNICAL.md`](src/mobilenet_v3_small/TECHNICAL.md) for hyperparameters and the mobile-export details.

### ViT-Video (PyTorch, video classifier)

```bash
cd src/vit_video
pip install -r requirements.txt

# 1. Fetch + split videos -> extract frames (idempotent: skips if data already present)
python run_pipeline.py --videos-per-keyword 15 --max-frames 60 --frame-size 224

# 2. Train (auto-picks ViT-B/16 on CUDA, MobileViT-XXS on CPU; BiLSTM temporal head)
python train.py --frames-root food_data/frames --epochs 20 --batch-size 8 --lr 3e-5 --patience 7

# 3. Resume from a checkpoint
python train.py --resume models/best_food_classifier.pth --epochs 10

# 4. Evaluate held-out test split
python validate_model.py --model models/best_food_classifier.pth

# 5. Export for deployment (TorchScript + ONNX + TFLite; adds .ptl Lite Interpreter build for mobile)
python export_mobile.py --model models/best_food_classifier.pth --format torchscript onnx tflite --quantize

# Inference on a single video
python inference.py --model models/best_food_classifier.pth --video path/to/clip.mp4
```

Full Colab training with HF dataset download + Drive checkpointing: open [`src/vit_video/vit_video.ipynb`](src/vit_video/vit_video.ipynb) (multi-class) or [`src/vit_video/vit_video_binary.ipynb`](src/vit_video/vit_video_binary.ipynb) (binary). See [`src/vit_video/TECHNICAL.md`](src/vit_video/TECHNICAL.md) for the model spec, export formats, and the TFJS incompatibility notes.

## Dataset Fetch (Optional)

If you only need the dataset fetch tool dependencies:

```bash
pip install -r requirements-fetch-only.txt
```

Dry-run fetch command:

```bash
python -m tools.fetch_google_dataset --dry-run
```

Windows helpers:

- `run_fetch_google_dataset.bat`
- `run_fetch_google_dataset_dry_run.bat`

## Documentation

- Project index: [`documentation/PROJECT_INDEX.md`](documentation/PROJECT_INDEX.md)
- Dataset pipelines: [`DATASET.md`](DATASET.md)
- Model specs:
  - [`src/efficientnet_lite_gpu/TECHNICAL.md`](src/efficientnet_lite_gpu/TECHNICAL.md)
  - [`src/mobilenet_v3_small/TECHNICAL.md`](src/mobilenet_v3_small/TECHNICAL.md)
  - [`src/vit_video/TECHNICAL.md`](src/vit_video/TECHNICAL.md)
- Module READMEs: [`src/efficientnet_lite_gpu/README.md`](src/efficientnet_lite_gpu/README.md), [`src/vit_video/README.md`](src/vit_video/README.md)
- Windows long path setup: [`documentation/WINDOWS_LONG_PATHS.md`](documentation/WINDOWS_LONG_PATHS.md)
- Architecture reference: [`documentation/1_architecture/1_system_design.md`](documentation/1_architecture/1_system_design.md)
