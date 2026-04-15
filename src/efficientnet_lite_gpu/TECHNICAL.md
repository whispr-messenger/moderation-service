# EfficientNet-B0 Food Classifier -- Technical Documentation

## Overview

Image classification model for food content moderation. Classifies a single image as **healthy**, **unhealthy**, or **not_food** (3-class softmax) using an EfficientNet-B0 backbone with frozen-backbone transfer learning.

| | |
|---|---|
| **Framework** | TensorFlow / Keras |
| **Backbone** | EfficientNetB0 (ImageNet-pretrained, frozen) |
| **Head** | GlobalAveragePooling2D → Dropout(0.2) → Dense(N, softmax) |
| **Input** | RGB image, 224×224 |
| **Output** | Softmax over `{healthy, unhealthy, not_food}` (or sigmoid scalar if only 2 classes are present) |
| **Mobile format** | TFLite float + int8-quantized |
| **HF model repo** | [`maia2000/efficientnet-food-binary`](https://huggingface.co/maia2000/efficientnet-food-binary) |
| **HF dataset repo** | [`maia2000/food-classifier-dataset`](https://huggingface.co/datasets/maia2000/food-classifier-dataset) |

---

## 1. Why EfficientNet-B0

- ~4 M parameters — smallest EfficientNet variant, mobile-friendly
- 77.1 % ImageNet top-1 with ~7× fewer params than ResNet-50
- Native TFLite conversion (float and int8 quantized)
- Strong ImageNet features transfer cleanly to food imagery

Compared to the ViT model in this repo: pure CNN, single-image (no temporal modeling), simpler and faster.

---

## 2. Classes

The number of classes is **detected at training time** from the folder layout under `/content/binary/`:

- **2 classes** (`healthy`, `unhealthy`) → sigmoid head + `binary_crossentropy`
- **3 classes** (`healthy`, `unhealthy`, `not_food`) → softmax head + `sparse_categorical_crossentropy`

The 3-class variant is produced when the Food-101 + Imagenette fallback runs (see [DATASET.md](../../DATASET.md)).

---

## 3. Training pipeline

The Colab notebook `efficientnet_colab.ipynb` is generated from `scripts/build_colab_notebooks.py` and is the canonical training pipeline. It is **disconnect-resilient**: every cell is self-contained, checkpoints are mirrored to Google Drive, and `model.fit(initial_epoch=N)` resumes from the last completed epoch.

### Cells

1. **Setup** — install `huggingface_hub` + `hf_transfer`, login, mount Drive at `/content/drive/MyDrive/whispr-checkpoints/efficientnet/`.
2. **Download** — try HF dataset; on <1000 frames fall back to Food-101 + Imagenette and write into `/content/frames/{healthy,unhealthy,not_food}/`.
3. **Organize** — symlink into `/content/binary/{class}/` (drops empty class dirs).
4. **Train** — see hyperparameters below.
5. **Push** — upload `model.h5`, `README.md`, `metrics.json` to the HF model repo; upload `frames/` to the HF dataset repo (with `traceback.print_exc()` on failure).
6. **Mobile export** — convert to `model.tflite` and `model_quantized.tflite`, generate `confusion_matrix.png`, upload all three to the model repo.

### Hyperparameters

| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Optimizer | Adam, lr=1e-3 |
| Loss | `binary_crossentropy` (2 cls) / `sparse_categorical_crossentropy` (3 cls) |
| Epochs | 10 (with `EarlyStopping(patience=3, restore_best_weights=True)`) |
| Augmentation | RandomFlip(horizontal) + RandomRotation(0.05) + RandomZoom(0.1) |
| Preprocessing | `efficientnet.preprocess_input` (ImageNet mean/std) |
| Validation split | 20 %, fixed `seed=42` |

### Resume logic

Per-epoch persistence callback writes `model.h5` + `train_state.json` (`{last_epoch, history}`) locally and to Drive. On rerun the train cell:

1. Pulls the checkpoint back from Drive if local is missing.
2. Reads `last_epoch` from `train_state.json`.
3. If `last_epoch >= TOTAL_EPOCHS` → skip training, just load the model.
4. Otherwise resume from `initial_epoch=last_epoch`, history is concatenated.

---

## 4. Mobile export

Cell 6 produces:

- `model.tflite` — float TFLite (~16 MB)
- `model_quantized.tflite` — int8-quantized TFLite (~4 MB), via `Optimize.DEFAULT`
- `confusion_matrix.png` — 5×4 figure on the validation split; the same cell prints `raw sigmoid probs`, `label counts`, `pred counts`, `accuracy`, and a full sklearn `classification_report` so you can spot a collapsed model immediately.

All three are uploaded to the HF model repo with 3-attempt retry.

---

## 5. App integration (`app_mockup_demo/app.py`)

`predict_efficientnet` loads `model.tflite` from `app_mockup_demo/models/` (downloaded via the HF panel in the sidebar). The function tolerates two output shapes:

- **Single sigmoid scalar** (binary classifier) → expanded to `[1-p, p]` and labelled `["healthy", "unhealthy"]` (alphabetical, matching `image_dataset_from_directory`).
- **Softmax vector** of length N → standard `argmax`.

Class names are taken from a sibling `labels.json` if present, otherwise inferred from the output dimension via `_classes_for_n` (n=1 and n=2 both → binary; n=16 → legacy 16-class).

> ⚠️ A 2-class model has **no `not_food` head**. If you need the app to reject empty rooms / faces, train the 3-class variant by enabling the Food-101 + Imagenette fallback.

---

## 6. Files

| Path | Purpose |
|---|---|
| `efficientnet_colab.ipynb` | Generated training notebook (do not hand-edit; regenerate via `scripts/build_colab_notebooks.py`) |
| `train/train.py` | Standalone (non-Colab) training entry point |
| `convert_to_tflite.py` | Standalone TFLite converter (used outside the notebook) |
| `config.yaml` | Hyperparameter overrides for the standalone trainer |
| `requirements.txt` | Full training deps |
| `requirements-fetch-only.txt` | Slim deps for the DuckDuckGo scraper in `tools/` |

See [`DATASET.md`](../../DATASET.md) for the dataset structure and Food-101 / Imagenette mapping.
