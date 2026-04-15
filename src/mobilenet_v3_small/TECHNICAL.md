# MobileNetV3-Small Food Classifier -- Technical Documentation

## Overview

Lightweight image classifier for food content moderation. Classifies a single image as **healthy**, **unhealthy**, or **not_food** using a MobileNetV3-Small backbone with frozen-backbone transfer learning.

Same training pipeline as the EfficientNet-B0 model in this repo — only the backbone differs. Use this when latency / model size matters more than peak accuracy.

| | |
|---|---|
| **Framework** | TensorFlow / Keras |
| **Backbone** | MobileNetV3Small (ImageNet-pretrained, frozen) |
| **Head** | GlobalAveragePooling2D → Dropout(0.2) → Dense(N, softmax) |
| **Input** | RGB image, 224×224 |
| **Output** | Softmax over `{healthy, unhealthy, not_food}` (or sigmoid scalar if 2 classes) |
| **Mobile formats** | TFLite float (~4 MB) + int8 quantized (~1 MB); TFJS uint8 (~1 MB, browser) |
| **HF model repo** | [`maia2000/mobilenet-food-binary`](https://huggingface.co/maia2000/mobilenet-food-binary) |
| **HF dataset repo** | [`maia2000/food-classifier-dataset`](https://huggingface.co/datasets/maia2000/food-classifier-dataset) |

---

## 1. Why MobileNetV3-Small

- ~2.5 M parameters — roughly half the size of EfficientNet-B0
- Designed for mobile CPUs (depthwise separable conv, h-swish activation, NAS-derived blocks)
- Lower accuracy ceiling than EfficientNet (typically 1–3 points lower on this dataset) in exchange for ~2-3× faster inference and ~4× smaller TFLite
- Use it when you want sub-10 ms moderation per image on commodity mobile hardware

---

## 2. Classes

Detected at training time from the folder layout under `/content/binary/`:

- **2 classes** → sigmoid head + `binary_crossentropy`
- **3 classes** → softmax head + `sparse_categorical_crossentropy`

The 3-class variant requires the Food-101 + Imagenette fallback (see [DATASET.md](../../DATASET.md)).

---

## 3. Training pipeline

`mobilenet_colab.ipynb` is generated from `scripts/build_colab_notebooks.py`. It is **disconnect-resilient**: each cell is self-contained, checkpoints sync to Google Drive, training resumes via `model.fit(initial_epoch=N)`.

### Cells (identical structure to EfficientNet)

1. **Setup** — install HF deps, mount Drive at `/content/drive/MyDrive/whispr-checkpoints/mobilenet/`.
2. **Download** — try HF dataset; on <1000 frames fall back to Food-101 + Imagenette into `/content/frames/{class}/`.
3. **Organize** — symlink into `/content/binary/{class}/`.
4. **Train** — frozen MobileNetV3Small + small head.
5. **Push** — model + dataset to HF; full traceback on failure.
6. **Mobile export** — TFLite (float + int8) + TFJS (uint8, browser) + confusion matrix PNG.

### Hyperparameters

| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Optimizer | Adam, lr=1e-3 |
| Loss | `binary_crossentropy` (2 cls) / `sparse_categorical_crossentropy` (3 cls) |
| Epochs | 10 (with `EarlyStopping(patience=3, restore_best_weights=True)`) |
| Augmentation | RandomFlip(horizontal) + RandomRotation(0.05) + RandomZoom(0.1) |
| Preprocessing | `mobilenet_v3.preprocess_input` |
| Validation split | 20 %, fixed `seed=42` |

### Resume logic

Per-epoch `PersistCallback` writes `model.h5` + `train_state.json` (`{last_epoch, history}`) locally and to Drive. On rerun the cell pulls the checkpoint, reads `last_epoch`, and either skips training (if already done) or resumes mid-run.

---

## 4. Mobile export

Cell 6 produces:

- `model.tflite` — float TFLite (~4 MB)
- `model_quantized.tflite` — int8 TFLite (~1 MB)
- `model_tfjs/` — TensorFlow.js Layers model, uint8-quantized (~0.5–1 MB across `model.json` + shards). Load in the browser via `tf.loadLayersModel('<path>/model.json')`. Produced via `tensorflowjs.converters.save_keras_model(..., quantization_dtype_map={"uint8": "*"})`.
- `confusion_matrix.png` — validation-split matrix, with diagnostic prints (raw sigmoid probability range, per-class label/pred counts, accuracy, full `classification_report`)

All are uploaded to the HF model repo with retries.

---

## 5. App integration (`app_mockup_demo/app.py`)

The TFLite runner (`predict_efficientnet`) handles MobileNet's binary head:

- The trained model emits a **single sigmoid scalar**, output shape `(1, 1)`.
- The app detects `preds.size == 1` and expands to `[1-p, p]` with labels `["healthy", "unhealthy"]`.
- The health rollup then maps `unhealthy` to the block decision.

> ⚠️ A 2-class model has **no `not_food` head** — it will force every input into healthy or unhealthy. To get a "not food" verdict, retrain with the Imagenette fallback enabled.

### Observed inference time

~5–10 ms per image on a modern laptop CPU via TFLite interpreter (the demo records this in its speed panel; "⚡ Instant" tier).

---

## 6. Files

| Path | Purpose |
|---|---|
| `main.py` | CLI entry point (`--action train|eval|eval_tflite|test`) |
| `mobilenet_colab.ipynb` | Colab training notebook with HF push + TFLite + TFJS export |
| `train/train.py` | Two-stage Keras trainer (frozen head, then optional backbone fine-tune) |
| `validation/validation.py` | Keras model validator (imports `HEALTH_LABELS` from `src/common/`) |
| `validation/validation_tflite.py` | TFLite model validator |
| `convert_to_tflite.py` | Standalone Keras → TFLite converter |
| `tools/fetch_google_dataset.py` | Thin shim — delegates to `src/common/fetch_google_dataset.py` |
| `tools/configuration_generator.py` | Writes `config.yaml` with hardware-aware defaults |
| `tools/config_validator.py` | CLI validation of `config.yaml` |
| `config.yaml` | Hyperparameter overrides (generated on first run) |
| `requirements.txt` | Full training deps |
| `requirements-fetch-only.txt` | Scraper-only deps |

Shared modules used by this pipeline:
- [`src/common/health_labels.py`](../common/health_labels.py) — canonical fine-class → healthy/unhealthy/not_food mapping
- [`src/common/fetch_google_dataset.py`](../common/fetch_google_dataset.py) — shared image scraper
- [`src/common/logging_config.py`](../common/logging_config.py) — root logging setup (called by `main.py`)

See [`DATASET.md`](../../DATASET.md) for the dataset structure.
