# ViT Video Classifier -- Technical Documentation

## Overview

Video / multi-frame classifier for food content moderation. Takes a sequence of 8 frames sampled from a clip (or a single image duplicated 8×), encodes each frame with a Vision Transformer backbone, and aggregates temporally with a small head (default: bidirectional LSTM).

| | |
|---|---|
| **Framework** | PyTorch |
| **Backbone (GPU)** | `vit_b_16` (torchvision, ImageNet-pretrained) — ~86 M params |
| **Backbone (CPU)** | `mobilevit_xxs` — ~1.3 M params, mobile-friendly fallback |
| **Temporal head** | BiLSTM (default) — also supports `avg`, `max`, `conv1d` via `--pool` |
| **Input** | 8 RGB frames × 224×224 |
| **Output** | Softmax over the dataset's class set (binary or 16-class depending on training data) |
| **Mobile formats** | `.ptl` PyTorch Mobile Lite (canonical on-device), `.pt` TorchScript, `.onnx`, `.tflite` (via `ai_edge_torch`), optional CoreML `.mlpackage`. **TFJS is not supported for this model — see §4a; use MobileNetV3 for browser.** |
| **HF model repos** | [`maia2000/food-classifier`](https://huggingface.co/maia2000/food-classifier), [`maia2000/food-classifier-binary-vit`](https://huggingface.co/maia2000/food-classifier-binary-vit) |
| **HF dataset repo** | [`maia2000/food-classifier-dataset`](https://huggingface.co/datasets/maia2000/food-classifier-dataset) |

---

## 1. Why ViT + temporal head

- **Self-attention** captures long-range relationships across an image (and across frames once temporally aggregated). Useful when the discriminating cue isn't local — e.g. judging a whole plate of food vs a small fried element.
- **Backbone auto-selects**: `vit_b_16` on CUDA, `mobilevit_xxs` on CPU. The training script picks based on `torch.cuda.is_available()`.
- **Pluggable temporal head** (`--pool {bilstm,avg,max,conv1d}`) lets you trade compute for temporal modeling. BiLSTM gives the best results in our runs but `avg` is essentially free.
- **Video-aware** — duplicates a single image 8× when given still input, so the same model serves images and clips.

---

## 2. Classes

The classifier head dimension is **inferred from the dataset directory** at training time. The two operating modes:

- **Binary** (`healthy`, `unhealthy`) when trained on `maia2000/food-binary-dataset` or the binary subset of `food-classifier-dataset`.
- **16-class** when trained on the legacy 16-class layout in `food-classifier-dataset`.

The fine-grained 16 classes still roll up to `{healthy, unhealthy, not_food}` at inference time using the mapping baked into `app.py` (`HEALTH_LABELS`).

---

## 3. Training pipeline

Two notebooks live in this directory:

- `vit_video.ipynb` — full multi-class video trainer (BiLSTM temporal head, 8 frames per clip, Colab-disconnect-resilient via Drive checkpoints).
- `vit_video_binary.ipynb` — lighter-weight binary single-frame trainer used for the `food-classifier-binary-vit` HF release; frozen backbone, ~10 min on a Colab T4.

The standalone CLI is `train.py`, which calls into `engine/trainer.py`. Both notebooks invoke the same trainer.

### Augmentation (training only)

`RandomResizedCrop(224, scale=(0.6, 1.0))`, `RandomHorizontalFlip`, `RandomRotation(15°)`, `RandomPerspective`, `ColorJitter`, `GaussianBlur`, `RandomErasing`. Validation uses `Resize(224) + CenterCrop(224)` only.

### Normalization

ImageNet mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]`.

### Hyperparameters — multi-class (`vit_video.ipynb`)

| Parameter | Value |
|---|---|
| Frames per clip | 8 |
| Image size | 224 × 224 |
| Batch size | 8 |
| Optimizer | AdamW, lr=3e-5, weight_decay=1e-3 |
| LR search | candidates `5e-6,1e-5,3e-5,5e-5,1e-4` (optional, via `hparam_search_epochs > 0`) |
| Loss | `CrossEntropyLoss`, optional class weights from training distribution |
| Epochs | 20 with `EarlyStopping(patience=7)` |
| Temporal pool | `lstm` (BiLSTM) |
| Backbone | `auto` (ViT-B/16 on CUDA, MobileViT-XXS on CPU) |
| Gradient clipping | `max_grad_norm=1.0` |
| Dropout (head) | 0.4 |
| Class weighting | enabled by default (computes inverse-frequency weights from train set) |

### Hyperparameters — binary (`vit_video_binary.ipynb`)

Simpler pipeline: frozen ViT-B/16 + linear head, single-frame (no temporal aggregation), trained on `maia2000/food-binary-dataset` (~35k frames).

| Parameter | Value |
|---|---|
| Backbone | ViT-B/16, **frozen** (only `model.heads` trained) |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-4 |
| Loss | `CrossEntropyLoss(label_smoothing=0.1)` |
| Epochs | 10 with `EarlyStopping(patience=3)` |
| AMP | enabled |

### Resume

`engine/trainer.py` saves `best_food_classifier.pth` after every val-improvement epoch. Pass `--resume <path>` to continue from a checkpoint; the optimizer state and epoch counter are restored.

---

## 4. Mobile export

Section 13 of `vit_video.ipynb` runs `export_mobile.py` with `EXPORT_FORMATS` (default `['torchscript', 'onnx', 'tflite']`; add `'coreml'` on macOS) and `QUANTIZE=True`. Outputs land in `exported_models/`:

- `best_food_classifier.pth` — full PyTorch checkpoint (model + optimizer state + metadata).
- `<name>.pt` — TorchScript, traced, `torch.utils.mobile_optimizer.optimize_for_mobile` applied (gracefully skipped if tracing doesn't tolerate it — e.g. on some LSTM builds).
- `<name>.ptl` — **PyTorch Mobile Lite Interpreter** build (produced alongside the `.pt` via `traced_model._save_for_lite_interpreter`). Smaller runtime footprint, optimized for on-device Android / iOS. **This is the canonical shipping format for the ViT-Video model on mobile.**
- `<name>.onnx` — ONNX export, opset 17, with **both batch and num_frames axes dynamic**.
- `<name>.tflite` — via `ai_edge_torch` (preferred) or ONNX→TF→TFLite fallback. `--quantize` applies float16 weight quantization (~2× smaller, no accuracy loss, no representative dataset needed).
- `<name>.mlpackage` — CoreML ML Program, iOS15+ target (only if `coremltools` is installed).
- `model_card.json` — records resolved backbone (not `"auto"`), classes, input shape, normalization, evaluation metrics.

The TorchScript file is what the demo app (`app.py`) loads via `torch.jit.load`.

### 4a. TFJS export — not supported for ViT (use MobileNetV3 instead)

TFJS is **deliberately excluded** from the default format list for this model.

#### Why — architectural and operator-level incompatibility

Vision Transformer (ViT) models are not well-suited for TensorFlow.js deployment due to architectural and operator-level incompatibilities. Unlike convolutional networks, ViT models rely heavily on transformer-specific operations:

- **Multi-head self-attention** — decomposed into scaled dot-product matmuls + softmax + head-wise reshape
- **Matrix multiplications with dynamic tensor reshaping** — patch embedding and attention heads reshape from shapes only known at runtime
- **Layer normalization** — computed over the last dimension with learned affine parameters, composed tightly with GELU activations
- **Attention-score computation** — numerically sensitive (softmax over `Q·Kᵀ / √dₖ`) and emitted as a fused subgraph by `torch.onnx.export`
- **BiLSTM temporal head** — adds stateful RNN ops on top of the transformer backbone

These are **fully supported in PyTorch** and **partially supported in ONNX**, but they are not consistently mapped into TensorFlow SavedModel graphs in a way that TensorFlow.js can reliably interpret.

#### Why the conversion chain breaks

In practice, the `PyTorch → ONNX → TensorFlow → TFJS` pipeline often fails because intermediate conversion tools — notably `onnx-tf` and the TensorFlow graph exporters — do not fully implement support for transformer-specific operators or dynamic computation graphs. Common symptoms:

- **Incomplete or empty exported models** (the SavedModel is produced but contains only a subset of the original graph)
- **Silent failures during graph conversion** (no exception, but the output is numerically wrong)
- **Op-not-supported errors at TFJS load time** (model ships but the browser runtime refuses to run it)

Additionally, **TensorFlow.js is primarily optimized for convolution-based architectures and lightweight sequential models**, not large transformer-based vision models. Even when a conversion technically succeeds, the browser-side execution is often too slow to be usable.

#### Analogy

CNNs are LEGO blocks: `Conv → BN → ReLU → Pool` translates cleanly to any framework. ViT is a robotic assembly — self-attention, dynamic shapes, and composite norm/activation blocks that only survive intact when the target runtime matches the source runtime's assumptions. TFJS is not that runtime.

#### How this model is actually deployed

For this reason, ViT models are typically deployed via runtimes that natively support transformer computation graphs:

- **ONNX Runtime / ONNX Runtime Web** — for web and edge deployment where a browser-compatible runtime is required
- **PyTorch Mobile** — for native Android / iOS inference

In this project, the ViT-Video model is shipped as a **PyTorch Mobile Lite** artifact (`.ptl`) — produced by calling `traced_model._save_for_lite_interpreter()` on the TorchScript module rather than the regular `.save()`. The Lite Interpreter guarantees full operator compatibility on-device, runs the optimized mobile kernel set, and eliminates graph-conversion risk entirely (no cross-framework round-trip). `.pt` TorchScript is still produced alongside for desktop / server use.

#### If you still want to try TFJS export

`export_tfjs` in `export_mobile.py` is kept as a best-effort entry point with fail-fast handling — it prints what broke at each stage (ONNX, `onnx-tf`, `tensorflowjs_converter`) so you know whether to blame the op set or the conversion:

```bash
python export_mobile.py --model ... --format tfjs
```

If it does succeed, **always validate numerical parity** against the PyTorch model on a held-out batch before shipping; silent wrong outputs are the default failure mode.

#### Browser deployment — use MobileNetV3 instead

The MobileNetV3-Small pipeline (`src/mobilenet_v3_small/`) is a pure CNN, trained directly in Keras, and converts to TFJS with one line:

```python
tfjs.converters.save_keras_model(model, "model_tfjs/", quantization_dtype_map={"uint8": "*"})
```

That produces a ~1 MB uint8-quantized bundle that loads via `tf.loadLayersModel("model.json")` with no op-compatibility issues. See [`src/mobilenet_v3_small/TECHNICAL.md`](../mobilenet_v3_small/TECHNICAL.md) §4.

#### Deployment target matrix

| Target | Runtime | Format | Model |
|---|---|---|---|
| Native Android / iOS | PyTorch Mobile (Lite Interpreter) | `.ptl` | ViT-Video |
| Web / edge (transformer-capable) | ONNX Runtime / ONNX Runtime Web | `.onnx` | ViT-Video |
| Server / desktop Python | PyTorch | `.pt` TorchScript | ViT-Video |
| Android (TF ecosystem) | TFLite | `.tflite` | ViT-Video |
| iOS (Apple ecosystem) | CoreML | `.mlpackage` | ViT-Video |
| **Browser (TFJS)** | **TensorFlow.js** | **`model.json` + shards** | **MobileNetV3-Small** |

---

## 5. App integration (`app_mockup_demo/app.py`)

`predict_vit` accepts either a single PIL image (duplicated 8×) or a list of frames (uniformly sampled / padded to exactly 8). It:

1. Looks for `vit_food.pt`, `best_food_classifier.pt`, `model.pt`, or `best_food_classifier.pth` under `app_mockup_demo/models/` and the legacy locations.
2. Tries `torch.jit.load` first (TorchScript). Falls back to `load_model_from_checkpoint` from `vit_video.utils.model_utils` for raw state-dict files.
3. Detects class count from the classifier weight shape and reads class names from the checkpoint metadata, then a sibling `labels.json`, then `_classes_for_n(n)` as fallback.
4. Returns `(class_name, confidence, probs, classes)` — all wrapped in try/except so a corrupt model never crashes the UI.

Video uploads in the app go through `_extract_video_frames` (OpenCV), which pulls 8 evenly spaced frames; the result feeds straight into `predict_vit`.

---

## 6. Files

| Path | Purpose |
|---|---|
| `vit_video.ipynb` | Multi-class Colab training notebook (video + BiLSTM temporal head) |
| `vit_video_binary.ipynb` | Binary single-frame training notebook |
| `train.py` | CLI entry point (training) |
| `run_pipeline.py` | Full data pipeline: YouTube fetch → frame extraction → train/val/test split |
| `inference.py` | Single-video or single-image inference CLI |
| `validate_model.py` | Held-out test evaluation + external K-fold validation |
| `export_mobile.py` | Mobile export (TorchScript `.pt`, Lite Interpreter `.ptl`, ONNX, TFLite, CoreML) |
| `generatedata.py` | YouTube scraping via `yt-dlp` |
| `upload_hf.py` | HuggingFace dataset + model push |
| `engine/trainer.py` | Training loop, AMP, checkpoint logic |
| `engine/dataset.py` | Frame sampling, augmentation, class-weight computation |
| `models/vit.py` | `MobileViTModel` — ViT-B/16 or MobileViT-XXS backbone + temporal head |
| `utils/model_utils.py` | Checkpoint load/remap + factory used by the demo app |
| `utils/hardware.py`, `utils/video.py`, `utils/data_utils.py` | Hardware probe, OpenCV frame IO, transforms |
| `_bootstrap.py` | sys.path helper (Colab-friendly; no-op in normal installs) |
| `requirements.txt` | Training deps (torch, timm, opencv, yt-dlp, …) |

See [`DATASET.md`](../../DATASET.md) for the dataset structure and class taxonomy.
