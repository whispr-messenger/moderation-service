# Food Moderation Chat Demo

A Streamlit chat mockup to test the ViT and EfficientNet food classifiers end-to-end.

Upload an image → the selected model predicts the class → moderation decision is shown inline.

---

## Setup

```bash
cd app_mockup_demo
pip install -r requirements.txt
```

## Models

Place the trained/exported models in `app_mockup_demo/models/`:

| File | Source | Size |
|---|---|---|
| `efficientnet_food.tflite` | `src/efficientnet_lite_gpu/BestModelEfficientNetLite_inference.tflite` | ~4 MB |
| `vit_food.pt` | `src/vit_video/exported_models/model.pt` (TorchScript) | ~5 MB |

Rename them to match the paths above.

Or download from HuggingFace:

```bash
# EfficientNet
huggingface-cli download maia2000/efficientnet-food-classifier \
  BestModelEfficientNetLite_inference.tflite \
  --local-dir models/ && \
  mv models/BestModelEfficientNetLite_inference.tflite models/efficientnet_food.tflite

# ViT
huggingface-cli download maia2000/food-classifier \
  model.pt \
  --local-dir models/ && \
  mv models/model.pt models/vit_food.pt
```

## Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Features

- **Chat UI** -- drag-and-drop image upload, inline moderation response
- **Two models** -- switch between EfficientNet (TFLite) and ViT (TorchScript) in the sidebar
- **Confidence threshold** -- slider to tune the flag threshold
- **Top-5 predictions** -- expandable panel showing probability distribution
- **Latency readout** -- ms-level timing for each inference
- **Health rollup** -- 16 fine-grained classes → healthy / unhealthy / not_food

## How it works

1. User uploads an image in the chat
2. Selected model (EfficientNet or ViT) runs inference
3. Prediction is mapped to `healthy` / `unhealthy` / `not_food` via `HEALTH_LABELS`
4. If `unhealthy` and confidence > threshold → content flagged 🚫
5. If `healthy` → content passes ✅
6. If `not_food` → ignored ℹ️

The ViT model expects a video (8 frames). For single-image inference, the notebook duplicates the image 8 times to fill the temporal dimension.
