"""
Streamlit chat mockup demo for food moderation.

Tests both ViT (video) and EfficientNet (image) models in a chat UI.
Users upload an image (or video), the selected model classifies it,
and the moderation decision is shown inline.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Whispr Food Moderation Demo", page_icon="🍔", layout="wide")

CLASSES_16 = [
    "fruits", "vegetables", "salads", "seafood", "grilled_meat", "grain_bowls",
    "soups", "smoothies", "burgers", "pizza", "fried_food", "desserts",
    "candy_sweets", "salty_snacks", "sugary_drinks", "not_food",
]

CLASSES_2 = ["healthy", "unhealthy"]


def _classes_from_labels_json(model_path: Path) -> list[str] | None:
    """Look for a labels.json or classes.json next to the model."""
    import json
    for name in ("labels.json", "classes.json"):
        for d in [model_path.parent, model_path.parent.parent]:
            p = d / name
            if p.exists():
                try:
                    data = json.loads(p.read_text())
                    if isinstance(data, list):
                        return data
                    if isinstance(data, dict):
                        if "class_names" in data:
                            return data["class_names"]
                        if "classes" in data:
                            return data["classes"]
                        if "id2label" in data:
                            return [data["id2label"][str(i)] for i in range(len(data["id2label"]))]
                except Exception:
                    pass
    return None


def _classes_for_n(n: int) -> list[str]:
    """Best-guess class list for a given output dimension."""
    if n == 16:
        return CLASSES_16
    if n == 2 or n == 1:  # n=1 = sigmoid binary classifier
        return CLASSES_2
    # fallback: generic names
    return [f"class_{i}" for i in range(n)]

HEALTH_LABELS = {
    "fruits": "healthy", "vegetables": "healthy", "salads": "healthy",
    "seafood": "healthy", "grilled_meat": "healthy", "grain_bowls": "healthy",
    "soups": "healthy", "smoothies": "healthy",
    "burgers": "unhealthy", "pizza": "unhealthy", "fried_food": "unhealthy",
    "desserts": "unhealthy", "candy_sweets": "unhealthy",
    "salty_snacks": "unhealthy", "sugary_drinks": "unhealthy",
    "not_food": "not_food",
}

HEALTH_EMOJI = {"healthy": "🥗", "unhealthy": "🍔", "not_food": "❓"}

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# HuggingFace model download
# ---------------------------------------------------------------------------
HF_USER = "maia2000"  # https://huggingface.co/maia2000

# File extensions we consider as "model" files
MODEL_EXTENSIONS = {".pt", ".pth", ".tflite", ".onnx", ".safetensors", ".bin", ".keras", ".h5"}


@st.cache_data(ttl=300)  # Cache for 5 minutes
def list_hf_models(user: str, token: str | None = None) -> list[dict]:
    """List all model repos under a HF user account."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []
    try:
        api = HfApi(token=token if token else None)
        models = api.list_models(author=user)
        return [{"id": m.id, "name": m.id.split("/")[-1]} for m in models]
    except Exception as e:
        st.error(f"Failed to list models: {e}")
        return []


@st.cache_data(ttl=300)
def list_repo_model_files(repo_id: str, token: str | None = None) -> list[str]:
    """List model files inside a given HF repo."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []
    try:
        api = HfApi(token=token if token else None)
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        return [f for f in files if Path(f).suffix.lower() in MODEL_EXTENSIONS]
    except Exception as e:
        st.error(f"Failed to list files in {repo_id}: {e}")
        return []


def download_from_hf(repo_id: str, filename: str, token: str | None = None) -> Path | None:
    """Download a single file from a HuggingFace repo into MODELS_DIR."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        st.error("`huggingface_hub` not installed. Run: pip install huggingface_hub")
        return None
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token if token else None,
            local_dir=str(MODELS_DIR),
        )
        return Path(downloaded)
    except Exception as e:
        st.error(f"Download failed: {e}")
        return None


def _find_model(filenames: list[str], search_roots: list[Path]) -> Path | None:
    """Look for a model file in a list of candidate directories."""
    for root in search_roots:
        if not root.exists():
            continue
        for name in filenames:
            # Exact match
            candidate = root / name
            if candidate.exists():
                return candidate
            # Recursive search (one level deep)
            for sub in root.rglob(name):
                return sub
    return None


# ---------------------------------------------------------------------------
# Model loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_efficientnet_model():
    """Load the EfficientNet TFLite model from common locations.
    Returns (interpreter, input_dtype, path, classes)."""
    try:
        import tensorflow as tf
    except ImportError:
        return None, None, None, []
    search = [
        MODELS_DIR,
        REPO_ROOT / "models",
        REPO_ROOT / "src" / "efficientnet_lite_gpu",
        REPO_ROOT / "src" / "efficientnet_lite_gpu" / "tflite",
    ]
    candidates = [
        "efficientnet_food.tflite",
        "BestModelEfficientNetLite_inference.tflite",
        "BestModelEfficientNetLite.tflite",
        "model.tflite",
    ]
    path = _find_model(candidates, search)
    if path is None:
        return None, None, None, []
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()

    # Determine classes
    n_classes = interpreter.get_output_details()[0]["shape"][-1]
    classes = _classes_from_labels_json(path) or _classes_for_n(int(n_classes))
    if len(classes) != n_classes:
        classes = _classes_for_n(int(n_classes))

    return interpreter, interpreter.get_input_details()[0]["dtype"], path, classes


@st.cache_resource(show_spinner=False)
def load_vit_model():
    """Load the ViT model from common locations.
    Returns (model, path, classes)."""
    try:
        import torch
    except ImportError:
        return None, None, []
    search = [
        MODELS_DIR,
        REPO_ROOT / "models",
        REPO_ROOT / "src" / "vit_video" / "models",
        REPO_ROOT / "src" / "vit_video" / "exported_models",
    ]
    candidates = [
        "vit_food.pt",
        "best_food_classifier.pt",
        "model.pt",
        "best_food_classifier.pth",
    ]
    path = _find_model(candidates, search)
    if path is None:
        return None, None, []

    # Detect class count from checkpoint first
    num_classes = 16
    classes_from_ckpt = None
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            # Look for classes in metadata
            meta = ckpt.get("metadata", {}) if isinstance(ckpt.get("metadata"), dict) else {}
            classes_from_ckpt = meta.get("classes") or ckpt.get("classes")
            # Detect class count from classifier weight shape
            sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
            if isinstance(sd, dict):
                for k, v in sd.items():
                    if "classifier" in k and k.endswith(".weight") and hasattr(v, "shape"):
                        num_classes = int(v.shape[0])
                        break
    except Exception:
        pass

    # Determine classes
    classes = (
        classes_from_ckpt
        or _classes_from_labels_json(path)
        or _classes_for_n(num_classes)
    )
    if len(classes) != num_classes:
        classes = _classes_for_n(num_classes)

    # Try TorchScript first
    try:
        model = torch.jit.load(str(path), map_location="cpu")
        model.eval()
        return model, path, classes
    except Exception:
        # Plain state_dict -- try loading into MobileViTModel
        try:
            import sys
            sys.path.insert(0, str(REPO_ROOT / "src"))
            from vit_video.utils.model_utils import load_model_from_checkpoint
            model = load_model_from_checkpoint(path, num_classes=num_classes, device=torch.device("cpu"))
            model.eval()
            return model, path, classes
        except Exception as e:
            st.error(f"Failed to load ViT model: {e}")
            return None, None, []


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_efficientnet(image: Image.Image) -> tuple[str, float, np.ndarray, list[str]]:
    """Run EfficientNet/MobileNet TFLite inference on a single image. Never raises."""
    try:
        interpreter, input_dtype, _, classes = load_efficientnet_model()
        if interpreter is None:
            return "(model not loaded)", 0.0, np.zeros(16), CLASSES_16

        img = image.convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]["index"], img_batch.astype(input_details[0]["dtype"]))
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]["index"])[0]

        if not np.any(np.isfinite(preds)):
            return "(invalid output)", 0.0, np.zeros(len(classes) or 16), classes or CLASSES_16

        # Single-scalar sigmoid output (binary classifier) -> expand to [P(class0), P(class1)]
        if preds.shape == (1,) or preds.size == 1:
            p1 = float(np.clip(preds.flatten()[0], 0.0, 1.0))
            preds = np.array([1.0 - p1, p1], dtype=np.float32)
            if len(classes) != 2:
                # alphabetical order matches Keras image_dataset_from_directory: healthy, unhealthy
                classes = ["healthy", "unhealthy"]

        idx = int(np.argmax(preds))
        name = classes[idx] if idx < len(classes) else f"class_{idx}"
        return name, float(preds[idx]), preds, classes
    except Exception as e:
        st.error(f"EfficientNet inference failed: {e}")
        return "(error)", 0.0, np.zeros(16), CLASSES_16


MAX_VIDEO_BYTES = 100 * 1024 * 1024  # 100 MB cap


def _extract_video_frames(video_bytes: bytes, n_frames: int = 8) -> list[Image.Image]:
    """Extract n evenly-spaced frames from an mp4. Never raises -- returns []."""
    import tempfile, os as _os
    if len(video_bytes) == 0:
        st.error("Video is empty.")
        return []
    if len(video_bytes) > MAX_VIDEO_BYTES:
        st.error(f"Video too large ({len(video_bytes) / 1024 / 1024:.1f} MB). Max {MAX_VIDEO_BYTES // 1024 // 1024} MB.")
        return []
    try:
        import cv2
    except ImportError:
        st.error("OpenCV required for video: pip install opencv-python-headless")
        return []

    tmp_path = None
    cap = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            st.error("Could not open video (unsupported codec?). Try re-encoding as H.264 MP4.")
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frames: list[Image.Image] = []
        if total > 0:
            idxs = [int(i * (total - 1) / max(n_frames - 1, 1)) for i in range(n_frames)]
            for i in idxs:
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))
                except Exception:
                    continue
        else:
            # Fallback: sequential read up to n_frames
            count = 0
            while count < n_frames * 4:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if count % 4 == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))
                count += 1

        while frames and len(frames) < n_frames:
            frames.append(frames[-1])
        if not frames:
            st.error("Could not decode any frames from the video.")
        return frames
    except Exception as e:
        st.error(f"Video decoding failed: {e}")
        return []
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if tmp_path:
                _os.unlink(tmp_path)
        except Exception:
            pass


def predict_vit(frames: Image.Image | list[Image.Image]) -> tuple[str, float, np.ndarray, list[str]]:
    """Run ViT inference on a single image (duplicated 8x) or a list of frames. Never raises."""
    try:
        import torch
        from torchvision import transforms
    except ImportError:
        return "(torch not installed)", 0.0, np.zeros(16), CLASSES_16

    try:
        model, _, classes = load_vit_model()
        if model is None:
            return "(model not loaded)", 0.0, np.zeros(16), CLASSES_16

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if isinstance(frames, list):
            if len(frames) == 0:
                return "(empty video)", 0.0, np.zeros(len(classes) or 16), classes or CLASSES_16
            if len(frames) >= 8:
                step = len(frames) / 8
                picked = [frames[int(i * step)] for i in range(8)]
            else:
                picked = frames + [frames[-1]] * (8 - len(frames))
            tensors = [transform(f.convert("RGB")) for f in picked]
        else:
            tensors = [transform(frames.convert("RGB"))] * 8

        video_tensor = torch.stack(tensors).unsqueeze(0)  # (1, 8, 3, 224, 224)

        with torch.no_grad():
            logits = model(video_tensor)
            probs = torch.softmax(logits, dim=1)[0].numpy()

        if not np.any(np.isfinite(probs)):
            return "(invalid output)", 0.0, np.zeros(len(classes) or 16), classes or CLASSES_16
        idx = int(np.argmax(probs))
        name = classes[idx] if idx < len(classes) else f"class_{idx}"
        return name, float(probs[idx]), probs, classes
    except Exception as e:
        st.error(f"ViT inference failed: {e}")
        return "(error)", 0.0, np.zeros(16), CLASSES_16


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown(
    "<h3 style='margin:0 0 0.5em 0'>🍔 Whispr Food Moderation</h3>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")

    # Check which models are available to set a sensible default
    _eff_check = load_efficientnet_model()
    _vit_check = load_vit_model()
    _eff_available = _eff_check[0] is not None
    _vit_available = _vit_check[0] is not None

    model_options = []
    if _eff_available:
        model_options.append("EfficientNet (TFLite)")
    if _vit_available:
        model_options.append("ViT (TorchScript)")
    if not model_options:
        model_options = ["EfficientNet (TFLite)", "ViT (TorchScript)"]

    model_choice = st.selectbox("Model", model_options, index=0)
    threshold = 0.4

    with st.expander("📥 Download from HF", expanded=True):
        hf_user_input = st.text_input("User/org", value=HF_USER, key="hf_user_input", label_visibility="collapsed")

        if st.button("🔍 List", use_container_width=True):
            st.session_state.hf_repos = list_hf_models(hf_user_input)

        repos = st.session_state.get("hf_repos", [])
        if repos:
            repo_name = st.selectbox(f"{len(repos)} repos", [r["id"].split("/", 1)[-1] for r in repos], key="hf_repo_choice")
            repo_full = f"{hf_user_input}/{repo_name}"
            files = list_repo_model_files(repo_full)
            if files:
                file_choice = st.selectbox(f"{len(files)} files", files, key="hf_file_choice")
                if st.button("⬇️ Download", use_container_width=True):
                    with st.spinner(f"Downloading {file_choice}..."):
                        path = download_from_hf(repo_id=repo_full, filename=file_choice)
                    if path:
                        st.success(f"{path.name} ({path.stat().st_size / (1024 * 1024):.1f} MB)")
                        load_efficientnet_model.clear()
                        load_vit_model.clear()
            else:
                st.caption("No model files in repo")

    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.processed_files = set()
        st.session_state.last_block = None
        st.rerun()

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey 👋 Send me a food image. I'll let it through if it's healthy, block it if it's unhealthy."}
    ]

if "last_block" not in st.session_state:
    st.session_state.last_block = None  # {"reason": str, "image": Image}

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()  # IDs of already-processed uploads

if "perf_log" not in st.session_state:
    # list of dicts: {model, class, confidence, latency_ms, health, image_kb}
    st.session_state.perf_log = []

chat_col, perf_col = st.columns([3, 2], gap="medium")

with chat_col:
    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "video" in msg:
                import base64
                b64 = base64.b64encode(msg["video"]).decode()
                st.markdown(
                    f'<video width="240" controls style="border-radius:8px">'
                    f'<source src="data:video/mp4;base64,{b64}" type="video/mp4">'
                    f'</video>',
                    unsafe_allow_html=True,
                )
            elif "image" in msg:
                st.image(msg["image"], width=160)
            if msg.get("content"):
                st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])

    # Show last-blocked alert at the top if there was one
    if st.session_state.last_block:
        st.error(st.session_state.last_block["reason"], icon="🚫")

# ---------------------------------------------------------------------------
# Image upload -- run moderation BEFORE adding to chat
# ---------------------------------------------------------------------------
with chat_col:
    uploaded = st.file_uploader(
        "Upload an image or video",
        type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "webm"],
        label_visibility="collapsed",
    )

# Only process each upload once -- Streamlit re-executes the whole script on every rerun
if uploaded is not None and uploaded.file_id not in st.session_state.processed_files:
    st.session_state.processed_files.add(uploaded.file_id)
    try:
        raw_bytes = uploaded.read()
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        st.stop()

    if not raw_bytes:
        st.error("Uploaded file is empty.")
        st.stop()

    suffix = Path(uploaded.name).suffix.lower()
    is_video = suffix in {".mp4", ".mov", ".webm"}

    image: Image.Image | None = None
    video_frames: list[Image.Image] = []

    if is_video:
        video_frames = _extract_video_frames(raw_bytes, n_frames=8)
        if not video_frames:
            st.stop()
        image = video_frames[len(video_frames) // 2]  # middle frame as preview
    else:
        try:
            from PIL import ImageOps
            image = Image.open(io.BytesIO(raw_bytes))
            image.load()  # force decode now so bad files fail here
            image = ImageOps.exif_transpose(image)  # honor phone rotation
        except Exception as e:
            st.error(f"Could not open image (corrupt or unsupported format): {e}")
            st.stop()

    # Run moderation (inference functions never raise, but wrap defensively)
    try:
        with st.spinner(f"Moderation check ({model_choice})..."):
            start = time.perf_counter()
            if model_choice.startswith("EfficientNet"):
                # EfficientNet is image-only; use middle frame for video
                cls, conf, probs, classes = predict_efficientnet(image)
            else:
                cls, conf, probs, classes = predict_vit(video_frames if is_video else image)
            latency_ms = (time.perf_counter() - start) * 1000.0
    except Exception as e:
        st.error(f"Moderation failed: {e}")
        st.stop()

    # If inference returned an error marker, don't pretend it was a real prediction
    if cls.startswith("(") and cls.endswith(")"):
        st.warning(f"Model result unavailable: {cls}. Upload will not be processed.")
        st.stop()

    # Log performance
    st.session_state.perf_log.append({
        "model": model_choice,
        "class": cls,
        "confidence": conf,
        "latency_ms": latency_ms,
        "image_kb": len(raw_bytes) / 1024,
    })

    # Map to health group: the class name itself may already be a health label
    # (if the model was trained on 2/3 classes directly) or a fine-grained class
    if cls in HEALTH_LABELS:
        health = HEALTH_LABELS[cls]
    elif cls in ("healthy", "unhealthy", "not_food"):
        health = cls
    else:
        health = "unknown"

    emoji = HEALTH_EMOJI.get(health, "❓")
    meta = f"{emoji} `{cls}` ({conf:.0%}) · {latency_ms:.0f} ms"

    media_word = "video" if is_video else "image"

    # Decision
    if health == "unhealthy" and conf >= threshold:
        # BLOCK: show alert, do NOT add to chat
        st.session_state.last_block = {
            "reason": (
                f"🚫 **Message blocked** — the moderation model detected "
                f"**unhealthy food** (`{cls}`, {conf:.0%} confidence). "
                f"This {media_word} cannot be sent."
            ),
        }
        st.rerun()
    else:
        # ALLOW: healthy, not_food, or low-confidence unhealthy → send
        st.session_state.last_block = None
        msg: dict = {
            "role": "user",
            "content": f"📎 sent `{uploaded.name}`",
            "meta": meta,
        }
        if is_video:
            msg["video"] = raw_bytes
        else:
            msg["image"] = image
        st.session_state.messages.append(msg)

        # Auto-reply based on the content
        if conf < threshold:
            reply = f"⚠️ I couldn't confidently identify that ({conf:.0%}). Delivered anyway."
        elif health == "healthy":
            reply = f"✅ Healthy food received! Looks like **{cls}**."
        else:  # not_food
            reply = f"ℹ️ Not food, but that's fine — delivered."

        st.session_state.messages.append({"role": "assistant", "content": reply, "meta": meta})
        st.rerun()

# ---------------------------------------------------------------------------
# Performance panel
# ---------------------------------------------------------------------------
perf_log = st.session_state.perf_log

def _human_time(ms: float) -> str:
    """Human-readable latency."""
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms/1000:.1f} s"


def _rating(avg_ms: float) -> tuple[str, str, str]:
    """(badge, short verdict, plain-language explanation) for a given latency."""
    if avg_ms < 100:
        return "⚡ Instant", "Faster than a blink",            "The user won't notice any delay — safe to run on every message."
    if avg_ms < 500:
        return "🟢 Fast",    "About as quick as a chat reply", "Feels snappy in a messaging app."
    if avg_ms < 1500:
        return "🟡 Noticeable", "A short pause",               "There's a visible delay, but still usable."
    return     "🔴 Slow",    "Users will wait",                "Needs a smaller model or a GPU for real use."


with perf_col, st.expander(f"📊 How fast is it? ({len(perf_log)} checked)", expanded=bool(perf_log)):
    if not perf_log:
        st.caption("Send an image to see how quickly the model checks it.")
    else:
        import statistics

        latencies = [r["latency_ms"] for r in perf_log]
        avg_ms = statistics.mean(latencies)
        per_minute = 60_000 / avg_ms if avg_ms > 0 else 0.0
        badge, verdict, explanation = _rating(avg_ms)

        st.markdown(f"### {badge}")
        st.markdown(f"**{verdict}** — takes about **{_human_time(avg_ms)}** per photo.")
        st.caption(explanation)

        col1, col2 = st.columns(2)
        col1.metric("Time per photo", _human_time(avg_ms))
        col2.metric("Photos per minute", f"{per_minute:,.0f}")

        # Per-model comparison (only if both models were tested)
        by_model: dict[str, list[float]] = {}
        for r in perf_log:
            by_model.setdefault(r["model"], []).append(r["latency_ms"])

        if len(by_model) > 1:
            st.markdown("**Which model is faster?**")
            ranked = sorted(by_model.items(), key=lambda kv: statistics.mean(kv[1]))
            for rank, (m, lats) in enumerate(ranked):
                m_avg = statistics.mean(lats)
                short_name = m.replace(" (TFLite)", "").replace(" (TorchScript)", "")
                tag = "🥇" if rank == 0 else "🥈"
                st.caption(f"{tag} **{short_name}** — {_human_time(m_avg)} per photo ({len(lats)} tries)")

        if len(perf_log) >= 2:
            import pandas as pd
            df = pd.DataFrame([{"Photo #": i + 1, "Seconds": r["latency_ms"] / 1000}
                              for i, r in enumerate(perf_log)])
            st.caption("Time per photo:")
            st.line_chart(df.set_index("Photo #"), height=120)

        with st.expander("🔍 Recent checks", expanded=False):
            for i, r in enumerate(reversed(perf_log[-10:])):
                pos = len(perf_log) - i
                st.text(
                    f"#{pos:2d} · {r['class']} ({r['confidence']:.0%}) · {_human_time(r['latency_ms'])}"
                )

        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.perf_log = []
            st.rerun()
