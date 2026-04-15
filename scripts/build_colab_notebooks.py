"""Generate disconnect-resilient Colab notebooks as proper .ipynb JSON."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

COMMON_TITLE = """# {model_name} -- Healthy / Unhealthy Food Classifier
Downloads frames from HF, remaps 16 classes to binary, trains {model_name} (frozen backbone), pushes to HF.

**Resilient to runtime disconnects**: each cell is self-contained and re-establishes state. Checkpoints live on Google Drive; training resumes from last completed epoch.
"""

CELL_SETUP = """# 1. Install + HF login + mount Drive (self-contained, safe to re-run)
import os, sys, subprocess
from pathlib import Path

try:
    subprocess.run(["pip", "-q", "install", "huggingface_hub", "hf_transfer"], check=True)
except Exception as e:
    print(f"pip install warning: {{e}}")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

HF_TOKEN = ""
try:
    from google.colab import userdata
    HF_TOKEN = (userdata.get("HF_TOKEN") or "").strip()
except Exception:
    pass
if not HF_TOKEN:
    HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

if HF_TOKEN:
    from huggingface_hub import login
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("HF: logged in")
    except Exception as e:
        print(f"HF login failed (continuing): {{e}}")
else:
    print("HF_TOKEN not found -- add it to Colab Secrets")

DRIVE_DIR = None
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE_DIR = Path("/content/drive/MyDrive/whispr-checkpoints/{slug}")
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Drive checkpoint dir: {{DRIVE_DIR}}")
except Exception as e:
    print(f"Drive not mounted (training still works, no resume): {{e}}")

HF_DATASET = "maia2000/food-classifier-dataset"
HF_MODEL   = "{hf_model}"
os.environ["HF_TOKEN"] = HF_TOKEN
"""

CELL_DOWNLOAD = """# 2. Get dataset: try HF -> fall back to Food-101 + Imagenette (idempotent)
import os, subprocess, sys
from pathlib import Path
from collections import Counter
from huggingface_hub import snapshot_download

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
HF_DATASET = "maia2000/food-classifier-dataset"

FRAMES = Path("/content/frames")
have = sum(1 for _ in FRAMES.rglob("*.jpg")) if FRAMES.exists() else 0

# 2a. Try the user's HF dataset first
if have < 1000:
    for attempt in range(3):
        try:
            snapshot_download(
                repo_id=HF_DATASET, repo_type="dataset",
                local_dir="/content", allow_patterns=["frames/**"],
                token=HF_TOKEN or None, max_workers=16,
            )
            break
        except Exception as e:
            print(f"HF download attempt {attempt+1} failed: {e}")
    have = sum(1 for _ in FRAMES.rglob("*.jpg")) if FRAMES.exists() else 0

print(f"frames from HF: {have}")

# 2b. Fall back to Food-101 + Imagenette if HF dataset is empty
USE_FALLBACK = have < 1000
if USE_FALLBACK:
    print("HF dataset empty -- falling back to Food-101 + Imagenette")
    try:
        from datasets import load_dataset
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)
        from datasets import load_dataset

    # Map Food-101 fine classes to healthy / unhealthy
    FOOD101_TO_HEALTH = {
        # healthy
        "beet_salad":"healthy","caesar_salad":"healthy","caprese_salad":"healthy",
        "greek_salad":"healthy","seaweed_salad":"healthy","ceviche":"healthy",
        "grilled_salmon":"healthy","mussels":"healthy","oysters":"healthy",
        "sashimi":"healthy","scallops":"healthy","sushi":"healthy",
        "tuna_tartare":"healthy","shrimp_and_grits":"healthy","crab_cakes":"healthy",
        "lobster_roll_sandwich":"healthy","beef_carpaccio":"healthy","beef_tartare":"healthy",
        "filet_mignon":"healthy","peking_duck":"healthy","pork_chop":"healthy",
        "prime_rib":"healthy","steak":"healthy","deviled_eggs":"healthy",
        "bibimbap":"healthy","paella":"healthy","risotto":"healthy",
        "miso_soup":"healthy","pho":"healthy","ramen":"healthy",
        "edamame":"healthy","hummus":"healthy","guacamole":"healthy",
        # unhealthy
        "hamburger":"unhealthy","hot_dog":"unhealthy","club_sandwich":"unhealthy",
        "grilled_cheese_sandwich":"unhealthy","pulled_pork_sandwich":"unhealthy",
        "pizza":"unhealthy","chicken_wings":"unhealthy","french_fries":"unhealthy",
        "fried_calamari":"unhealthy","fried_rice":"unhealthy","onion_rings":"unhealthy",
        "spring_rolls":"unhealthy","beignets":"unhealthy","churros":"unhealthy",
        "donuts":"unhealthy","fish_and_chips":"unhealthy","samosa":"unhealthy",
        "apple_pie":"unhealthy","baklava":"unhealthy","bread_pudding":"unhealthy",
        "cannoli":"unhealthy","carrot_cake":"unhealthy","cheesecake":"unhealthy",
        "chocolate_cake":"unhealthy","chocolate_mousse":"unhealthy",
        "creme_brulee":"unhealthy","cup_cakes":"unhealthy","panna_cotta":"unhealthy",
        "red_velvet_cake":"unhealthy","strawberry_shortcake":"unhealthy",
        "tiramisu":"unhealthy","waffles":"unhealthy","pancakes":"unhealthy",
        "french_toast":"unhealthy","macarons":"unhealthy","ice_cream":"unhealthy",
        "frozen_yogurt":"unhealthy","nachos":"unhealthy","garlic_bread":"unhealthy",
    }

    PER_CLASS_CAP = 250    # per Food-101 source class
    NOT_FOOD_CAP  = 4000   # total non-food images

    FRAMES.mkdir(parents=True, exist_ok=True)
    for tgt in ("healthy", "unhealthy", "not_food"):
        (FRAMES / tgt).mkdir(parents=True, exist_ok=True)

    # Load Food-101 (~75k train images, streams quickly)
    print("Loading Food-101...")
    food = load_dataset("ethz/food101", split="train")
    label_names = food.features["label"].names

    per_source = Counter()
    written = Counter()
    for row in food:
        src = label_names[row["label"]]
        tgt = FOOD101_TO_HEALTH.get(src)
        if not tgt or per_source[src] >= PER_CLASS_CAP:
            continue
        idx = per_source[src]
        dst = FRAMES / tgt / f"food101_{src}_{idx:04d}.jpg"
        per_source[src] += 1
        if dst.exists():
            written[tgt] += 1
            continue
        try:
            row["image"].convert("RGB").save(dst, "JPEG", quality=88)
            written[tgt] += 1
        except Exception:
            pass

    # Load Imagenette for not_food (10 non-food ImageNet classes, ~9k images)
    print("Loading Imagenette for not_food...")
    try:
        imnet = load_dataset("frgfm/imagenette", "320px", split="train")
        n_written = 0
        for i, row in enumerate(imnet):
            if n_written >= NOT_FOOD_CAP:
                break
            dst = FRAMES / "not_food" / f"imagenette_{i:05d}.jpg"
            if dst.exists():
                n_written += 1
                continue
            try:
                row["image"].convert("RGB").save(dst, "JPEG", quality=88)
                n_written += 1
            except Exception:
                pass
        written["not_food"] = n_written
    except Exception as e:
        print(f"Imagenette load failed (continuing without not_food): {e}")

    for tgt in ("healthy", "unhealthy", "not_food"):
        total = len(list((FRAMES / tgt).glob("*.jpg")))
        print(f"  {tgt:10s} {total:5d} images (added {written[tgt]})")
"""

CELL_BINARY = """# 3. Organize into /content/binary/{healthy,unhealthy,not_food}/ (symlinks)
import os, shutil
from pathlib import Path

FRAMES = Path("/content/frames")
BIN    = Path("/content/binary")
TARGETS = ("healthy", "unhealthy", "not_food")

UNHEALTHY = ["burgers","candy_sweets","desserts","fried_food","pizza","salty_snacks","sugary_drinks"]
HEALTHY   = ["fruits","grain_bowls","grilled_meat","salads","seafood","smoothies","soups","vegetables"]
NOT_FOOD  = ["not_food"]
LABEL_MAP = {**{c:"unhealthy" for c in UNHEALTHY}, **{c:"healthy" for c in HEALTHY}, "not_food": "not_food"}
# If the fallback was used, source folders are already named healthy/unhealthy/not_food
LABEL_MAP.update({"healthy": "healthy", "unhealthy": "unhealthy"})

if BIN.exists():
    shutil.rmtree(BIN)
for lbl in TARGETS:
    (BIN / lbl).mkdir(parents=True)

counts = {lbl: 0 for lbl in TARGETS}
for cls in FRAMES.iterdir():
    if not cls.is_dir():
        continue
    lbl = LABEL_MAP.get(cls.name)
    if not lbl:
        continue
    for img in cls.rglob("*.jpg"):
        dst = BIN / lbl / f"{cls.name}_{img.parent.name}_{img.name}"
        if not dst.exists():
            try:
                os.symlink(img.resolve(), dst)
            except FileExistsError:
                pass
        counts[lbl] += 1
print(counts)

# Drop empty class dirs so Keras doesn't complain
for lbl in TARGETS:
    if counts[lbl] == 0:
        shutil.rmtree(BIN / lbl)
        print(f"  dropped empty class: {lbl}")
"""

CELL_TRAIN_TEMPLATE = """# 4. Train -- {model_name} frozen. Resumes from Drive checkpoint if present.
import os, json, shutil
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import {tf_class}
from tensorflow.keras.applications.{tf_module} import preprocess_input

BIN       = Path("/content/binary")
DRIVE_DIR = Path("/content/drive/MyDrive/whispr-checkpoints/{slug}") if Path("/content/drive/MyDrive").exists() else None
LOCAL_CKPT = Path("/content/model.h5")
LOCAL_STATE = Path("/content/train_state.json")
DRIVE_CKPT  = (DRIVE_DIR / "model.h5") if DRIVE_DIR else None
DRIVE_STATE = (DRIVE_DIR / "train_state.json") if DRIVE_DIR else None

IMG, BATCH, TOTAL_EPOCHS = 224, 32, 10

if DRIVE_CKPT and DRIVE_CKPT.exists() and not LOCAL_CKPT.exists():
    shutil.copy2(DRIVE_CKPT, LOCAL_CKPT)
    print(f"Restored model from {{DRIVE_CKPT}}")
if DRIVE_STATE and DRIVE_STATE.exists() and not LOCAL_STATE.exists():
    shutil.copy2(DRIVE_STATE, LOCAL_STATE)

initial_epoch = 0
prior_history = {{}}
if LOCAL_STATE.exists():
    try:
        st = json.loads(LOCAL_STATE.read_text())
        initial_epoch = int(st.get("last_epoch", 0))
        prior_history = st.get("history", {{}})
        print(f"Resuming from epoch {{initial_epoch}}")
    except Exception as e:
        print(f"state read failed: {{e}}")

if initial_epoch >= TOTAL_EPOCHS:
    print(f"Already trained for {{initial_epoch}}/{{TOTAL_EPOCHS}} epochs -- skipping training")
    model = tf.keras.models.load_model(str(LOCAL_CKPT))
    class_names = sorted(d.name for d in BIN.iterdir() if d.is_dir())
    val_acc = max(prior_history.get("val_accuracy", [0.0]))
    hist_history = prior_history
else:
    n_classes = sum(1 for d in BIN.iterdir() if d.is_dir())
    label_mode = "binary" if n_classes == 2 else "int"
    train_ds = tf.keras.utils.image_dataset_from_directory(
        BIN, validation_split=0.2, subset="training", seed=42,
        image_size=(IMG, IMG), batch_size=BATCH, label_mode=label_mode)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        BIN, validation_split=0.2, subset="validation", seed=42,
        image_size=(IMG, IMG), batch_size=BATCH, label_mode=label_mode)
    class_names = train_ds.class_names
    print(f"classes ({{n_classes}}): {{class_names}}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)

    if LOCAL_CKPT.exists():
        print(f"Loading existing model from {{LOCAL_CKPT}}")
        model = tf.keras.models.load_model(str(LOCAL_CKPT))
    else:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ])
        base = {tf_class}(include_top=False, weights="imagenet", input_shape=(IMG, IMG, 3))
        base.trainable = False
        inputs = tf.keras.Input((IMG, IMG, 3))
        x = aug(inputs)
        x = preprocess_input(x)
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        if n_classes == 2:  # noqa
            out = layers.Dense(1, activation="sigmoid")(x)
            loss = "binary_crossentropy"
        else:
            out = layers.Dense(n_classes, activation="softmax")(x)
            loss = "sparse_categorical_crossentropy"
        model = models.Model(inputs, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss=loss, metrics=["accuracy"])
    model.summary()

    class PersistCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            try:
                self.model.save(str(LOCAL_CKPT))
            except Exception as e:
                print(f"local save failed: {{e}}")
            merged = dict(prior_history)
            hist = getattr(self.model, "history", None)
            if hist is not None and hasattr(hist, "history"):
                for k, vs in hist.history.items():
                    merged[k] = list(prior_history.get(k, []))[:initial_epoch] + [float(x) for x in vs]
            try:
                LOCAL_STATE.write_text(json.dumps({{"last_epoch": epoch + 1, "history": merged}}))
            except Exception as e:
                print(f"state save failed: {{e}}")
            if DRIVE_CKPT:
                try:
                    shutil.copy2(LOCAL_CKPT, DRIVE_CKPT)
                    shutil.copy2(LOCAL_STATE, DRIVE_STATE)
                except Exception as e:
                    print(f"drive sync failed (continuing): {{e}}")

    es = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    hist = model.fit(
        train_ds, validation_data=val_ds,
        epochs=TOTAL_EPOCHS, initial_epoch=initial_epoch,
        callbacks=[es, PersistCallback()],
    )
    hist_history = dict(prior_history)
    for k, vs in hist.history.items():
        hist_history[k] = list(prior_history.get(k, []))[:initial_epoch] + [float(x) for x in vs]
    val_acc = max(hist_history.get("val_accuracy", [0.0]))
    try:
        model.save(str(LOCAL_CKPT))
        if DRIVE_CKPT:
            shutil.copy2(LOCAL_CKPT, DRIVE_CKPT)
    except Exception as e:
        print(f"final save failed: {{e}}")
    print(f"\\nBest val acc: {{val_acc:.4f}}")
"""

CELL_UPLOAD_TEMPLATE = """# 5. Push to HuggingFace (best-effort; won't fail the notebook)
import os, json
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL = "{hf_model}"
HF_DATASET = "maia2000/food-classifier-dataset"
TAG = "{tag}"

UNHEALTHY = ["burgers","candy_sweets","desserts","fried_food","pizza","salty_snacks","sugary_drinks"]
HEALTHY   = ["fruits","grain_bowls","grilled_meat","salads","seafood","smoothies","soups","vegetables"]

MODEL_PATH = Path("/content/model.h5")
STATE_PATH = Path("/content/train_state.json")

val_acc = 0.0
BIN_DIR = Path("/content/binary")
class_names = sorted(d.name for d in BIN_DIR.iterdir() if d.is_dir()) if BIN_DIR.exists() else ["healthy", "unhealthy"]
history = {{}}
if STATE_PATH.exists():
    try:
        st = json.loads(STATE_PATH.read_text())
        history = st.get("history", {{}})
        val_acc = max(history.get("val_accuracy", [0.0]))
    except Exception as e:
        print(f"state read failed: {{e}}")

card = f'''---
license: apache-2.0
tags: [image-classification, food, binary-classification, {{TAG}}]
---
# Binary Healthy/Unhealthy Food Classifier -- {model_name}

Frozen {model_name} + sigmoid head. Best val accuracy: **{{val_acc:.4f}}**.

- unhealthy: {{", ".join(UNHEALTHY)}}
- healthy: {{", ".join(HEALTHY)}}
- dataset: [{{HF_DATASET}}](https://huggingface.co/datasets/{{HF_DATASET}})
'''
Path("/content/README.md").write_text(card)
Path("/content/metrics.json").write_text(json.dumps({{
    "val_accuracy": float(val_acc), "classes": class_names, "history": history,
}}, indent=2))

if not HF_TOKEN:
    print("No HF_TOKEN -- skipping upload")
elif not MODEL_PATH.exists():
    print("No model found -- skipping upload")
else:
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=HF_TOKEN)
        create_repo(HF_MODEL, repo_type="model", token=HF_TOKEN, exist_ok=True)
        for f in [str(MODEL_PATH), "/content/README.md", "/content/metrics.json"]:
            for attempt in range(3):
                try:
                    api.upload_file(path_or_fileobj=f, path_in_repo=Path(f).name,
                                    repo_id=HF_MODEL, token=HF_TOKEN)
                    break
                except Exception as e:
                    print(f"upload {{f}} attempt {{attempt+1}} failed: {{e}}")
        print(f"https://huggingface.co/{{HF_MODEL}}")
    except Exception as e:
        print(f"HF model upload failed (artifacts are local + on Drive): {{e}}")

# 5b. Push the dataset (frames/) so future runs can skip the Food-101 fallback
FRAMES_DIR = Path("/content/frames")
if not HF_TOKEN:
    print("No HF_TOKEN -- skipping dataset upload")
elif not FRAMES_DIR.exists() or not any(FRAMES_DIR.rglob("*.jpg")):
    print("No frames/ dir -- skipping dataset upload")
else:
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=HF_TOKEN)
        create_repo(HF_DATASET, repo_type="dataset", token=HF_TOKEN, exist_ok=True)

        # Dataset card with class counts
        ds_counts = {{}}
        for cls in sorted(d for d in FRAMES_DIR.iterdir() if d.is_dir()):
            ds_counts[cls.name] = sum(1 for _ in cls.rglob("*.jpg"))
        ds_card = "---\\nlicense: apache-2.0\\ntask_categories: [image-classification]\\ntags: [food, healthy-eating]\\n---\\n"
        ds_card += "# Food classifier dataset\\n\\nClass counts:\\n\\n"
        for k, v in ds_counts.items():
            ds_card += f"- **{{k}}**: {{v}} images\\n"
        Path("/content/dataset_README.md").write_text(ds_card)

        try:
            api.upload_file(
                path_or_fileobj="/content/dataset_README.md", path_in_repo="README.md",
                repo_id=HF_DATASET, repo_type="dataset", token=HF_TOKEN,
            )
        except Exception as e:
            print(f"dataset README upload failed: {{e}}")

        print(f"Uploading {{sum(ds_counts.values())}} images to {{HF_DATASET}} (single commit via upload_large_folder)...")
        try:
            # upload_large_folder batches into bulk commits -- avoids the 128-commits/hour rate limit
            api.upload_large_folder(
                folder_path=str(FRAMES_DIR),
                repo_id=HF_DATASET, repo_type="dataset",
                ignore_patterns=["*.tmp", ".ipynb_checkpoints/*"],
            )
        except AttributeError:
            # Older huggingface_hub without upload_large_folder -- fall back to upload_folder (still one commit)
            api.upload_folder(
                folder_path=str(FRAMES_DIR), path_in_repo="frames",
                repo_id=HF_DATASET, repo_type="dataset", token=HF_TOKEN,
                ignore_patterns=["*.tmp", ".ipynb_checkpoints/*"],
            )
        print(f"https://huggingface.co/datasets/{{HF_DATASET}}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"HF dataset upload failed (frames still local): {{e}}")
"""

CELL_MOBILE_EXPORT = """# 6. Mobile export: TFLite (float + int8) + confusion matrix PNG, push to model repo
import os, json
from pathlib import Path
import numpy as np
import tensorflow as tf

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
HF_MODEL  = "{hf_model}"
# Support either model format (Keras 3 prefers .keras; older code used .h5)
MODEL_PATH = next((p for p in [Path("/content/model.keras"), Path("/content/model.h5")] if p.exists()), Path("/content/model.h5"))
TFLITE_PATH   = Path("/content/model.tflite")
TFLITE_Q_PATH = Path("/content/model_quantized.tflite")
CM_PNG        = Path("/content/confusion_matrix.png")
BIN           = Path("/content/binary")

# 6a. Convert to TFLite
if MODEL_PATH.exists():
    model = tf.keras.models.load_model(str(MODEL_PATH))
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        TFLITE_PATH.write_bytes(conv.convert())
        print(f"TFLite (float):     {{TFLITE_PATH.name}}  {{TFLITE_PATH.stat().st_size/1024:.0f}} KB")
    except Exception as e:
        print(f"TFLite float export failed: {{e}}")
    try:
        conv_q = tf.lite.TFLiteConverter.from_keras_model(model)
        conv_q.optimizations = [tf.lite.Optimize.DEFAULT]
        TFLITE_Q_PATH.write_bytes(conv_q.convert())
        print(f"TFLite (quantized): {{TFLITE_Q_PATH.name}}  {{TFLITE_Q_PATH.stat().st_size/1024:.0f}} KB")
    except Exception as e:
        print(f"TFLite quantized export failed: {{e}}")
else:
    print("No model.h5 -- skipping TFLite export")

# 6b. Confusion matrix PNG (with diagnostics)
if MODEL_PATH.exists() and BIN.exists():
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns

        n_classes = sum(1 for d in BIN.iterdir() if d.is_dir())
        label_mode = "binary" if n_classes == 2 else "int"
        val_ds = tf.keras.utils.image_dataset_from_directory(
            BIN, validation_split=0.2, subset="validation", seed=42,
            image_size=(224, 224), batch_size=32, label_mode=label_mode, shuffle=False)
        class_names = list(val_ds.class_names)
        print(f"classes: {{class_names}}  (n={{n_classes}})")

        # Collect predictions over the whole val set
        all_preds = []
        all_labels = []
        for x, y in val_ds:
            p = model.predict(x, verbose=0)
            all_preds.append(p)
            all_labels.append(y.numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        if n_classes == 2:
            # sigmoid -> shape (N, 1) probability of class 1
            probs = all_preds.reshape(-1)
            y_pred = (probs > 0.5).astype(int)
            y_true = all_labels.reshape(-1).astype(int)
            print(f"raw sigmoid probs: min={{probs.min():.3f}}  max={{probs.max():.3f}}  mean={{probs.mean():.3f}}")
        else:
            # softmax -> shape (N, C)
            y_pred = np.argmax(all_preds, axis=1).astype(int)
            y_true = all_labels.reshape(-1).astype(int)

        print(f"label counts:    " + "  ".join(f"{{class_names[i]}}={{int((y_true==i).sum())}}" for i in range(n_classes)))
        print(f"pred counts:     " + "  ".join(f"{{class_names[i]}}={{int((y_pred==i).sum())}}" for i in range(n_classes)))
        print(f"accuracy: {{(y_true == y_pred).mean():.4f}}")

        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(CM_PNG, dpi=120)
        plt.show()
        print(f"Saved: {{CM_PNG.name}}")
        print("\\n" + classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Confusion matrix failed: {{e}}")

# 6c. Upload mobile artifacts to the model repo
if HF_TOKEN:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        for f in [TFLITE_PATH, TFLITE_Q_PATH, CM_PNG]:
            if not f.exists():
                continue
            for attempt in range(3):
                try:
                    api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                                    repo_id=HF_MODEL, token=HF_TOKEN)
                    print(f"  uploaded {{f.name}} -> {{HF_MODEL}}")
                    break
                except Exception as e:
                    print(f"  upload {{f.name}} attempt {{attempt+1}} failed: {{e}}")
    except Exception as e:
        print(f"Mobile artifact upload failed: {{e}}")
else:
    print("No HF_TOKEN -- skipping mobile artifact upload")
"""


def make_cells(model_name: str, slug: str, hf_model: str, tf_class: str, tf_module: str, tag: str):
    md = COMMON_TITLE.format(model_name=model_name)
    setup = CELL_SETUP.format(slug=slug, hf_model=hf_model)
    train = CELL_TRAIN_TEMPLATE.format(model_name=model_name, slug=slug, tf_class=tf_class, tf_module=tf_module)
    upload = CELL_UPLOAD_TEMPLATE.format(model_name=model_name, hf_model=hf_model, tag=tag)
    mobile = CELL_MOBILE_EXPORT.format(hf_model=hf_model)
    cells_src = [
        ("markdown", md),
        ("code", setup),
        ("code", CELL_DOWNLOAD),
        ("code", CELL_BINARY),
        ("code", train),
        ("code", upload),
        ("code", mobile),
    ]
    cells = []
    for ct, src in cells_src:
        c = {"cell_type": ct, "metadata": {}, "source": src.splitlines(keepends=True)}
        if ct == "code":
            c["execution_count"] = None
            c["outputs"] = []
        cells.append(c)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main():
    eff = make_cells("EfficientNetB0", "efficientnet", "maia2000/efficientnet-food-binary",
                     "EfficientNetB0", "efficientnet", "efficientnetb0")
    mob = make_cells("MobileNetV3Small", "mobilenet", "maia2000/mobilenet-food-binary",
                     "MobileNetV3Small", "mobilenet_v3", "mobilenetv3small")
    (REPO / "src/efficientnet_lite_gpu/efficientnet_colab.ipynb").write_text(json.dumps(eff, indent=1), encoding="utf-8")
    (REPO / "src/mobilenet_v3_small/mobilenet_colab.ipynb").write_text(json.dumps(mob, indent=1), encoding="utf-8")
    print("wrote both notebooks")


if __name__ == "__main__":
    main()
