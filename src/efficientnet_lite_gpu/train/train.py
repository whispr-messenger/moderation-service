import os
import json
from pathlib import Path
import shutil
import hashlib

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import Image, UnidentifiedImageError
from tools.split_dataset import build_clean_split

def _get_img_size(train_cfg: dict) -> tuple[int, int]:
    image_size = train_cfg["image_size"]
    if isinstance(image_size, int):
        return (image_size, image_size)
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))
    raise ValueError(f"Invalid image_size in config: {image_size}")


def _apply_sys_config(sys_cfg: dict, train_cfg: dict):
    if sys_cfg.get("disable_XLA_logs", True):
        tf.config.optimizer.set_jit(False)

    if sys_cfg.get("tf_force_gpu_allow_growth", True):
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    if train_cfg.get("mixed_precision", False):
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")


def _build_paths(train_cfg: dict) -> dict:

    cwd = Path.cwd()

    dataset_root = cwd / train_cfg["dataset_dir"]        # e.g. train/dataset
    results_root = cwd / train_cfg["results_dir"]        # e.g. train/results

    paths = {
        "dataset_root": dataset_root,
        "train_dir": dataset_root / train_cfg["train_dir"],
        "val_dir":   dataset_root / train_cfg["val_dir"],
        "test_dir":  dataset_root / train_cfg["test_dir"],

        "results_root": results_root,
        "data_exploration_dir": results_root / train_cfg["data_exploration_dir"],
        "evaluation_results_dir": results_root / train_cfg["evaluation_results_dir"],
        "training_logs_dir": results_root / train_cfg["training_logs_dir"],
        "training_results_dir": results_root / train_cfg["training_results_dir"],
    }

    for k in ("data_exploration_dir", "evaluation_results_dir",
              "training_logs_dir", "training_results_dir"):
        paths[k].mkdir(parents=True, exist_ok=True)

    return paths


def _resolve_raw_dataset_dir(train_cfg: dict) -> Path:
    cwd = Path.cwd()
    raw_cfg = train_cfg.get("raw_dataset_dir")
    if raw_cfg:
        p = Path(str(raw_cfg).strip())
        return p if p.is_absolute() else (cwd / p)

    # Fallback to a single raw dataset source directory.
    dataset_root = cwd / train_cfg["dataset_dir"]
    fallback = dataset_root / "_raw_source"
    if fallback.exists():
        return fallback
    raise ValueError(
        "Missing raw dataset source. Set train_config.raw_dataset_dir "
        "or provide dataset_dir/_raw_source as the unique source dataset."
    )


def _prepare_clean_dataset_split(train_cfg: dict):
    # Rebuild a clean split from one raw source before training.
    enabled = bool(train_cfg.get("rebuild_clean_split_before_train", True))
    if not enabled:
        print("Clean split rebuild disabled by config (rebuild_clean_split_before_train=false).")
        return

    cwd = Path.cwd()
    output_dir = cwd / train_cfg["dataset_dir"]
    raw_input_dir = _resolve_raw_dataset_dir(train_cfg)
    split_seed = int(train_cfg.get("split_seed", 42))
    train_ratio = float(train_cfg.get("split_train_ratio", 0.7))
    val_ratio = float(train_cfg.get("split_val_ratio", 0.15))
    test_ratio = float(train_cfg.get("split_test_ratio", 0.15))
    split_mode = str(train_cfg.get("split_mode", "copy"))

    print("Rebuilding clean dataset split from unique raw source...")
    print(f"Raw source dir: {raw_input_dir}")
    print(f"Output split dir: {output_dir}")
    result = build_clean_split(
        raw_input_dir=raw_input_dir,
        output_dir=output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        mode=split_mode,
        seed=split_seed,
        train_dir_name=train_cfg["train_dir"],
        val_dir_name=train_cfg["val_dir"],
        test_dir_name=train_cfg["test_dir"],
    )
    print("Clean split generated.")
    print("Split overlap report:", result["overlap_report"])


def _is_valid_image_file(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def _sanitize_image_directory(root_dir: Path, quarantine_dir: Path) -> dict:
    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    moved = []
    checked = 0

    for class_dir in sorted(root_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for file_path in class_dir.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in allowed_ext:
                continue
            checked += 1
            if _is_valid_image_file(file_path):
                continue

            target_dir = quarantine_dir / root_dir.name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / file_path.name

            # Avoid collisions if same filename appears multiple times.
            if target_path.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                idx = 1
                while True:
                    candidate = target_dir / f"{stem}__dup{idx}{suffix}"
                    if not candidate.exists():
                        target_path = candidate
                        break
                    idx += 1

            shutil.move(str(file_path), str(target_path))
            moved.append((file_path, target_path))

    return {"checked": checked, "moved": moved}


def _sanitize_datasets_if_needed(train_cfg: dict, paths: dict):
    sanitize_enabled = train_cfg.get("sanitize_invalid_images", True)
    if not sanitize_enabled:
        print("Image sanitization disabled by config (sanitize_invalid_images=false).")
        return

    quarantine_dir = paths["dataset_root"] / "_invalid_images"

    print(f"Scanning training images in: {paths['train_dir']}")
    train_report = _sanitize_image_directory(paths["train_dir"], quarantine_dir)
    print(
        f"Train scan complete: checked={train_report['checked']} moved_invalid={len(train_report['moved'])}"
    )

    if paths["test_dir"].exists():
        print(f"Scanning test images in: {paths['test_dir']}")
        test_report = _sanitize_image_directory(paths["test_dir"], quarantine_dir)
        print(
            f"Test scan complete: checked={test_report['checked']} moved_invalid={len(test_report['moved'])}"
        )

    if train_report["moved"]:
        preview = train_report["moved"][:5]
        print("Examples of moved invalid train files:")
        for src, dst in preview:
            print(f" - {src} -> {dst}")


def _build_datasets(train_cfg: dict, paths: dict, img_size: tuple[int, int]):
    batch_size = train_cfg["batch_size"]
    model_cfg = train_cfg.get("model_config", {})
    label_smoothing = float(model_cfg.get("label_smoothing", 0.0))

    # Use categorical labels when label smoothing is enabled.
    train_label_mode = "categorical" if label_smoothing > 0 else "int"

    train_dir = paths["train_dir"]
    test_dir = paths["test_dir"]
    val_dir = paths["val_dir"]

    print("Using train dir:", train_dir)
    print("Using val dir:", val_dir)
    print("Using test dir:", test_dir)
    print(f"Label mode (train/val): {train_label_mode}  (label_smoothing={label_smoothing})")

    # Mapping de classes unique et strict sur les trois splits.
    class_names_train_raw = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if not class_names_train_raw:
        raise ValueError(f"No class directories found in train dir: {train_dir}")
    print("Train class names (raw):", class_names_train_raw)

    val_class_names_disk = sorted([d.name for d in val_dir.iterdir() if d.is_dir()]) if val_dir.exists() else []
    test_class_names_disk = sorted([d.name for d in test_dir.iterdir() if d.is_dir()]) if test_dir.exists() else []
    print("Val class names on disk:", val_class_names_disk)
    print("Test class names on disk:", test_class_names_disk)

    if val_class_names_disk != class_names_train_raw:
        raise ValueError(
            "Class mismatch between Train and Val.\n"
            f"Train classes: {class_names_train_raw}\n"
            f"Val classes:   {val_class_names_disk}"
        )
    if test_class_names_disk != class_names_train_raw:
        raise ValueError(
            "Class mismatch between Train and Test.\n"
            f"Train classes: {class_names_train_raw}\n"
            f"Test classes:  {test_class_names_disk}"
        )

    class_names = class_names_train_raw

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)
    print("Train class names (effective):", class_names)
    print("Train class indices:", class_to_idx)
    print("Detected num_classes:", num_classes)

    val_class_names = list(class_names)
    test_class_names = list(class_names)

    # Interdiction de re-splitter train pour fabriquer val.
    print("Validation split source: dedicated val directory")
    val_ds = image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode=train_label_mode,
        class_names=class_names,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    train_ds = image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode=train_label_mode,
        class_names=class_names,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )

    print("Val class names:", val_class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    print("Test class names (effective):", test_class_names)

    print("Test shuffle is fixed to False for stable evaluation order.")

    dataset_meta = {
        "val_class_names": val_class_names,
        "test_class_names": test_class_names,
    }
    return train_ds, val_ds, test_ds, class_names, class_to_idx, num_classes, dataset_meta


def _build_data_augmentation(train_cfg: dict) -> tf.keras.Sequential:
    da = train_cfg["data_augmentation"]
    layers = [
        tf.keras.layers.RandomFlip(da.get("randomFlip", "horizontal")),
        tf.keras.layers.RandomRotation(da.get("randomRotation", 0.15)),
        tf.keras.layers.RandomZoom(da.get("randomZoom", 0.2)),
        tf.keras.layers.RandomContrast(da.get("randomContrast", 0.2)),
    ]
    if da.get("randomBrightness"):
        layers.append(tf.keras.layers.RandomBrightness(da["randomBrightness"]))
    if da.get("randomTranslation"):
        t = da["randomTranslation"]
        layers.append(tf.keras.layers.RandomTranslation(t, t))
    return tf.keras.Sequential(layers, name="data_augmentation")


def _get_backbone_class(model_name: str):
    name = model_name.lower()
    if name in ("efficientnet-b0", "efficientnet_b0"):
        return tf.keras.applications.EfficientNetB0
    elif name in ("efficientnet-b1", "efficientnet_b1"):
        return tf.keras.applications.EfficientNetB1
    elif name in ("efficientnet-b2", "efficientnet_b2"):
        return tf.keras.applications.EfficientNetB2
    elif name in ("efficientnet-b3", "efficientnet_b3"):
        return tf.keras.applications.EfficientNetB3
    elif name in ("mobilenet-v2-035", "mobilenet-v2-050", "mobilenet-v2-100"):
        return tf.keras.applications.MobileNetV2
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def _get_preprocess_input(model_name: str):
    name = model_name.lower()
    if name.startswith("efficientnet"):
        return tf.keras.applications.efficientnet.preprocess_input
    elif name.startswith("mobilenet-v2"):
        return tf.keras.applications.mobilenet_v2.preprocess_input
    else:
        raise ValueError(f"Unsupported model_name for preprocess: {model_name}")


def _count_images_per_class(root_dir: Path, class_names: list[str]) -> dict[str, int]:
    counts = {}
    for cls in class_names:
        class_dir = root_dir / cls
        if not class_dir.exists() or not class_dir.is_dir():
            counts[cls] = 0
            continue
        counts[cls] = sum(1 for p in class_dir.rglob("*") if p.is_file())
    return counts


def _check_data_leakage(train_dir: Path, val_dir: Path, test_dir: Path) -> dict:
    # Double vérification par chemin source et hash de contenu.
    def collect_paths_and_hashes(root: Path):
        # Lecture unique pour produire les ensembles de chemins et les empreintes SHA256.
        paths = set()
        hashes = {}
        if not root.exists():
            return paths, hashes
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            ap = str(p.resolve())
            paths.add(ap)
            try:
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                hashes.setdefault(h, []).append(ap)
            except OSError:
                continue
        return paths, hashes

    train_paths, train_hashes = collect_paths_and_hashes(train_dir)
    val_paths, val_hashes = collect_paths_and_hashes(val_dir)
    test_paths, test_hashes = collect_paths_and_hashes(test_dir)

    path_tv = sorted(train_paths & val_paths)
    path_tt = sorted(train_paths & test_paths)
    path_vt = sorted(val_paths & test_paths)

    hash_tv = sorted(set(train_hashes.keys()) & set(val_hashes.keys()))
    hash_tt = sorted(set(train_hashes.keys()) & set(test_hashes.keys()))
    hash_vt = sorted(set(val_hashes.keys()) & set(test_hashes.keys()))

    return {
        "path_overlap": {
            "train_val_duplicates": len(path_tv),
            "train_test_duplicates": len(path_tt),
            "val_test_duplicates": len(path_vt),
            "train_val_examples": path_tv[:20],
            "train_test_examples": path_tt[:20],
            "val_test_examples": path_vt[:20],
        },
        "hash_overlap": {
            "train_val_duplicates": len(hash_tv),
            "train_test_duplicates": len(hash_tt),
            "val_test_duplicates": len(hash_vt),
            "train_hash_count": len(train_hashes),
            "val_hash_count": len(val_hashes),
            "test_hash_count": len(test_hashes),
            "train_val_hash_examples": hash_tv[:20],
            "train_test_hash_examples": hash_tt[:20],
            "val_test_hash_examples": hash_vt[:20],
        },
    }


def _quarantine_train_test_duplicates_if_needed(train_cfg: dict, paths: dict):
    # Move duplicated Train/Test content out of test set.
    enabled = bool(train_cfg.get("quarantine_train_test_duplicates", True))
    if not enabled:
        print("Duplicate quarantine disabled by config (quarantine_train_test_duplicates=false).")
        return

    train_dir = paths["train_dir"]
    test_dir = paths["test_dir"]
    if not train_dir.exists() or not test_dir.exists():
        print("Skip duplicate quarantine: train_dir or test_dir missing.")
        return

    quarantine_root = paths["dataset_root"] / "_quarantine_duplicates" / "Test"
    log_path = paths["evaluation_results_dir"] / "quarantined_test_duplicates.json"

    def collect_hash_map(root: Path):
        hash_map = {}
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            try:
                h = hashlib.md5(p.read_bytes()).hexdigest()
            except OSError:
                continue
            hash_map.setdefault(h, []).append(p)
        return hash_map

    train_hashes = collect_hash_map(train_dir)
    test_hashes = collect_hash_map(test_dir)
    overlap_hashes = set(train_hashes.keys()) & set(test_hashes.keys())

    moved_records = []
    for h in sorted(overlap_hashes):
        for src in test_hashes[h]:
            try:
                rel = src.relative_to(test_dir)
            except ValueError:
                rel = Path(src.name)
            dst = quarantine_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)

            if dst.exists():
                stem = dst.stem
                suffix = dst.suffix
                idx = 1
                while True:
                    candidate = dst.with_name(f"{stem}__dup{idx}{suffix}")
                    if not candidate.exists():
                        dst = candidate
                        break
                    idx += 1

            shutil.move(str(src), str(dst))
            moved_records.append({
                "md5": h,
                "source_test_path": str(src),
                "quarantine_path": str(dst),
                "matching_train_example": str(train_hashes[h][0]),
            })

    summary = {
        "enabled": enabled,
        "overlap_hash_count": len(overlap_hashes),
        "moved_test_files_count": len(moved_records),
        "quarantine_root": str(quarantine_root),
        "moved_examples_preview": moved_records[:20],
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Train/Test duplicate quarantine summary: {summary['moved_test_files_count']} files moved.")
    print(f"Duplicate quarantine log saved to: {log_path}")


def _build_and_train_model(train_cfg: dict,
                           model_cfg: dict,
                           comp_cfg: dict,
                           img_size: tuple[int, int],
                           num_classes: int,
                           train_ds,
                           val_ds):
    STAGE1_EPOCHS = int(train_cfg.get("stage1_epochs",
                        train_cfg.get("initial_epochs", 12)))
    STAGE2_EPOCHS = int(train_cfg.get("stage2_epochs",
                        train_cfg.get("fine_tune_epochs", 6)))
    fine_tune = train_cfg["fine_tune"]
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 3))
    reduce_lr_patience = int(train_cfg.get("reduce_lr_patience", 2))

    # Fine-tuning defaults: stable stage-1 LR + smaller stage-2 LR.
    stage1_lr = float(train_cfg.get("stage1_learning_rate",
                      model_cfg.get("learning_rate", 1e-3)))
    stage2_lr = float(train_cfg.get("stage2_learning_rate",
                      train_cfg.get("fine_tune_lr", 2e-5)))

    data_augmentation = _build_data_augmentation(train_cfg)

    model_name = model_cfg["model_name"]
    BackboneClass = _get_backbone_class(model_name)
    preprocess_input = _get_preprocess_input(model_name)

    if model_name.lower().startswith("mobilenet-v2"):
        alpha = {"035": 0.35, "050": 0.50, "100": 1.0}[model_name.split("-")[-1]]
        base_model = BackboneClass(
            include_top=model_cfg["include_top"],
            weights=model_cfg["weights"],
            input_shape=img_size + (3,),
            alpha=alpha,
        )
    else:
        base_model = BackboneClass(
            include_top=model_cfg["include_top"],
            weights=model_cfg["weights"],
            input_shape=img_size + (3,),
        )
    base_model.trainable = model_cfg["trainable"]

    inputs = tf.keras.Input(shape=img_size + (3,))
    # Augmentation first (on raw [0,255] pixels), then preprocess for backbone.
    x = data_augmentation(inputs)
    x = tf.keras.layers.Lambda(preprocess_input, name="backbone_preprocess")(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation=model_cfg["output_activation"],
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)

    model = tf.keras.Model(inputs, outputs)

    optimizer_name = model_cfg.get("optimizer", "adam")
    optimizer = tf.keras.optimizers.get({
        "class_name": optimizer_name,
        "config": {"learning_rate": stage1_lr}
    })

    label_smoothing = float(model_cfg.get("label_smoothing", 0.0))
    if label_smoothing > 0:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing
        )
        print(f"Using CategoricalCrossentropy with label_smoothing={label_smoothing}")
    else:
        loss_fn = model_cfg["loss"]

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=model_cfg["metrics"]
    )

    model.summary()

    best_model_path = str(Path(comp_cfg["model_name"]).with_suffix(".keras"))

    print("Training schedule:")
    print(f"  Stage 1 epochs: {STAGE1_EPOCHS}")
    print(f"  Stage 2 epochs: {STAGE2_EPOCHS}")
    print(f"  EarlyStopping patience: {early_stopping_patience}")
    print(f"  ReduceLROnPlateau patience: {reduce_lr_patience}")
    print(f"  Stage 1 learning rate: {stage1_lr}")
    print(f"  Stage 2 learning rate: {stage2_lr}")

    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            mode="max",
        )
    ]

    # Stage 1: train classification head with frozen backbone.
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE1_EPOCHS,
        callbacks=callbacks_stage1,
    )

    best_val_acc_stage1 = max(history_stage1.history["val_accuracy"])
    print(f"Stage 1 best val accuracy: {best_val_acc_stage1:.4f}")

    history_stage2 = None

    if fine_tune:
        # Stage 2: fine-tuning contrôlé du backbone avec LR plus faible.
        base_model.trainable = True
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        optimizer_ft = tf.keras.optimizers.get({
            "class_name": optimizer_name,
            "config": {"learning_rate": stage2_lr}
        })

        model.compile(
            optimizer=optimizer_ft,
            loss=loss_fn,
            metrics=model_cfg["metrics"]
        )

        trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        non_trainable_count = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
        print(f"Trainable params: {trainable_count}, Non-trainable params: {non_trainable_count}")

        callbacks_stage2 = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=early_stopping_patience,
                restore_best_weights=True,
                mode="max",
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=best_model_path,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-6,
                mode="max",
            ),
        ]

        history_stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=STAGE2_EPOCHS,
            callbacks=callbacks_stage2,
        )

        if history_stage2.history.get("val_accuracy"):
            print(f"Stage 2 best val accuracy: {max(history_stage2.history['val_accuracy']):.4f}")
        else:
            print("Stage 2 best val accuracy: N/A")
    else:
        print("Stage 2 disabled (fine_tune=false).")

    return model, history_stage1, history_stage2, best_val_acc_stage1


def _evaluate_and_save(train_cfg: dict,
                       comp_cfg: dict,
                       paths: dict,
                       model,
                       train_ds,
                       val_ds,
                       test_ds,
                       class_names,
                       class_to_idx,
                       dataset_meta,
                       history_stage1,
                       history_stage2,
                       best_val_acc_stage1):

    NUM_CLASSES = len(class_names)
    IMG_SIZE = _get_img_size(train_cfg)
    BATCH_SIZE = train_cfg["batch_size"]

    # Final validation evaluation for tracked metrics.
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy after fine-tuning: {val_accuracy:.4f}")

    model_out_name = str(Path(comp_cfg["model_name"]).with_suffix(".keras"))
    model.save(model_out_name)
    print(f"Model training complete and saved to {model_out_name}")

    data_exp_dir = paths["data_exploration_dir"]
    eval_dir = paths["evaluation_results_dir"]
    train_results_dir = paths["training_results_dir"]
    logs_dir = paths["training_logs_dir"]

    train_counts = _count_images_per_class(paths["train_dir"], class_names)
    val_counts = _count_images_per_class(paths["val_dir"], class_names) if paths["val_dir"].exists() else {}
    test_counts = _count_images_per_class(paths["test_dir"], class_names) if paths["test_dir"].exists() else {}

    leakage_report = _check_data_leakage(paths["train_dir"], paths["val_dir"], paths["test_dir"])
    print("Potential duplicate leakage counts:", leakage_report)
    print("Train class names:", class_names)
    print("Val class names:", dataset_meta.get("val_class_names", class_names))
    print("Test class names:", dataset_meta.get("test_class_names", class_names))
    print("Unified class_to_idx:", class_to_idx)
    print("Train samples per class:", train_counts)
    print("Val samples per class:", val_counts)
    print("Test samples per class:", test_counts)

    # Arrêt immédiat si fuite détectée.
    path_overlap = leakage_report["path_overlap"]
    hash_overlap = leakage_report["hash_overlap"]
    if (
        path_overlap["train_val_duplicates"]
        or path_overlap["train_test_duplicates"]
        or path_overlap["val_test_duplicates"]
        or hash_overlap["train_val_duplicates"]
        or hash_overlap["train_test_duplicates"]
        or hash_overlap["val_test_duplicates"]
    ):
        raise RuntimeError(
            "Leakage detected before evaluation.\n"
            f"path_overlap={path_overlap}\n"
            f"hash_overlap={hash_overlap}"
        )

    plt.figure(figsize=(10, 6))
    classes_sorted = sorted(train_counts.keys())
    counts_sorted = [train_counts[c] for c in classes_sorted]
    plt.bar(range(len(classes_sorted)), counts_sorted)
    plt.xticks(range(len(classes_sorted)), classes_sorted, rotation=45, ha="right")
    plt.title("Training Set Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(data_exp_dir / "class_distribution.png", dpi=300)
    plt.close()

    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values()) if val_counts else int(tf.data.experimental.cardinality(val_ds).numpy()) * BATCH_SIZE
    total_test = sum(test_counts.values()) if test_counts else int(tf.data.experimental.cardinality(test_ds).numpy()) * BATCH_SIZE

    plt.figure(figsize=(6, 4))
    text_lines = [
        f"Number of classes: {NUM_CLASSES}",
        f"Train images: {total_train}",
        f"Val images: {total_val}",
        f"Test images: {total_test}",
        f"Image size: {IMG_SIZE[0]}x{IMG_SIZE[1]}",
        f"Batch size: {BATCH_SIZE}",
    ]
    plt.axis("off")
    plt.text(0.01, 0.99, "\n".join(text_lines),
             va="top", ha="left", fontsize=12)
    plt.title("Dataset Statistics", pad=20)
    plt.tight_layout()
    plt.savefig(data_exp_dir / "dataset_statistics.png", dpi=300)
    plt.close()

    def collect_sample_paths(root_dir: Path, max_per_class=3):
        sample_paths = []
        for class_name in sorted(os.listdir(root_dir)):
            class_path = root_dir / class_name
            if not class_path.is_dir():
                continue
            images = [
                class_path / f for f in os.listdir(class_path)
                if (class_path / f).is_file()
            ]
            images = images[:max_per_class]
            for p in images:
                sample_paths.append((p, class_name))
        return sample_paths

    sample_paths = collect_sample_paths(paths["train_dir"], max_per_class=2)
    num_samples = len(sample_paths)
    cols = 4
    rows = int(np.ceil(num_samples / cols)) if num_samples > 0 else 1

    plt.figure(figsize=(cols * 3, rows * 3))
    for idx, (img_path, cls) in enumerate(sample_paths):
        plt.subplot(rows, cols, idx + 1)
        img = load_img(img_path, target_size=IMG_SIZE)
        plt.imshow(img)
        plt.title(cls)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(data_exp_dir / "sample_images.png", dpi=300)
    plt.close()

    # Independent test evaluation for generalization metrics.
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise RuntimeError(
            f"y_true and y_pred length mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}"
        )

    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall:    {recall_macro:.4f}")
    print(f"Macro F1-score:  {f1_macro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall:    {recall_weighted:.4f}")
    print(f"Weighted F1-score:  {f1_weighted:.4f}")
    print("Preview true/pred (first 15):")
    for idx in range(min(15, len(y_true))):
        print(f"  #{idx:02d} true={class_names[int(y_true[idx])]} pred={class_names[int(y_pred[idx])]}")

    all_labels = list(range(NUM_CLASSES))
    class_report_dict = classification_report(
        y_true, y_pred, labels=all_labels, target_names=class_names,
        output_dict=True, zero_division=0
    )
    class_report_text = classification_report(
        y_true, y_pred, labels=all_labels, target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(class_report_text)

    np.save(eval_dir / "test_confusion_matrix.npy", cm)

    with open(eval_dir / "test_class_report.json", "w", encoding="utf-8") as f:
        json.dump(class_report_dict, f, ensure_ascii=False, indent=2)
    with open(eval_dir / "test_class_report.txt", "w", encoding="utf-8") as f:
        f.write(class_report_text)

    test_metrics = {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "test_shuffle": False,
    }
    with open(eval_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    per_class_f1 = [class_report_dict[c]["f1-score"] for c in class_names]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(class_names)), per_class_f1)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.ylabel("F1-score")
    plt.title("Per-class F1-score on Test Set")
    plt.tight_layout()
    plt.savefig(eval_dir / "class_performance.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation="nearest")
    plt.colorbar(im)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Test Set)")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(eval_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    metrics_values = [acc, precision_weighted, recall_weighted, f1_weighted]

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(metrics_names)), metrics_values)
    plt.xticks(range(len(metrics_names)), metrics_names, rotation=0)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Overall Performance on Test Set")
    plt.tight_layout()
    plt.savefig(eval_dir / "performance_metrics.png", dpi=300)
    plt.close()

    training_config = {
        "IMG_SIZE": IMG_SIZE,
        "BATCH_SIZE": train_cfg["batch_size"],
        "INITIAL_EPOCHS": train_cfg["initial_epochs"],
        "FINE_TUNE_EPOCHS": train_cfg["fine_tune_epochs"],
        "num_classes": NUM_CLASSES,
        "class_names": class_names,
        "model_config": train_cfg["model_config"],
    }
    with open(train_results_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(training_config), f, ensure_ascii=False, indent=2)

    combined_history = {
        "stage1": history_stage1.history,
        "stage2": history_stage2.history if history_stage2 is not None else {},
    }

    combined_history = _to_json_safe(combined_history)

    with open(train_results_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(combined_history, f, ensure_ascii=False, indent=2)

    acc1 = history_stage1.history.get("accuracy", [])
    val_acc1 = history_stage1.history.get("val_accuracy", [])
    loss1 = history_stage1.history.get("loss", [])
    val_loss1 = history_stage1.history.get("val_loss", [])

    acc2 = history_stage2.history.get("accuracy", []) if history_stage2 else []
    val_acc2 = history_stage2.history.get("val_accuracy", []) if history_stage2 else []
    loss2 = history_stage2.history.get("loss", []) if history_stage2 else []
    val_loss2 = history_stage2.history.get("val_loss", []) if history_stage2 else []

    epochs1 = list(range(1, len(acc1) + 1))
    epochs2 = list(range(len(acc1) + 1, len(acc1) + len(acc2) + 1))

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs1, acc1, label="Train Acc (Stage1)")
    plt.plot(epochs1, val_acc1, label="Val Acc (Stage1)")
    if acc2:
        plt.plot(epochs2, acc2, label="Train Acc (Stage2)")
    if val_acc2:
        plt.plot(epochs2, val_acc2, label="Val Acc (Stage2)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs1, loss1, label="Train Loss (Stage1)")
    plt.plot(epochs1, val_loss1, label="Val Loss (Stage1)")
    if loss2:
        plt.plot(epochs2, loss2, label="Train Loss (Stage2)")
    if val_loss2:
        plt.plot(epochs2, val_loss2, label="Val Loss (Stage2)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(train_results_dir / "training_history.png", dpi=300)
    plt.close()

    # best metrics
    best_val_acc_stage2 = max(
        history_stage2.history.get("val_accuracy", [0.0])
    ) if history_stage2 and history_stage2.history.get("val_accuracy") else 0.0
    best_val_acc_overall = max(best_val_acc_stage1, best_val_acc_stage2)
    print(f"Stage 1 best val accuracy (summary): {best_val_acc_stage1:.4f}")
    print(f"Stage 2 best val accuracy (summary): {best_val_acc_stage2:.4f}")
    print(f"Final validation accuracy: {val_accuracy:.4f}")
    print(f"Final test accuracy: {acc:.4f}")

    best_metrics = {
        "best_val_acc_stage1": float(best_val_acc_stage1),
        "best_val_acc_stage2": float(best_val_acc_stage2),
        "best_val_acc_overall": float(best_val_acc_overall),
        "final_val_loss": float(val_loss),
        "final_val_accuracy": float(val_accuracy),
        "test_accuracy": float(acc),
        "test_precision_macro": float(precision_macro),
        "test_recall_macro": float(recall_macro),
        "test_f1_macro": float(f1_macro),
        "test_precision_weighted": float(precision_weighted),
        "test_recall_weighted": float(recall_weighted),
        "test_f1_weighted": float(f1_weighted),
        "class_to_idx": class_to_idx,
        "leakage_report": leakage_report,
    }

    with open(logs_dir / "best_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(best_metrics), f, ensure_ascii=False, indent=2)

    with open(logs_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(combined_history, f, ensure_ascii=False, indent=2)

    print(f"Data exploration saved to {paths['data_exploration_dir']}")
    print(f"Evaluation results saved to {paths['evaluation_results_dir']}")
    print(f"Training results saved to {paths['training_results_dir']}")
    print(f"Training logs saved to {paths['training_logs_dir']}")


def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    elif isinstance(obj, (np.floating, tf.Tensor)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    else:
        return obj

def run(cfg: dict):
    # End-to-end flow: config -> datasets -> training -> artifacts.
    train_cfg = cfg["train_config"]
    sys_cfg = cfg["sys_config"]
    comp_cfg = cfg["compilation_config"]

    _apply_sys_config(sys_cfg, train_cfg)
    # Étape obligatoire pour éviter la contamination entre train/val/test.
    _prepare_clean_dataset_split(train_cfg)

    img_size = _get_img_size(train_cfg)
    paths = _build_paths(train_cfg)
    _sanitize_datasets_if_needed(train_cfg, paths)

    train_ds, val_ds, test_ds, class_names, class_to_idx, num_classes, dataset_meta = _build_datasets(
        train_cfg, paths, img_size
    )

    model_cfg = train_cfg["model_config"]
    model, history_stage1, history_stage2, best_val_acc_stage1 = _build_and_train_model(
        train_cfg, model_cfg, comp_cfg, img_size, num_classes, train_ds, val_ds
    )

    _evaluate_and_save(
        train_cfg, comp_cfg, paths,
        model,
        train_ds, val_ds, test_ds,
        class_names, class_to_idx, dataset_meta,
        history_stage1, history_stage2,
        best_val_acc_stage1,
    )
