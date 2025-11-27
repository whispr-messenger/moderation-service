import os
import json
from pathlib import Path

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

    if sys_cfg.get("tf_fore_gpu_allow_growth", True):
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
        "train_dir": dataset_root / train_cfg["train_dir"],   # e.g. .../Train
        "val_dir":   dataset_root / train_cfg["val_dir"],     # e.g. .../Val（如果你以后用单独 Val）
        "test_dir":  dataset_root / train_cfg["test_dir"],    # e.g. .../Test

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


def _build_datasets(train_cfg: dict, paths: dict, img_size: tuple[int, int]):
    batch_size = train_cfg["batch_size"]

    train_dir = paths["train_dir"]
    test_dir = paths["test_dir"]

    print("Using train dir:", train_dir)
    print("Using test dir:", test_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode="int",
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode="int",
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Detected num_classes:", num_classes)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    test_class_names = test_ds.class_names
    print("Test classes:", test_class_names)
    if test_class_names != class_names:
        print("⚠️ Warning: Test class names do not match training class names!")

    return train_ds, val_ds, test_ds, class_names, num_classes


def _build_data_augmentation(train_cfg: dict) -> tf.keras.Sequential:
    da = train_cfg["data_augmentation"]
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip(da.get("randomFlip", "horizontal")),
        tf.keras.layers.RandomRotation(da.get("randomRotation", 0.05)),
        tf.keras.layers.RandomZoom(da.get("randomZoom", 0.1)),
        tf.keras.layers.RandomContrast(da.get("randomContrast", 0.1)),
    ], name="data_augmentation")


def _get_efficientnet_class(model_name: str):
    name = model_name.lower()
    if name in ("efficientnet-b0", "efficientnet_b0"):
        return tf.keras.applications.EfficientNetB0
    elif name in ("efficientnet-b1", "efficientnet_b1"):
        return tf.keras.applications.EfficientNetB1
    elif name in ("efficientnet-b2", "efficientnet_b2"):
        return tf.keras.applications.EfficientNetB2
    elif name in ("efficientnet-b3", "efficientnet_b3"):
        return tf.keras.applications.EfficientNetB3
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def _build_and_train_model(train_cfg: dict,
                           model_cfg: dict,
                           img_size: tuple[int, int],
                           num_classes: int,
                           train_ds,
                           val_ds):
    INITIAL_EPOCHS = train_cfg["initial_epochs"]
    FINE_TUNE_EPOCHS = train_cfg["fine_tune_epochs"]
    fine_tune = train_cfg["fine_tune"]

    data_augmentation = _build_data_augmentation(train_cfg)

    EfficientNetClass = _get_efficientnet_class(model_cfg["model_name"])

    base_model = EfficientNetClass(
        include_top=model_cfg["include_top"],
        weights=model_cfg["weights"],
        input_shape=img_size + (3,)
    )
    base_model.trainable = model_cfg["trainable"]

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation=model_cfg["output_activation"]
    )(x)

    model = tf.keras.Model(inputs, outputs)

    optimizer_name = model_cfg.get("optimizer", "adam")
    lr = float(model_cfg.get("learning_rate", 1e-2))

    optimizer = tf.keras.optimizers.get({
        "class_name": optimizer_name,
        "config": {"learning_rate": lr}
    })

    model.compile(
        optimizer=optimizer,
        loss=model_cfg["loss"],
        metrics=model_cfg["metrics"]
    )

    model.summary()

    es_cfg = model_cfg["EarlyStopping"]
    rlr_cfg = model_cfg["ReduceLROnPlateau"]

    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor=es_cfg["monitor"],
            patience=es_cfg["patience"],
            restore_best_weights=es_cfg["restore_best_weights"],
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=rlr_cfg["monitor"],
            factor=rlr_cfg["factor"],
            patience=rlr_cfg["patience"],
            min_lr=rlr_cfg["min_lr"],
        )
    ]

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks_stage1,
    )

    best_val_acc_stage1 = max(history_stage1.history["val_accuracy"])
    print(f"Stage 1 best val accuracy: {best_val_acc_stage1:.4f}")

    history_stage2 = None

    if fine_tune:
        base_model.trainable = True
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        fine_tune_lr = train_cfg.get("fine_tune_lr", lr / 100.0)
        optimizer_ft = tf.keras.optimizers.get({
            "class_name": optimizer_name,
            "config": {"learning_rate": fine_tune_lr}
        })

        model.compile(
            optimizer=optimizer_ft,
            loss=model_cfg["loss"],
            metrics=model_cfg["metrics"]
        )

        trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        non_trainable_count = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
        print(f"Trainable params: {trainable_count}, Non-trainable params: {non_trainable_count}")

        callbacks_stage2 = [
            tf.keras.callbacks.EarlyStopping(
                monitor=es_cfg["monitor"],
                patience=5,
                restore_best_weights=True,
            )
        ]

        history_stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=FINE_TUNE_EPOCHS,
            callbacks=callbacks_stage2,
        )

    return model, history_stage1, history_stage2, best_val_acc_stage1


def _evaluate_and_save(train_cfg: dict,
                       comp_cfg: dict,
                       paths: dict,
                       model,
                       train_ds,
                       val_ds,
                       test_ds,
                       class_names,
                       history_stage1,
                       history_stage2,
                       best_val_acc_stage1):

    NUM_CLASSES = len(class_names)
    IMG_SIZE = _get_img_size(train_cfg)
    BATCH_SIZE = train_cfg["batch_size"]

    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy after fine-tuning: {val_accuracy:.4f}")

    model_out_name = comp_cfg["model_name"]
    model.save(model_out_name)
    print(f"✅ Model training complete and saved to {model_out_name}")

    data_exp_dir = paths["data_exploration_dir"]
    eval_dir = paths["evaluation_results_dir"]
    train_results_dir = paths["training_results_dir"]
    logs_dir = paths["training_logs_dir"]

    def count_images_in_dir(root_dir: Path):
        class_counts = {}
        for class_name in sorted(os.listdir(root_dir)):
            class_path = root_dir / class_name
            if not class_path.is_dir():
                continue
            num_images = sum(len(files) for _, _, files in os.walk(class_path))
            class_counts[class_name] = num_images
        return class_counts

    train_counts = count_images_in_dir(paths["train_dir"])
    test_counts = count_images_in_dir(paths["test_dir"]) if paths["test_dir"].exists() else {}

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
    total_test = sum(test_counts.values()) if test_counts else 0

    plt.figure(figsize=(6, 4))
    text_lines = [
        f"Number of classes: {NUM_CLASSES}",
        f"Train images: {total_train}",
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

    y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1-score:  {f1:.4f}")

    class_report_dict = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    np.save(eval_dir / "test_confusion_matrix.npy", cm)

    with open(eval_dir / "test_class_report.json", "w", encoding="utf-8") as f:
        json.dump(class_report_dict, f, ensure_ascii=False, indent=2)

    test_metrics = {
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
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
    metrics_values = [acc, precision, recall, f1]

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
    best_metrics = {
        "best_val_acc_stage1": float(best_val_acc_stage1),
        "best_val_acc_stage2": float(best_val_acc_stage2),
        "best_val_acc_overall": float(best_val_acc_overall),
        "final_val_loss": float(val_loss),
        "final_val_accuracy": float(val_accuracy),
        "test_accuracy": float(acc),
        "test_precision_weighted": float(precision),
        "test_recall_weighted": float(recall),
        "test_f1_weighted": float(f1),
    }

    with open(logs_dir / "best_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(best_metrics), f, ensure_ascii=False, indent=2)

    combined_history = _to_json_safe(combined_history)

    with open(logs_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(combined_history, f, ensure_ascii=False, indent=2)

    print(f"📊 Data exploration saved to {paths['data_exploration_dir']}")
    print(f"📈 Evaluation results saved to {paths['evaluation_results_dir']}")
    print(f"📝 Training results saved to {paths['training_results_dir']}")
    print(f"🧾 Training logs saved to {paths['training_logs_dir']}")


def _to_json_safe(obj):
    import numpy as np
    import tensorflow as tf

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
    train_cfg = cfg["train_config"]
    sys_cfg = cfg["sys_config"]
    comp_cfg = cfg["compilation_config"]

    _apply_sys_config(sys_cfg, train_cfg)

    img_size = _get_img_size(train_cfg)
    paths = _build_paths(train_cfg)

    train_ds, val_ds, test_ds, class_names, num_classes = _build_datasets(
        train_cfg, paths, img_size
    )

    model_cfg = train_cfg["model_config"]
    model, history_stage1, history_stage2, best_val_acc_stage1 = _build_and_train_model(
        train_cfg, model_cfg, img_size, num_classes, train_ds, val_ds
    )

    _evaluate_and_save(
        train_cfg, comp_cfg, paths,
        model,
        train_ds, val_ds, test_ds,
        class_names,
        history_stage1, history_stage2,
        best_val_acc_stage1,
    )
