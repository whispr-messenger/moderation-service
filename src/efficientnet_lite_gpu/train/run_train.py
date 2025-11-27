import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import tensorflow as tf
# from tensorflow.keras import mixed_precision  # 混合精度需TensorFlow>=2.10

DATA_DIR = Path("../../Food Classification dataset/Train")
TEST_DIR = DATA_DIR.parent / "Test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
INITIAL_EPOCHS = 100
FINE_TUNE_EPOCHS = 100

# mixed_precision.set_global_policy('mixed_float16')

tf.config.optimizer.set_jit(False)  # 关闭XLA以避免无关日志
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
print("Physical GPUs:", tf.config.list_physical_devices('GPU'))

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode="int",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("Classes:", class_names)
print("Detected num_classes:", NUM_CLASSES)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="data_augmentation")

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,)
)

base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks_stage1 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
]
history_stage1 = model.fit(train_ds,
                           validation_data=val_ds,
                           epochs=INITIAL_EPOCHS,
                           callbacks=callbacks_stage1)

best_val_acc_stage1 = max(history_stage1.history["val_accuracy"])
print(f"Stage 1 best val accuracy: {best_val_acc_stage1:.4f}")

base_model.trainable = True
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_count = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
print(f"Trainable params: {trainable_count}, Non-trainable params: {non_trainable_count}")

callbacks_stage2 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]
history_stage2 = model.fit(train_ds,
                           validation_data=val_ds,
                           epochs=FINE_TUNE_EPOCHS,
                           callbacks=callbacks_stage2)

val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
print(f"Validation accuracy after fine-tuning: {val_accuracy:.4f}")

model.save("efficientnet_food_finetuned.h5")
print("✅ Model training complete and saved to efficientnet_food_finetuned.h5")

os.makedirs("../efficientnet_lite/data_exploration", exist_ok=True)

def count_images_in_dir(root_dir: Path):
    class_counts = {}
    for class_name in sorted(os.listdir(root_dir)):
        class_path = root_dir / class_name
        if not class_path.is_dir():
            continue
        num_images = sum(
            len(files)
            for _, _, files in os.walk(class_path)
        )
        class_counts[class_name] = num_images
    return class_counts

train_counts = count_images_in_dir(DATA_DIR)
test_counts  = count_images_in_dir(TEST_DIR) if TEST_DIR.exists() else {}

plt.figure(figsize=(10, 6))
classes_sorted = sorted(train_counts.keys())
counts_sorted = [train_counts[c] for c in classes_sorted]
plt.bar(range(len(classes_sorted)), counts_sorted)
plt.xticks(range(len(classes_sorted)), classes_sorted, rotation=45, ha="right")
plt.title("Training Set Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig("data_exploration/class_distribution.png", dpi=300)
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
plt.savefig("data_exploration/dataset_statistics.png", dpi=300)
plt.close()

from tensorflow.keras.utils import load_img, img_to_array

def collect_sample_paths(root_dir: Path, max_per_class=3):
    sample_paths = []
    for class_name in sorted(os.listdir(root_dir)):
        class_path = root_dir / class_name
        if not class_path.is_dir():
            continue
        images = [class_path / f for f in os.listdir(class_path)
                  if (class_path / f).is_file()]
        images = images[:max_per_class]
        for p in images:
            sample_paths.append((p, class_name))
    return sample_paths

sample_paths = collect_sample_paths(DATA_DIR, max_per_class=2)
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
plt.savefig("data_exploration/sample_images.png", dpi=300)
plt.close()

os.makedirs("../efficientnet_lite/evaluation_results", exist_ok=True)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_class_names = test_ds.class_names
print("Test classes:", test_class_names)

if test_class_names != class_names:
    print("⚠️ Warning: Test class names do not match training class names!")

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
    y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
)
cm = confusion_matrix(y_true, y_pred)

np.save("../efficientnet_lite/evaluation_results/test_confusion_matrix.npy", cm)

with open("../efficientnet_lite/evaluation_results/test_class_report.json", "w", encoding="utf-8") as f:
    json.dump(class_report_dict, f, ensure_ascii=False, indent=2)

test_metrics = {
    "accuracy": float(acc),
    "precision_weighted": float(precision),
    "recall_weighted": float(recall),
    "f1_weighted": float(f1),
}
with open("../efficientnet_lite/evaluation_results/test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(test_metrics, f, ensure_ascii=False, indent=2)

per_class_f1 = [class_report_dict[c]["f1-score"] for c in class_names]

plt.figure(figsize=(10, 6))
plt.bar(range(len(class_names)), per_class_f1)
plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
plt.ylabel("F1-score")
plt.title("Per-class F1-score on Test Set")
plt.tight_layout()
plt.savefig("evaluation_results/class_performance.png", dpi=300)
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
plt.savefig("evaluation_results/confusion_matrix.png", dpi=300)
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
plt.savefig("evaluation_results/performance_metrics.png", dpi=300)
plt.close()

os.makedirs("training_results", exist_ok=True)

training_config = {
    "DATA_DIR": str(DATA_DIR),
    "TEST_DIR": str(TEST_DIR),
    "IMG_SIZE": IMG_SIZE,
    "BATCH_SIZE": BATCH_SIZE,
    "INITIAL_EPOCHS": INITIAL_EPOCHS,
    "FINE_TUNE_EPOCHS": FINE_TUNE_EPOCHS,
    "optimizer": "Adam",
    "initial_lr": 1e-2,
    "fine_tune_lr": 1e-5,
    "num_classes": NUM_CLASSES,
    "class_names": class_names,
}
with open("training_results/training_config.json", "w", encoding="utf-8") as f:
    json.dump(training_config, f, ensure_ascii=False, indent=2)

combined_history = {
    "stage1": history_stage1.history,
    "stage2": history_stage2.history,
}

with open("training_results/training_history.json", "w", encoding="utf-8") as f:
    json.dump(combined_history, f, ensure_ascii=False, indent=2)

acc1 = history_stage1.history.get("accuracy", [])
val_acc1 = history_stage1.history.get("val_accuracy", [])
loss1 = history_stage1.history.get("loss", [])
val_loss1 = history_stage1.history.get("val_loss", [])

acc2 = history_stage2.history.get("accuracy", [])
val_acc2 = history_stage2.history.get("val_accuracy", [])
loss2 = history_stage2.history.get("loss", [])
val_loss2 = history_stage2.history.get("val_loss", [])

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
plt.savefig("training_results/training_history.png", dpi=300)
plt.close()

os.makedirs("../efficientnet_lite/training_logs", exist_ok=True)

best_val_acc_stage2 = max(history_stage2.history.get("val_accuracy", [0.0])) if history_stage2.history.get("val_accuracy") else 0.0
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

with open("../efficientnet_lite/training_logs/best_metrics.json", "w", encoding="utf-8") as f:
    json.dump(best_metrics, f, ensure_ascii=False, indent=2)

# 再保存一份训练历史到日志目录（可与results里相同或额外信息）
with open("../efficientnet_lite/training_logs/training_history.json", "w", encoding="utf-8") as f:
    json.dump(combined_history, f, ensure_ascii=False, indent=2)

print("📊 Data exploration saved to data_exploration/")
print("📈 Evaluation results saved to evaluation_results/")
print("📝 Training results saved to training_results/")
print("🧾 Training logs saved to training_logs/")