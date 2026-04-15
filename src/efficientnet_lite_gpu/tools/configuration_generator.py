import sys
import yaml
from .hardware_test import check_gpus
from pathlib import Path

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

CONFIG_PATH = Path("config.yaml")

def config(model_name : str):
    if CONFIG_PATH.exists():
        # Keep user-managed config values; do not overwrite existing file.
        return load_config()

    device = check_gpus()

    cfg = {
        "train_config": {
            "training_device" : device,
            "mixed_precision" : False,
            "dataset_dir" : "train/dataset",
            "results_dir" : "train/results",
            "data_exploration_dir" : "data_exploration",
            "evaluation_results_dir" : "evaluation_results",
            "training_logs_dir" : "training_logs",
            "training_results_dir" : "training_results",
            "train_dir" : "Train",
            "val_dir" : "Val",
            "test_dir" : "Test",
            "batch_size" : 8,
            "image_size" : 224,
            "initial_epochs" : 5,
            "fine_tune" : True,
            "fine_tune_epochs" : 2,
            "data_augmentation": {
                "randomFlip": "horizontal",
                "randomRotation": 0.05,
                "randomZoom": 0.1,
                "randomContrast": 0.1,
            },
            "model_config": {
                "model_name": model_name,
                "include_top" : False,
                "weights" : 'imagenet',
                "input_shape" : "(224, 224, 3)",
                "trainable" : False,
                "optimizer" : "adam",
                "output_activation" : "softmax",
                "learning_rate" : 1e-2,
                "loss" : "sparse_categorical_crossentropy",
                "metrics" : ["accuracy"],
                "EarlyStopping" : {
                    "monitor" : "val_loss",
                    "patience" : 8,
                    "restore_best_weights" : True,
                },
                "ReduceLROnPlateau" : {
                    "monitor" : "val_loss",
                    "factor" : 0.5,
                    "patience" : 3,
                    "min_lr" : 1e-6,
                }
            }
        },
        "sys_config": {
            "disable_XLA_logs" : True,
            "tf_force_gpu_allow_growth" : True,
        },
        "compilation_config": {
            "compiler" : "gcc",
            "compiler_args" : [],
            "model_name" : "BestModelEfficientNetLite.h5",
        }
    }

    try:
        CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return cfg
    except Exception as e:
        print(f"{RED} ❌ {e} !")
        print(f"{RED} ❌ Can't save config!")
        sys.exit(1)

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("config.yaml not found, please generate it first.")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
