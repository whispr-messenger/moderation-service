import sys
from pathlib import Path

from .configuration_generator import load_config
from .hardware_test import RED, GREEN, YELLOW, RESET


def _require_key(d: dict, key: str, parent: str):
    if key not in d:
        print(f"{RED}❌ Missing key '{key}' in {parent}{RESET}")
        sys.exit(1)
    return d[key]


def _require_int(value, name: str, min_value: int | None = None):
    if not isinstance(value, int):
        print(f"{RED}❌ {name} must be int, got {type(value).__name__}{RESET}")
        sys.exit(1)
    if min_value is not None and value < min_value:
        print(f"{RED}❌ {name} must be >= {min_value}, got {value}{RESET}")
        sys.exit(1)


def _require_bool(value, name: str):
    if not isinstance(value, bool):
        print(f"{RED}❌ {name} must be bool, got {type(value).__name__}{RESET}")
        sys.exit(1)


def _require_str(value, name: str, allow_empty: bool = False):
    if not isinstance(value, str):
        print(f"{RED}❌ {name} must be str, got {type(value).__name__}{RESET}")
        sys.exit(1)
    if not allow_empty and not value.strip():
        print(f"{RED}❌ {name} must not be empty{RESET}")
        sys.exit(1)


def _require_list(value, name: str):
    if not isinstance(value, (list, tuple)):
        print(f"{RED}❌ {name} must be list/tuple, got {type(value).__name__}{RESET}")
        sys.exit(1)


def validate_config() -> dict:
    try:
        cfg = load_config()
    except FileNotFoundError:
        print(f"{RED}❌ config.yaml not found. Please generate config first.{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{RED}❌ Failed to load config.yaml: {e}{RESET}")
        sys.exit(1)

    train_cfg = _require_key(cfg, "train_config", "root config")
    sys_cfg = _require_key(cfg, "sys_config", "root config")
    comp_cfg = _require_key(cfg, "compilation_config", "root config")

    if not isinstance(train_cfg, dict) or not isinstance(sys_cfg, dict) or not isinstance(comp_cfg, dict):
        print(f"{RED}❌ train_config/sys_config/compilation_config must all be dict{RESET}")
        sys.exit(1)

    # ======================
    #   train_config
    # ======================
    training_device = _require_key(train_cfg, "training_device", "train_config")
    _require_str(training_device, "train_config.training_device")
    if training_device not in ("cpu", "gpu"):
        print(f"{YELLOW}⚠️ training_device is '{training_device}', expected 'cpu' or 'gpu'{RESET}")

    mixed_precision = _require_key(train_cfg, "mixed_precision", "train_config")
    _require_bool(mixed_precision, "train_config.mixed_precision")

    dataset_dir = _require_key(train_cfg, "dataset_dir", "train_config")
    _require_str(dataset_dir, "train_config.dataset_dir")
    dataset_dir_path = Path(dataset_dir)
    if not dataset_dir_path.exists():
        print(f"{RED}❌ dataset_dir does not exist: {dataset_dir_path.resolve()}{RESET}")
        sys.exit(1)
    if not dataset_dir_path.is_dir():
        print(f"{RED}❌ dataset_dir is not a directory: {dataset_dir_path.resolve()}{RESET}")
        sys.exit(1)

    train_dir = _require_key(train_cfg, "train_dir", "train_config")
    val_dir = _require_key(train_cfg, "val_dir", "train_config")
    test_dir = _require_key(train_cfg, "test_dir", "train_config")
    _require_str(train_dir, "train_config.train_dir")
    _require_str(val_dir, "train_config.val_dir")
    _require_str(test_dir, "train_config.test_dir")

    # Warn if subdirectory starts with "/" which breaks Path.join
    for name, sub in [("train_dir", train_dir), ("val_dir", val_dir), ("test_dir", test_dir)]:
        if sub.startswith("/"):
            print(f"{YELLOW}⚠️ {name} starts with '/', Path join will treat it as absolute path. "
                  f"Consider removing leading '/'. Current: {sub}{RESET}")

    batch_size = _require_key(train_cfg, "batch_size", "train_config")
    _require_int(batch_size, "train_config.batch_size", min_value=1)

    image_size = _require_key(train_cfg, "image_size", "train_config")
    if isinstance(image_size, int):
        if image_size <= 0:
            print(f"{RED}❌ train_config.image_size must be > 0, got {image_size}{RESET}")
            sys.exit(1)
    elif isinstance(image_size, (list, tuple)):
        if len(image_size) != 2 or any((not isinstance(v, int) or v <= 0) for v in image_size):
            print(f"{RED}❌ train_config.image_size as list must be [H, W] with positive ints, got {image_size}{RESET}")
            sys.exit(1)
    else:
        print(f"{RED}❌ train_config.image_size must be int or [H, W], got {type(image_size).__name__}{RESET}")
        sys.exit(1)

    initial_epochs = _require_key(train_cfg, "initial_epochs", "train_config")
    _require_int(initial_epochs, "train_config.initial_epochs", min_value=1)

    fine_tune = _require_key(train_cfg, "fine_tune", "train_config")
    _require_bool(fine_tune, "train_config.fine_tune")

    fine_tune_epochs = _require_key(train_cfg, "fine_tune_epochs", "train_config")
    _require_int(fine_tune_epochs, "train_config.fine_tune_epochs", min_value=0)

    # data_augmentation
    da_cfg = _require_key(train_cfg, "data_augmentation", "train_config")
    if not isinstance(da_cfg, dict):
        print(f"{RED}❌ train_config.data_augmentation must be dict{RESET}")
        sys.exit(1)

    # ======================
    #   model_config
    # ======================
    model_cfg = _require_key(train_cfg, "model_config", "train_config")
    if not isinstance(model_cfg, dict):
        print(f"{RED}❌ train_config.model_config must be dict{RESET}")
        sys.exit(1)

    model_name = _require_key(model_cfg, "model_name", "model_config")
    _require_str(model_name, "model_config.model_name")

    include_top = _require_key(model_cfg, "include_top", "model_config")
    _require_bool(include_top, "model_config.include_top")

    trainable = _require_key(model_cfg, "trainable", "model_config")
    _require_bool(trainable, "model_config.trainable")

    optimizer = _require_key(model_cfg, "optimizer", "model_config")
    _require_str(optimizer, "model_config.optimizer")

    output_activation = _require_key(model_cfg, "output_activation", "model_config")
    _require_str(output_activation, "model_config.output_activation")

    learning_rate = _require_key(model_cfg, "learning_rate", "model_config")
    if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
        print(f"{RED}❌ model_config.learning_rate must be > 0, got {learning_rate}{RESET}")
        sys.exit(1)

    loss = _require_key(model_cfg, "loss", "model_config")
    _require_str(loss, "model_config.loss")

    metrics = _require_key(model_cfg, "metrics", "model_config")
    _require_list(metrics, "model_config.metrics")
    if not metrics:
        print(f"{RED}❌ model_config.metrics must not be empty{RESET}")
        sys.exit(1)

    # EarlyStopping & ReduceLROnPlateau
    es_cfg = _require_key(model_cfg, "EarlyStopping", "model_config")
    rlr_cfg = _require_key(model_cfg, "ReduceLROnPlateau", "model_config")
    if not isinstance(es_cfg, dict) or not isinstance(rlr_cfg, dict):
        print(f"{RED}❌ EarlyStopping and ReduceLROnPlateau must be dict{RESET}")
        sys.exit(1)

    # ======================
    #   sys_config
    # ======================
    disable_xla = _require_key(sys_cfg, "disable_XLA_logs", "sys_config")
    _require_bool(disable_xla, "sys_config.disable_XLA_logs")

    allow_growth = _require_key(sys_cfg, "tf_force_gpu_allow_growth", "sys_config")
    _require_bool(allow_growth, "sys_config.tf_force_gpu_allow_growth")

    # ======================
    #   compilation_config
    # ======================
    compiler = _require_key(comp_cfg, "compiler", "compilation_config")
    _require_str(compiler, "compilation_config.compiler")

    compiler_args = _require_key(comp_cfg, "compiler_args", "compilation_config")
    if not isinstance(compiler_args, list):
        print(f"{RED}❌ compilation_config.compiler_args must be list{RESET}")
        sys.exit(1)

    model_out_name = _require_key(comp_cfg, "model_name", "compilation_config")
    _require_str(model_out_name, "compilation_config.model_name")

    print(f"{GREEN}✅ Config validation passed{RESET}")
    return cfg
