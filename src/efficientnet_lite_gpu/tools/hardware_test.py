import sys
import pkg_resources
from pathlib import Path
from .tools_nvidia_cuda import *

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def check_gpus():
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print(f"{GREEN}✅ GPU detected:{RESET}")
        for gpu in gpus:
            print(f"   - {gpu}")
        device = "gpu"

        check_nvidia_driver_and_cuda()
        check_nvcc()
        check_tf_cuda_build_info()
    else:
        print(f"{YELLOW}⚠️ No GPU detected. Falling back to CPU.{RESET}")
        device = "cpu"

    return device

def check_requirement(requirement_file="requirements.txt"):
    print("Checking requirements...")

    try:
        with open(requirement_file, "r") as f:
            requirements = f.readlines()
    except FileNotFoundError:
        print("Requirements file not found")
        return

    for requirement in requirements:
        requirement = requirement.strip()
        if not requirement or requirement.startswith("#"):
            continue

        try:
            pkg_resources.require(requirement)
            print(f"{GREEN}✅ {requirement} OK")
        except pkg_resources.DistributionNotFound:
            print(f"{RED}❌ Missing: {requirement}")
            sys.exit(1)
        except pkg_resources.VersionConflict as e:
            print(f"{YELLOW}⚠️ Version mismatch for {requirement}: {e}")

    print("Environment check complete.\n")

def check_python_version():
    if sys.version >= "3.10":
        print(f"{GREEN}✅ Python version: {sys.version} OK")
    else:
        print(f"{RED}❌ Python version: {sys.version} NOT OK")
        print(f"{RED} ------------- Python version requirement >= 3.11 -------------------")
        sys.exit(1)

def check_dataset_dir(root: str | Path, sub: str | None = None):
    root = Path(root)
    path = root / sub if sub else root

    if not path.exists():
        print(f"{RED}❌ Dataset directory does not exist: {path}{RESET}")
        sys.exit(1)
    if not any(path.iterdir()):
        print(f"{RED}❌ Dataset directory is empty: {path}{RESET}")
        sys.exit(1)

    print(f"{GREEN}✅ Dataset directory OK: {path}{RESET}")
    return path

def check_train_results_dir(cfg):
    results_dir = cfg["train_config"]["results_dir"]
    ensure_dir_writable(results_dir)

    data_exploration_dir = Path.cwd() / results_dir / cfg["train_config"]["data_exploration_dir"]
    ensure_dir_writable(data_exploration_dir)

    evaluation_results_dir = Path.cwd() / results_dir / cfg["train_config"]["evaluation_results_dir"]
    ensure_dir_writable(evaluation_results_dir)

    training_logs_dir = Path.cwd() / results_dir / cfg["train_config"]["training_logs_dir"]
    ensure_dir_writable(training_logs_dir)

    training_results_dir = Path.cwd() / results_dir / cfg["train_config"]["training_results_dir"]
    ensure_dir_writable(training_results_dir)

def _scan_bad_images(cfg):
    import tensorflow as tf

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    print(f"Scanning {cfg} ...")
    bad_files = []

    for p in cfg.rglob("*"):
        if not p.is_file():
            continue

        ext = p.suffix.lower()
        if ext not in supported_exts:
            print(f"[Format invalid] {p}")
            bad_files.append(p)
            continue

        try:
            img_bytes = tf.io.read_file(str(p))
            _ = tf.image.decode_image(img_bytes)  # 尝试解码
        except Exception as e:
            print(f"[Error decode] {p} -> {e}")
            bad_files.append(p)

    print(f"\nFound {len(bad_files)} error files。")
    return bad_files

def check_train_dataset_dir(cfg):
    # Check the train dataset
    dataset_dir = cfg["train_config"]["dataset_dir"]

    train_dataset_dir = Path.cwd() / dataset_dir / cfg["train_config"]["train_dir"]
    # val_dataset_dir = Path.cwd() / dataset_dir/ cfg["train_config"]["val_dir"]
    test_dataset_dir = Path.cwd() / dataset_dir / cfg["train_config"]["test_dir"]

    ensure_dir_writable(train_dataset_dir)
    # ensure_dir_writable(val_dataset_dir)
    ensure_dir_writable(test_dataset_dir)

    check_dataset_dir(train_dataset_dir)
    count_images_in_folder(train_dataset_dir, recursive=True)

    # check_dataset_dir(val_dataset_dir)
    # check_dataset_dir(val_dataset_dir , recursive=True)

    check_dataset_dir(test_dataset_dir)
    count_images_in_folder(test_dataset_dir, recursive=True)

    _scan_bad_images(train_dataset_dir)

def ensure_dir_writable(path: str | Path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    test_file = p / ".write_test"
    try:
        with open(test_file, "w") as f:
            f.write("test")
        test_file.unlink()
        print(f"{GREEN}✅ Output directory writable: {p}{RESET}")
    except Exception as e:
        print(f"{RED}❌ Cannot write to directory {p}: {e}{RESET}")
        sys.exit(1)

def count_images_in_folder(folder_path: str | Path, recursive: bool = False) -> int:
    folder = Path(folder_path)

    if not folder.exists():
        print(f"{RED}❌ Folder does not exist: {folder.resolve()}{RESET}")
        sys.exit(1)

    if not folder.is_dir():
        print(f"{RED}❌ Path is not a directory: {folder.resolve()}{RESET}")
        sys.exit(1)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

    if recursive:
        files = folder.rglob("*")
    else:
        files = folder.iterdir()

    count = 0
    for p in files:
        if p.is_file() and p.suffix.lower() in image_exts:
            if not p.name.startswith("."):
                count += 1

    if count > 0:
        print(f"{GREEN}✅ Found {count} images in: {folder.resolve()}{RESET}")
    else:
        print(f"{RED}❌ No valid images in: {folder.resolve()}{RESET}")
        sys.exit(1)

    return count