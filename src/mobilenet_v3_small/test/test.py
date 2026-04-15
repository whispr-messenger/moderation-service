from pathlib import Path
import imghdr

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
TF_ACCEPTED_TYPES = {"bmp", "gif", "jpeg", "png"}


def scan_bad_images(root_dir: Path) -> list[Path]:
    """Scan a directory for corrupt or unsupported image files."""
    bad_files = []
    for filepath in root_dir.rglob("*"):
        if not filepath.is_file():
            continue
        ext = filepath.suffix.lower()
        if ext not in IMAGE_EXTS:
            continue
        img_type = imghdr.what(filepath)
        if img_type is None:
            print(f"Not a valid image (content): {filepath}")
            bad_files.append(filepath)
        elif img_type not in TF_ACCEPTED_TYPES:
            print(f"Type not supported by TF ({img_type}): {filepath}")
            bad_files.append(filepath)
    return bad_files


def run(cfg: dict):
    """Scan the training dataset for corrupt images."""
    train_cfg = cfg["train_config"]
    dataset_root = Path.cwd() / train_cfg["dataset_dir"]
    train_dir = dataset_root / train_cfg["train_dir"]
    test_dir = dataset_root / train_cfg["test_dir"]

    for label, directory in [("Train", train_dir), ("Test", test_dir)]:
        if not directory.exists():
            print(f"{label} directory not found: {directory}")
            continue
        print(f"\nScanning {label} directory: {directory}")
        bad = scan_bad_images(directory)
        print(f"Found {len(bad)} problematic files in {label}.")
