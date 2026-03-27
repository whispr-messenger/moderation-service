import sys
import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        except Exception:
            pass
        return torch.device("cuda")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def print_device_info() -> None:
    """Print device and CUDA availability info."""
    device = get_device()
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("\n[!] GPU not available. To enable CUDA:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print()

def get_num_workers() -> int:
    """Get platform-aware number of workers for DataLoader."""
    return 0 if sys.platform == "win32" else 4
