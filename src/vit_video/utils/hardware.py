import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        # cuDNN flags are best-effort; some torch builds without cuDNN raise AttributeError.
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        except (AttributeError, RuntimeError):
            pass
        return torch.device("cuda")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_device_info(*, hint_cuda: bool = False) -> None:
    """One-line device summary. Set ``hint_cuda=True`` to print CUDA wheel hint when on CPU."""
    device = get_device()
    cuda = torch.cuda.is_available()
    print(f"Device: {device}  (CUDA: {cuda})")
    if hint_cuda and not cuda:
        print(
            "Tip: CUDA build — pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu124"
        )
