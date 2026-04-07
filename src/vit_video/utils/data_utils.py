from typing import List, Tuple
from torchvision import transforms


def parse_normalization_values(mean_str: str, std_str: str) -> Tuple[List[float], List[float]]:
    norm_mean = [float(x.strip()) for x in mean_str.split(",")] if mean_str else [0.485, 0.456, 0.406]
    norm_std = [float(x.strip()) for x in std_str.split(",")] if std_str else [0.229, 0.224, 0.225]
    if len(norm_mean) != 3 or len(norm_std) != 3:
        raise ValueError("Normalization mean/std must have exactly 3 comma-separated values.")
    return norm_mean, norm_std


def build_transform(mean: List[float], std: List[float]) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
