from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from .hardware import get_device
from ..models.vit import MobileViTModel


def detect_backbone_from_checkpoint(state_dict: dict) -> str:
    keys = list(state_dict.keys())

    if any(k.startswith("vit.") for k in keys):
        for k in keys:
            if "classifier" in k and "weight" in k:
                feat_dim = state_dict[k].shape[1]
                if feat_dim == 768:
                    return "vit_b_16"
                elif feat_dim == 1024:
                    return "vit_l_16"
        return "vit_b_16"

    if any(k.startswith("backbone.stages") for k in keys):
        for k in keys:
            if "classifier" in k and "weight" in k:
                feat_dim = state_dict[k].shape[1]
                if feat_dim == 320:
                    return "mobilevit_xxs"
                elif feat_dim == 384:
                    return "mobilevit_xs"
                elif feat_dim == 640:
                    return "mobilevit_s"
        return "mobilevit_s"

    return "mobilevit_s"


def extract_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint
    return checkpoint


def remap_state_dict(sd: dict) -> dict:
    new_sd = {}
    for k, v in sd.items():
        nk = k.replace("module.", "")
        if nk.startswith("vit."):
            nk = "backbone." + nk[4:]
        if nk.startswith("heads.head."):
            nk = "classifier." + nk[11:]
        new_sd[nk] = v
    return new_sd


def load_model_from_checkpoint(
    model_path: Path,
    num_classes: int,
    model_name: str = "auto",
    device: torch.device | None = None,
) -> nn.Module:
    device = device or get_device()
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)

    if model_name == "auto":
        backbone = None
        if isinstance(checkpoint, dict):
            backbone = checkpoint.get("metadata", {}).get("backbone") or checkpoint.get("backbone")
        if not backbone:
            backbone = detect_backbone_from_checkpoint(state_dict)
            print(f"Using backbone: {backbone} (detected from layer sizes)")
        else:
            print(f"Using backbone: {backbone} (from checkpoint metadata)")
    else:
        backbone = model_name

    model = MobileViTModel(num_classes=num_classes, model_name=backbone, pretrained=False)
    model.load_state_dict(remap_state_dict(state_dict), strict=False)
    model = model.to(device)
    model.eval()
    return model
