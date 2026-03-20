from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

import timm

try:
    from diffusers import StableDiffusionPipeline
except Exception:
    StableDiffusionPipeline = None

from tqdm import tqdm


def get_device() -> torch.device:
    if torch.cuda.is_available():
        # Enable cuDNN autotuner for faster convolutions on GPU.
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        except Exception:
            pass
        return torch.device("cuda")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
                if feat_dim == 640:
                    return "mobilevit_s"
                elif feat_dim == 384:
                    return "mobilevit_xs"
        return "mobilevit_s"
    return "mobilevit_s"


def extract_state_dict(checkpoint: object) -> dict:
    """Normalize checkpoint payloads to a plain state_dict."""
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
        nk = k.replace('module.', '')
        if nk.startswith('vit.'):
            nk = 'backbone.' + nk[4:]
        if nk.startswith('heads.head.'):
            nk = 'classifier.' + nk[11:]
        new_sd[nk] = v
    return new_sd


class DataGenerator:

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_with_diffusers(
        self,
        prompts: Dict[str, str],
        num_images_per_class: int = 50,
        device: Optional[torch.device] = None,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
    ) -> Dict[str, List[Path]]:
        if StableDiffusionPipeline is None:
            raise RuntimeError(
                "diffusers StableDiffusionPipeline is not available in the venv."
            )

        device = device or get_device()
        results: Dict[str, List[Path]] = {}

        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            safety_checker=None,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        pipeline = pipeline.to(device)

        rng = random.Random(seed)

        for label, prompt in prompts.items():
            label_dir = self.output_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            saved_paths: List[Path] = []
            for i in range(num_images_per_class):
                generator = None
                if seed is not None:
                    generator = torch.Generator(device=device).manual_seed(
                        rng.randint(0, 2 ** 31 - 1)
                    )

                out = pipeline(
                    prompt,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                    num_inference_steps=20,
                )
                image = out.images[0]
                path = label_dir / f"{label}_{i:05d}.png"
                image.save(path)
                saved_paths.append(path)
            results[label] = saved_paths

        return results

    def ensure_existing_examples(
        self, source_examples: Dict[str, List[Path]]
    ) -> Dict[str, List[Path]]:
        results: Dict[str, List[Path]] = {}
        for label, paths in source_examples.items():
            label_dir = self.output_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            dests: List[Path] = []
            for i, p in enumerate(paths):
                target = label_dir / f"{label}_{i:05d}{Path(p).suffix}"
                if not Path(p).exists():
                    continue
                if not target.exists():
                    try:
                        Image.open(p).save(target)
                    except Exception:
                        continue
                dests.append(target)
            results[label] = dests
        return results


class VideoProcessor:

    def __init__(self, num_frames: int = 16, out_frame_size: Tuple[int, int] = (224, 224)):
        self.num_frames = num_frames
        self.out_frame_size = out_frame_size

    def _frame_transform(
        self, image: Image.Image, t: float, max_rotation: float = 4.0, max_scale: float = 0.06
    ) -> Image.Image:
        angle = math.sin(2 * math.pi * t) * max_rotation
        scale = 1.0 + math.sin(2 * math.pi * t + math.pi / 4) * max_scale

        w, h = image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = image.resize((new_w, new_h), Image.BICUBIC)
        img = ImageOps.fit(img, (w, h), method=Image.BICUBIC, centering=(0.5, 0.5))

        img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
        img = img.resize(self.out_frame_size, Image.BICUBIC)
        return img

    def image_to_video_frames(self, image_path: Path) -> List[Image.Image]:
        with Image.open(image_path) as img_obj:
            img = img_obj.convert("RGB")
        frames: List[Image.Image] = []
        for f in range(self.num_frames):
            t = f / max(1, (self.num_frames - 1))
            frame = self._frame_transform(img, t)
            frames.append(frame)
        return frames

    def save_video_frames(
        self, frames: List[Image.Image], out_dir: Path, prefix: str
    ) -> List[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        for i, frame in enumerate(frames):
            p = out_dir / f"{prefix}_frame_{i:03d}.png"
            frame.save(p)
            paths.append(p)
        return paths


class VideoDataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        classes: Optional[List[str]] = None,
        frames_per_video: int = 16,
        transform: Optional[transforms.Compose] = None,
        img_size: int = 224,
        augment: bool = False,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> None:
        self.root = Path(root)
        top_dirs = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        # Support nested layout: root/frames/healthy, root/raw_videos/healthy, etc.
        container_names = {"frames", "raw_videos"}
        if classes is not None:
            self.classes = classes
        elif top_dirs and set(top_dirs).issubset(container_names):
            class_set = set()
            for name in top_dirs:
                for sub in (self.root / name).iterdir():
                    if sub.is_dir():
                        class_set.add(sub.name)
            self.classes = sorted(class_set)
        else:
            self.classes = top_dirs

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.items: List[Tuple[Path, int]] = []
        self.frames_per_video = frames_per_video
        self.img_size = img_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        if transform is not None:
            self.transform = transform
        else:
            t_list: List[object] = []
            if augment:
                # Light spatial/color augmentation to improve generalization.
                t_list.extend(
                    [
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ColorJitter(
                            brightness=0.15,
                            contrast=0.15,
                            saturation=0.15,
                            hue=0.02,
                        ),
                    ]
                )
            t_list.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        self.mean,
                        self.std,
                    ),
                ]
            )
            self.transform = transforms.Compose(t_list)

        image_exts = (".png", ".jpg", ".jpeg")
        video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")

        if top_dirs and set(top_dirs).issubset(container_names):
            for c in self.classes:
                for cont_name in top_dirs:
                    class_dir = self.root / cont_name / c
                    if not class_dir.exists():
                        continue
                    for item in class_dir.iterdir():
                        if item.is_dir():
                            self.items.append((item, self.class_to_idx[c]))
                        elif item.suffix.lower() in video_exts:
                            self.items.append((item, self.class_to_idx[c]))
                        elif item.suffix.lower() in image_exts:
                            self.items.append((item, self.class_to_idx[c]))
        else:
            for c in self.classes:
                class_dir = self.root / c
                if not class_dir.exists():
                    continue
                for item in class_dir.iterdir():
                    if item.is_dir():
                        self.items.append((item, self.class_to_idx[c]))
                    elif item.suffix.lower() in video_exts:
                        self.items.append((item, self.class_to_idx[c]))
                    elif item.suffix.lower() in image_exts:
                        self.items.append((item, self.class_to_idx[c]))

    def __len__(self) -> int:
        return len(self.items)

    def _load_video_from_file(self, video_path: Path) -> torch.Tensor:
        import cv2
        import numpy as np
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise RuntimeError(f"Could not read video: {video_path}")
        
        if total_frames >= self.frames_per_video:
            indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
        else:
            indices = np.concatenate([
                np.arange(total_frames),
                np.full(self.frames_per_video - total_frames, total_frames - 1)
            ]).astype(int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frame_pil = Image.fromarray(frame)
                frames.append(self.transform(frame_pil))
            else:
                # Fallback: repeat last frame or use zeros
                if frames:
                    frames.append(frames[-1].clone())
                else:
                    frames.append(torch.zeros(3, self.img_size, self.img_size))
        
        cap.release()
        return torch.stack(frames, dim=0)

    def _load_video_from_dir(self, d: Path) -> torch.Tensor:
        # load frames sorted
        paths = sorted([p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
        if len(paths) == 0:
            raise RuntimeError(f"No frames found in {d}")
        # optionally crop/pad to frames_per_video
        if len(paths) >= self.frames_per_video:
            paths = paths[: self.frames_per_video]
        else:
            # repeat last frame
            paths += [paths[-1]] * (self.frames_per_video - len(paths))

        tensors = []
        for p in paths:
            with Image.open(str(p)) as img_obj:
                tensors.append(self.transform(img_obj.convert("RGB")))
        # shape: (T, C, H, W)
        return torch.stack(tensors, dim=0)

    def _load_video_from_image(self, img: Path) -> torch.Tensor:
        # repeat the same image to build T frames
        with Image.open(str(img)) as pil_img:
            img_obj = pil_img.convert("RGB")
            img_obj = img_obj.resize((self.img_size, self.img_size))
        tensors = [self.transform(img_obj) for _ in range(self.frames_per_video)]
        return torch.stack(tensors, dim=0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")

        try:
            if path.is_dir():
                vid = self._load_video_from_dir(path)
            elif path.suffix.lower() in video_exts:
                vid = self._load_video_from_file(path)
            else:
                vid = self._load_video_from_image(path)
        except Exception:
            # Keep training/eval resilient to single corrupt samples.
            vid = torch.zeros(self.frames_per_video, 3, self.img_size, self.img_size)
        # return (T, C, H, W), label
        return vid, label


class MobileViTModel(nn.Module):

    def __init__(
        self,
        num_classes: int,
        model_name: str = "mobilevit_xxs",
        pretrained: bool = True,
        temporal_pool: str = "avg",
        dropout: float = 0.0,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.temporal_pool = temporal_pool
        self._use_torchvision = False

        torchvision_vit_models = {
            "vit_b_16": ("vit_b_16", 768),
            "vit_b_32": ("vit_b_32", 768),
            "vit_l_16": ("vit_l_16", 1024),
            "vit_l_32": ("vit_l_32", 1024),
            "vit_h_14": ("vit_h_14", 1280),
        }
        
        # Map our names to timm names when torchvision ViT is unavailable (e.g. "no module named vit")
        timm_vit_models = {
            "vit_b_16": "vit_base_patch16_224",
            "vit_b_32": "vit_base_patch32_224",
            "vit_l_16": "vit_large_patch16_224",
            "vit_l_32": "vit_large_patch32_224",
            "vit_h_14": "vit_huge_patch14_224.in21k_ft_in1k",
        }

        if model_name in torchvision_vit_models:
            backbone = None
            try:
                from torchvision.models.vision_transformer import (
                    vit_b_16,
                    vit_b_32,
                    vit_l_16,
                    vit_l_32,
                    vit_h_14,
                    ViT_B_16_Weights,
                    ViT_B_32_Weights,
                    ViT_L_16_Weights,
                    ViT_L_32_Weights,
                    ViT_H_14_Weights,
                )
                weight_classes = {
                    "vit_b_16": ViT_B_16_Weights,
                    "vit_b_32": ViT_B_32_Weights,
                    "vit_l_16": ViT_L_16_Weights,
                    "vit_l_32": ViT_L_32_Weights,
                    "vit_h_14": ViT_H_14_Weights,
                }
                vit_constructors = {
                    "vit_b_16": vit_b_16,
                    "vit_b_32": vit_b_32,
                    "vit_l_16": vit_l_16,
                    "vit_l_32": vit_l_32,
                    "vit_h_14": vit_h_14,
                }
                constructor = vit_constructors[model_name]
                weights_cls = weight_classes[model_name]
                weights = weights_cls.DEFAULT if pretrained else None
                backbone = constructor(weights=weights)
            except (ImportError, ModuleNotFoundError):
                pass

            if backbone is not None:
                feat_dim = torchvision_vit_models[model_name][1]
                backbone.heads.head = nn.Identity()
                self._use_torchvision = True
            else:
                # Fallback to timm ViT (avoids "no module named vit" from torchvision)
                if timm is None:
                    raise RuntimeError("timm is required for ViT backbones. Install with: pip install timm")
                timm_name = timm_vit_models.get(model_name, "vit_base_patch16_224")
                backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="avg")
                feat_dim = torchvision_vit_models[model_name][1]
                self._use_torchvision = False
        else:
            if timm is None:
                raise RuntimeError("timm is required for MobileViTModel. Please install timm in the venv.")

            try:
                backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
            except Exception:
                backbone = timm.create_model("mobilenetv3_small_100", pretrained=pretrained, num_classes=0, global_pool="avg")

            feat_dim = getattr(backbone, "num_features", None)
            if feat_dim is None:
                feat_dim = 256

        self.backbone = backbone
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.temporal_conv = None
        if temporal_pool == "conv1d":
            # Lightweight temporal modeling on top of per-frame embeddings.
            self.temporal_conv = nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1)
        self.classifier = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from frames using backbone.

        Accepts input x with shape (N, C, H, W) and returns (N, feat_dim).
        """
        if self._use_torchvision:
            # torchvision ViT: we've replaced heads.head with Identity
            # so forward() returns (N, feat_dim) features directly
            feats = self.backbone(x)
        elif hasattr(self.backbone, "forward_features"):
            # timm models: prefer forward_features to get pre-classifier features
            feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone(x)
        
        # Handle various output formats
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if isinstance(feats, dict):
            for v in feats.values():
                feats = v
                break
        
        # Ensure output is (N, feat_dim)
        if feats.dim() == 3:
            # (N, seq_len, feat_dim) - take CLS token or mean pool
            feats = feats[:, 0]
        elif feats.dim() == 4:
            # (N, C, H, W) - global average pool to (N, C)
            feats = feats.mean(dim=[2, 3])

        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self._extract_frame_features(x)  # (B*T, feat_dim)
        feat_dim = feats.shape[1]
        feats = feats.view(b, t, feat_dim)

        if self.temporal_pool == "avg":
            pooled = feats.mean(dim=1)
        elif self.temporal_pool == "max":
            pooled, _ = feats.max(dim=1)
        elif self.temporal_pool == "conv1d" and self.temporal_conv is not None:
            temporal = feats.transpose(1, 2)
            temporal = self.temporal_conv(temporal)
            pooled = temporal.mean(dim=2)
        else:
            pooled = feats.mean(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class Trainer:
    """Trainer class to run training and validation loops."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        output_path: Optional[Path] = None,
        max_grad_norm: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.output_path = Path(output_path) if output_path is not None else Path("./models")
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.max_grad_norm = max_grad_norm

        # Use AMP when running on CUDA for faster training and lower memory
        self.use_amp = True if getattr(device, "type", "cpu") == "cuda" else False
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    def _train_one_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        preds = []
        targets = []
        for x, y in tqdm(self.train_loader, desc="train", leave=False):
            x = x.to(self.device)
            y = y.to(self.device)
            # x shape: (B, T, C, H, W) -> model expects same
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                # Unscale then clip
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(x)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            targets.extend(y.cpu().numpy().tolist())

        avg_loss = running_loss / len(self.train_loader.dataset)
        acc = 100.0 * (sum(1 for i, j in zip(targets, preds) if i == j) / len(targets)) if len(targets) > 0 else 0.0
        return avg_loss, acc

    def _validate(self) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        preds = []
        targets = []
        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="val", leave=False):
                x = x.to(self.device)
                y = y.to(self.device)
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model(x)
                        loss = self.criterion(logits, y)
                else:
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                running_loss += loss.item() * x.size(0)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
                targets.extend(y.cpu().numpy().tolist())

        avg_loss = running_loss / len(self.val_loader.dataset)
        acc = 100.0 * (sum(1 for i, j in zip(targets, preds) if i == j) / len(targets)) if len(targets) > 0 else 0.0
        return avg_loss, acc

    def fit(
        self,
        epochs: int = 10,
        early_stopping_patience: int = 3,
        min_delta: float = 1e-4,
        checkpoint_name: str = "best_mobilevit.pth",
        resume_from: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        # Optionally resume
        if resume_from is not None and Path(resume_from).exists():
            ck = torch.load(resume_from, map_location=self.device)
            sd = extract_state_dict(ck)
            self.model.load_state_dict(remap_state_dict(sd), strict=False)
            if isinstance(ck, dict) and "optimizer_state_dict" in ck:
                try:
                    self.optimizer.load_state_dict(ck["optimizer_state_dict"])
                except Exception:
                    pass
            print(f"Resumed training from checkpoint: {resume_from}")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc = self._validate()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                ckpt = self.output_path / checkpoint_name
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "epoch": epoch,
                    },
                    ckpt,
                )
                print(f"Saved improved model to {ckpt} (val_loss={best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"No improvement (patience {patience_counter}/{early_stopping_patience})")

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        return history


def build_dataloaders(
    dataset_root: str | Path,
    frames_per_video: int = 16,
    batch_size: int = 8,
    val_split: float = 0.15,
    num_workers: int = 2,
    img_size: int = 224,
    train_augment: bool = True,
    norm_mean: Optional[List[float]] = None,
    norm_std: Optional[List[float]] = None,
):
    base_ds = VideoDataset(
        root=dataset_root,
        frames_per_video=frames_per_video,
        img_size=img_size,
        augment=False,
        mean=norm_mean,
        std=norm_std,
    )
    n = len(base_ds)
    if n == 0:
        raise RuntimeError(f"No data found in {dataset_root}")
    indices = list(range(n))
    labels = [base_ds.items[i][1] for i in indices]

    # Only stratify when there are at least 2 samples per class and val set is large enough
    do_stratify = True
    cnt = Counter(labels)
    n_classes = len(cnt)
    n_val = max(1, int(n * val_split))
    if any(v < 2 for v in cnt.values()) or n < 2 or n_val < n_classes:
        do_stratify = False

    if do_stratify:
        train_idx, val_idx = train_test_split(indices, test_size=val_split, stratify=labels)
    else:
        train_idx, val_idx = train_test_split(indices, test_size=val_split)

    # Separate train/val datasets so we can apply augmentation only on train.
    train_ds = VideoDataset(
        root=dataset_root,
        classes=base_ds.classes,
        frames_per_video=frames_per_video,
        img_size=img_size,
        augment=train_augment,
        mean=norm_mean,
        std=norm_std,
    )
    val_ds = VideoDataset(
        root=dataset_root,
        classes=base_ds.classes,
        frames_per_video=frames_per_video,
        img_size=img_size,
        augment=False,
        mean=norm_mean,
        std=norm_std,
    )

    train_subset = torch.utils.data.Subset(train_ds, train_idx)
    val_subset = torch.utils.data.Subset(val_ds, val_idx)

    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, base_ds.classes


def compute_class_weights_from_dataset(dataset: Dataset, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    label_counts = [0] * num_classes

    if isinstance(dataset, torch.utils.data.Subset):
        for idx in dataset.indices:
            _, label = dataset.dataset.items[idx]
            label_counts[label] += 1
    else:
        for _, label in getattr(dataset, "items", []):
            label_counts[label] += 1

    total = sum(label_counts)
    if total == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    weights = []
    for count in label_counts:
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def main_example():
    """Demonstrative main function. Adjust parameters for real runs."""
    device = get_device()
    print("Using device:", device)

    synthetic_root = Path("./data/synthetic_food_videos")
    synthetic_root.mkdir(parents=True, exist_ok=True)

    prompts = {
        "healthy": "A high-quality photo of a healthy food plate, fresh vegetables and balanced meal",
        "unhealthy": "A high-quality photo of an unhealthy fast food meal, greasy burger and fries",
    }

    generator = DataGenerator(output_dir=synthetic_root)
    try:
        generator.generate_with_diffusers(prompts, num_images_per_class=20, device=device)
    except Exception as e:
        print("Stable Diffusion generation skipped or failed:", e)
        # if generation fails, the user should populate synthetic_root manually

    # Convert images to frame folders
    vp = VideoProcessor(num_frames=16, out_frame_size=(224, 224))
    for label in prompts.keys():
        label_img_dir = synthetic_root / label
        label_img_dir.mkdir(parents=True, exist_ok=True)
        # if images were generated, convert them; otherwise assume images are already in dir
        image_files = sorted([p for p in label_img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg")])
        for i, img_path in enumerate(image_files):
            frames = vp.image_to_video_frames(img_path)
            out_folder = label_img_dir / f"video_{i:04d}"
            vp.save_video_frames(frames, out_folder, prefix=f"{label}_{i:04d}")

    train_loader, val_loader, classes = build_dataloaders(synthetic_root, frames_per_video=16, batch_size=4)

    # Create model
    model = MobileViTModel(num_classes=len(classes), model_name="mobilevit_xxs", pretrained=True)

    trainer = Trainer(model=model, device=device, train_loader=train_loader, val_loader=val_loader, output_path=Path("./models"))
    history = trainer.fit(epochs=10, early_stopping_patience=3)
    print("Training complete. History keys:", list(history.keys()))


if __name__ == "__main__":
    main_example()
