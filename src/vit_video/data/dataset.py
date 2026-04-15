from __future__ import annotations

import logging
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
from collections import Counter, defaultdict
from typing import List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

from vit_video.data.splits import (
    keys_from_manifest_split,
    load_split_manifest,
    manifest_path_for_frames_dir,
    sync_manifest_with_frames_dir,
    video_stem_from_path,
    warn_if_new_videos_not_in_manifest,
    write_split_manifest,
)

_STEM_PATTERN = re.compile(r"^(.+)_frame_\d+$")


class VideoDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        classes: Optional[List[str]] = None,
        frames_per_video: int = 16,
        img_size: int = 224,
        augment: bool = False,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> None:
        self.root = Path(root)
        top_dirs = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
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
        self._random_temporal = bool(augment)
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

        t_list: list = []
        if augment:
            t_list.extend([
                transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ])
        else:
            t_list.append(transforms.Resize((img_size, img_size)))
        t_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        if augment:
            t_list.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)))
        self.transform = transforms.Compose(t_list)

        image_exts = (".png", ".jpg", ".jpeg", ".webp")
        video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")

        if top_dirs and set(top_dirs).issubset(container_names):
            for c in self.classes:
                for cont_name in top_dirs:
                    class_dir = self.root / cont_name / c
                    if not class_dir.exists():
                        continue
                    for item in class_dir.iterdir():
                        if item.is_dir() or item.suffix.lower() in video_exts or item.suffix.lower() in image_exts:
                            self.items.append((item, self.class_to_idx[c]))
        else:
            for c in self.classes:
                class_dir = self.root / c
                if not class_dir.exists():
                    continue
                for item in class_dir.iterdir():
                    if item.is_dir() or item.suffix.lower() in video_exts or item.suffix.lower() in image_exts:
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
            # Evenly spread available frames across slots to avoid last-frame bias
            indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frames.append(self.transform(Image.fromarray(frame)))
            else:
                logger.debug("Frame %d failed to decode in %s", idx, video_path)
                if frames:
                    frames.append(frames[-1].clone())
                else:
                    frames.append(torch.zeros(3, self.img_size, self.img_size))

        cap.release()
        return torch.stack(frames, dim=0)

    def _pick_frame_paths(self, paths: List[Path]) -> List[Path]:
        paths = sorted(paths)
        n = self.frames_per_video
        if len(paths) >= n:
            if self._random_temporal:
                pick = sorted(random.sample(range(len(paths)), n))
                return [paths[i] for i in pick]
            # Evenly-spaced sampling across the full video
            indices = np.linspace(0, len(paths) - 1, n, dtype=int)
            return [paths[i] for i in indices]
        if not paths:
            raise RuntimeError("No frame paths to sample")
        # Spread available frames evenly across n slots
        indices = np.linspace(0, len(paths) - 1, n, dtype=int)
        return [paths[i] for i in indices]

    def _tensor_stack_from_paths(self, paths: List[Path]) -> torch.Tensor:
        tensors = []
        for p in paths:
            with Image.open(str(p)) as img_obj:
                tensors.append(self.transform(img_obj.convert("RGB")))
        return torch.stack(tensors, dim=0)

    def _same_video_frame_paths(self, path: Path) -> List[Path]:
        stem_key = video_stem_from_path(path)
        exts = (".png", ".jpg", ".jpeg")
        return sorted(
            p for p in path.parent.iterdir()
            if p.suffix.lower() in exts and video_stem_from_path(p) == stem_key
        )

    def _load_video_from_dir(self, d: Path) -> torch.Tensor:
        paths = [p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        if not paths:
            raise RuntimeError(f"No frames found in {d}")
        return self._tensor_stack_from_paths(self._pick_frame_paths(paths))

    def _load_video_from_image(self, img: Path) -> torch.Tensor:
        with Image.open(str(img)) as pil_img:
            img_obj = pil_img.convert("RGB").resize((self.img_size, self.img_size))
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
            elif path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                mates = self._same_video_frame_paths(path)
                if _STEM_PATTERN.match(path.stem) or len(mates) > 1:
                    vid = self._tensor_stack_from_paths(self._pick_frame_paths(mates))
                else:
                    vid = self._load_video_from_image(path)
            else:
                vid = self._load_video_from_image(path)
        except Exception as exc:
            logger.warning("Failed to load %s: %s — returning zero tensor.", path, exc)
            vid = torch.zeros(self.frames_per_video, 3, self.img_size, self.img_size)
        return vid, label


def _video_level_split(
    items: List[Tuple[Path, int]], val_split: float, seed: int,
) -> Tuple[List[int], List[int]]:
    video_groups: dict[Tuple[int, str], List[int]] = defaultdict(list)
    for idx, (path, label) in enumerate(items):
        video_groups[(label, video_stem_from_path(path))].append(idx)

    group_keys = list(video_groups.keys())
    group_labels = [k[0] for k in group_keys]

    cnt = Counter(group_labels)
    do_stratify = all(v >= 2 for v in cnt.values()) and len(group_keys) >= 2

    if do_stratify:
        train_gk, val_gk = train_test_split(
            group_keys, test_size=val_split, stratify=group_labels, random_state=seed,
        )
    else:
        train_gk, val_gk = train_test_split(
            group_keys, test_size=val_split, random_state=seed,
        )

    train_idx = [i for k in set(train_gk) for i in video_groups[k]]
    val_idx = [i for k in set(val_gk) for i in video_groups[k]]

    print(
        f"[Split] Video-level split: {len(set(train_gk))} train videos "
        f"({len(train_idx)} frames), {len(set(val_gk))} val videos "
        f"({len(val_idx)} frames)"
    )
    return train_idx, val_idx


def _indices_for_video_keys(
    items: List[Tuple[Path, int]], classes: List[str], keys: Set[Tuple[str, str]],
) -> List[int]:
    return [
        i for i, (path, label) in enumerate(items)
        if (classes[label], video_stem_from_path(path)) in keys
    ]


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
    seed: int = 42,
    split_manifest: Optional[Union[str, Path]] = None,
    auto_write_manifest: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    base_ds = VideoDataset(
        root=dataset_root, frames_per_video=frames_per_video,
        img_size=img_size, augment=False, mean=norm_mean, std=norm_std,
    )
    if len(base_ds) == 0:
        raise RuntimeError(f"No data found in {dataset_root}")

    root = Path(dataset_root)
    mpath = Path(split_manifest) if split_manifest else manifest_path_for_frames_dir(root)

    if mpath.exists():
        sync_manifest_with_frames_dir(
            root, mpath, train_ratio, val_ratio, test_ratio, seed,
        )
        manifest = load_split_manifest(mpath)
        warn_if_new_videos_not_in_manifest(root, manifest)
        train_keys = keys_from_manifest_split(manifest, "train")
        val_keys = keys_from_manifest_split(manifest, "val")
        train_idx = _indices_for_video_keys(base_ds.items, base_ds.classes, train_keys)
        val_idx = _indices_for_video_keys(base_ds.items, base_ds.classes, val_keys)
        n_test = len(keys_from_manifest_split(manifest, "test"))
        print(
            f"[Split] Manifest {mpath.name}: "
            f"{len(train_keys)} train videos ({len(train_idx)} frame-rows), "
            f"{len(val_keys)} val videos ({len(val_idx)} frame-rows), "
            f"{n_test} test videos (held out for test.py)"
        )
        if not train_idx or not val_idx:
            raise RuntimeError(
                "Manifest produced empty train or val indices. "
                "Try --regenerate-splits or check frames_root in the manifest."
            )
    elif auto_write_manifest:
        write_split_manifest(root, mpath, train_ratio, val_ratio, test_ratio, seed)
        manifest = load_split_manifest(mpath)
        train_keys = keys_from_manifest_split(manifest, "train")
        val_keys = keys_from_manifest_split(manifest, "val")
        train_idx = _indices_for_video_keys(base_ds.items, base_ds.classes, train_keys)
        val_idx = _indices_for_video_keys(base_ds.items, base_ds.classes, val_keys)
        n_test = len(keys_from_manifest_split(manifest, "test"))
        print(
            f"[Split] Created manifest {mpath.name}: "
            f"{len(train_keys)} train / {len(val_keys)} val / {n_test} test videos"
        )
    else:
        train_idx, val_idx = _video_level_split(base_ds.items, val_split, seed)

    train_ds = VideoDataset(
        root=dataset_root, classes=base_ds.classes,
        frames_per_video=frames_per_video, img_size=img_size,
        augment=train_augment, mean=norm_mean, std=norm_std,
    )
    val_ds = VideoDataset(
        root=dataset_root, classes=base_ds.classes,
        frames_per_video=frames_per_video, img_size=img_size,
        augment=False, mean=norm_mean, std=norm_std,
    )

    train_subset = torch.utils.data.Subset(train_ds, train_idx)
    val_subset = torch.utils.data.Subset(val_ds, val_idx)

    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, base_ds.classes
