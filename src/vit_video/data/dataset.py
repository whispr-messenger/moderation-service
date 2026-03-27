import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
from collections import Counter
from typing import List, Optional, Tuple

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
                t_list.extend([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
                ])
            t_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
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
                if frames:
                    frames.append(frames[-1].clone())
                else:
                    frames.append(torch.zeros(3, self.img_size, self.img_size))
        
        cap.release()
        return torch.stack(frames, dim=0)

    def _load_video_from_dir(self, d: Path) -> torch.Tensor:
        paths = sorted([p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
        if len(paths) == 0:
            raise RuntimeError(f"No frames found in {d}")
        if len(paths) >= self.frames_per_video:
            paths = paths[: self.frames_per_video]
        else:
            paths += [paths[-1]] * (self.frames_per_video - len(paths))

        tensors = []
        for p in paths:
            with Image.open(str(p)) as img_obj:
                tensors.append(self.transform(img_obj.convert("RGB")))
        return torch.stack(tensors, dim=0)

    def _load_video_from_image(self, img: Path) -> torch.Tensor:
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
            vid = torch.zeros(self.frames_per_video, 3, self.img_size, self.img_size)
        return vid, label


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

    do_stratify = True
    cnt = Counter(labels)
    n_classes = len(cnt)
    n_val = max(1, int(n * val_split))
    if any(v < 2 for v in cnt.values()) or n < 2 or n_val < n_classes:
        do_stratify = False

    if do_stratify:
        train_idx, val_idx = train_test_split(indices, test_size=val_split, stratify=labels, random_state=seed)
    else:
        train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=seed)

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
