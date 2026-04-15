from __future__ import annotations

import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from ..utils.model_utils import extract_state_dict, remap_state_dict


def compute_class_weights_from_dataset(dataset: Dataset, num_classes: int) -> torch.Tensor:
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

    weights = [total / (num_classes * c) if c > 0 else 0.0 for c in label_counts]
    return torch.tensor(weights, dtype=torch.float32)


class Trainer:
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

        label_smoothing = 0.1
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.output_path = Path(output_path) if output_path is not None else Path("./models")
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.max_grad_norm = max_grad_norm
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.warmup_epochs: int = 3

        self.use_amp = getattr(device, "type", "cpu") == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    def _train_one_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in tqdm(self.train_loader, desc="train", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
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
            correct += (torch.argmax(logits, dim=1) == y).sum().item()
            total += y.size(0)

        return running_loss / total, 100.0 * correct / total if total else 0.0

    def _validate(self) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="val", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                # Validate in fp32 -- AMP fp16 can overflow on extreme logits and produce NaN
                logits = self.model(x).float()
                loss = self.criterion(logits, y)
                if not torch.isfinite(loss):
                    continue
                running_loss += loss.item() * x.size(0)
                correct += (torch.argmax(logits, dim=1) == y).sum().item()
                total += y.size(0)

        return running_loss / total, 100.0 * correct / total if total else 0.0

    def fit(
        self,
        epochs: int = 10,
        early_stopping_patience: int = 3,
        min_delta: float = 1e-4,
        checkpoint_name: str = "best_mobilevit.pth",
        resume_from: Optional[Path] = None,
        drive_checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        best_val_loss = float("inf")
        patience_counter = 0
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, epochs - self.warmup_epochs), eta_min=1e-7
        )
        if self.warmup_epochs > 0 and epochs > self.warmup_epochs:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, total_iters=self.warmup_epochs
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs]
            )
        else:
            self.scheduler = cosine

        if resume_from is not None and Path(resume_from).exists():
            ck = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(remap_state_dict(extract_state_dict(ck)), strict=False)
            if isinstance(ck, dict) and "optimizer_state_dict" in ck:
                try:
                    self.optimizer.load_state_dict(ck["optimizer_state_dict"])
                except (ValueError, KeyError, RuntimeError) as e:
                    # Common when resuming with a different param group layout (e.g. freeze toggled).
                    print(f"[trainer] Optimizer state not restored ({e}); continuing with fresh optimizer.")
            print(f"Resumed from: {resume_from}")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc = self._validate()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"train_acc={train_acc:.2f}% val_acc={val_acc:.2f}% lr={lr:.2e}")

            self.scheduler.step()

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                ckpt = self.output_path / checkpoint_name
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                }, ckpt)
                print(f"Saved model to {ckpt} (val_loss={best_val_loss:.4f})")
                if drive_checkpoint_dir:
                    drive_ckpt = Path(drive_checkpoint_dir) / checkpoint_name
                    drive_ckpt.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(ckpt), str(drive_ckpt))
                    print(f"Synced checkpoint to Drive: {drive_ckpt}")
            else:
                patience_counter += 1
                print(f"No improvement (patience {patience_counter}/{early_stopping_patience})")

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        return history
