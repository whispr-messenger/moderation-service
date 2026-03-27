import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from ..utils.model_utils import extract_state_dict, remap_state_dict

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

        self.use_amp = True if getattr(device, "type", "cpu") == "cuda" else False
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    def _train_one_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        preds = []
        targets = []
        for x, y in tqdm(self.train_loader, desc="train", leave=False):
            x = x.to(self.device)
            y = y.to(self.device)
            if self.use_amp:
                with torch.cuda.amp.autocast():
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
