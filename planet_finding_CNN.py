import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import torchvision.models as models


# =========================
# Config
# =========================

@dataclass
class Config:
    data_root: str = "./DATA_ROOT"          # <- change this
    num_epochs: int = 60
    batch_size: int = 8                      # cubes are heavy; start small
    lr: float = 1e-3
    patience: int = 8                        # match paper: stop after 8 epochs no val-loss improvement
    weight_decay: float = 0.0                # paper mentions Adam; but i dont think they did wd
    num_workers: int = 4
    use_amp: bool = True
    model_name: str = "regnet_y_8gf"         # recommended start; "regnet_y_16gf" is heavier
    save_path: str = "planet_finding_CNN.pt"
    normalize: str = "none"      # "none", "per_cube_channel", "robust_per_cube_channel"


cfg = Config()


# =========================
# Dataset
# =========================

class PlanetCubeFolderDataset(Dataset):
    """
    Folder structure:
      split/planet/*.npy
      split/no_planet/*.npy

    Each .npy is expected to be:
      - shape (C,H,W)  (preferred)
      OR shape (H,W,C) (we will transpose)
    """
    def __init__(self, split_dir: str, normalize: str = "per_cube_channel"):
        self.samples = []
        self.normalize = normalize

        planet_dir = os.path.join(split_dir, "planet")
        noplanet_dir = os.path.join(split_dir, "no_planet")

        for label_dir, y in [(planet_dir, 1), (noplanet_dir, 0)]:
            if not os.path.isdir(label_dir):
                raise FileNotFoundError(f"Missing directory: {label_dir}")
            for fn in os.listdir(label_dir):
                if fn.endswith(".npy"):
                    self.samples.append((os.path.join(label_dir, fn), y))

        if len(self.samples) == 0:
            raise RuntimeError(f"No .npy files found under {split_dir}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _per_channel_standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # x: (C,H,W)
        C = x.shape[0]
        x_flat = x.view(C, -1)
        mu = x_flat.mean(dim=1).view(C, 1, 1)
        sig = x_flat.std(dim=1).clamp_min(eps).view(C, 1, 1)
        return (x - mu) / sig

    @staticmethod
    def _per_channel_robust(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # x: (C,H,W) robust per channel using median/IQR
        C = x.shape[0]
        x_flat = x.view(C, -1)
        med = x_flat.median(dim=1).values.view(C, 1, 1)
        q1 = x_flat.quantile(0.25, dim=1).view(C, 1, 1)
        q3 = x_flat.quantile(0.75, dim=1).view(C, 1, 1)
        iqr = (q3 - q1).clamp_min(eps)
        return (x - med) / iqr

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        arr = np.load(path)  # (C,H,W) or (H,W,C)

        if arr.ndim != 3:
            raise ValueError(f"{path}: expected 3D array, got {arr.shape}")

        # Ensure channels-first
        if arr.shape[0] in (40, 47, 61, 75):  # likely already (C,H,W)
            x = arr
        else:
            # assume (H,W,C)
            x = np.transpose(arr, (2, 0, 1))

        x = torch.tensor(x, dtype=torch.float32)  # (C,H,W)

        # Normalization
        if self.normalize == "none":
            pass
        elif self.normalize == "per_cube_channel":
            x = self._per_channel_standardize(x)
        elif self.normalize == "robust_per_cube_channel":
            x = self._per_channel_robust(x)
        else:
            raise ValueError(f"Unknown normalize='{self.normalize}'")

        y = torch.tensor(y, dtype=torch.long)
        return x, y


# =========================
# Model helpers: modify input stem to C channels
# =========================

def replace_first_conv(module: nn.Module, in_channels: int) -> None:
    """
    Replace the first Conv2d found in `module` with an identical Conv2d except for in_channels.
    This is a practical way to adapt torchvision models to C not =3.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            old = child
            new = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                dilation=old.dilation,
                groups=old.groups,
                bias=(old.bias is not None),
                padding_mode=old.padding_mode,
            )
            setattr(module, name, new)
            return
        else:
            replace_first_conv(child, in_channels)
            # if replaced deeper, stop recursion by checking again
            # (simple approach: if first conv is already in_channels, we can stop early)
            # but safe enough to let it run; it returns once it finds a conv


def build_model(model_name: str, in_channels: int, num_classes: int = 2) -> nn.Module:
    """
    Paper uses EfficientNetV2 and RegNet variants WITHOUT pretrained weights.
    We'll offer both. Default: RegNetY (recommended for your replication pass).
    """
    model_name = model_name.lower()

    if model_name == "regnet_y_16gf":
        model = models.regnet_y_16gf(weights=None)
    elif model_name == "regnet_y_8gf":
        model = models.regnet_y_8gf(weights=None)
    elif model_name == "regnet_y_400mf":
        model = models.regnet_y_400mf(weights=None)
    elif model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
    else:
        raise ValueError("model_name must be one of: regnet_y_400mf, regnet_y_8gf, regnet_y_16gf, efficientnet_v2_s")

    # Change input stem to accept C channels
    replace_first_conv(model, in_channels)

    # Replace classifier head to output 2 classes
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):  # regnet
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif hasattr(model, "classifier"):  # efficientnet
        # efficientnet classifier is Sequential([... , Linear])
        if isinstance(model.classifier, nn.Sequential):
            # find last linear
            last = model.classifier[-1]
            if isinstance(last, nn.Linear):
                in_f = last.in_features
                model.classifier[-1] = nn.Linear(in_f, num_classes)
            else:
                raise RuntimeError("Unexpected EfficientNet classifier structure.")
        else:
            raise RuntimeError("Unexpected EfficientNet classifier structure.")
    else:
        raise RuntimeError("Unknown model head structure.")

    return model


# =========================
# Train / eval
# =========================

def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    all_probs = []
    all_preds = []
    all_targets = []
    total_loss = 0.0
    n = 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)  # (B,C,H,W)
        y = y.to(device, non_blocking=True)  # (B,)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), enabled=(scaler is not None)):
            logits = model(X)  # (B,2)
            loss = criterion(logits, y)

        if train_mode:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * X.size(0)
        n += X.size(0)

        probs = torch.softmax(logits.detach(), dim=1)[:, 1]  # P(planet)
        preds = logits.detach().argmax(dim=1)

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    avg_loss = total_loss / max(n, 1)
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    acc = accuracy_score(y_true, y_pred)
    # AUC requires both classes present; guard for edge cases
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return avg_loss, acc, auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = os.path.join(cfg.data_root, "train")
    val_dir   = os.path.join(cfg.data_root, "val")
    test_dir  = os.path.join(cfg.data_root, "test")

    train_ds = PlanetCubeFolderDataset(train_dir, normalize=cfg.normalize)
    val_ds   = PlanetCubeFolderDataset(val_dir,   normalize=cfg.normalize)
    test_ds  = PlanetCubeFolderDataset(test_dir,  normalize=cfg.normalize)

    # Infer C from first item
    C, H, W = train_ds[0][0].shape

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    model = build_model(cfg.model_name, in_channels=C, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (cfg.use_amp and device.type == "cuda") else None

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    best_epoch = -1

    for epoch in range(1, cfg.num_epochs + 1):
        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, device, criterion, optimizer, scaler)
        va_loss, va_acc, va_auc = run_epoch(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} auc {tr_auc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} auc {va_auc:.3f}"
        )

        # Early stopping on validation loss (paper-style)
        if va_loss < best_val_loss - 1e-6:
            best_val_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, best val loss {best_val_loss:.4f})")
                break

    # Load best model and evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    te_loss, te_acc, te_auc = run_epoch(model, test_loader, device, criterion)
    print(f"\nTEST | loss {te_loss:.4f} acc {te_acc:.3f} auc {te_auc:.3f}")

    # Save
    torch.save(model.state_dict(), cfg.save_path)
    print(f"Saved model state_dict to: {cfg.save_path}")


if __name__ == "__main__":
    main()
