import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np

from .data_utils import data_init, TrainDataset, filepath, _group_shuffle_split, tta_predict
from .model.resnet1D import ResNet1D
from .config import MODEL_DIR, set_seed


def _atomic_save(obj, path):
    """原子写入：先写 .tmp，再替换目标文件。"""
    tmp_path = path.with_suffix(".tmp")
    torch.save(obj, tmp_path)
    try:
        tmp_path.replace(path)
    except PermissionError:
        # Windows fallback: delete target first, then rename
        path.unlink(missing_ok=True)
        tmp_path.rename(path)


def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前使用设备: %s" % device)
    model_path = MODEL_DIR / "model.pth"

    # ---- Load & split data (group-aware) ----
    df = data_init(filepath["train"])
    groups = df["GroupId"].values
    train_idx, val_idx = _group_shuffle_split(groups, test_size=0.15, random_state=42)

    train_dataset = TrainDataset(df.iloc[train_idx], training=True)
    val_dataset = TrainDataset(df.iloc[val_idx], training=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    print(
        f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}"
    )

    # ---- Model ----
    num_classes = 2
    model = ResNet1D(1, num_classes, dropout=0.3)
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        print("Loaded existing model weights.")
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    mixup_alpha = 0.4

    best_val_loss = float("inf")
    patience = 50
    patience_counter = 0

    for epoch in range(50):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        total_sample = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            batch_size = y.shape[0]
            total_sample += batch_size

            # --- MixUp ---
            if mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                lam = max(lam, 1.0 - lam)  # symmetric: keep lam >= 0.5
                index = torch.randperm(batch_size, device=device)
                x_mixed = lam * x + (1.0 - lam) * x[index]
                y_onehot = F.one_hot(y, num_classes).float()
                y_mixed = lam * y_onehot + (1.0 - lam) * y_onehot[index]
                x_input = x_mixed
            else:
                x_input = x
                y_onehot = F.one_hot(y, num_classes).float()

            x_input = x_input.unsqueeze(1)  # [B, 1, F]
            pred = model(x_input)

            loss = criterion(pred, y_mixed if mixup_alpha > 0 else y)

            total_loss += loss.item() * batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        train_loss = total_loss / total_sample

        # ---- Validate (TTA) ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            batch_size = y.shape[0]
            val_total += batch_size

            probs = tta_predict(model, x, num_aug=5)
            loss = F.nll_loss(torch.log(probs + 1e-8), y)
            val_loss += loss.item() * batch_size
            val_correct += (probs.argmax(dim=1) == y).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(
            "Epoch %2d | train_loss: %.4f | val_loss: %.4f | val_acc: %.4f"
            % (epoch, train_loss, val_loss, val_acc)
        )

        # Save best model (atomic write: .tmp → .pth)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            _atomic_save(model.state_dict(), model_path)
            print("  -> Best model saved (val_loss=%.4f)" % val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping at epoch %d" % epoch)
                break

    print("Training complete. Best val_loss: %.4f" % best_val_loss)


if __name__ == "__main__":
    train()
