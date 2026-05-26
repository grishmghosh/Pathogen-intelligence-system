"""
training/train_resnet.py  –  Anti-overfit ResNet-50 training script
====================================================================
Mirrors all fixes from train_efficientnet.py:
  1. Label smoothing (0.1)
  2. Dropout on classifier head (0.5 — ResNet-50 is larger, needs more)
  3. Weight decay (1e-4)
  4. Cosine LR schedule with warm restarts
  5. Early stopping on val loss
  6. Stronger data augmentation
  7. Gradient clipping
  8. Temperature scaling calibration saved after training
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_ROOT   = r"C:\Pathogen-intelligence-system\dataset_split"
CHECKPOINT_DIR = r"C:\Pathogen-intelligence-system\checkpoints"
NUM_CLASSES    = 4
BATCH_SIZE     = 32
NUM_EPOCHS     = 50
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTH   = 0.1
DROPOUT_RATE   = 0.5       # Higher than EfficientNet — ResNet-50 has 25M params
PATIENCE       = 8
SEED           = 42
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Transforms (same augmentation policy as efficientnet script)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
train_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, "train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATASET_ROOT, "val"),   transform=val_transform)
test_dataset  = datasets.ImageFolder(os.path.join(DATASET_ROOT, "test"),  transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

print(f"Classes: {train_dataset.classes}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

from collections import Counter
val_counts = Counter(y for _, y in val_dataset.samples)
print(f"Val class counts: {val_counts}")
if max(val_counts.values()) / len(val_dataset) > 0.8:
    print("[WARN] Val set is >80% one class — check for plate-level leakage!")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model() -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # FIX: Replace the single Linear head with Dropout + Linear
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model.to(DEVICE)

model = build_model()

# ---------------------------------------------------------------------------
# Loss / Optimiser / Scheduler
# ---------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
scaler    = GradScaler()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
best_val_loss    = float("inf")
patience_counter = 0

for epoch in range(1, NUM_EPOCHS + 1):
    # --- Train ---
    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * imgs.size(0)

    scheduler.step()
    train_loss /= len(train_dataset)

    # --- Validate ---
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits  = model(imgs)
            val_loss += criterion(logits, labels).item() * imgs.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += labels.size(0)

    val_loss /= len(val_dataset)
    val_acc   = correct / total
    scheduler.step()

    print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | "
          f"Val: {val_loss:.4f} | Acc: {val_acc:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(),
             "val_loss": val_loss, "val_acc": val_acc,
             "classes": train_dataset.classes},
            os.path.join(CHECKPOINT_DIR, "resnet50_best.pth"),
        )
        print("  → Saved best checkpoint")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    if epoch > 5 and train_loss < 0.5 * val_loss:
        print("  [WARN] Overfit gap widening — check augmentation / data split.")

# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
print("\n--- Temperature Scaling Calibration ---")
from torch.nn.functional import cross_entropy

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    def forward(self, logits):
        return logits / self.temperature

ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "resnet50_best.pth"), map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

all_logits, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        all_logits.append(model(imgs.to(DEVICE)).cpu())
        all_labels.append(labels)
all_logits = torch.cat(all_logits)
all_labels = torch.cat(all_labels)

ts          = TemperatureScaler()
ts_optim    = optim.LBFGS([ts.temperature], lr=0.01, max_iter=50)

def ts_eval():
    ts_optim.zero_grad()
    loss = cross_entropy(ts(all_logits), all_labels)
    loss.backward()
    return loss

ts_optim.step(ts_eval)
print(f"Optimal temperature: {ts.temperature.item():.4f}")
torch.save({"temperature": ts.temperature.item()},
           os.path.join(CHECKPOINT_DIR, "resnet50_temperature.pth"))

# ---------------------------------------------------------------------------
# Final test evaluation
# ---------------------------------------------------------------------------
print("\n--- Final Test Evaluation (temperature-calibrated) ---")
correct, total = 0, 0
all_max_confs  = []
with torch.no_grad():
    for imgs, labels in test_loader:
        logits = model(imgs.to(DEVICE)) / ts.temperature.item()
        probs  = torch.softmax(logits, dim=1).cpu()
        preds  = probs.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_max_confs.extend(probs.max(1).values.tolist())

test_acc  = correct / total
mean_conf = float(np.mean(all_max_confs))
print(f"Test Accuracy:       {test_acc:.4f}")
print(f"Mean Max Confidence: {mean_conf:.4f}")

if mean_conf > 0.95 and test_acc < 0.90:
    print("[WARN] Overconfident relative to accuracy — check for leakage.")
print("\nTraining complete.")
