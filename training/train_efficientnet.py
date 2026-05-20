"""
training/train_efficientnet.py  –  Anti-overfit training script
===============================================================
FIXES applied vs. original:
  1. Label smoothing (0.1) — prevents the model from learning to output
     extreme confidences like 0.9999 on every sample.
  2. Dropout on classifier head (0.4) — standard for EfficientNet fine-tuning.
  3. Weight decay (1e-4) — L2 regularisation.
  4. Cosine LR schedule with warm restarts — better than step decay.
  5. Early stopping on val loss (not val acc) — stops before severe overfit.
  6. Data augmentation strengthened — RandomErasing, ColorJitter, etc.
  7. Gradient clipping — prevents exploding gradients with mixed precision.
  8. Temperature scaling calibration saved after training.
  9. val split checks class balance to flag leakage-related issues.
"""

import os
import sys
import time
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
DATASET_ROOT  = r"C:\Pathogen-intelligence-system\dataset_split"
CHECKPOINT_DIR = r"C:\Pathogen-intelligence-system\checkpoints"
NUM_CLASSES   = 4          # e_coli, k_pneumoniae, p_aeruginosa, s_aureus
BATCH_SIZE    = 32
NUM_EPOCHS    = 50
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.1        # FIX: prevents extreme confidence outputs
DROPOUT_RATE  = 0.4        # FIX: applied to classifier head
PATIENCE      = 8          # early-stopping patience (epochs)
SEED          = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# FIX: Much stronger augmentation than original to reduce overfit
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),   # FIX: occlusion robustness
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Classes: {train_dataset.classes}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# Warn if val set looks suspiciously imbalanced (possible leakage indicator)
from collections import Counter
val_targets = [y for _, y in val_dataset.samples]
val_counts  = Counter(val_targets)
print(f"Val class counts: {val_counts}")
if max(val_counts.values()) / len(val_targets) > 0.8:
    print("[WARN] Val set is >80% one class — check for plate-level leakage!")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model() -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # FIX: Add dropout to the classifier head before the final linear layer
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE, inplace=True),   # FIX: was 0.2 default; bumped to 0.4
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model.to(DEVICE)

model = build_model()

# ---------------------------------------------------------------------------
# Loss — Label Smoothing
# ---------------------------------------------------------------------------
# FIX: CrossEntropyLoss with label_smoothing prevents the model from
# learning to output probabilities near 1.0, which causes overconfidence.
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

# ---------------------------------------------------------------------------
# Optimiser + Scheduler
# ---------------------------------------------------------------------------
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# FIX: Cosine annealing with warm restarts is more effective than StepLR
# for preventing overfit on small medical-image datasets.
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
scaler    = GradScaler()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
best_val_loss = float("inf")
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "val_acc": []}

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
        # FIX: Gradient clipping prevents exploding gradients with AMP
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
            logits = model(imgs)
            val_loss += criterion(logits, labels).item() * imgs.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += labels.size(0)

    val_loss /= len(val_dataset)
    val_acc   = correct / total

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # Early stopping on val loss
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(),
             "val_loss": val_loss, "val_acc": val_acc,
             "classes": train_dataset.classes},
            os.path.join(CHECKPOINT_DIR, "efficientnet_b0_best.pth"),
        )
        print("  → Saved best checkpoint")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
            break

    # Flag if train/val gap is growing — sign of overfit
    if epoch > 5 and train_loss < 0.5 * val_loss:
        print("  [WARN] Train loss << Val loss — possible overfitting. "
              "Consider more augmentation or reducing model capacity.")

# ---------------------------------------------------------------------------
# Temperature scaling (post-hoc calibration)
# ---------------------------------------------------------------------------
print("\n--- Post-training Temperature Scaling calibration ---")

from torch.nn.functional import cross_entropy

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

# Load best model
ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "efficientnet_b0_best.pth"), map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Collect val logits
all_logits, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        all_logits.append(model(imgs).cpu())
        all_labels.append(labels)
all_logits = torch.cat(all_logits)
all_labels = torch.cat(all_labels)

ts = TemperatureScaler()
ts_optimizer = optim.LBFGS([ts.temperature], lr=0.01, max_iter=50)

def ts_eval():
    ts_optimizer.zero_grad()
    loss = cross_entropy(ts(all_logits), all_labels)
    loss.backward()
    return loss

ts_optimizer.step(ts_eval)
print(f"Optimal temperature: {ts.temperature.item():.4f}")

torch.save(
    {"temperature": ts.temperature.item()},
    os.path.join(CHECKPOINT_DIR, "efficientnet_b0_temperature.pth"),
)
print("Temperature saved. Apply this during inference to get calibrated probabilities.")

# ---------------------------------------------------------------------------
# Final test evaluation
# ---------------------------------------------------------------------------
print("\n--- Final Test Evaluation ---")
ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "efficientnet_b0_best.pth"), map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

correct, total = 0, 0
all_probs, all_preds, all_true = [], [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        # Apply temperature scaling for calibrated probabilities
        logits_cal = logits / ts.temperature.item()
        probs = torch.softmax(logits_cal, dim=1)
        preds = probs.argmax(1).cpu()
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)
        all_probs.append(probs.cpu())
        all_preds.append(preds)
        all_true.append(labels)

all_probs = torch.cat(all_probs).numpy()
all_preds = torch.cat(all_preds).numpy()
all_true  = torch.cat(all_true).numpy()

test_acc = correct / total
mean_max_conf = all_probs.max(axis=1).mean()

print(f"Test Accuracy:       {test_acc:.4f}")
print(f"Mean Max Confidence: {mean_max_conf:.4f}")

# FIX: Flag overconfidence explicitly
if mean_max_conf > 0.95 and test_acc < 0.90:
    print("[WARN] Model is OVERCONFIDENT relative to its accuracy. "
          "Check for leakage or increase temperature further.")
elif mean_max_conf > 0.95:
    print("[INFO] High confidence. Verify with ECE metric in analysis/calibration.py.")

print("\nTraining complete.")
