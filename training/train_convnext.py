"""
training/train_convnext.py  –  Training script for ConvNeXt-Tiny
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.convnext_setup import build_convnext_tiny

# ---------------------------------------------------------------------------
# Configuration & Environment
# ---------------------------------------------------------------------------
ROOT_DIR       = Path(__file__).resolve().parent.parent
DATASET_ROOT   = Path(r"F:\Pathogen-intelligence-system-old\dataset_split")
if not (DATASET_ROOT / "train").exists():
    DATASET_ROOT = Path(r"C:\Pathogen-intelligence-system\dataset_split")
if not (DATASET_ROOT / "train").exists():
    DATASET_ROOT = ROOT_DIR / "dataset_split"

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
NUM_CLASSES    = 4
BATCH_SIZE     = 32
NUM_EPOCHS     = 50
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTH   = 0.1
PATIENCE       = 8
SEED           = 42
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP        = torch.cuda.is_available()

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Training ConvNeXt-Tiny on device: {DEVICE}")
print(f"Mixed precision (AMP): {USE_AMP}")

# ---------------------------------------------------------------------------
# Data Preprocessing Transforms
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
# Main Training Routine
# ---------------------------------------------------------------------------
def main():
    if not (DATASET_ROOT / "train").exists():
        print(f"[ERROR] Dataset train directory not found at {DATASET_ROOT / 'train'}")
        print("Please run `python split_dataset.py` first to generate plate-aware dataset splits.")
        return

    train_dataset = datasets.ImageFolder(str(DATASET_ROOT / "train"), transform=train_transform)
    val_dataset   = datasets.ImageFolder(str(DATASET_ROOT / "val"),   transform=val_transform)
    test_dataset  = datasets.ImageFolder(str(DATASET_ROOT / "test"),  transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    print(f"Classes ({len(train_dataset.classes)}): {train_dataset.classes}")
    print(f"Dataset split size -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    model = build_convnext_tiny(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = CHECKPOINT_DIR / "convnext_tiny_best.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc  = train_correct / train_total
        epoch_val_loss   = val_loss / val_total
        epoch_val_acc    = val_correct / val_total

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
              f"Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f} -- "
              f"Val Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": epoch_val_acc,
            }, checkpoint_path)
            print(f"  --> Saved best model checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print("\nTraining completed.")


if __name__ == "__main__":
    main()
