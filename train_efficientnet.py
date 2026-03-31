import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_data_loaders
from efficientnet_setup import build_efficientnet_b0


from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    """Run one training epoch and return average training loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


def validate(model, loader, criterion, device):
    """Run validation and return average loss and accuracy percentage."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    accuracy = (correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    return avg_loss, accuracy


def save_training_log(log_rows, csv_path):
    """Save epoch-wise metrics to a CSV file."""
    fieldnames = ["epoch", "train_loss", "val_loss", "accuracy"]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)


def main():
    print("Training EfficientNet-B0")
    # 1) Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2) Load train/validation data
    _, _, train_loader, val_loader = get_data_loaders(
        batch_size=16, num_workers=2, pin_memory=True
    )

    # 3) Build and move model
    model = build_efficientnet_b0(num_classes=4)
    model = model.to(device)

    # 4) Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    scaler = GradScaler()

    # 5) Training configuration
    num_epochs = 5
    training_log = []
    best_acc = 0.0
    save_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "efficientnet_b0_best.pth")

    # 6) Train + validate loop
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": val_accuracy,
        }
        training_log.append(log_row)

        print(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

        # Save best model
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch} with accuracy {val_accuracy:.2f}%")

    # 7) Save training log
    log_path = os.path.join(save_dir, "efficientnet_log.csv")
    save_training_log(training_log, log_path)
    print(f"Training log saved to: {log_path}")
    print(f"Model weights saved to: {model_path}")


if __name__ == "__main__":
    main()
