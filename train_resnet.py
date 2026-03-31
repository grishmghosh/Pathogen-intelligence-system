import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_data_loaders
from resnet_setup import build_resnet50


from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0

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

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(loader.dataset)
    val_accuracy = (correct / total) * 100 if total > 0 else 0.0
    return val_loss, val_accuracy


def main():
    print("Training ResNet-50")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, train_loader, val_loader = get_data_loaders(batch_size=16)

    model = build_resnet50(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    log_rows = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_accuracy = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": val_accuracy,
            }
        )

        print(
            f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%"
        )

    log_file = os.path.join(os.getcwd(), "resnet_log.csv")
    with open(log_file, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch", "train_loss", "val_loss", "accuracy"],
        )
        def main():
            print("Training ResNet-50")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            # DataLoader with required params
            _, _, train_loader, val_loader = get_data_loaders(
                batch_size=16, num_workers=2, pin_memory=True
            )
            model = build_resnet50(num_classes=4).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0003)
            scaler = GradScaler()

            num_epochs = 5
            log_rows = []
            best_acc = 0.0
            save_dir = os.path.join(os.getcwd(), "checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            best_model_path = os.path.join(save_dir, "resnet50_best.pth")

            for epoch in range(1, num_epochs + 1):
                train_loss = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    scaler=scaler,
                )
                val_loss, val_accuracy = validate(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                )

                log_rows.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "accuracy": val_accuracy,
                    }
                )

                print(
                    f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%"
                )

                # Save best model
                if val_accuracy > best_acc:
                    best_acc = val_accuracy
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Best model saved at epoch {epoch} with accuracy {val_accuracy:.2f}%")

            log_file = os.path.join(save_dir, "resnet_log.csv")
            with open(log_file, mode="w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=["epoch", "train_loss", "val_loss", "accuracy"],
                )
                writer.writeheader()
                writer.writerows(log_rows)
            print(f"Training log saved to: {log_file}")
