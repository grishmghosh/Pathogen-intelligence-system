# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def build_efficientnet_b0(num_classes: int = 4) -> nn.Module:
    # Load EfficientNet-B0 with pretrained ImageNet weights
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace the final classifier layer to match the target number of classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def main():
    # Select device: GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build and move model to selected device
    model = build_efficientnet_b0(num_classes=4)
    model = model.to(device)

    # Print full model architecture
    print("\nModel Architecture:")
    print(model)

    # Print output features of the final classification layer to confirm 4 classes
    out_features = model.classifier[1].out_features
    print(f"\nFinal classifier output features: {out_features}")


if __name__ == "__main__":
    main()
