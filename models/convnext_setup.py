import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights


def build_convnext_tiny(num_classes: int = 4) -> nn.Module:
    """Build ConvNeXt-Tiny with pretrained weights adapted for pathogen classification."""
    model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_convnext_tiny(num_classes=4).to(device)
    print("\nConvNeXt-Tiny Model Architecture:")
    print(model)

    out_features = model.classifier[2].out_features
    print(f"\nFinal classifier output features: {out_features}")


if __name__ == "__main__":
    main()
