import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import Swin_T_Weights


def build_swin_t(num_classes: int = 4) -> nn.Module:
    """Build Swin-T (Swin Transformer Tiny) with pretrained weights adapted for pathogen classification."""
    model = models.swin_t(weights=Swin_T_Weights.DEFAULT)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_swin_t(num_classes=4).to(device)
    print("\nSwin-T Model Architecture:")
    print(model)

    out_features = model.head.out_features
    print(f"\nFinal head output features: {out_features}")


if __name__ == "__main__":
    main()
