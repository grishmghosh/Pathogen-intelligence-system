import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


def build_resnet50(num_classes=4):
    """Build a pretrained ResNet-50 and adapt it for the target class count."""
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet50(num_classes=4).to(device)

    print(f"Using device: {device}")
    print(model)
    print(f"Final output layer size: {model.fc.out_features}")


if __name__ == "__main__":
    main()
