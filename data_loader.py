import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Default dataset root (can be overridden when calling get_data_loaders)
DEFAULT_DATASET_ROOT = r"C:\Pathogen-intelligence-system\dataset_split"

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms():
    """Create image preprocessing transforms for EfficientNet-style input."""
    common_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return common_transform


def create_datasets(dataset_root=DEFAULT_DATASET_ROOT):
    """
    Create train and validation datasets from folder structure:
    dataset_root/train/<class_name>/*
    dataset_root/val/<class_name>/*
    """
    transform = build_transforms()

    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")

    train_dataset = ImageFolder(root=train_dir, transform=transform)
    val_dataset = ImageFolder(root=val_dir, transform=transform)

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, batch_size=32, num_workers=0):
    """
    Create DataLoaders for train and validation datasets.
    - train_loader: shuffled
    - val_loader: not shuffled
    """
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def print_dataset_info(train_dataset, val_dataset):
    """Print required dataset information."""
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Class names: {train_dataset.classes}")


def get_data_loaders(dataset_root=DEFAULT_DATASET_ROOT, batch_size=32, num_workers=0):
    """
    Reusable helper to build datasets and dataloaders.
    Returns:
        train_dataset, val_dataset, train_loader, val_loader
    """
    train_dataset, val_dataset = create_datasets(dataset_root=dataset_root)

    # Optional safety check for class-index consistency between train and val
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Class mapping mismatch between train and val datasets.\n"
            f"Train mapping: {train_dataset.class_to_idx}\n"
            f"Val mapping: {val_dataset.class_to_idx}"
        )

    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def main():
    train_dataset, val_dataset, train_loader, val_loader = get_data_loaders(
        dataset_root=DEFAULT_DATASET_ROOT,
        batch_size=32,
        num_workers=0,
    )

    print_dataset_info(train_dataset, val_dataset)

    # Keep references available when running this file directly
    _ = train_loader, val_loader


if __name__ == "__main__":
    main()
