"""
Perturbation engine for generating controlled image variants.
Generates perturbations dynamically in memory without saving to disk.
"""

import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.perturbation_config import ENABLED_PERTURBATIONS, PERTURBATION_CONFIG


def validate_image(image):
    """Validate image format and properties."""
    if image is None:
        raise ValueError("Image is None")

    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")

    if image.dtype != np.uint8:
        raise ValueError(f"Image dtype must be uint8, got {image.dtype}")

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must have 3 channels, got shape {image.shape}")


def load_image(image_path):
    """Load image from disk and return as numpy array in RGB format."""
    image_path = str(Path(image_path).expanduser())
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is not None:
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            validate_image(image)
            return image
    except Exception:
        pass

    try:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except (ImportError, Exception):
            pass

        with Image.open(image_path) as handle:
            if getattr(handle, "n_frames", 1) > 1:
                handle.seek(0)
            rgb_image = handle.convert("RGB")
            image = np.asarray(rgb_image, dtype=np.uint8)
            validate_image(image)
            return image
    except Exception as exc:
        raise ValueError(f"Failed to load image: {image_path}: {exc}") from exc


def adjust_brightness(image, factor):
    """Adjust image brightness by multiplying pixel values."""
    validate_image(image)
    image_float = image.astype(np.float32)
    adjusted = np.clip(image_float * factor, 0, 255).astype(np.uint8)
    return adjusted


def adjust_contrast(image, factor):
    """Adjust image contrast around the mean intensity."""
    validate_image(image)
    image_float = image.astype(np.float32)
    mean = np.mean(image_float)
    adjusted = np.clip((image_float - mean) * factor + mean, 0, 255).astype(np.uint8)
    return adjusted


def add_gaussian_noise(image, sigma, seed=None):
    """Add Gaussian noise to the image with reproducible results."""
    validate_image(image)

    if seed is None:
        seed = PERTURBATION_CONFIG["random_seed"]

    rng = np.random.default_rng(seed)
    image_float = image.astype(np.float32)
    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image_float + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_gaussian_blur(image, kernel_size):
    """Apply Gaussian blur with specified kernel size."""
    validate_image(image)

    if kernel_size < 1:
        raise ValueError(f"Kernel size must be >= 1, got {kernel_size}")

    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred


def generate_perturbations(image_path):
    """Generate all enabled perturbations from a single input image."""
    original = load_image(image_path)

    perturbations = {
        "original": {
            "image": original,
            "type": "none",
            "parameter": None,
            "id": "original",
        }
    }

    config = PERTURBATION_CONFIG

    if "bright" in ENABLED_PERTURBATIONS:
        param = config["brightness_increase_factor"]
        perturbations["bright"] = {
            "image": adjust_brightness(original, param),
            "type": "brightness",
            "parameter": param,
            "id": f"brightness_increase_{param}",
        }

    if "dark" in ENABLED_PERTURBATIONS:
        param = config["brightness_decrease_factor"]
        perturbations["dark"] = {
            "image": adjust_brightness(original, param),
            "type": "brightness",
            "parameter": param,
            "id": f"brightness_decrease_{param}",
        }

    if "high_contrast" in ENABLED_PERTURBATIONS:
        param = config["contrast_increase_factor"]
        perturbations["high_contrast"] = {
            "image": adjust_contrast(original, param),
            "type": "contrast",
            "parameter": param,
            "id": f"contrast_increase_{param}",
        }

    if "low_contrast" in ENABLED_PERTURBATIONS:
        param = config["contrast_decrease_factor"]
        perturbations["low_contrast"] = {
            "image": adjust_contrast(original, param),
            "type": "contrast",
            "parameter": param,
            "id": f"contrast_decrease_{param}",
        }

    if "gaussian_noise" in ENABLED_PERTURBATIONS:
        param = config["gaussian_noise_sigma"]
        perturbations["gaussian_noise"] = {
            "image": add_gaussian_noise(original, param),
            "type": "noise",
            "parameter": param,
            "id": f"gaussian_noise_sigma_{param}",
        }

    if "gaussian_blur" in ENABLED_PERTURBATIONS:
        param = config["gaussian_blur_kernel_size"]
        perturbations["gaussian_blur"] = {
            "image": apply_gaussian_blur(original, param),
            "type": "blur",
            "parameter": param,
            "id": f"gaussian_blur_kernel_{param}",
        }

    return perturbations


def main():
    """Test perturbation generation on a sample image with debug logging and visualization."""
    sample_image_path = input("Enter image path: ").strip()

    if not sample_image_path:
        print("No path provided. Exiting.")
        return

    if not os.path.exists(sample_image_path):
        print(f"Image not found: {sample_image_path}")
        return

    print(f"\nLoading image: {sample_image_path}")
    perturbations = generate_perturbations(sample_image_path)

    print(f"\nGenerated {len(perturbations)} perturbation variants:")
    for name, data in perturbations.items():
        img = data["image"]
        print(f"  - {name}:")
        print(f"      type: {data['type']}")
        print(f"      parameter: {data['parameter']}")
        print(f"      id: {data['id']}")
        print(f"      shape: {img.shape}")
        print(f"      dtype: {img.dtype}")

    print("\nPerturbation generation completed successfully.")

    print("\nDisplaying perturbations...")
    for name, data in perturbations.items():
        plt.figure(figsize=(6, 6))
        plt.imshow(data["image"])
        plt.title(f"{name}\n{data['id']}")
        plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
