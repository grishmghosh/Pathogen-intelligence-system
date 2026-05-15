"""
Perturbation engine for generating controlled image variants.
Generates perturbations dynamically in memory without saving to disk.
"""

import os
import cv2
import numpy as np

from perturbation_config import PERTURBATION_CONFIG, ENABLED_PERTURBATIONS


def load_image(image_path):
    """Load image from disk and return as numpy array in RGB format."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def adjust_brightness(image, factor):
    """Adjust image brightness by multiplying pixel values."""
    adjusted = np.clip(image * factor, 0, 255).astype(np.uint8)
    return adjusted


def adjust_contrast(image, factor):
    """Adjust image contrast around the mean intensity."""
    mean = np.mean(image)
    adjusted = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    return adjusted


def add_gaussian_noise(image, sigma):
    """Add Gaussian noise to the image."""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_gaussian_blur(image, kernel_size):
    """Apply Gaussian blur with specified kernel size."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred


def generate_perturbations(image_path):
    """
    Generate all enabled perturbations from a single input image.
    
    Args:
        image_path: Path to input image
        
    Returns:
        Dictionary mapping perturbation names to structured metadata:
        {
            "perturbation_name": {
                "image": numpy array,
                "type": perturbation type,
                "parameter": applied parameter value
            }
        }
    """
    original = load_image(image_path)
    
    perturbations = {
        "original": {
            "image": original,
            "type": "none",
            "parameter": None
        }
    }
    
    config = PERTURBATION_CONFIG
    
    if "bright" in ENABLED_PERTURBATIONS:
        param = config["brightness_increase_factor"]
        perturbations["bright"] = {
            "image": adjust_brightness(original, param),
            "type": "brightness",
            "parameter": param
        }
    
    if "dark" in ENABLED_PERTURBATIONS:
        param = config["brightness_decrease_factor"]
        perturbations["dark"] = {
            "image": adjust_brightness(original, param),
            "type": "brightness",
            "parameter": param
        }
    
    if "high_contrast" in ENABLED_PERTURBATIONS:
        param = config["contrast_increase_factor"]
        perturbations["high_contrast"] = {
            "image": adjust_contrast(original, param),
            "type": "contrast",
            "parameter": param
        }
    
    if "low_contrast" in ENABLED_PERTURBATIONS:
        param = config["contrast_decrease_factor"]
        perturbations["low_contrast"] = {
            "image": adjust_contrast(original, param),
            "type": "contrast",
            "parameter": param
        }
    
    if "gaussian_noise" in ENABLED_PERTURBATIONS:
        param = config["gaussian_noise_sigma"]
        perturbations["gaussian_noise"] = {
            "image": add_gaussian_noise(original, param),
            "type": "noise",
            "parameter": param
        }
    
    if "gaussian_blur" in ENABLED_PERTURBATIONS:
        param = config["gaussian_blur_kernel_size"]
        perturbations["gaussian_blur"] = {
            "image": apply_gaussian_blur(original, param),
            "type": "blur",
            "parameter": param
        }
    
    return perturbations


def main():
    """Test perturbation generation on a sample image."""
    sample_image_path = r"C:\Pathogen-intelligence-system\dataset_split\val\e_coli\Plate 1\IMG_5240.JPG"
    
    if not os.path.exists(sample_image_path):
        print(f"Sample image not found: {sample_image_path}")
        print("Please update the path to a valid image in your dataset.")
        return
    
    print(f"Loading image: {sample_image_path}")
    perturbations = generate_perturbations(sample_image_path)
    
    print(f"\nGenerated {len(perturbations)} perturbation variants:")
    for name, data in perturbations.items():
        print(f"  - {name}: type={data['type']}, parameter={data['parameter']}")
    
    print("\nPerturbation generation completed successfully.")


if __name__ == "__main__":
    main()
