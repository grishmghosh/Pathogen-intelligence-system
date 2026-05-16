"""
Perturbation configuration for robustness testing.
Stores all perturbation parameters and experiment settings.
"""

PERTURBATION_CONFIG = {
    "brightness_increase_factor": 1.2,
    "brightness_decrease_factor": 0.8,
    "contrast_increase_factor": 1.15,
    "contrast_decrease_factor": 0.85,
    "gaussian_noise_sigma": 8,
    "gaussian_blur_kernel_size": 5,
    "random_seed": 42,
}

ENABLED_PERTURBATIONS = [
    "bright",
    "dark",
    "high_contrast",
    "low_contrast",
    "gaussian_noise",
    "gaussian_blur",
]

PERTURBATION_ORDER = [
    "original",
    "bright",
    "dark",
    "high_contrast",
    "low_contrast",
    "gaussian_noise",
    "gaussian_blur",
]
