"""
Visualization Utilities for Pathogen Intelligence System.

Shared helper functions for all visualization modules.
Provides:
    - Plot theme / style setup
    - Output directory management
    - Data loading from CSV/JSON
    - Common formatting utilities
    - Color palette definitions
    - Graceful error handling wrappers

Philosophy:
    Centralize all reusable plotting infrastructure here so that
    individual plot modules remain focused on rendering logic only.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("visualization")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s — %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
ROBUSTNESS_PLOTS_DIR = os.path.join(PLOTS_DIR, "robustness")
CALIBRATION_PLOTS_DIR = os.path.join(PLOTS_DIR, "calibration")
HEATMAPS_PLOTS_DIR = os.path.join(PLOTS_DIR, "heatmaps")
SUMMARIES_DIR = os.path.join(PLOTS_DIR, "summaries")

OUTPUT_DIRS = [
    RESULTS_DIR,
    PLOTS_DIR,
    ROBUSTNESS_PLOTS_DIR,
    CALIBRATION_PLOTS_DIR,
    HEATMAPS_PLOTS_DIR,
    SUMMARIES_DIR,
]

# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "efficientnet_b0": "#2196F3",
    "resnet50": "#FF5722",
    "default_0": "#4CAF50",
    "default_1": "#9C27B0",
    "default_2": "#FF9800",
    "default_3": "#00BCD4",
}

PERTURBATION_COLORS = {
    "bright": "#FFC107",
    "dark": "#795548",
    "high_contrast": "#E91E63",
    "low_contrast": "#9E9E9E",
    "gaussian_noise": "#3F51B5",
    "gaussian_blur": "#009688",
    "original": "#4CAF50",
}

SEVERITY_CMAP = "YlOrRd"
HEATMAP_CMAP = "RdYlGn_r"

# ---------------------------------------------------------------------------
# Theme Setup
# ---------------------------------------------------------------------------

def setup_plot_theme():
    """
    Configure matplotlib / seaborn for scientific publication-quality plots.

    Sets:
        - seaborn 'whitegrid' style
        - DPI = 150
        - Tight layout
        - Consistent font sizes
    """
    matplotlib.use("Agg")  # Non-interactive backend for server / CI
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.figsize": (10, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.autolayout": True,
    })
    logger.info("Plot theme configured (Agg backend, seaborn whitegrid).")


# ---------------------------------------------------------------------------
# Directory Management
# ---------------------------------------------------------------------------

def ensure_output_dirs():
    """Create all required output directories if they do not exist."""
    for dirpath in OUTPUT_DIRS:
        os.makedirs(dirpath, exist_ok=True)
    logger.info("Output directories verified / created.")


# ---------------------------------------------------------------------------
# Data Loading Helpers
# ---------------------------------------------------------------------------

def load_csv_safe(filepath):
    """
    Load a CSV file into a pandas DataFrame with graceful error handling.

    Args:
        filepath: Absolute or relative path to a CSV file.

    Returns:
        pandas.DataFrame or None if the file is missing / malformed.
    """
    if not os.path.isfile(filepath):
        logger.warning("CSV not found: %s", filepath)
        return None
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            logger.warning("CSV is empty: %s", filepath)
            return None
        logger.info("Loaded CSV: %s (%d rows, %d cols)", filepath, len(df), len(df.columns))
        return df
    except Exception as exc:
        logger.error("Failed to read CSV %s: %s", filepath, exc)
        return None


def load_json_safe(filepath):
    """
    Load a JSON file with graceful error handling.

    Args:
        filepath: Absolute or relative path to a JSON file.

    Returns:
        Parsed JSON object (dict/list) or None on failure.
    """
    if not os.path.isfile(filepath):
        logger.warning("JSON not found: %s", filepath)
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Loaded JSON: %s", filepath)
        return data
    except Exception as exc:
        logger.error("Failed to read JSON %s: %s", filepath, exc)
        return None


# ---------------------------------------------------------------------------
# Data Extraction from In-Memory Structures
# ---------------------------------------------------------------------------

def extract_robustness_dataframe(robustness_report):
    """
    Convert a robustness report dict (from robustness_analyzer) into a tidy
    pandas DataFrame suitable for plotting.

    Columns produced:
        model, perturbation, prediction, confidence, prediction_changed,
        confidence_drop, impact_score

    Args:
        robustness_report: Dict returned by generate_robustness_report()

    Returns:
        pandas.DataFrame (may be empty if report has errors)
    """
    rows = []
    for model_name, model_report in robustness_report.items():
        if model_name == "model_comparison":
            continue
        if "error" in model_report:
            continue

        sensitivity = model_report.get("sensitivity_analysis", {})
        confidence_analysis = model_report.get("confidence_analysis", {})
        consistency = model_report.get("consistency_analysis", {})

        original_prediction = consistency.get("original_prediction", None)
        original_confidence = confidence_analysis.get("original_confidence", None)

        # Add original row
        if original_prediction is not None and original_confidence is not None:
            rows.append({
                "model": model_name,
                "perturbation": "original",
                "prediction": original_prediction,
                "confidence": original_confidence,
                "prediction_changed": False,
                "confidence_drop": 0.0,
                "impact_score": 0.0,
            })

        # Add perturbation rows from impact ranking
        for item in sensitivity.get("perturbation_impact_ranking", []):
            rows.append({
                "model": model_name,
                "perturbation": item.get("name", "unknown"),
                "prediction": item.get("new_prediction", original_prediction)
                              if item.get("prediction_changed") else original_prediction,
                "confidence": (original_confidence - item.get("confidence_drop", 0))
                              if original_confidence is not None else None,
                "prediction_changed": item.get("prediction_changed", False),
                "confidence_drop": item.get("confidence_drop", 0),
                "impact_score": item.get("impact_score", 0),
            })

    if not rows:
        logger.warning("No valid data found in robustness report.")
        return pd.DataFrame()

    return pd.DataFrame(rows)


def extract_inference_dataframe(inference_results):
    """
    Convert raw inference results dict (from batch_inference) into a tidy
    pandas DataFrame.

    Columns produced:
        model, perturbation, prediction, confidence, predicted_idx,
        perturbation_type, perturbation_parameter

    Args:
        inference_results: Dict returned by run_batch_inference()

    Returns:
        pandas.DataFrame (may be empty)
    """
    rows = []
    for model_name, model_data in inference_results.items():
        if isinstance(model_data, dict) and "error" in model_data:
            continue
        for pert_name, pert_data in model_data.items():
            if "error" in pert_data:
                continue
            metadata = pert_data.get("metadata", {})
            rows.append({
                "model": model_name,
                "perturbation": pert_name,
                "prediction": pert_data.get("prediction"),
                "confidence": pert_data.get("confidence"),
                "predicted_idx": pert_data.get("predicted_idx"),
                "perturbation_type": metadata.get("type", "none"),
                "perturbation_parameter": metadata.get("parameter"),
            })

    if not rows:
        logger.warning("No valid data found in inference results.")
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Color Helpers
# ---------------------------------------------------------------------------

def get_model_color(model_name, index=0):
    """Return a consistent color for a given model name."""
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    fallback_key = f"default_{index % 4}"
    return MODEL_COLORS.get(fallback_key, "#607D8B")


def get_perturbation_color(pert_name):
    """Return a consistent color for a given perturbation name."""
    return PERTURBATION_COLORS.get(pert_name, "#607D8B")


# ---------------------------------------------------------------------------
# Save Helpers
# ---------------------------------------------------------------------------

def save_figure(fig, filepath, close=True):
    """
    Save a matplotlib figure to disk, creating directories as needed.

    Args:
        fig: matplotlib Figure object
        filepath: Destination path (PNG)
        close: Whether to close the figure after saving (default True)
    """
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    fig.savefig(filepath, bbox_inches="tight", facecolor="white")
    logger.info("Saved plot: %s", filepath)
    if close:
        plt.close(fig)
