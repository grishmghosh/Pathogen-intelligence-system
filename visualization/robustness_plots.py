"""
Robustness Degradation Plots for Pathogen Intelligence System.

Generates scientific visualizations of model robustness under perturbations:
    - Accuracy vs severity
    - Confidence vs severity
    - Multi-model comparison bar charts
    - Per-class robustness degradation

All functions accept either in-memory data structures (dicts / DataFrames)
or CSV file paths, and save output as PNG to results/plots/robustness/.

Architecture:
    This module depends ONLY on visualization_utils for shared helpers.
    It does NOT import from inference or analysis directly.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.visualization_utils import (
    logger,
    setup_plot_theme,
    ensure_output_dirs,
    ROBUSTNESS_PLOTS_DIR,
    get_model_color,
    get_perturbation_color,
    save_figure,
    extract_robustness_dataframe,
    extract_inference_dataframe,
    load_csv_safe,
)

# Canonical perturbation ordering (matches configs/perturbation_config.py)
PERTURBATION_ORDER = [
    "original",
    "bright",
    "dark",
    "high_contrast",
    "low_contrast",
    "gaussian_noise",
    "gaussian_blur",
]


# ---------------------------------------------------------------------------
# 1. Accuracy vs Severity
# ---------------------------------------------------------------------------

def plot_accuracy_vs_severity(
    inference_results=None,
    robustness_report=None,
    csv_path=None,
    output_path=None,
    title="Accuracy vs Perturbation Severity",
):
    """
    Plot prediction accuracy (consistency with original prediction) for each
    perturbation, grouped by model.

    Accepts ONE of:
        - inference_results: dict from run_batch_inference()
        - robustness_report: dict from generate_robustness_report()
        - csv_path: path to a CSV with columns [model, perturbation, prediction_changed]

    Args:
        inference_results: Raw inference output dict (optional).
        robustness_report: Robustness report dict (optional).
        csv_path: Path to pre-exported CSV (optional).
        output_path: Destination PNG path. Defaults to ROBUSTNESS_PLOTS_DIR.
        title: Plot title string.

    Returns:
        Path to saved PNG, or None on failure.
    """
    setup_plot_theme()
    ensure_output_dirs()

    # --- resolve data source ---
    df = _resolve_robustness_df(inference_results, robustness_report, csv_path)
    if df is None or df.empty:
        logger.warning("No data available for accuracy vs severity plot.")
        return None

    # Derive prediction_changed if not present
    if "prediction_changed" not in df.columns:
        if "perturbation" in df.columns and "prediction" in df.columns and "model" in df.columns:
            original_preds = (
                df[df["perturbation"] == "original"]
                .set_index("model")["prediction"]
                .to_dict()
            )
            if original_preds:
                df = df.copy()
                df["prediction_changed"] = df.apply(
                    lambda row: row["prediction"] != original_preds.get(row["model"]),
                    axis=1,
                )
            else:
                logger.warning("No original predictions found; cannot derive accuracy.")
                return None
        else:
            logger.warning("Column 'prediction_changed' missing and cannot be derived.")
            return None

    df["accurate"] = (~df["prediction_changed"]).astype(int)

    # Order perturbations
    ordered_perts = [p for p in PERTURBATION_ORDER if p in df["perturbation"].unique()]
    if not ordered_perts:
        ordered_perts = sorted(df["perturbation"].unique())

    models = sorted(df["model"].unique())
    x = np.arange(len(ordered_perts))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        model_df = df[df["model"] == model]
        accuracies = []
        for pert in ordered_perts:
            subset = model_df[model_df["perturbation"] == pert]
            acc = subset["accurate"].mean() * 100 if not subset.empty else 0
            accuracies.append(acc)
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            accuracies,
            width,
            label=model,
            color=get_model_color(model, i),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Perturbation")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_perts, rotation=30, ha="right")
    ax.set_ylim(0, 110)
    ax.legend(title="Model")
    ax.yaxis.grid(True, alpha=0.3)

    if output_path is None:
        output_path = os.path.join(ROBUSTNESS_PLOTS_DIR, "accuracy_vs_severity.png")
    save_figure(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# 2. Confidence vs Severity
# ---------------------------------------------------------------------------

def plot_confidence_vs_severity(
    inference_results=None,
    robustness_report=None,
    csv_path=None,
    output_path=None,
    title="Confidence vs Perturbation Severity",
):
    """
    Plot mean confidence score for each perturbation, per model.

    Accepts the same data sources as plot_accuracy_vs_severity().

    Returns:
        Path to saved PNG, or None on failure.
    """
    setup_plot_theme()
    ensure_output_dirs()

    df = _resolve_confidence_df(inference_results, robustness_report, csv_path)
    if df is None or df.empty:
        logger.warning("No data available for confidence vs severity plot.")
        return None

    if "confidence" not in df.columns:
        logger.warning("Column 'confidence' missing; cannot plot.")
        return None

    ordered_perts = [p for p in PERTURBATION_ORDER if p in df["perturbation"].unique()]
    if not ordered_perts:
        ordered_perts = sorted(df["perturbation"].unique())

    models = sorted(df["model"].unique())

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        model_df = df[df["model"] == model]
        means = []
        for pert in ordered_perts:
            subset = model_df[model_df["perturbation"] == pert]
            mean_val = subset["confidence"].mean() if not subset.empty else 0
            means.append(mean_val)
        ax.plot(
            ordered_perts,
            means,
            marker="o",
            linewidth=2,
            markersize=8,
            label=model,
            color=get_model_color(model, i),
        )

    ax.set_xlabel("Perturbation")
    ax.set_ylabel("Mean Confidence")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model")
    ax.yaxis.grid(True, alpha=0.3)
    plt.xticks(rotation=30, ha="right")

    if output_path is None:
        output_path = os.path.join(ROBUSTNESS_PLOTS_DIR, "confidence_vs_severity.png")
    save_figure(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# 3. Model Comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
    robustness_report=None,
    csv_path=None,
    output_path=None,
    title="Model Robustness Comparison",
):
    """
    Generate a grouped bar chart comparing robustness sub-scores across models.

    Metrics compared:
        - Robustness Score
        - Consistency Score
        - Stability Score
        - Resistance Score

    Args:
        robustness_report: Dict from generate_robustness_report() (optional).
        csv_path: Path to summary CSV (optional).
        output_path: Destination PNG path.
        title: Plot title.

    Returns:
        Path to saved PNG, or None on failure.
    """
    setup_plot_theme()
    ensure_output_dirs()

    # --- Extract model scores ---
    records = []
    if robustness_report is not None:
        for model_name, model_report in robustness_report.items():
            if model_name == "model_comparison":
                continue
            if "error" in model_report:
                continue
            scores = model_report.get("robustness_score", {})
            if "error" in scores:
                continue
            records.append({
                "model": model_name,
                "Robustness": scores.get("robustness_score", 0),
                "Consistency": scores.get("consistency_score", 0),
                "Stability": scores.get("stability_score", 0),
                "Resistance": scores.get("resistance_score", 0),
            })
    elif csv_path is not None:
        df_csv = load_csv_safe(csv_path)
        if df_csv is not None:
            required = {"model", "Robustness", "Consistency", "Stability", "Resistance"}
            if required.issubset(df_csv.columns):
                records = df_csv.to_dict("records")

    if not records:
        logger.warning("No model score data available for comparison plot.")
        return None

    df = pd.DataFrame(records)
    metrics = ["Robustness", "Consistency", "Stability", "Resistance"]
    models = df["model"].tolist()

    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]

    for i, model in enumerate(models):
        values = [df.loc[df["model"] == model, m].values[0] for m in metrics]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=model,
            color=palette[i % len(palette)],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score (0–100)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 110)
    ax.legend(title="Model")
    ax.yaxis.grid(True, alpha=0.3)

    if output_path is None:
        output_path = os.path.join(ROBUSTNESS_PLOTS_DIR, "model_comparison.png")
    save_figure(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# 4. Per-Class Robustness Degradation
# ---------------------------------------------------------------------------

def plot_per_class_robustness_degradation(
    inference_results=None,
    csv_path=None,
    output_path=None,
    title="Per-Class Robustness Degradation",
):
    """
    Plot confidence degradation for each predicted class across perturbations.

    This shows how each pathogen class is affected differently by perturbations,
    enabling identification of class-specific vulnerabilities.

    Args:
        inference_results: Raw inference output dict (optional).
        csv_path: Path to inference CSV (optional).
        output_path: Destination PNG path.
        title: Plot title.

    Returns:
        Path to saved PNG, or None on failure.
    """
    setup_plot_theme()
    ensure_output_dirs()

    df = _resolve_confidence_df(inference_results, None, csv_path)
    if df is None or df.empty:
        logger.warning("No data for per-class degradation plot.")
        return None

    if "prediction" not in df.columns or "confidence" not in df.columns:
        logger.warning("Required columns missing for per-class plot.")
        return None

    ordered_perts = [p for p in PERTURBATION_ORDER if p in df["perturbation"].unique()]
    if not ordered_perts:
        ordered_perts = sorted(df["perturbation"].unique())

    classes = sorted(df["prediction"].dropna().unique())
    if not classes:
        logger.warning("No prediction classes found.")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    class_palette = sns.color_palette("husl", len(classes))

    for idx, cls in enumerate(classes):
        cls_df = df[df["prediction"] == cls]
        means = []
        for pert in ordered_perts:
            subset = cls_df[cls_df["perturbation"] == pert]
            means.append(subset["confidence"].mean() if not subset.empty else np.nan)
        ax.plot(
            ordered_perts,
            means,
            marker="s",
            linewidth=2,
            markersize=7,
            label=cls,
            color=class_palette[idx],
        )

    ax.set_xlabel("Perturbation")
    ax.set_ylabel("Mean Confidence")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Predicted Class", loc="lower left")
    ax.yaxis.grid(True, alpha=0.3)
    plt.xticks(rotation=30, ha="right")

    if output_path is None:
        output_path = os.path.join(ROBUSTNESS_PLOTS_DIR, "per_class_degradation.png")
    save_figure(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _resolve_robustness_df(inference_results, robustness_report, csv_path):
    """Resolve the best available data source into a DataFrame."""
    if csv_path is not None:
        return load_csv_safe(csv_path)
    if robustness_report is not None:
        return extract_robustness_dataframe(robustness_report)
    if inference_results is not None:
        return extract_inference_dataframe(inference_results)
    return None


def _resolve_confidence_df(inference_results, robustness_report, csv_path):
    """Resolve data source with confidence column."""
    if csv_path is not None:
        return load_csv_safe(csv_path)
    if inference_results is not None:
        return extract_inference_dataframe(inference_results)
    if robustness_report is not None:
        return extract_robustness_dataframe(robustness_report)
    return None
