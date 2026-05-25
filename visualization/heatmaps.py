"""
Prediction Flip Heatmaps for Pathogen Intelligence System.

Generates heatmap visualizations for:
    - Perturbation type vs prediction flip frequency
    - Severity vs instability
    - Model instability comparison

Heatmaps provide a dense, at-a-glance view of which perturbations and models
are most susceptible to prediction changes.

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
    HEATMAPS_PLOTS_DIR,
    HEATMAP_CMAP,
    save_figure,
    extract_robustness_dataframe,
    extract_inference_dataframe,
    load_csv_safe,
)

# Canonical perturbation ordering
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
# 1. Perturbation Type vs Prediction Flip Frequency
# ---------------------------------------------------------------------------

def plot_prediction_flip_heatmap(
    inference_results=None,
    robustness_report=None,
    csv_path=None,
    output_path=None,
    title="Prediction Flip Frequency by Perturbation",
):
    """
    Generate a heatmap showing prediction flip frequency for each
    perturbation type across models.

    Rows = perturbation types, Columns = models.
    Cell value = 1 if prediction flipped, 0 if not (or flip rate if
    multiple samples per cell).

    Args:
        inference_results: Raw inference output dict (optional).
        robustness_report: Robustness report dict (optional).
        csv_path: CSV path (optional).
        output_path: Destination PNG.
        title: Plot title.

    Returns:
        Path to saved PNG, or None on failure.
    """
    setup_plot_theme()
    ensure_output_dirs()

    df = _resolve_df(inference_results, robustness_report, csv_path)
    if df is None or df.empty:
        logger.warning("No data for prediction flip heatmap.")
        return None

    # Ensure prediction_changed column exists
    df = _ensure_flip_column(df)
    if df is None or "prediction_changed" not in df.columns:
        logger.warning("Cannot determine prediction flips.")
        return None

    # Filter out 'original' perturbation (it cannot flip by definition)
    df_filtered = df[df["perturbation"] != "original"].copy()
    if df_filtered.empty:
        logger.warning("No non-original perturbations found.")
        return None

    # Build pivot table: perturbation × model → flip rate
    pivot = df_filtered.pivot_table(
        values="prediction_changed",
        index="perturbation",
        columns="model",
        aggfunc="mean",
        fill_value=0,
    )

    # Reorder rows to canonical order
    ordered = [p for p in PERTURBATION_ORDER if p in pivot.index]
    remaining = [p for p in pivot.index if p not in ordered]
    pivot = pivot.reindex(ordered + remaining)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 3), max(6, len(pivot) * 0.8)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=HEATMAP_CMAP,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Flip Rate"},
        ax=ax,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Perturbation")
    ax.set_title(title)
    plt.yticks(rotation=0)

    if output_path is None:
        output_path = os.path.join(HEATMAPS_PLOTS_DIR, "prediction_flip_heatmap.png")
    save_figure(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# 2. Severity vs Instability
# ---------------------------------------------------------------------------

def plot_severity_vs_instability_heatmap(
    inference_results=None,
    robustness_report=None,
    csv_path=None,
    output_path=None,
    title="Severity vs Instability",
):
    """
    Generate a heatmap showing confidence drop (instability) for each
    perturbation type and model.

    Rows = perturbation types, Columns = models.
    Cell value = confidence drop from original (higher = less stable).

    Args:
        inference_results: Raw inference output dict (optional).
        robustness_report: Robustness report dict (optional).
        csv_path: CSV path (optional).
        output_path: Destination PNG.
        title: Plot title.

    Returns:
        Path to saved PNG, or None on failure.
    """
    setup_plot_theme()
    ensure_output_dirs()

    df = _resolve_df(inference_results, robustness_report, csv_path)
    if df is None or df.empty:
        logger.warning("No data for severity vs instability heatmap.")
        return None

    # Compute confidence drop relative to original per model
    df = _compute_confidence_drop(df)
    if df is None or "confidence_drop" not in df.columns:
        logger.warning("Cannot compute confidence drops.")
        return None

    df_filtered = df[df["perturbation"] != "original"].copy()
    if df_filtered.empty:
        logger.warning("No non-original perturbations found.")
        return None

    pivot = df_filtered.pivot_table(
        values="confidence_drop",
        index="perturbation",
        columns="model",
        aggfunc="mean",
        fill_value=0,
    )

    ordered = [p for p in PERTURBATION_ORDER if p in pivot.index]
    remaining = [p for p in pivot.index if p not in ordered]
    pivot = pivot.reindex(ordered + remaining)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 3), max(6, len(pivot) * 0.8)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        vmin=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Confidence Drop"},
        ax=ax,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Perturbation")
    ax.set_title(title)
    plt.yticks(rotation=0)

    if output_path is None:
        output_path = os.path.join(HEATMAPS_PLOTS_DIR, "severity_vs_instability.png")
    save_figure(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# 3. Model Instability Comparison
# ---------------------------------------------------------------------------

def plot_model_instability_comparison(
    inference_results=None,
    robustness_report=None,
    csv_path=None,
    output_path=None,
    title="Model Instability Comparison",
):
    """
    Generate a combined heatmap comparing multiple instability metrics
    across models.

    Rows = models, Columns = instability metrics.
    Metrics:
        - flip_rate: fraction of perturbations causing prediction change
        - mean_confidence_drop: average confidence decrease
        - max_confidence_drop: worst-case confidence decrease
        - confidence_std: standard deviation of confidence across perturbations

    Args:
        inference_results: Raw inference output dict (optional).
        robustness_report: Robustness report dict (optional).
        csv_path: CSV path (optional).
        output_path: Destination PNG.
        title: Plot title.

    Returns:
        Path to saved PNG, or None on failure.
    """
    setup_plot_theme()
    ensure_output_dirs()

    df = _resolve_df(inference_results, robustness_report, csv_path)
    if df is None or df.empty:
        logger.warning("No data for instability comparison heatmap.")
        return None

    df = _ensure_flip_column(df)
    df = _compute_confidence_drop(df)
    if df is None:
        logger.warning("Cannot prepare instability data.")
        return None

    df_filtered = df[df["perturbation"] != "original"].copy()
    if df_filtered.empty:
        logger.warning("No non-original perturbations found.")
        return None

    models = sorted(df_filtered["model"].unique())
    records = []

    for model in models:
        m_df = df_filtered[df_filtered["model"] == model]
        flip_rate = m_df["prediction_changed"].mean() if "prediction_changed" in m_df.columns else 0
        mean_drop = m_df["confidence_drop"].mean() if "confidence_drop" in m_df.columns else 0
        max_drop = m_df["confidence_drop"].max() if "confidence_drop" in m_df.columns else 0
        conf_std = m_df["confidence"].std() if "confidence" in m_df.columns else 0
        records.append({
            "Model": model,
            "Flip Rate": flip_rate,
            "Mean Conf. Drop": mean_drop,
            "Max Conf. Drop": max_drop,
            "Conf. Std Dev": conf_std if not np.isnan(conf_std) else 0,
        })

    summary_df = pd.DataFrame(records).set_index("Model")

    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 1.5)))
    sns.heatmap(
        summary_df,
        annot=True,
        fmt=".4f",
        cmap=HEATMAP_CMAP,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Metric Value"},
        ax=ax,
    )
    ax.set_title(title)
    plt.yticks(rotation=0)

    if output_path is None:
        output_path = os.path.join(HEATMAPS_PLOTS_DIR, "model_instability_comparison.png")
    save_figure(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _resolve_df(inference_results, robustness_report, csv_path):
    """Resolve data source into a DataFrame."""
    if csv_path is not None:
        return load_csv_safe(csv_path)
    if inference_results is not None:
        return extract_inference_dataframe(inference_results)
    if robustness_report is not None:
        return extract_robustness_dataframe(robustness_report)
    return None


def _ensure_flip_column(df):
    """Ensure 'prediction_changed' column exists."""
    if df is None or df.empty:
        return df
    if "prediction_changed" in df.columns:
        return df
    # Derive from comparing predictions to original per model
    if "perturbation" in df.columns and "prediction" in df.columns and "model" in df.columns:
        df = df.copy()
        original_preds = (
            df[df["perturbation"] == "original"]
            .set_index("model")["prediction"]
            .to_dict()
        )
        if original_preds:
            df["prediction_changed"] = df.apply(
                lambda row: row["prediction"] != original_preds.get(row["model"]),
                axis=1,
            )
            return df
    return df


def _compute_confidence_drop(df):
    """Add 'confidence_drop' column relative to original confidence per model."""
    if df is None or df.empty:
        return df
    if "confidence_drop" in df.columns:
        return df
    if "confidence" not in df.columns or "model" not in df.columns or "perturbation" not in df.columns:
        return df

    df = df.copy()
    original_confs = (
        df[df["perturbation"] == "original"]
        .set_index("model")["confidence"]
        .to_dict()
    )
    if original_confs:
        df["confidence_drop"] = df.apply(
            lambda row: original_confs.get(row["model"], 0) - row["confidence"],
            axis=1,
        )
    else:
        df["confidence_drop"] = 0.0
    return df
