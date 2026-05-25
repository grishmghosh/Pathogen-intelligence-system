"""
Calibration Visualizations for Pathogen Intelligence System.

Generates calibration-focused scientific plots:
    - Reliability diagrams (calibration plots)
    - Confidence histograms
    - Calibration curves (ECE-based)
    - Expected vs actual accuracy plots

Calibration measures how well a model's confidence scores correspond to
actual correctness. A perfectly calibrated model is correct X% of the time
when it reports X% confidence.

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
    CALIBRATION_PLOTS_DIR,
    get_model_color,
    save_figure,
    extract_inference_dataframe,
    load_csv_safe,
)


# ---------------------------------------------------------------------------
# Calibration Computation Helpers
# ---------------------------------------------------------------------------

def _compute_calibration_bins(confidences, correctness, n_bins=10):
    """
    Compute calibration statistics for binned confidence scores.

    Args:
        confidences: array-like of confidence scores (0-1).
        correctness: array-like of boolean correctness indicators.
        n_bins: Number of bins to divide the confidence range.

    Returns:
        dict with keys:
            bin_edges, bin_centers, bin_accuracies, bin_confidences,
            bin_counts, ece (Expected Calibration Error).
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            bin_accuracies[i] = correctness[mask].mean()
            bin_confidences[i] = confidences[mask].mean()

    # ECE: weighted average of |accuracy - confidence| per bin
    total = len(confidences)
    ece = 0.0
    if total > 0:
        for i in range(n_bins):
            ece += (bin_counts[i] / total) * abs(bin_accuracies[i] - bin_confidences[i])

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "ece": ece,
    }


def _derive_correctness(df):
    """
    Derive a 'correct' column from available data.

    Strategy:
        1. If 'correct' column already exists, use it.
        2. If 'prediction_changed' exists, treat prediction_changed == False
           as correct (relative to original prediction robustness framing).
        3. If 'true_label' and 'prediction' exist, compare them.
        4. If 'perturbation' exists, mark 'original' as correct baseline
           and compare other predictions to the original prediction.

    Args:
        df: pandas DataFrame with at least 'confidence' column.

    Returns:
        Modified DataFrame with 'correct' column, or None if derivation fails.
    """
    if df is None or df.empty:
        return None

    df = df.copy()

    if "correct" in df.columns:
        return df

    if "prediction_changed" in df.columns:
        df["correct"] = (~df["prediction_changed"]).astype(int)
        return df

    if "true_label" in df.columns and "prediction" in df.columns:
        df["correct"] = (df["true_label"] == df["prediction"]).astype(int)
        return df

    # Robustness-aware correctness: compare to original prediction per model
    if "perturbation" in df.columns and "prediction" in df.columns and "model" in df.columns:
        original_preds = (
            df[df["perturbation"] == "original"]
            .set_index("model")["prediction"]
            .to_dict()
        )
        if original_preds:
            df["correct"] = df.apply(
                lambda row: int(row["prediction"] == original_preds.get(row["model"])),
                axis=1,
            )
            return df

    # Fallback: cannot determine correctness
    logger.warning("Cannot derive correctness from available columns: %s", list(df.columns))
    return None


# ---------------------------------------------------------------------------
# 1. Reliability Diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    inference_results=None,
    csv_path=None,
    n_bins=10,
    output_path=None,
    title="Reliability Diagram",
):
    """
    Plot a reliability diagram showing calibration quality per model.

    The diagonal represents perfect calibration. Bars above the diagonal
    indicate under-confidence; bars below indicate over-confidence.

    Args:
        inference_results: Raw inference output dict (optional).
        csv_path: Path to CSV with columns [model, confidence, correct] (optional).
        n_bins: Number of calibration bins.
        output_path: Destination PNG path.
        title: Plot title.

    Returns:
        Path to saved PNG, or None on failure.
    """
    setup_plot_theme()
    ensure_output_dirs()

    df = _resolve_df(inference_results, csv_path)
    df = _derive_correctness(df)
    if df is None or df.empty or "correct" not in df.columns:
        logger.warning("Insufficient data for reliability diagram.")
        return None

    models = sorted(df["model"].unique())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[0, idx]
        model_df = df[df["model"] == model]

        cal = _compute_calibration_bins(
            model_df["confidence"].values,
            model_df["correct"].values,
            n_bins=n_bins,
        )

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

        # Bar chart of bin accuracies
        bar_width = 1.0 / n_bins
        color = get_model_color(model, idx)
        ax.bar(
            cal["bin_centers"],
            cal["bin_accuracies"],
            width=bar_width * 0.9,
            alpha=0.7,
            color=color,
            edgecolor="white",
            label=f"{model} (ECE={cal['ece']:.3f})",
        )

        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Fraction of Correct Predictions")
        ax.set_title(f"{model}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=9)
        ax.set_aspect("equal")

    fig.suptitle(title, fontsize=14, y=1.02)

    if output_path is None:
        output_path = os.path.join(CALIBRATION_PLOTS_DIR, "reliability_diagram.png")
    save_figure(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# 2. Confidence Histogram
# ---------------------------------------------------------------------------

def plot_confidence_histogram(
    inference_results=None,
    csv_path=None,
    n_bins=20,
    output_path=None,
    title="Confidence Score Distribution",
):
    """
    Plot confidence score distribution histograms for each model.

    Args:
        inference_results: Raw inference output dict (optional).
        csv_path: Path to CSV with [model, confidence] (optional).
        n_bins: Number of histogram bins.
        output_path: Destination PNG path.
        title: Plot title.

    Returns:
        Path to saved PNG, or None on failure.
    """
    setup_plot_theme()
    ensure_output_dirs()

    df = _resolve_df(inference_results, csv_path)
    if df is None or df.empty or "confidence" not in df.columns:
        logger.warning("Insufficient data for confidence histogram.")
        return None

    models = sorted(df["model"].unique())

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, model in enumerate(models):
        model_df = df[df["model"] == model]
        ax.hist(
            model_df["confidence"].dropna(),
            bins=n_bins,
            alpha=0.6,
            label=model,
            color=get_model_color(model, idx),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(title="Model")
    ax.set_xlim(0, 1)
    ax.yaxis.grid(True, alpha=0.3)

    if output_path is None:
        output_path = os.path.join(CALIBRATION_PLOTS_DIR, "confidence_histogram.png")
    save_figure(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# 3. Calibration Curve (ECE overlay)
# ---------------------------------------------------------------------------

def plot_calibration_curve(
    inference_results=None,
    csv_path=None,
    n_bins=10,
    output_path=None,
    title="Calibration Curve",
):
    """
    Plot calibration curves for each model on a single axis.

    Each curve shows how mean bin accuracy relates to mean bin confidence,
    with ECE annotated in the legend.

    Args:
        inference_results: Raw inference output dict (optional).
        csv_path: CSV path (optional).
        n_bins: Number of calibration bins.
        output_path: Destination PNG.
        title: Plot title.

    Returns:
        Path to saved PNG, or None on failure.
    """
    setup_plot_theme()
    ensure_output_dirs()

    df = _resolve_df(inference_results, csv_path)
    df = _derive_correctness(df)
    if df is None or df.empty or "correct" not in df.columns:
        logger.warning("Insufficient data for calibration curve.")
        return None

    models = sorted(df["model"].unique())
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")

    for idx, model in enumerate(models):
        model_df = df[df["model"] == model]
        cal = _compute_calibration_bins(
            model_df["confidence"].values,
            model_df["correct"].values,
            n_bins=n_bins,
        )
        # Only plot bins with data
        mask = cal["bin_counts"] > 0
        ax.plot(
            cal["bin_confidences"][mask],
            cal["bin_accuracies"][mask],
            marker="o",
            linewidth=2,
            markersize=7,
            label=f"{model} (ECE={cal['ece']:.3f})",
            color=get_model_color(model, idx),
        )

    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction Correct")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.set_aspect("equal")

    if output_path is None:
        output_path = os.path.join(CALIBRATION_PLOTS_DIR, "calibration_curve.png")
    save_figure(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# 4. Expected vs Actual Accuracy
# ---------------------------------------------------------------------------

def plot_expected_vs_actual_accuracy(
    inference_results=None,
    csv_path=None,
    n_bins=10,
    output_path=None,
    title="Expected vs Actual Accuracy",
):
    """
    Scatter plot of expected (mean confidence per bin) vs actual (fraction
    correct per bin) accuracy for each model.

    Points on the diagonal indicate perfect calibration.

    Args:
        inference_results: Raw inference output dict (optional).
        csv_path: CSV path (optional).
        n_bins: Number of bins.
        output_path: Destination PNG.
        title: Plot title.

    Returns:
        Path to saved PNG, or None on failure.
    """
    setup_plot_theme()
    ensure_output_dirs()

    df = _resolve_df(inference_results, csv_path)
    df = _derive_correctness(df)
    if df is None or df.empty or "correct" not in df.columns:
        logger.warning("Insufficient data for expected vs actual plot.")
        return None

    models = sorted(df["model"].unique())
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")

    for idx, model in enumerate(models):
        model_df = df[df["model"] == model]
        cal = _compute_calibration_bins(
            model_df["confidence"].values,
            model_df["correct"].values,
            n_bins=n_bins,
        )
        mask = cal["bin_counts"] > 0
        sizes = cal["bin_counts"][mask] * 5 + 20  # size proportional to count
        ax.scatter(
            cal["bin_confidences"][mask],
            cal["bin_accuracies"][mask],
            s=sizes,
            alpha=0.8,
            label=model,
            color=get_model_color(model, idx),
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_xlabel("Expected Accuracy (Mean Confidence)")
    ax.set_ylabel("Actual Accuracy (Fraction Correct)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(title="Model")
    ax.set_aspect("equal")

    if output_path is None:
        output_path = os.path.join(CALIBRATION_PLOTS_DIR, "expected_vs_actual.png")
    save_figure(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _resolve_df(inference_results, csv_path):
    """Resolve the best available data source into a DataFrame."""
    if csv_path is not None:
        return load_csv_safe(csv_path)
    if inference_results is not None:
        return extract_inference_dataframe(inference_results)
    return None
