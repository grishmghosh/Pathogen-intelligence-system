"""
Experiment Summary Report Generator for Pathogen Intelligence System.

Generates aggregated experiment statistics from robustness and inference
outputs, including:
    - Best / worst robustness scores
    - Calibration summary (ECE per model)
    - Degradation summary per perturbation
    - Aggregated per-model statistics

Exports results as CSV and JSON to results/plots/summaries/.

Architecture:
    This module depends ONLY on visualization_utils for shared helpers.
    It does NOT import from inference or analysis directly.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.visualization_utils import (
    logger,
    ensure_output_dirs,
    SUMMARIES_DIR,
    extract_robustness_dataframe,
    extract_inference_dataframe,
    load_csv_safe,
    load_json_safe,
)

# Reuse calibration bin computation for ECE
from visualization.calibration_plots import _compute_calibration_bins, _derive_correctness


# ---------------------------------------------------------------------------
# 1. Generate Experiment Summary
# ---------------------------------------------------------------------------

def generate_experiment_summary(
    inference_results=None,
    robustness_report=None,
    inference_csv_path=None,
    robustness_csv_path=None,
):
    """
    Generate a comprehensive experiment summary dictionary.

    The summary contains:
        - per_model_stats: dict of per-model aggregated metrics
        - best_robustness: model with highest robustness score
        - worst_robustness: model with lowest robustness score
        - calibration_summary: ECE per model
        - degradation_summary: mean confidence drop per perturbation
        - overall_statistics: global aggregates

    Args:
        inference_results: Raw inference output dict (optional).
        robustness_report: Robustness report dict (optional).
        inference_csv_path: Path to inference CSV (optional).
        robustness_csv_path: Path to robustness CSV (optional).

    Returns:
        dict with experiment summary, or empty dict on total failure.
    """
    summary = {
        "per_model_stats": {},
        "best_robustness": None,
        "worst_robustness": None,
        "calibration_summary": {},
        "degradation_summary": {},
        "overall_statistics": {},
    }

    # --- Load inference data ---
    inf_df = None
    if inference_csv_path is not None:
        inf_df = load_csv_safe(inference_csv_path)
    elif inference_results is not None:
        inf_df = extract_inference_dataframe(inference_results)

    # --- Load robustness data ---
    rob_df = None
    if robustness_csv_path is not None:
        rob_df = load_csv_safe(robustness_csv_path)
    elif robustness_report is not None:
        rob_df = extract_robustness_dataframe(robustness_report)

    # --- Per-model stats from robustness report ---
    if robustness_report is not None:
        best_score = -1
        worst_score = 101
        for model_name, model_report in robustness_report.items():
            if model_name == "model_comparison":
                continue
            if "error" in model_report:
                summary["per_model_stats"][model_name] = {"error": model_report["error"]}
                continue

            scores = model_report.get("robustness_score", {})
            consistency = model_report.get("consistency_analysis", {})
            confidence = model_report.get("confidence_analysis", {})
            sensitivity = model_report.get("sensitivity_analysis", {})

            robustness_score = scores.get("robustness_score", 0)
            stats = {
                "robustness_score": robustness_score,
                "interpretation": scores.get("interpretation", "Unknown"),
                "consistency_score": scores.get("consistency_score", 0),
                "stability_score": scores.get("stability_score", 0),
                "resistance_score": scores.get("resistance_score", 0),
                "consistency_rate": consistency.get("consistency_rate", 0),
                "original_confidence": confidence.get("original_confidence", 0),
                "mean_confidence": confidence.get("mean_confidence", 0),
                "confidence_drop": confidence.get("confidence_drop", 0),
                "confidence_std": confidence.get("std_confidence", 0),
                "prediction_flips": len(
                    sensitivity.get("prediction_flip_perturbations", [])
                ),
                "most_damaging": (
                    sensitivity.get("most_damaging_perturbation", {}).get("name", "N/A")
                    if sensitivity.get("most_damaging_perturbation") else "N/A"
                ),
            }
            summary["per_model_stats"][model_name] = stats

            if robustness_score > best_score:
                best_score = robustness_score
                summary["best_robustness"] = {
                    "model": model_name,
                    "score": robustness_score,
                    "interpretation": scores.get("interpretation", "Unknown"),
                }
            if robustness_score < worst_score:
                worst_score = robustness_score
                summary["worst_robustness"] = {
                    "model": model_name,
                    "score": robustness_score,
                    "interpretation": scores.get("interpretation", "Unknown"),
                }

    # --- Calibration summary (ECE per model) ---
    if inf_df is not None and not inf_df.empty:
        cal_df = _derive_correctness(inf_df)
        if cal_df is not None and "correct" in cal_df.columns:
            models = sorted(cal_df["model"].unique())
            for model in models:
                m_df = cal_df[cal_df["model"] == model]
                cal = _compute_calibration_bins(
                    m_df["confidence"].values,
                    m_df["correct"].values,
                )
                summary["calibration_summary"][model] = {
                    "ece": round(cal["ece"], 6),
                    "num_samples": len(m_df),
                }

    # --- Degradation summary (mean confidence drop per perturbation) ---
    working_df = inf_df if inf_df is not None else rob_df
    if working_df is not None and not working_df.empty:
        if (
            "confidence" in working_df.columns
            and "perturbation" in working_df.columns
            and "model" in working_df.columns
        ):
            original_confs = (
                working_df[working_df["perturbation"] == "original"]
                .set_index("model")["confidence"]
                .to_dict()
            )
            if original_confs:
                non_original = working_df[working_df["perturbation"] != "original"].copy()
                non_original["conf_drop"] = non_original.apply(
                    lambda r: original_confs.get(r["model"], 0) - r["confidence"],
                    axis=1,
                )
                deg = (
                    non_original.groupby("perturbation")["conf_drop"]
                    .agg(["mean", "std", "max"])
                    .rename(columns={"mean": "mean_drop", "std": "std_drop", "max": "max_drop"})
                )
                summary["degradation_summary"] = deg.round(6).to_dict("index")

    # --- Overall statistics ---
    if working_df is not None and not working_df.empty:
        summary["overall_statistics"] = {
            "total_models": int(working_df["model"].nunique()) if "model" in working_df.columns else 0,
            "total_perturbations": int(working_df["perturbation"].nunique()) if "perturbation" in working_df.columns else 0,
            "total_records": len(working_df),
            "mean_confidence": round(float(working_df["confidence"].mean()), 6)
            if "confidence" in working_df.columns else None,
        }

    logger.info("Experiment summary generated.")
    return summary


# ---------------------------------------------------------------------------
# 2. Export as CSV
# ---------------------------------------------------------------------------

def export_summary_csv(summary, output_path=None):
    """
    Export per-model experiment summary statistics as a CSV file.

    Args:
        summary: Dict from generate_experiment_summary().
        output_path: Destination CSV path (optional).

    Returns:
        Path to saved CSV, or None on failure.
    """
    ensure_output_dirs()

    per_model = summary.get("per_model_stats", {})
    if not per_model:
        logger.warning("No per-model stats to export.")
        return None

    rows = []
    for model_name, stats in per_model.items():
        if isinstance(stats, dict) and "error" not in stats:
            row = {"model": model_name}
            row.update(stats)
            rows.append(row)

    if not rows:
        logger.warning("No valid model stats for CSV export.")
        return None

    df = pd.DataFrame(rows)

    if output_path is None:
        output_path = os.path.join(SUMMARIES_DIR, "experiment_summary.csv")

    dirpath = os.path.dirname(output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info("Exported summary CSV: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 3. Export as JSON
# ---------------------------------------------------------------------------

def export_summary_json(summary, output_path=None):
    """
    Export the full experiment summary as a JSON file.

    Args:
        summary: Dict from generate_experiment_summary().
        output_path: Destination JSON path (optional).

    Returns:
        Path to saved JSON, or None on failure.
    """
    ensure_output_dirs()

    if not summary:
        logger.warning("Empty summary; nothing to export.")
        return None

    if output_path is None:
        output_path = os.path.join(SUMMARIES_DIR, "experiment_summary.json")

    dirpath = os.path.dirname(output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # Ensure JSON-serializable (convert numpy types)
    clean = _make_json_serializable(summary)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2, ensure_ascii=False)

    logger.info("Exported summary JSON: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _make_json_serializable(obj):
    """Recursively convert numpy/pandas types to native Python types."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj
