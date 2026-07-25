"""
Attention drift analysis for the Pathogen Intelligence System (Step 7).

This module compares clean and perturbed attention maps to quantify how much
attention shifts under perturbation severity. It remains deterministic and
uses only map geometry and overlap-based metrics.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.explainability.attention_analysis import normalize_attention_map


_SEVERITY_ORDER = {
    "clean": 0,
    "none": 0,
    "original": 0,
    "baseline": 0,
    "very_low": 1,
    "low": 2,
    "moderate": 3,
    "high": 4,
    "critical": 5,
}


def _severity_rank(value: Any) -> int:
    return _SEVERITY_ORDER.get(str(value).strip().lower(), 99)


def _center_of_mass(attention_map: np.ndarray) -> Tuple[float, float]:
    total = float(attention_map.sum())
    if total <= 0.0:
        return 0.5, 0.5
    y_coords, x_coords = np.indices(attention_map.shape)
    center_y = float((attention_map * y_coords).sum() / total) / max(attention_map.shape[0] - 1, 1)
    center_x = float((attention_map * x_coords).sum() / total) / max(attention_map.shape[1] - 1, 1)
    return float(np.clip(center_x, 0.0, 1.0)), float(np.clip(center_y, 0.0, 1.0))


def _top_attention_mask(attention_map: np.ndarray, threshold_fraction: float = 0.60) -> np.ndarray:
    peak = float(attention_map.max())
    if peak <= 0.0:
        return np.zeros_like(attention_map, dtype=bool)
    return attention_map >= (peak * threshold_fraction)


def compute_attention_drift_metrics(
    clean_attention: Any,
    perturbed_attention: Any,
    threshold_fraction: float = 0.60,
) -> Dict[str, Any]:
    """Compare two attention maps and produce deterministic drift metrics."""
    clean_map = normalize_attention_map(clean_attention)
    perturbed_map = normalize_attention_map(perturbed_attention)
    if clean_map is None or perturbed_map is None:
        return {
            "attention_drift_score": None,
            "activation_displacement": None,
            "overlap_score": None,
            "overlap_reduction": None,
            "attention_l1_shift": None,
            "attention_entropy_shift": None,
            "collapse_flag": False,
            "drift_reason": "invalid_attention_map",
        }

    if clean_map.shape != perturbed_map.shape:
        target_height = max(clean_map.shape[0], perturbed_map.shape[0])
        target_width = max(clean_map.shape[1], perturbed_map.shape[1])
        clean_map = _resize_like(clean_map, (target_height, target_width))
        perturbed_map = _resize_like(perturbed_map, (target_height, target_width))

    clean_center = _center_of_mass(clean_map)
    perturbed_center = _center_of_mass(perturbed_map)
    displacement = float(np.sqrt((clean_center[0] - perturbed_center[0]) ** 2 + (clean_center[1] - perturbed_center[1]) ** 2) / np.sqrt(2.0))

    clean_mask = _top_attention_mask(clean_map, threshold_fraction=threshold_fraction)
    perturbed_mask = _top_attention_mask(perturbed_map, threshold_fraction=threshold_fraction)
    union = np.logical_or(clean_mask, perturbed_mask)
    intersection = np.logical_and(clean_mask, perturbed_mask)
    overlap_score = float(intersection.sum() / union.sum()) if union.any() else 0.0
    overlap_reduction = float(np.clip(1.0 - overlap_score, 0.0, 1.0))

    l1_shift = float(np.mean(np.abs(clean_map - perturbed_map)))
    clean_entropy = _map_entropy(clean_map)
    perturbed_entropy = _map_entropy(perturbed_map)
    entropy_shift = float(np.clip(perturbed_entropy - clean_entropy, 0.0, 1.0))

    drift_score = float(np.clip(0.45 * displacement + 0.35 * overlap_reduction + 0.20 * entropy_shift, 0.0, 1.0))
    collapse_flag = drift_score >= 0.65 or overlap_score <= 0.30 or entropy_shift >= 0.25
    drift_reason = []
    if displacement >= 0.25:
        drift_reason.append("activation_displacement")
    if overlap_reduction >= 0.35:
        drift_reason.append("overlap_reduction")
    if entropy_shift >= 0.25:
        drift_reason.append("entropy_spike")

    return {
        "attention_drift_score": round(drift_score, 6),
        "activation_displacement": round(displacement, 6),
        "overlap_score": round(overlap_score, 6),
        "overlap_reduction": round(overlap_reduction, 6),
        "attention_l1_shift": round(l1_shift, 6),
        "attention_entropy_shift": round(entropy_shift, 6),
        "collapse_flag": bool(collapse_flag),
        "drift_reason": "|".join(drift_reason) if drift_reason else None,
    }


def _map_entropy(attention_map: np.ndarray) -> float:
    values = np.maximum(attention_map.astype(float), 0.0).ravel()
    total = float(values.sum())
    if total <= 0.0:
        return 1.0
    probabilities = np.clip(values / total, 1e-12, 1.0)
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return float(np.clip(entropy / np.log(len(probabilities)), 0.0, 1.0)) if len(probabilities) > 1 else 0.0


def _resize_like(attention_map: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    target_height, target_width = shape
    source_height, source_width = attention_map.shape
    if (source_height, source_width) == shape:
        return attention_map
    y_indices = np.linspace(0, source_height - 1, target_height)
    x_indices = np.linspace(0, source_width - 1, target_width)
    y_indices = np.round(y_indices).astype(int)
    x_indices = np.round(x_indices).astype(int)
    resized = attention_map[np.ix_(y_indices, x_indices)]
    return resized


def compute_attention_drift_analysis(
    attention_data: Any,
    attention_column: str = "attention_map",
    sample_column: str = "sample_id",
    model_column: str = "model_name",
    severity_column: str = "severity_level",
    perturbation_column: str = "perturbation_type",
) -> pd.DataFrame:
    """Compare clean and perturbed attention maps for each sample/model."""
    frame = pd.DataFrame(attention_data).copy()
    if frame.empty or attention_column not in frame.columns:
        return pd.DataFrame(columns=[
            sample_column,
            model_column,
            severity_column,
            perturbation_column,
            "attention_drift_score",
            "activation_displacement",
            "overlap_score",
            "overlap_reduction",
            "attention_l1_shift",
            "attention_entropy_shift",
            "collapse_flag",
            "drift_reason",
            "baseline_severity",
        ])

    grouping_columns = [sample_column]
    if model_column in frame.columns:
        grouping_columns.append(model_column)

    records: List[Dict[str, Any]] = []
    for key_values, group in frame.groupby(grouping_columns):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        key_map = dict(zip(grouping_columns, [str(value) for value in key_values]))

        ordered = group.copy()
        if severity_column in ordered.columns:
            ordered = ordered.assign(_severity_rank=ordered[severity_column].apply(_severity_rank)).sort_values(["_severity_rank", severity_column])
        else:
            ordered = ordered.reset_index(drop=True)
            ordered["_severity_rank"] = range(len(ordered))

        baseline_rows = ordered[ordered[severity_column].astype(str).str.lower().isin({"clean", "none", "original", "baseline"})] if severity_column in ordered.columns else pd.DataFrame()
        baseline_row = baseline_rows.iloc[0] if not baseline_rows.empty else ordered.iloc[0]
        baseline_attention = baseline_row.get(attention_column)
        baseline_severity = str(baseline_row.get(severity_column, "baseline")) if severity_column in baseline_row else "baseline"

        for _, row in ordered.iterrows():
            current_attention = row.get(attention_column)
            metrics = compute_attention_drift_metrics(baseline_attention, current_attention)
            if metrics["attention_drift_score"] is None:
                continue
            if row.get("_severity_rank", 0) == baseline_row.get("_severity_rank", 0):
                continue

            records.append({
                sample_column: key_map.get(sample_column, ""),
                model_column: key_map.get(model_column, None),
                severity_column: str(row.get(severity_column, "unknown")) if severity_column in row else "unknown",
                perturbation_column: str(row.get(perturbation_column, "unknown")) if perturbation_column in row else "unknown",
                "attention_drift_score": metrics["attention_drift_score"],
                "activation_displacement": metrics["activation_displacement"],
                "overlap_score": metrics["overlap_score"],
                "overlap_reduction": metrics["overlap_reduction"],
                "attention_l1_shift": metrics["attention_l1_shift"],
                "attention_entropy_shift": metrics["attention_entropy_shift"],
                "collapse_flag": metrics["collapse_flag"],
                "drift_reason": metrics["drift_reason"],
                "baseline_severity": baseline_severity,
                "severity_rank": int(row.get("_severity_rank", _severity_rank(row.get(severity_column, "unknown")))),
            })

    if not records:
        return pd.DataFrame(columns=[
            sample_column,
            model_column,
            severity_column,
            perturbation_column,
            "attention_drift_score",
            "activation_displacement",
            "overlap_score",
            "overlap_reduction",
            "attention_l1_shift",
            "attention_entropy_shift",
            "collapse_flag",
            "drift_reason",
            "baseline_severity",
            "severity_rank",
        ])

    return pd.DataFrame(records).sort_values(
        ["attention_drift_score", "activation_displacement"], ascending=[False, False]
    ).reset_index(drop=True)


def compute_attention_stability_curve(drift_analysis: pd.DataFrame) -> pd.DataFrame:
    """Build a severity curve describing how attention stability degrades."""
    if drift_analysis is None or drift_analysis.empty or "severity_rank" not in drift_analysis.columns:
        return pd.DataFrame(columns=[
            "severity_level",
            "severity_rank",
            "total_observations",
            "mean_attention_drift_score",
            "mean_activation_displacement",
            "mean_overlap_score",
            "collapse_rate",
        ])

    rows: List[Dict[str, Any]] = []
    for severity_label, group in drift_analysis.groupby("severity_level"):
        drift = pd.to_numeric(group["attention_drift_score"], errors="coerce").dropna() if "attention_drift_score" in group.columns else pd.Series(dtype=float)
        displacement = pd.to_numeric(group["activation_displacement"], errors="coerce").dropna() if "activation_displacement" in group.columns else pd.Series(dtype=float)
        overlap = pd.to_numeric(group["overlap_score"], errors="coerce").dropna() if "overlap_score" in group.columns else pd.Series(dtype=float)
        if drift.empty:
            continue
        rows.append({
            "severity_level": str(severity_label),
            "severity_rank": _severity_rank(severity_label),
            "total_observations": int(len(drift)),
            "mean_attention_drift_score": round(float(drift.mean()), 6),
            "mean_activation_displacement": round(float(displacement.mean()), 6) if not displacement.empty else 0.0,
            "mean_overlap_score": round(float(overlap.mean()), 6) if not overlap.empty else 0.0,
            "collapse_rate": round(float(group["collapse_flag"].mean()), 6) if "collapse_flag" in group.columns else 0.0,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "severity_level",
            "severity_rank",
            "total_observations",
            "mean_attention_drift_score",
            "mean_activation_displacement",
            "mean_overlap_score",
            "collapse_rate",
        ])

    return pd.DataFrame(rows).sort_values("severity_rank").reset_index(drop=True)


def compute_attention_collapse_report(drift_analysis: pd.DataFrame) -> pd.DataFrame:
    """Extract the first attention collapse point for each sample/model pair."""
    if drift_analysis is None or drift_analysis.empty or "collapse_flag" not in drift_analysis.columns:
        return pd.DataFrame(columns=[
            "sample_id",
            "model_name",
            "collapse_severity",
            "collapse_severity_rank",
            "collapse_trigger",
            "attention_drift_score",
            "activation_displacement",
            "overlap_score",
            "collapse_reason",
        ])

    grouping_columns = [col for col in ["sample_id", "model_name"] if col in drift_analysis.columns]
    if not grouping_columns:
        return pd.DataFrame(columns=[
            "sample_id",
            "model_name",
            "collapse_severity",
            "collapse_severity_rank",
            "collapse_trigger",
            "attention_drift_score",
            "activation_displacement",
            "overlap_score",
            "collapse_reason",
        ])

    rows: List[Dict[str, Any]] = []
    for key_values, group in drift_analysis.groupby(grouping_columns):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        key_map = dict(zip(grouping_columns, [str(value) for value in key_values]))
        collapsed = group[group["collapse_flag"].astype(bool)]
        if collapsed.empty:
            continue
        collapsed = collapsed.sort_values(["severity_rank", "attention_drift_score"], ascending=[True, False])
        row = collapsed.iloc[0]
        rows.append({
            "sample_id": key_map.get("sample_id", ""),
            "model_name": key_map.get("model_name", ""),
            "collapse_severity": str(row.get("severity_level", "unknown")),
            "collapse_severity_rank": int(row.get("severity_rank", _severity_rank(row.get("severity_level", "unknown")))),
            "collapse_trigger": str(row.get("perturbation_type", row.get("severity_level", "unknown"))),
            "attention_drift_score": round(float(row.get("attention_drift_score", 0.0)), 6),
            "activation_displacement": round(float(row.get("activation_displacement", 0.0)), 6),
            "overlap_score": round(float(row.get("overlap_score", 0.0)), 6),
            "collapse_reason": str(row.get("drift_reason", "")),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "sample_id",
            "model_name",
            "collapse_severity",
            "collapse_severity_rank",
            "collapse_trigger",
            "attention_drift_score",
            "activation_displacement",
            "overlap_score",
            "collapse_reason",
        ])

    return pd.DataFrame(rows).sort_values(["attention_drift_score", "sample_id"], ascending=[False, True]).reset_index(drop=True)
