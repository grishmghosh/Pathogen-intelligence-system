"""
Confidence Dispersion Analysis for Pathogen Intelligence System (Step 6).

This module measures how concentrated or flat a prediction distribution is.
The computations are deterministic and only rely on probability vectors that
already exist in the inference output.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.uncertainty.uncertainty_metrics import normalize_probability_vector


def _distribution_shape(concentration: float) -> str:
    if concentration >= 0.75:
        return "peaked"
    if concentration >= 0.50:
        return "balanced"
    return "flat"


def compute_confidence_dispersion(
    data: Any,
    probability_column: str = "probabilities",
    confidence_column: str = "confidence",
    sample_column: str = "sample_id",
    model_column: str = "model_name",
    severity_column: str = "severity_level",
    perturbation_column: str = "perturbation_type",
) -> pd.DataFrame:
    """
    Measure spread and concentration of each prediction probability vector.

    The output preserves the input columns and appends the following fields:
    ``confidence_variance``, ``confidence_std``, ``confidence_spread``,
    ``confidence_concentration``, ``confidence_flatness``, ``top2_gap``,
    ``distribution_shape``, ``normalization_status``.
    """
    frame = pd.DataFrame(data).copy()
    if frame.empty:
        return pd.DataFrame(columns=[
            sample_column,
            model_column,
            severity_column,
            perturbation_column,
            "confidence_variance",
            "confidence_std",
            "confidence_spread",
            "confidence_concentration",
            "confidence_flatness",
            "top2_gap",
            "distribution_shape",
            "normalization_status",
        ])

    records: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        probability_value = row.get(probability_column) if probability_column in frame.columns else None
        confidence_value = row.get(confidence_column) if confidence_column in frame.columns else None
        vector, status = normalize_probability_vector(
            probability_value,
            confidence=float(confidence_value) if pd.notna(confidence_value) else None,
        )

        if vector.size == 0:
            continue

        ordered = np.sort(vector)[::-1]
        top1 = float(ordered[0])
        top2 = float(ordered[1]) if ordered.size > 1 else 0.0
        variance = float(np.var(vector))
        std = float(np.std(vector))
        spread = float(np.max(vector) - np.min(vector))
        concentration = top1
        flatness = 1.0 - concentration
        top2_gap = top1 - top2

        record = row.to_dict()
        record.update({
            "confidence_variance": round(variance, 6),
            "confidence_std": round(std, 6),
            "confidence_spread": round(spread, 6),
            "confidence_concentration": round(concentration, 6),
            "confidence_flatness": round(flatness, 6),
            "top2_gap": round(top2_gap, 6),
            "distribution_shape": _distribution_shape(concentration),
            "normalization_status": status,
        })
        records.append(record)

    result = pd.DataFrame(records)
    if sample_column in result.columns:
        result[sample_column] = result[sample_column].astype(str)
    if model_column in result.columns:
        result[model_column] = result[model_column].astype(str)
    return result


def aggregate_confidence_dispersion(
    dispersion: pd.DataFrame,
    group_column: str = "sample_id",
) -> pd.DataFrame:
    """Aggregate confidence dispersion metrics for samples or models."""
    if dispersion is None or dispersion.empty or group_column not in dispersion.columns:
        return pd.DataFrame(columns=[
            group_column,
            "mean_confidence_variance",
            "mean_confidence_std",
            "mean_confidence_spread",
            "mean_confidence_concentration",
            "mean_top2_gap",
            "observation_count",
        ])

    rows: List[Dict[str, Any]] = []
    for group_value, group in dispersion.groupby(group_column):
        variance = pd.to_numeric(group["confidence_variance"], errors="coerce").dropna() if "confidence_variance" in group.columns else pd.Series(dtype=float)
        std = pd.to_numeric(group["confidence_std"], errors="coerce").dropna() if "confidence_std" in group.columns else pd.Series(dtype=float)
        spread = pd.to_numeric(group["confidence_spread"], errors="coerce").dropna() if "confidence_spread" in group.columns else pd.Series(dtype=float)
        concentration = pd.to_numeric(group["confidence_concentration"], errors="coerce").dropna() if "confidence_concentration" in group.columns else pd.Series(dtype=float)
        top2_gap = pd.to_numeric(group["top2_gap"], errors="coerce").dropna() if "top2_gap" in group.columns else pd.Series(dtype=float)

        if variance.empty and std.empty and spread.empty and concentration.empty:
            continue

        rows.append({
            group_column: str(group_value),
            "mean_confidence_variance": round(float(variance.mean()), 6) if not variance.empty else None,
            "mean_confidence_std": round(float(std.mean()), 6) if not std.empty else None,
            "mean_confidence_spread": round(float(spread.mean()), 6) if not spread.empty else None,
            "mean_confidence_concentration": round(float(concentration.mean()), 6) if not concentration.empty else None,
            "mean_top2_gap": round(float(top2_gap.mean()), 6) if not top2_gap.empty else None,
            "observation_count": int(len(group)),
        })

    if not rows:
        return pd.DataFrame(columns=[
            group_column,
            "mean_confidence_variance",
            "mean_confidence_std",
            "mean_confidence_spread",
            "mean_confidence_concentration",
            "mean_top2_gap",
            "observation_count",
        ])

    return pd.DataFrame(rows).sort_values(group_column).reset_index(drop=True)


def compute_dispersion_summary(dispersion: pd.DataFrame) -> Dict[str, Any]:
    """Summarise confidence dispersion metrics."""
    if dispersion is None or dispersion.empty or "confidence_variance" not in dispersion.columns:
        return {
            "total_observations": 0,
            "mean_confidence_variance": 0.0,
            "mean_confidence_std": 0.0,
            "mean_confidence_spread": 0.0,
            "mean_confidence_concentration": 0.0,
            "flat_count": 0,
            "peaked_count": 0,
        }

    variance = pd.to_numeric(dispersion["confidence_variance"], errors="coerce").dropna()
    std = pd.to_numeric(dispersion["confidence_std"], errors="coerce").dropna() if "confidence_std" in dispersion.columns else pd.Series(dtype=float)
    spread = pd.to_numeric(dispersion["confidence_spread"], errors="coerce").dropna() if "confidence_spread" in dispersion.columns else pd.Series(dtype=float)
    concentration = pd.to_numeric(dispersion["confidence_concentration"], errors="coerce").dropna() if "confidence_concentration" in dispersion.columns else pd.Series(dtype=float)
    shapes = dispersion["distribution_shape"].astype(str) if "distribution_shape" in dispersion.columns else pd.Series(dtype=str)

    if variance.empty:
        return {
            "total_observations": 0,
            "mean_confidence_variance": 0.0,
            "mean_confidence_std": 0.0,
            "mean_confidence_spread": 0.0,
            "mean_confidence_concentration": 0.0,
            "flat_count": 0,
            "peaked_count": 0,
        }

    return {
        "total_observations": int(len(variance)),
        "mean_confidence_variance": round(float(variance.mean()), 6),
        "mean_confidence_std": round(float(std.mean()), 6) if not std.empty else 0.0,
        "mean_confidence_spread": round(float(spread.mean()), 6) if not spread.empty else 0.0,
        "mean_confidence_concentration": round(float(concentration.mean()), 6) if not concentration.empty else 0.0,
        "flat_count": int((shapes == "flat").sum()) if not shapes.empty else 0,
        "peaked_count": int((shapes == "peaked").sum()) if not shapes.empty else 0,
    }
