"""
Attention analysis utilities for the Pathogen Intelligence System (Step 7).

This module extracts deterministic statistics from Grad-CAM or attention maps.
It does not generate visuals; it only computes metrics that downstream modules
can aggregate, compare, or export.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _safe_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value)
    elif isinstance(value, pd.Series):
        array = value.to_numpy()
    else:
        return None
    array = np.asarray(array, dtype=float)
    if array.size == 0:
        return None
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array


def normalize_attention_map(attention_map: Any) -> Optional[np.ndarray]:
    """Normalize an attention map to the range [0, 1]."""
    array = _safe_array(attention_map)
    if array is None:
        return None
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        return None
    array = np.maximum(array, 0.0)
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum <= minimum:
        return np.zeros_like(array, dtype=float)
    return (array - minimum) / (maximum - minimum)


def _attention_entropy(flat_map: np.ndarray) -> float:
    values = np.asarray(flat_map, dtype=float).ravel()
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= 0.0:
        return 1.0
    probabilities = values / total
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return float(np.clip(entropy / np.log(len(probabilities)), 0.0, 1.0)) if len(probabilities) > 1 else 0.0


def _center_of_mass(attention_map: np.ndarray) -> Tuple[float, float]:
    height, width = attention_map.shape
    total = float(attention_map.sum())
    if total <= 0.0:
        return 0.5, 0.5
    y_coords, x_coords = np.indices(attention_map.shape)
    center_y = float((attention_map * y_coords).sum() / total) / max(height - 1, 1)
    center_x = float((attention_map * x_coords).sum() / total) / max(width - 1, 1)
    return float(np.clip(center_x, 0.0, 1.0)), float(np.clip(center_y, 0.0, 1.0))


def compute_attention_features(attention_map: Any) -> Dict[str, Any]:
    """Compute deterministic summary statistics for a single attention map."""
    normalized = normalize_attention_map(attention_map)
    if normalized is None:
        return {
            "attention_status": "invalid",
            "attention_height": None,
            "attention_width": None,
            "attention_mean": None,
            "attention_variance": None,
            "attention_std": None,
            "attention_entropy": None,
            "attention_concentration": None,
            "attention_spread": None,
            "attention_peak": None,
            "attention_center_x": None,
            "attention_center_y": None,
            "attention_mass_above_mean": None,
            "attention_active_ratio": None,
        }

    height, width = normalized.shape
    mean = float(normalized.mean())
    variance = float(normalized.var())
    std = float(normalized.std())
    entropy = _attention_entropy(normalized)
    peak = float(normalized.max())
    minimum = float(normalized.min())
    spread = peak - minimum
    concentration = peak
    center_x, center_y = _center_of_mass(normalized)
    above_mean = float((normalized >= mean).mean())
    active_ratio = float((normalized >= (0.5 * peak if peak > 0 else 0.0)).mean())

    return {
        "attention_status": "ok",
        "attention_height": int(height),
        "attention_width": int(width),
        "attention_mean": round(mean, 6),
        "attention_variance": round(variance, 6),
        "attention_std": round(std, 6),
        "attention_entropy": round(entropy, 6),
        "attention_concentration": round(concentration, 6),
        "attention_spread": round(spread, 6),
        "attention_peak": round(peak, 6),
        "attention_center_x": round(center_x, 6),
        "attention_center_y": round(center_y, 6),
        "attention_mass_above_mean": round(above_mean, 6),
        "attention_active_ratio": round(active_ratio, 6),
        "normalized_attention_map": normalized,
    }


def compute_attention_analysis(
    data: Any,
    attention_column: str = "attention_map",
    sample_column: str = "sample_id",
    model_column: str = "model_name",
    severity_column: str = "severity_level",
    perturbation_column: str = "perturbation_type",
) -> pd.DataFrame:
    """Build a tabular attention analysis frame from input records."""
    frame = pd.DataFrame(data).copy()
    if frame.empty:
        return pd.DataFrame(columns=[
            sample_column,
            model_column,
            severity_column,
            perturbation_column,
            "attention_status",
            "attention_height",
            "attention_width",
            "attention_mean",
            "attention_variance",
            "attention_std",
            "attention_entropy",
            "attention_concentration",
            "attention_spread",
            "attention_peak",
            "attention_center_x",
            "attention_center_y",
            "attention_mass_above_mean",
            "attention_active_ratio",
            "normalized_attention_map",
        ])

    records: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        metrics = compute_attention_features(row.get(attention_column))
        record = row.to_dict()
        record.update(metrics)
        records.append(record)

    result = pd.DataFrame(records)
    for column in [sample_column, model_column, severity_column, perturbation_column]:
        if column in result.columns:
            result[column] = result[column].astype(str)
    return result


def aggregate_attention_summary(attention_analysis: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate attention metrics into a compact summary."""
    if attention_analysis is None or attention_analysis.empty or "attention_entropy" not in attention_analysis.columns:
        return {
            "total_observations": 0,
            "mean_attention_entropy": 0.0,
            "mean_attention_concentration": 0.0,
            "mean_attention_variance": 0.0,
            "mean_attention_std": 0.0,
            "most_focused_sample_ids": [],
            "least_focused_sample_ids": [],
        }

    entropy = pd.to_numeric(attention_analysis["attention_entropy"], errors="coerce").dropna()
    concentration = pd.to_numeric(attention_analysis["attention_concentration"], errors="coerce").dropna() if "attention_concentration" in attention_analysis.columns else pd.Series(dtype=float)
    variance = pd.to_numeric(attention_analysis["attention_variance"], errors="coerce").dropna() if "attention_variance" in attention_analysis.columns else pd.Series(dtype=float)
    std = pd.to_numeric(attention_analysis["attention_std"], errors="coerce").dropna() if "attention_std" in attention_analysis.columns else pd.Series(dtype=float)

    if entropy.empty:
        return {
            "total_observations": 0,
            "mean_attention_entropy": 0.0,
            "mean_attention_concentration": 0.0,
            "mean_attention_variance": 0.0,
            "mean_attention_std": 0.0,
            "most_focused_sample_ids": [],
            "least_focused_sample_ids": [],
        }

    focused_ids: List[str] = []
    least_focused_ids: List[str] = []
    if "sample_id" in attention_analysis.columns:
        sample_means = attention_analysis.groupby("sample_id")["attention_concentration"].mean().sort_values(ascending=False) if "attention_concentration" in attention_analysis.columns else pd.Series(dtype=float)
        if not sample_means.empty:
            focused_ids = [str(value) for value in sample_means.head(5).index.tolist()]
            least_focused_ids = [str(value) for value in sample_means.tail(5).index.tolist()]

    return {
        "total_observations": int(len(entropy)),
        "mean_attention_entropy": round(float(entropy.mean()), 6),
        "mean_attention_concentration": round(float(concentration.mean()), 6) if not concentration.empty else 0.0,
        "mean_attention_variance": round(float(variance.mean()), 6) if not variance.empty else 0.0,
        "mean_attention_std": round(float(std.mean()), 6) if not std.empty else 0.0,
        "most_focused_sample_ids": focused_ids,
        "least_focused_sample_ids": least_focused_ids,
    }
