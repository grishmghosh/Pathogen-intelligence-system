"""
Cross-model attention comparison for the Pathogen Intelligence System (Step 7).

This module compares attention maps across different models to determine how
consistently models focus on similar regions. It uses deterministic overlap,
correlation, and divergence metrics only.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.explainability.attention_analysis import normalize_attention_map
from analysis.explainability.attention_drift import compute_attention_drift_metrics


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    pair = np.vstack([left.ravel(), right.ravel()])
    if pair.shape[1] < 2:
        return 0.0
    if np.std(pair[0]) == 0.0 or np.std(pair[1]) == 0.0:
        return 0.0
    corr = np.corrcoef(pair[0], pair[1])[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_vec = left.ravel().astype(float)
    right_vec = right.ravel().astype(float)
    denom = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
    if denom <= 0.0:
        return 0.0
    return float(np.clip(np.dot(left_vec, right_vec) / denom, -1.0, 1.0))


def _top_mask(attention_map: np.ndarray, threshold_fraction: float = 0.60) -> np.ndarray:
    peak = float(attention_map.max())
    if peak <= 0.0:
        return np.zeros_like(attention_map, dtype=bool)
    return attention_map >= (peak * threshold_fraction)


def _overlap_score(left: np.ndarray, right: np.ndarray, threshold_fraction: float = 0.60) -> float:
    left_mask = _top_mask(left, threshold_fraction=threshold_fraction)
    right_mask = _top_mask(right, threshold_fraction=threshold_fraction)
    union = np.logical_or(left_mask, right_mask)
    if not union.any():
        return 0.0
    intersection = np.logical_and(left_mask, right_mask)
    return float(intersection.sum() / union.sum())


def compare_attention_maps(
    map_a: Any,
    map_b: Any,
    threshold_fraction: float = 0.60,
) -> Dict[str, Any]:
    """Compare two attention maps and return similarity/divergence metrics."""
    left = normalize_attention_map(map_a)
    right = normalize_attention_map(map_b)
    if left is None or right is None:
        return {
            "attention_similarity": None,
            "attention_divergence": None,
            "overlap_score": None,
            "correlation_score": None,
            "cosine_similarity": None,
            "mean_absolute_difference": None,
            "same_focus_flag": False,
            "comparison_status": "invalid_attention_map",
        }

    if left.shape != right.shape:
        target_height = max(left.shape[0], right.shape[0])
        target_width = max(left.shape[1], right.shape[1])
        left = _resize_like(left, (target_height, target_width))
        right = _resize_like(right, (target_height, target_width))

    overlap = _overlap_score(left, right, threshold_fraction=threshold_fraction)
    correlation = _safe_corr(left, right)
    cosine = _cosine_similarity(left, right)
    mad = float(np.mean(np.abs(left - right)))

    similarity = float(np.clip((overlap + max(correlation, 0.0) + max(cosine, 0.0)) / 3.0, 0.0, 1.0))
    divergence = float(np.clip(1.0 - similarity, 0.0, 1.0))
    same_focus_flag = overlap >= 0.60 and correlation >= 0.40

    return {
        "attention_similarity": round(similarity, 6),
        "attention_divergence": round(divergence, 6),
        "overlap_score": round(overlap, 6),
        "correlation_score": round(correlation, 6),
        "cosine_similarity": round(cosine, 6),
        "mean_absolute_difference": round(mad, 6),
        "same_focus_flag": bool(same_focus_flag),
        "comparison_status": "ok",
    }


def _resize_like(attention_map: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    target_height, target_width = shape
    source_height, source_width = attention_map.shape
    if (source_height, source_width) == shape:
        return attention_map
    y_indices = np.linspace(0, source_height - 1, target_height)
    x_indices = np.linspace(0, source_width - 1, target_width)
    y_indices = np.round(y_indices).astype(int)
    x_indices = np.round(x_indices).astype(int)
    return attention_map[np.ix_(y_indices, x_indices)]


def compute_cross_model_attention_comparison(
    attention_data: Any,
    attention_column: str = "attention_map",
    sample_column: str = "sample_id",
    model_column: str = "model_name",
    severity_column: str = "severity_level",
    perturbation_column: str = "perturbation_type",
    prediction_column: str = "predicted_class",
) -> pd.DataFrame:
    """Compare model attention maps within each sample/perturbation group."""
    frame = pd.DataFrame(attention_data).copy()
    if frame.empty or attention_column not in frame.columns or model_column not in frame.columns:
        return pd.DataFrame(columns=[
            sample_column,
            severity_column,
            perturbation_column,
            "model_a",
            "model_b",
            "attention_similarity",
            "attention_divergence",
            "overlap_score",
            "correlation_score",
            "cosine_similarity",
            "mean_absolute_difference",
            "same_focus_flag",
            "disagreement_attention_flag",
            "comparison_status",
        ])

    grouping_columns = [col for col in [sample_column, severity_column, perturbation_column] if col in frame.columns]
    if not grouping_columns:
        grouping_columns = [sample_column]

    rows: List[Dict[str, Any]] = []
    for key_values, group in frame.groupby(grouping_columns):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        key_map = dict(zip(grouping_columns, [str(value) for value in key_values]))

        models = sorted(group[model_column].dropna().astype(str).unique())
        if len(models) < 2:
            continue

        for model_a, model_b in combinations(models, 2):
            left_rows = group[group[model_column].astype(str) == model_a]
            right_rows = group[group[model_column].astype(str) == model_b]
            if left_rows.empty or right_rows.empty:
                continue
            left_row = left_rows.iloc[0]
            right_row = right_rows.iloc[0]
            comparison = compare_attention_maps(left_row[attention_column], right_row[attention_column])
            if comparison["attention_similarity"] is None:
                continue

            disagreement_flag = False
            if prediction_column in group.columns:
                left_prediction = left_row.get(prediction_column)
                right_prediction = right_row.get(prediction_column)
                disagreement_flag = str(left_prediction) != str(right_prediction)

            rows.append({
                sample_column: key_map.get(sample_column, ""),
                severity_column: key_map.get(severity_column, "unknown") if severity_column in key_map else "unknown",
                perturbation_column: key_map.get(perturbation_column, "unknown") if perturbation_column in key_map else "unknown",
                "model_a": model_a,
                "model_b": model_b,
                "attention_similarity": comparison["attention_similarity"],
                "attention_divergence": comparison["attention_divergence"],
                "overlap_score": comparison["overlap_score"],
                "correlation_score": comparison["correlation_score"],
                "cosine_similarity": comparison["cosine_similarity"],
                "mean_absolute_difference": comparison["mean_absolute_difference"],
                "same_focus_flag": comparison["same_focus_flag"],
                "disagreement_attention_flag": bool(disagreement_flag and comparison["attention_divergence"] >= 0.25),
                "comparison_status": comparison["comparison_status"],
            })

    if not rows:
        return pd.DataFrame(columns=[
            sample_column,
            severity_column,
            perturbation_column,
            "model_a",
            "model_b",
            "attention_similarity",
            "attention_divergence",
            "overlap_score",
            "correlation_score",
            "cosine_similarity",
            "mean_absolute_difference",
            "same_focus_flag",
            "disagreement_attention_flag",
            "comparison_status",
        ])

    return pd.DataFrame(rows).sort_values(
        ["attention_divergence", "attention_similarity"], ascending=[False, True]
    ).reset_index(drop=True)


def compute_attention_divergence_ranking(comparison_analysis: pd.DataFrame) -> pd.DataFrame:
    """Rank model pairs by attention divergence."""
    if comparison_analysis is None or comparison_analysis.empty or "attention_divergence" not in comparison_analysis.columns:
        return pd.DataFrame(columns=[
            "model_a",
            "model_b",
            "mean_attention_divergence",
            "max_attention_divergence",
            "mean_attention_similarity",
            "comparison_count",
        ])

    rows: List[Dict[str, Any]] = []
    for (model_a, model_b), group in comparison_analysis.groupby(["model_a", "model_b"]):
        divergence = pd.to_numeric(group["attention_divergence"], errors="coerce").dropna()
        similarity = pd.to_numeric(group["attention_similarity"], errors="coerce").dropna() if "attention_similarity" in group.columns else pd.Series(dtype=float)
        if divergence.empty:
            continue
        rows.append({
            "model_a": str(model_a),
            "model_b": str(model_b),
            "mean_attention_divergence": round(float(divergence.mean()), 6),
            "max_attention_divergence": round(float(divergence.max()), 6),
            "mean_attention_similarity": round(float(similarity.mean()), 6) if not similarity.empty else 0.0,
            "comparison_count": int(len(group)),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "model_a",
            "model_b",
            "mean_attention_divergence",
            "max_attention_divergence",
            "mean_attention_similarity",
            "comparison_count",
        ])

    return pd.DataFrame(rows).sort_values(
        ["mean_attention_divergence", "max_attention_divergence"], ascending=[False, False]
    ).reset_index(drop=True)


def compute_attention_consistency_summary(comparison_analysis: pd.DataFrame) -> Dict[str, Any]:
    """Summarise cross-model attention consistency."""
    if comparison_analysis is None or comparison_analysis.empty or "attention_divergence" not in comparison_analysis.columns:
        return {
            "total_comparisons": 0,
            "mean_attention_similarity": 0.0,
            "mean_attention_divergence": 0.0,
            "high_consistency_count": 0,
            "low_consistency_count": 0,
            "most_divergent_pair": None,
            "model_pair_rankings": [],
        }

    divergence = pd.to_numeric(comparison_analysis["attention_divergence"], errors="coerce").dropna()
    similarity = pd.to_numeric(comparison_analysis["attention_similarity"], errors="coerce").dropna() if "attention_similarity" in comparison_analysis.columns else pd.Series(dtype=float)
    ranking = compute_attention_divergence_ranking(comparison_analysis)

    most_divergent_pair = None
    if not ranking.empty:
        most_divergent_pair = {
            "model_a": str(ranking.iloc[0]["model_a"]),
            "model_b": str(ranking.iloc[0]["model_b"]),
            "mean_attention_divergence": round(float(ranking.iloc[0]["mean_attention_divergence"]), 6),
        }

    model_pair_rankings = ranking.to_dict(orient="records") if not ranking.empty else []

    return {
        "total_comparisons": int(len(divergence)),
        "mean_attention_similarity": round(float(similarity.mean()), 6) if not similarity.empty else 0.0,
        "mean_attention_divergence": round(float(divergence.mean()), 6) if not divergence.empty else 0.0,
        "high_consistency_count": int((comparison_analysis["attention_similarity"] >= 0.70).sum()) if "attention_similarity" in comparison_analysis.columns else 0,
        "low_consistency_count": int((comparison_analysis["attention_similarity"] < 0.40).sum()) if "attention_similarity" in comparison_analysis.columns else 0,
        "most_divergent_pair": most_divergent_pair,
        "model_pair_rankings": model_pair_rankings,
    }
