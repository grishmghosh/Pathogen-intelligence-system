"""
Uncertainty Metrics for Pathogen Intelligence System (Step 6).

This module provides deterministic, rule-based probability handling and
entropy-based uncertainty metrics. It deliberately avoids probabilistic
inference and only normalises / summarises the probability vectors that are
already available from inference outputs.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _coerce_probability_array(probabilities: Any) -> Optional[np.ndarray]:
    """Convert common probability containers into a 1-D float array."""
    if probabilities is None:
        return None

    if isinstance(probabilities, pd.Series):
        values = probabilities.to_numpy()
    elif isinstance(probabilities, pd.DataFrame):
        values = probabilities.to_numpy().ravel()
    elif isinstance(probabilities, dict):
        values = list(probabilities.values())
    elif isinstance(probabilities, np.ndarray):
        values = probabilities
    elif isinstance(probabilities, (list, tuple)):
        values = probabilities
    else:
        return None

    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return array.astype(float, copy=False)


def normalize_probability_vector(
    probabilities: Any,
    fallback_size: int = 2,
    confidence: Optional[float] = None,
) -> Tuple[np.ndarray, str]:
    """
    Safely normalise a probability vector.

    Returns a tuple of ``(vector, status)`` where status is one of:
    ``valid``, ``normalized``, ``confidence_proxy``, or ``uniform_fallback``.
    """
    array = _coerce_probability_array(probabilities)

    if array is None or array.size == 0:
        if confidence is not None and pd.notna(confidence):
            conf = float(np.clip(float(confidence), 0.0, 1.0))
            vector = np.array([conf, 1.0 - conf], dtype=float)
            total = vector.sum()
            if total <= 0:
                return np.array([0.5, 0.5], dtype=float), "uniform_fallback"
            return vector / total, "confidence_proxy"

        size = max(int(fallback_size), 2)
        return np.full(size, 1.0 / size, dtype=float), "uniform_fallback"

    array = np.clip(array.astype(float, copy=False), 0.0, None)
    if array.size == 0:
        size = max(int(fallback_size), 2)
        return np.full(size, 1.0 / size, dtype=float), "uniform_fallback"

    total = float(np.sum(array))
    if not np.isfinite(total) or total <= 0.0:
        size = max(int(fallback_size), int(array.size), 2)
        return np.full(size, 1.0 / size, dtype=float), "uniform_fallback"

    normalized = array / total
    if abs(total - 1.0) <= 1e-6:
        return normalized, "valid"
    return normalized, "normalized"


def compute_prediction_entropy(
    probabilities: Any,
    confidence: Optional[float] = None,
    fallback_size: int = 2,
) -> Dict[str, Any]:
    """
    Compute entropy-based uncertainty metrics for a single probability vector.

    The resulting uncertainty score is normalised to [0, 1]:
        0   -> highly certain / peaked distribution
        1   -> maximally uncertain / flat distribution
    """
    vector, status = normalize_probability_vector(
        probabilities,
        fallback_size=fallback_size,
        confidence=confidence,
    )

    size = int(vector.size)
    if size <= 1:
        entropy = 0.0
        normalized_entropy = 0.0
    else:
        safe_vector = np.clip(vector, 1e-12, 1.0)
        entropy = float(-np.sum(safe_vector * np.log(safe_vector)))
        entropy = float(max(entropy, 0.0))
        normalized_entropy = float(entropy / np.log(size)) if size > 1 else 0.0
        normalized_entropy = float(np.clip(normalized_entropy, 0.0, 1.0))

    peak_probability = float(np.max(vector)) if size > 0 else 0.0
    probability_variance = float(np.var(vector)) if size > 0 else 0.0
    probability_std = float(np.std(vector)) if size > 0 else 0.0
    probability_spread = float(np.max(vector) - np.min(vector)) if size > 0 else 0.0

    return {
        "probability_count": size,
        "probability_sum": round(float(np.sum(vector)), 6),
        "normalization_status": status,
        "entropy": round(entropy, 6),
        "normalized_entropy": round(normalized_entropy, 6),
        "uncertainty_score": round(normalized_entropy, 6),
        "peak_probability": round(peak_probability, 6),
        "probability_variance": round(probability_variance, 6),
        "probability_std": round(probability_std, 6),
        "probability_spread": round(probability_spread, 6),
        "confidence_proxy_used": status == "confidence_proxy",
    }


def compute_entropy_metrics(
    data: Any,
    probability_column: str = "probabilities",
    confidence_column: str = "confidence",
    sample_column: str = "sample_id",
    model_column: str = "model_name",
    severity_column: str = "severity_level",
    perturbation_column: str = "perturbation_type",
    predicted_class_column: str = "predicted_class",
) -> pd.DataFrame:
    """
    Add entropy-based uncertainty metrics to a tabular prediction set.

    The function preserves existing columns and appends:
    ``entropy``, ``normalized_entropy``, ``uncertainty_score``,
    ``peak_probability``, ``probability_variance``, ``probability_std``,
    ``probability_spread``, ``probability_count``, and ``normalization_status``.
    """
    frame = pd.DataFrame(data).copy()
    if frame.empty:
        empty = frame.copy()
        for column in [
            "probability_count",
            "probability_sum",
            "normalization_status",
            "entropy",
            "normalized_entropy",
            "uncertainty_score",
            "peak_probability",
            "probability_variance",
            "probability_std",
            "probability_spread",
            "confidence_proxy_used",
        ]:
            empty[column] = pd.Series(dtype=float if column != "normalization_status" and column != "confidence_proxy_used" else object)
        return empty

    records = []
    for _, row in frame.iterrows():
        probability_value = row.get(probability_column) if probability_column in frame.columns else None
        confidence_value = row.get(confidence_column) if confidence_column in frame.columns else None
        metrics = compute_prediction_entropy(
            probability_value,
            confidence=float(confidence_value) if pd.notna(confidence_value) else None,
        )

        record = row.to_dict()
        record.update(metrics)
        record[sample_column] = record.get(sample_column)
        record[model_column] = record.get(model_column)
        record[severity_column] = record.get(severity_column)
        record[perturbation_column] = record.get(perturbation_column)
        record[predicted_class_column] = record.get(predicted_class_column)
        records.append(record)

    result = pd.DataFrame(records)
    if sample_column in result.columns:
        result[sample_column] = result[sample_column].astype(str)
    if model_column in result.columns:
        result[model_column] = result[model_column].astype(str)
    return result


def compute_entropy_summary(entropy_metrics: pd.DataFrame) -> Dict[str, Any]:
    """Summarise entropy-based uncertainty metrics."""
    if entropy_metrics is None or entropy_metrics.empty or "uncertainty_score" not in entropy_metrics.columns:
        return {
            "total_observations": 0,
            "mean_entropy": 0.0,
            "mean_normalized_entropy": 0.0,
            "max_uncertainty": 0.0,
            "min_uncertainty": 0.0,
            "high_uncertainty_count": 0,
            "critical_uncertainty_count": 0,
        }

    scores = pd.to_numeric(entropy_metrics["uncertainty_score"], errors="coerce").dropna()
    entropies = pd.to_numeric(entropy_metrics["entropy"], errors="coerce").dropna() if "entropy" in entropy_metrics.columns else scores
    normalized = pd.to_numeric(entropy_metrics["normalized_entropy"], errors="coerce").dropna() if "normalized_entropy" in entropy_metrics.columns else scores

    if scores.empty:
        return {
            "total_observations": 0,
            "mean_entropy": 0.0,
            "mean_normalized_entropy": 0.0,
            "max_uncertainty": 0.0,
            "min_uncertainty": 0.0,
            "high_uncertainty_count": 0,
            "critical_uncertainty_count": 0,
        }

    return {
        "total_observations": int(len(scores)),
        "mean_entropy": round(float(entropies.mean()), 6) if not entropies.empty else 0.0,
        "mean_normalized_entropy": round(float(normalized.mean()), 6) if not normalized.empty else round(float(scores.mean()), 6),
        "max_uncertainty": round(float(scores.max()), 6),
        "min_uncertainty": round(float(scores.min()), 6),
        "high_uncertainty_count": int((scores >= 0.60).sum()),
        "critical_uncertainty_count": int((scores >= 0.80).sum()),
    }
