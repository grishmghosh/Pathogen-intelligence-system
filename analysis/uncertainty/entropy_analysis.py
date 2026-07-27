"""
Entropy Analysis for Pathogen Intelligence System (Step 6).

This module builds on the deterministic uncertainty metrics and provides
analysis views for entropy growth, perturbation sensitivity, and confidence
collapse detection. It intentionally keeps scoring separate from any later
classification layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.uncertainty.uncertainty_metrics import compute_entropy_metrics


_SEVERITY_RANKS: Dict[str, int] = {
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
    label = str(value).strip().lower()
    return _SEVERITY_RANKS.get(label, 99)


def _sort_severity_labels(labels: Sequence[str]) -> List[str]:
    return sorted([str(label) for label in labels], key=lambda item: (_severity_rank(item), item))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        return default
    return float(numeric)


def compute_entropy_analysis(
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
    Compute row-level entropy uncertainty analysis.

    The returned DataFrame is a direct extension of the input predictions
    with deterministic entropy metrics added.
    """
    return compute_entropy_metrics(
        data,
        probability_column=probability_column,
        confidence_column=confidence_column,
        sample_column=sample_column,
        model_column=model_column,
        severity_column=severity_column,
        perturbation_column=perturbation_column,
        predicted_class_column=predicted_class_column,
    )


def aggregate_sample_uncertainty(
    entropy_analysis: pd.DataFrame,
    sample_column: str = "sample_id",
) -> pd.DataFrame:
    """Aggregate entropy uncertainty to the sample level."""
    if entropy_analysis is None or entropy_analysis.empty or sample_column not in entropy_analysis.columns:
        return pd.DataFrame(columns=[
            sample_column,
            "mean_uncertainty_score",
            "max_uncertainty_score",
            "min_uncertainty_score",
            "mean_entropy",
            "mean_confidence",
            "observation_count",
        ])

    records: List[Dict[str, Any]] = []
    for sample_id, group in entropy_analysis.groupby(sample_column):
        uncertainty = pd.to_numeric(group["uncertainty_score"], errors="coerce").dropna()
        entropy = pd.to_numeric(group["normalized_entropy"], errors="coerce").dropna() if "normalized_entropy" in group.columns else uncertainty
        confidence = pd.to_numeric(group["confidence"], errors="coerce").dropna() if "confidence" in group.columns else pd.Series(dtype=float)

        if uncertainty.empty:
            continue

        records.append({
            sample_column: str(sample_id),
            "mean_uncertainty_score": round(float(uncertainty.mean()), 6),
            "max_uncertainty_score": round(float(uncertainty.max()), 6),
            "min_uncertainty_score": round(float(uncertainty.min()), 6),
            "mean_entropy": round(float(entropy.mean()), 6) if not entropy.empty else round(float(uncertainty.mean()), 6),
            "mean_confidence": round(float(confidence.mean()), 6) if not confidence.empty else None,
            "observation_count": int(len(uncertainty)),
        })

    if not records:
        return pd.DataFrame(columns=[
            sample_column,
            "mean_uncertainty_score",
            "max_uncertainty_score",
            "min_uncertainty_score",
            "mean_entropy",
            "mean_confidence",
            "observation_count",
        ])

    return pd.DataFrame(records).sort_values(
        ["mean_uncertainty_score", sample_column], ascending=[False, True]
    ).reset_index(drop=True)


def compute_uncertainty_severity_curve(
    entropy_analysis: pd.DataFrame,
    severity_column: str = "severity_level",
) -> pd.DataFrame:
    """Measure how uncertainty changes across perturbation severity."""
    if entropy_analysis is None or entropy_analysis.empty or severity_column not in entropy_analysis.columns:
        return pd.DataFrame(columns=[
            severity_column,
            "severity_rank",
            "total_observations",
            "mean_uncertainty_score",
            "max_uncertainty_score",
            "min_uncertainty_score",
            "mean_entropy",
        ])

    rows: List[Dict[str, Any]] = []
    for severity_label, group in entropy_analysis.groupby(severity_column):
        uncertainty = pd.to_numeric(group["uncertainty_score"], errors="coerce").dropna()
        if uncertainty.empty:
            continue
        entropy = pd.to_numeric(group["normalized_entropy"], errors="coerce").dropna() if "normalized_entropy" in group.columns else uncertainty
        rows.append({
            severity_column: str(severity_label),
            "severity_rank": _severity_rank(severity_label),
            "total_observations": int(len(uncertainty)),
            "mean_uncertainty_score": round(float(uncertainty.mean()), 6),
            "max_uncertainty_score": round(float(uncertainty.max()), 6),
            "min_uncertainty_score": round(float(uncertainty.min()), 6),
            "mean_entropy": round(float(entropy.mean()), 6) if not entropy.empty else round(float(uncertainty.mean()), 6),
        })

    if not rows:
        return pd.DataFrame(columns=[
            severity_column,
            "severity_rank",
            "total_observations",
            "mean_uncertainty_score",
            "max_uncertainty_score",
            "min_uncertainty_score",
            "mean_entropy",
        ])

    return pd.DataFrame(rows).sort_values("severity_rank").reset_index(drop=True)


def rank_perturbation_uncertainty(
    entropy_analysis: pd.DataFrame,
    perturbation_column: str = "perturbation_type",
) -> pd.DataFrame:
    """Rank perturbation types by uncertainty impact."""
    if entropy_analysis is None or entropy_analysis.empty or perturbation_column not in entropy_analysis.columns:
        return pd.DataFrame(columns=[
            perturbation_column,
            "total_observations",
            "mean_uncertainty_score",
            "max_uncertainty_score",
            "min_uncertainty_score",
            "uncertainty_spread",
        ])

    rows: List[Dict[str, Any]] = []
    for perturbation_type, group in entropy_analysis.groupby(perturbation_column):
        uncertainty = pd.to_numeric(group["uncertainty_score"], errors="coerce").dropna()
        if uncertainty.empty:
            continue
        rows.append({
            perturbation_column: str(perturbation_type),
            "total_observations": int(len(uncertainty)),
            "mean_uncertainty_score": round(float(uncertainty.mean()), 6),
            "max_uncertainty_score": round(float(uncertainty.max()), 6),
            "min_uncertainty_score": round(float(uncertainty.min()), 6),
            "uncertainty_spread": round(float(uncertainty.max() - uncertainty.min()), 6),
        })

    if not rows:
        return pd.DataFrame(columns=[
            perturbation_column,
            "total_observations",
            "mean_uncertainty_score",
            "max_uncertainty_score",
            "min_uncertainty_score",
            "uncertainty_spread",
        ])

    return pd.DataFrame(rows).sort_values(
        ["mean_uncertainty_score", perturbation_column], ascending=[False, True]
    ).reset_index(drop=True)


def detect_confidence_collapse(
    entropy_analysis: pd.DataFrame,
    sample_column: str = "sample_id",
    model_column: str = "model_name",
    severity_column: str = "severity_level",
    perturbation_column: str = "perturbation_type",
    entropy_column: str = "normalized_entropy",
    confidence_column: str = "confidence",
    entropy_shift_threshold: float = 0.25,
    confidence_drop_threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Detect rapid confidence degradation and entropy spikes across perturbations.

    The first row that crosses the entropy or confidence threshold is
    reported as the collapse trigger.
    """
    if entropy_analysis is None or entropy_analysis.empty:
        return pd.DataFrame(columns=[
            sample_column,
            model_column,
            "collapse_trigger",
            "collapse_severity",
            "collapse_severity_rank",
            "baseline_entropy",
            "collapse_entropy",
            "entropy_shift",
            "baseline_confidence",
            "collapse_confidence",
            "confidence_drop",
            "collapse_reason",
        ])

    if sample_column not in entropy_analysis.columns:
        return pd.DataFrame(columns=[
            sample_column,
            model_column,
            "collapse_trigger",
            "collapse_severity",
            "collapse_severity_rank",
            "baseline_entropy",
            "collapse_entropy",
            "entropy_shift",
            "baseline_confidence",
            "collapse_confidence",
            "confidence_drop",
            "collapse_reason",
        ])

    grouping_columns = [sample_column]
    if model_column in entropy_analysis.columns:
        grouping_columns.append(model_column)

    records: List[Dict[str, Any]] = []
    for key_values, group in entropy_analysis.groupby(grouping_columns):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        key_map = dict(zip(grouping_columns, [str(value) for value in key_values]))

        ordered = group.copy()
        if severity_column in ordered.columns:
            ordered = ordered.assign(
                _severity_rank=ordered[severity_column].apply(_severity_rank)
            ).sort_values(["_severity_rank", severity_column], ascending=[True, True])
        else:
            ordered = ordered.reset_index(drop=True)
            ordered["_severity_rank"] = range(len(ordered))

        if ordered.empty:
            continue

        baseline_row = None
        if severity_column in ordered.columns:
            clean_rows = ordered[ordered[severity_column].astype(str).str.lower().isin({"clean", "none", "original", "baseline"})]
            if not clean_rows.empty:
                baseline_row = clean_rows.iloc[0]
        if baseline_row is None:
            baseline_row = ordered.iloc[0]

        baseline_entropy = _safe_float(baseline_row.get(entropy_column, 0.0), 0.0)
        baseline_confidence = _safe_float(baseline_row.get(confidence_column, 0.0), 0.0)

        collapse_row = None
        collapse_reason = None
        for _, row in ordered.iterrows():
            entropy_value = _safe_float(row.get(entropy_column, baseline_entropy), baseline_entropy)
            confidence_value = _safe_float(row.get(confidence_column, baseline_confidence), baseline_confidence)
            entropy_shift = entropy_value - baseline_entropy
            confidence_drop = baseline_confidence - confidence_value

            reasons = []
            if entropy_shift >= entropy_shift_threshold:
                reasons.append("entropy_spike")
            if confidence_drop >= confidence_drop_threshold:
                reasons.append("confidence_degradation")

            if reasons:
                collapse_row = row
                collapse_reason = "|".join(reasons)
                break

        if collapse_row is None:
            continue

        entropy_value = _safe_float(collapse_row.get(entropy_column, baseline_entropy), baseline_entropy)
        confidence_value = _safe_float(collapse_row.get(confidence_column, baseline_confidence), baseline_confidence)
        collapse_severity = str(collapse_row.get(severity_column, "unknown")) if severity_column in collapse_row else "unknown"
        collapse_trigger = str(collapse_row.get(perturbation_column, collapse_severity)) if perturbation_column in collapse_row else collapse_severity

        records.append({
            sample_column: key_map.get(sample_column, ""),
            model_column: key_map.get(model_column, None),
            "collapse_trigger": collapse_trigger,
            "collapse_severity": collapse_severity,
            "collapse_severity_rank": int(collapse_row.get("_severity_rank", _severity_rank(collapse_severity))),
            "baseline_entropy": round(baseline_entropy, 6),
            "collapse_entropy": round(entropy_value, 6),
            "entropy_shift": round(entropy_value - baseline_entropy, 6),
            "baseline_confidence": round(baseline_confidence, 6),
            "collapse_confidence": round(confidence_value, 6),
            "confidence_drop": round(baseline_confidence - confidence_value, 6),
            "collapse_reason": collapse_reason,
        })

    if not records:
        return pd.DataFrame(columns=[
            sample_column,
            model_column,
            "collapse_trigger",
            "collapse_severity",
            "collapse_severity_rank",
            "baseline_entropy",
            "collapse_entropy",
            "entropy_shift",
            "baseline_confidence",
            "collapse_confidence",
            "confidence_drop",
            "collapse_reason",
        ])

    result = pd.DataFrame(records)
    sort_columns = ["entropy_shift", "confidence_drop"]
    return result.sort_values(sort_columns, ascending=[False, False]).reset_index(drop=True)
