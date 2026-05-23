"""
Risk Classification for Pathogen Intelligence System (Step 5).

This module maps the deterministic reliability risk score produced by
``risk_estimation`` into clear, rule-based labels. Classification is kept
separate from scoring so the thresholds can evolve independently.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


_RISK_LEVELS: List[str] = ["minimal", "low", "moderate", "high", "critical"]

_RISK_THRESHOLDS: Dict[str, float] = {
    "minimal": 0.15,
    "low": 0.35,
    "moderate": 0.60,
    "high": 0.80,
}

_RISK_LEVEL_RANKS: Dict[str, int] = {
    "minimal": 1,
    "low": 2,
    "moderate": 3,
    "high": 4,
    "critical": 5,
}


def _clamp01(value: object, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if not np.isfinite(numeric):
        numeric = default
    return float(np.clip(numeric, 0.0, 1.0))


def classify_reliability_risk_level(risk_score: float) -> str:
    """
    Map a normalized reliability risk score to a label.

    Labels are ordered from best to worst:
        minimal, low, moderate, high, critical
    """
    score = _clamp01(risk_score)
    if score >= _RISK_THRESHOLDS["high"]:
        return "critical"
    if score >= _RISK_THRESHOLDS["moderate"]:
        return "high"
    if score >= _RISK_THRESHOLDS["low"]:
        return "moderate"
    if score >= _RISK_THRESHOLDS["minimal"]:
        return "low"
    return "minimal"


def assign_reliability_risk_labels(
    risk_scores: pd.DataFrame,
    risk_column: str = "reliability_risk_score",
) -> pd.DataFrame:
    """
    Add deterministic risk labels to a sample- or model-level risk table.

    The function preserves the original DataFrame and adds:
        * ``risk_level``
        * ``risk_level_rank``
    """
    if risk_scores is None or risk_scores.empty:
        result = risk_scores.copy() if risk_scores is not None else pd.DataFrame()
        result["risk_level"] = pd.Series(dtype=str)
        result["risk_level_rank"] = pd.Series(dtype=int)
        return result

    if risk_column not in risk_scores.columns:
        result = risk_scores.copy()
        result["risk_level"] = pd.Series(dtype=str)
        result["risk_level_rank"] = pd.Series(dtype=int)
        return result

    result = risk_scores.copy()
    result[risk_column] = pd.to_numeric(result[risk_column], errors="coerce")

    labels: List[str] = []
    ranks: List[int] = []
    for _, row in result.iterrows():
        label = classify_reliability_risk_level(row.get(risk_column))
        labels.append(label)
        ranks.append(_RISK_LEVEL_RANKS[label])

    result["risk_level"] = labels
    result["risk_level_rank"] = ranks
    return result


def compute_risk_classification_summary(
    labelled_risk: pd.DataFrame,
) -> Dict:
    """
    Summarise the distribution of risk labels.
    """
    if labelled_risk is None or labelled_risk.empty or "risk_level" not in labelled_risk.columns:
        return {
            "total_records": 0,
            "risk_distribution": {level: 0 for level in _RISK_LEVELS},
            "risk_rates": {level: 0.0 for level in _RISK_LEVELS},
            "most_common_risk_level": None,
        }

    counts = labelled_risk["risk_level"].astype(str).value_counts().to_dict()
    for level in _RISK_LEVELS:
        counts.setdefault(level, 0)
    total = int(sum(counts.values()))
    rates = {level: round(counts[level] / total, 6) if total else 0.0 for level in _RISK_LEVELS}
    most_common = None
    if total:
        most_common = max(_RISK_LEVELS, key=lambda level: counts.get(level, 0))

    return {
        "total_records": total,
        "risk_distribution": counts,
        "risk_rates": rates,
        "most_common_risk_level": most_common,
    }
