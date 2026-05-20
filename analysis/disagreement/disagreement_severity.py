"""
Disagreement Severity Classification for Pathogen Intelligence System.

Provides deterministic, rule-based severity classification for each
disagreement event based on predicted classes and confidence signals.

Severity levels:
    low      - same class, small confidence variation
    moderate - different classes, at least one model is uncertain
    high     - different classes, both models moderately confident
    critical - different classes, both models highly confident

Architecture note:
    This module is pure classification logic.  It receives enriched
    disagreement records (from ``confidence_disagreement``) and labels
    each row with a severity tier.  Scoring is handled separately in
    ``disagreement_scoring``.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity thresholds (deterministic, tunable)
# ---------------------------------------------------------------------------

# Confidence below this is considered "uncertain"
UNCERTAIN_THRESHOLD: float = 0.50

# Confidence above this is considered "highly confident"
HIGH_CONFIDENCE_THRESHOLD: float = 0.80

# Confidence gap below this is considered "small"
SMALL_GAP_THRESHOLD: float = 0.10


def _classify_single(
    classes_agree: bool,
    confidence_a: Optional[float],
    confidence_b: Optional[float],
    confidence_gap: Optional[float],
) -> str:
    """
    Classify a single pairwise disagreement event.

    Decision tree (evaluated top-to-bottom):

    1. If classes agree:
       - gap <= SMALL_GAP_THRESHOLD  -> "low"
       - gap >  SMALL_GAP_THRESHOLD  -> "low"   (same class is never severe)

    2. If classes disagree:
       a. Both confident (>= HIGH_CONFIDENCE)         -> "critical"
       b. Both above UNCERTAIN but not both high      -> "high"
       c. At least one uncertain (< UNCERTAIN)         -> "moderate"
       d. Confidence data missing                      -> "moderate" (default)

    Returns:
        One of ``"low"``, ``"moderate"``, ``"high"``, ``"critical"``.
    """
    # --- Classes agree ---
    if classes_agree:
        return "low"

    # --- Classes disagree ---
    # If confidence is missing, default to moderate
    if confidence_a is None or confidence_b is None:
        return "moderate"

    if not np.isfinite(confidence_a) or not np.isfinite(confidence_b):
        return "moderate"

    both_high = (confidence_a >= HIGH_CONFIDENCE_THRESHOLD
                 and confidence_b >= HIGH_CONFIDENCE_THRESHOLD)
    if both_high:
        return "critical"

    both_above_uncertain = (confidence_a >= UNCERTAIN_THRESHOLD
                            and confidence_b >= UNCERTAIN_THRESHOLD)
    if both_above_uncertain:
        return "high"

    # At least one model is uncertain
    return "moderate"


# ---------------------------------------------------------------------------
# Public API - classify pairwise records
# ---------------------------------------------------------------------------

def classify_pairwise_severity(
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a ``severity`` column to a pairwise confidence analysis DataFrame.

    Args:
        pairwise: Output of
            :func:`confidence_disagreement.compute_pairwise_confidence_analysis`.
            Expected columns: ``classes_agree``, ``confidence_a``,
            ``confidence_b``, ``confidence_diff``.

    Returns:
        A copy of *pairwise* with an added ``severity`` column
        (values: ``"low"`` | ``"moderate"`` | ``"high"`` | ``"critical"``).

    Raises:
        ValueError: If input is empty or missing required columns.
    """
    if pairwise.empty:
        out = pairwise.copy()
        out["severity"] = pd.Series(dtype=str)
        return out

    required = {"classes_agree", "confidence_a", "confidence_b"}
    missing = required - set(pairwise.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = pairwise.copy()
    gap_col = "confidence_diff" if "confidence_diff" in result.columns else None

    severities: List[str] = []
    for _, row in result.iterrows():
        gap = float(row[gap_col]) if gap_col and pd.notna(row[gap_col]) else None
        sev = _classify_single(
            classes_agree=bool(row["classes_agree"]),
            confidence_a=float(row["confidence_a"]) if pd.notna(row["confidence_a"]) else None,
            confidence_b=float(row["confidence_b"]) if pd.notna(row["confidence_b"]) else None,
            confidence_gap=gap,
        )
        severities.append(sev)

    result["severity"] = severities
    logger.info(
        "Severity distribution: %s",
        result["severity"].value_counts().to_dict(),
    )
    return result


# ---------------------------------------------------------------------------
# Public API - per-sample severity (multi-model roll-up)
# ---------------------------------------------------------------------------

def classify_sample_severity(
    pairwise_with_severity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Roll up pairwise severities to a single per-sample severity.

    When a sample has multiple model pairs, the *worst* (highest)
    severity across all pairs is taken as the sample-level severity.

    Ordering: ``low < moderate < high < critical``.

    Args:
        pairwise_with_severity: Output of :func:`classify_pairwise_severity`.

    Returns:
        DataFrame with columns ``sample_id``, ``severity``,
        ``n_pairs``, ``n_critical``, ``n_high``, ``n_moderate``, ``n_low``.
    """
    _SEVERITY_ORDER = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
    _REVERSE = {v: k for k, v in _SEVERITY_ORDER.items()}

    if pairwise_with_severity.empty:
        return pd.DataFrame(columns=[
            "sample_id", "severity", "n_pairs",
            "n_critical", "n_high", "n_moderate", "n_low",
        ])

    records: List[Dict] = []
    for sample_id, grp in pairwise_with_severity.groupby("sample_id"):
        counts = grp["severity"].value_counts().to_dict()
        worst_rank = max(_SEVERITY_ORDER.get(s, 0) for s in grp["severity"])
        records.append({
            "sample_id": str(sample_id),
            "severity": _REVERSE[worst_rank],
            "n_pairs": len(grp),
            "n_critical": counts.get("critical", 0),
            "n_high": counts.get("high", 0),
            "n_moderate": counts.get("moderate", 0),
            "n_low": counts.get("low", 0),
        })

    result = pd.DataFrame(records)
    result = result.sort_values("sample_id").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Public API - severity summary statistics
# ---------------------------------------------------------------------------

def compute_severity_summary(
    pairwise_with_severity: pd.DataFrame,
) -> Dict:
    """
    Compute summary statistics from severity-classified pairwise records.

    Args:
        pairwise_with_severity: Output of :func:`classify_pairwise_severity`.

    Returns:
        Dictionary with:

        * ``total_pairs``          - total pairwise records
        * ``severity_counts``      - dict mapping severity -> count
        * ``severity_rates``       - dict mapping severity -> proportion (0-1)
        * ``critical_sample_ids``  - list of sample_ids with critical severity
        * ``high_sample_ids``      - list of sample_ids with high severity
    """
    if pairwise_with_severity.empty:
        return {
            "total_pairs": 0,
            "severity_counts": {"low": 0, "moderate": 0, "high": 0, "critical": 0},
            "severity_rates": {"low": 0.0, "moderate": 0.0, "high": 0.0, "critical": 0.0},
            "critical_sample_ids": [],
            "high_sample_ids": [],
        }

    total = len(pairwise_with_severity)
    counts = pairwise_with_severity["severity"].value_counts().to_dict()

    # Ensure all keys present
    for key in ("low", "moderate", "high", "critical"):
        counts.setdefault(key, 0)

    rates = {k: round(v / total, 6) for k, v in counts.items()}

    critical_ids = sorted(
        pairwise_with_severity.loc[
            pairwise_with_severity["severity"] == "critical", "sample_id"
        ].unique().tolist()
    )
    high_ids = sorted(
        pairwise_with_severity.loc[
            pairwise_with_severity["severity"] == "high", "sample_id"
        ].unique().tolist()
    )

    return {
        "total_pairs": total,
        "severity_counts": counts,
        "severity_rates": rates,
        "critical_sample_ids": critical_ids,
        "high_sample_ids": high_ids,
    }
