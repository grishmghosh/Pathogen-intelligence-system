"""
Confidence-Aware Disagreement Detection for Pathogen Intelligence System.

Extends Step 1 disagreement detection with confidence gap analysis.
For each disagreement sample this module computes per-pair confidence
differences, average confidence spreads, and maximum disagreement spreads.

Depends on:
    analysis.disagreement.disagreement_utils.load_predictions
    analysis.disagreement.disagreement_utils.detect_disagreements

Architecture Flow:
    Step 1 Disagreements -> Confidence Gap Analysis -> Enriched Records
"""

import itertools
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_confidence_available(df: pd.DataFrame) -> bool:
    """Return True if at least some confidence values are present."""
    if "confidence" not in df.columns:
        return False
    return df["confidence"].notna().any()


def _safe_confidence(val) -> Optional[float]:
    """Coerce a value to float or return None."""
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Public API - Per-sample confidence gap
# ---------------------------------------------------------------------------

def compute_confidence_gaps(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-sample confidence statistics across all models.

    For every ``sample_id`` that has predictions from >= 2 models this
    function computes:

    * ``confidence_min``  - lowest confidence across models
    * ``confidence_max``  - highest confidence across models
    * ``confidence_mean`` - average confidence across models
    * ``confidence_gap``  - max - min  (the *spread*)
    * ``n_models``        - number of models with predictions for this sample
    * ``classes_agree``   - boolean, True when all models predict the same class

    Samples with fewer than 2 model predictions are excluded.

    Args:
        predictions: Normalised DataFrame (output of ``load_predictions``).

    Returns:
        DataFrame indexed by ``sample_id`` with the columns above.

    Raises:
        ValueError: If predictions is empty or confidence is entirely absent.
    """
    if predictions.empty:
        raise ValueError("Predictions DataFrame is empty.")

    has_conf = _validate_confidence_available(predictions)
    if not has_conf:
        raise ValueError(
            "No valid confidence values found. "
            "Confidence-aware analysis requires at least some numeric confidence scores."
        )

    # Group by sample
    groups = predictions.groupby("sample_id")

    records: List[Dict] = []
    for sample_id, grp in groups:
        if len(grp) < 2:
            continue  # need at least 2 models

        confs = grp["confidence"].dropna()
        if confs.empty:
            continue

        classes = grp["predicted_class"].unique()

        records.append({
            "sample_id": sample_id,
            "confidence_min": float(confs.min()),
            "confidence_max": float(confs.max()),
            "confidence_mean": float(confs.mean()),
            "confidence_gap": float(confs.max() - confs.min()),
            "n_models": int(len(grp["model_name"].unique())),
            "classes_agree": len(classes) == 1,
        })

    if not records:
        logger.warning("No samples with >= 2 models and valid confidence found.")
        return pd.DataFrame(columns=[
            "sample_id", "confidence_min", "confidence_max",
            "confidence_mean", "confidence_gap", "n_models", "classes_agree",
        ])

    result = pd.DataFrame(records)
    result = result.sort_values("sample_id").reset_index(drop=True)
    logger.info("Confidence gaps computed for %d samples.", len(result))
    return result


# ---------------------------------------------------------------------------
# Public API - Pairwise confidence analysis
# ---------------------------------------------------------------------------

def compute_pairwise_confidence_analysis(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute pairwise model confidence differences for every sample.

    For every unique pair of models (A, B) and every sample that has
    predictions from both, one record is emitted containing:

    * ``sample_id``
    * ``model_a``, ``model_b``
    * ``class_a``, ``class_b``
    * ``confidence_a``, ``confidence_b``
    * ``confidence_diff``    - abs(confidence_a - confidence_b)
    * ``classes_agree``      - boolean

    Args:
        predictions: Normalised predictions DataFrame.

    Returns:
        DataFrame with one row per (sample, model_pair).

    Raises:
        ValueError: If fewer than 2 models or empty input.
    """
    if predictions.empty:
        raise ValueError("Predictions DataFrame is empty.")

    model_names = sorted(predictions["model_name"].unique())
    if len(model_names) < 2:
        raise ValueError(
            f"At least 2 models required; found {len(model_names)}: {model_names}"
        )

    # Pivot predicted_class and confidence
    cls_pivot = predictions.pivot_table(
        index="sample_id", columns="model_name",
        values="predicted_class", aggfunc="first",
    )
    conf_pivot = predictions.pivot_table(
        index="sample_id", columns="model_name",
        values="confidence", aggfunc="first",
    )

    records: List[Dict] = []
    for model_a, model_b in itertools.combinations(model_names, 2):
        if model_a not in cls_pivot.columns or model_b not in cls_pivot.columns:
            continue

        # Restrict to samples present in both
        mask = (
            cls_pivot[[model_a, model_b]].notna().all(axis=1)
            & conf_pivot[[model_a, model_b]].notna().all(axis=1)
        )
        for sample_id in cls_pivot.loc[mask].index:
            ca = str(cls_pivot.loc[sample_id, model_a])
            cb = str(cls_pivot.loc[sample_id, model_b])
            conf_a = _safe_confidence(conf_pivot.loc[sample_id, model_a])
            conf_b = _safe_confidence(conf_pivot.loc[sample_id, model_b])

            if conf_a is None or conf_b is None:
                continue

            records.append({
                "sample_id": str(sample_id),
                "model_a": model_a,
                "model_b": model_b,
                "class_a": ca,
                "class_b": cb,
                "confidence_a": conf_a,
                "confidence_b": conf_b,
                "confidence_diff": abs(conf_a - conf_b),
                "classes_agree": ca == cb,
            })

    if not records:
        logger.warning("No pairwise records could be computed.")
        return pd.DataFrame(columns=[
            "sample_id", "model_a", "model_b", "class_a", "class_b",
            "confidence_a", "confidence_b", "confidence_diff", "classes_agree",
        ])

    result = pd.DataFrame(records)
    result = result.sort_values(["sample_id", "model_a", "model_b"]).reset_index(drop=True)
    logger.info("Pairwise confidence analysis: %d records.", len(result))
    return result


# ---------------------------------------------------------------------------
# Public API - Confidence spread summary
# ---------------------------------------------------------------------------

def compute_confidence_spread_summary(
    pairwise: pd.DataFrame,
) -> Dict:
    """
    Aggregate summary statistics from pairwise confidence analysis.

    Args:
        pairwise: Output of :func:`compute_pairwise_confidence_analysis`.

    Returns:
        Dictionary with:

        * ``mean_confidence_diff``    - average absolute confidence difference
        * ``max_confidence_diff``     - largest confidence difference observed
        * ``median_confidence_diff``  - median confidence difference
        * ``mean_confidence_diff_agree``    - mean diff when classes agree
        * ``mean_confidence_diff_disagree`` - mean diff when classes disagree
        * ``total_pairs``             - number of pairwise records
        * ``agree_pairs``             - pairs where classes agree
        * ``disagree_pairs``          - pairs where classes disagree
    """
    if pairwise.empty:
        return {
            "mean_confidence_diff": 0.0,
            "max_confidence_diff": 0.0,
            "median_confidence_diff": 0.0,
            "mean_confidence_diff_agree": 0.0,
            "mean_confidence_diff_disagree": 0.0,
            "total_pairs": 0,
            "agree_pairs": 0,
            "disagree_pairs": 0,
        }

    diffs = pairwise["confidence_diff"]
    agree_mask = pairwise["classes_agree"]

    agree_diffs = diffs[agree_mask]
    disagree_diffs = diffs[~agree_mask]

    return {
        "mean_confidence_diff": round(float(diffs.mean()), 6),
        "max_confidence_diff": round(float(diffs.max()), 6),
        "median_confidence_diff": round(float(diffs.median()), 6),
        "mean_confidence_diff_agree": round(float(agree_diffs.mean()), 6) if not agree_diffs.empty else 0.0,
        "mean_confidence_diff_disagree": round(float(disagree_diffs.mean()), 6) if not disagree_diffs.empty else 0.0,
        "total_pairs": int(len(pairwise)),
        "agree_pairs": int(agree_mask.sum()),
        "disagree_pairs": int((~agree_mask).sum()),
    }
