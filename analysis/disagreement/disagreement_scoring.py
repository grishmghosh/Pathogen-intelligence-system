"""
Disagreement Scoring and Export for Pathogen Intelligence System (Step 2).

Computes a normalised disagreement score for every pairwise model
comparison and provides CSV/JSON export for both confidence analysis
and severity analysis results.

Score formula (0-1 scale):
    score = class_weight * (class_factor + confidence_factor)

    class_factor:
        0.0 if same class, 0.5 if different class

    confidence_factor:
        Weighted combination of confidence gap and average confidence
        that captures how *confidently* models disagree.

    class_weight:
        0.3 for agreement, 1.0 for disagreement (amplifies class conflicts)

Architecture note:
    - Scoring is separated from severity classification.
    - Export is separated from computation.
    - The ``_make_serialisable`` helper is reused from ``disagreement_export``.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import shared serialisation helper from Step 1
from analysis.disagreement.disagreement_export import _make_serialisable


# ===========================================================================
# Scoring
# ===========================================================================

# Weight applied when classes agree (keeps score low for same-class pairs)
_AGREE_WEIGHT: float = 0.3

# Weight applied when classes disagree
_DISAGREE_WEIGHT: float = 1.0

# Relative importance of confidence gap vs average confidence in the
# confidence factor (must sum to 1.0)
_GAP_IMPORTANCE: float = 0.6
_AVG_IMPORTANCE: float = 0.4


def _score_single(
    classes_agree: bool,
    confidence_a: Optional[float],
    confidence_b: Optional[float],
) -> float:
    """
    Compute the disagreement score for a single pairwise comparison.

    Returns a value in [0.0, 1.0] where:
        0.0 = perfect agreement (same class, same confidence)
        1.0 = maximal disagreement (different class, both 100 % confident)

    When confidence values are unavailable the score is based solely
    on class agreement (0.0 or 0.5).
    """
    # --- Class factor ---
    class_factor = 0.0 if classes_agree else 0.5

    # --- Confidence factor ---
    if (confidence_a is not None and confidence_b is not None
            and np.isfinite(confidence_a) and np.isfinite(confidence_b)):
        gap = abs(confidence_a - confidence_b)          # 0-1
        avg = (confidence_a + confidence_b) / 2.0       # 0-1

        if classes_agree:
            # When classes agree, higher avg confidence is *good* -> lower score
            confidence_factor = _GAP_IMPORTANCE * gap + _AVG_IMPORTANCE * (1.0 - avg)
        else:
            # When classes disagree, higher avg confidence is *bad* -> higher score
            confidence_factor = _GAP_IMPORTANCE * gap + _AVG_IMPORTANCE * avg
    else:
        confidence_factor = 0.0

    # --- Combine ---
    weight = _AGREE_WEIGHT if classes_agree else _DISAGREE_WEIGHT
    raw = class_factor + 0.5 * confidence_factor   # max theoretical = 1.0
    score = weight * raw

    # Clamp to [0, 1]
    return float(np.clip(score, 0.0, 1.0))


def compute_disagreement_scores(pairwise: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``disagreement_score`` column to pairwise records.

    Args:
        pairwise: DataFrame from
            :func:`confidence_disagreement.compute_pairwise_confidence_analysis`
            (must include ``classes_agree``, ``confidence_a``, ``confidence_b``).

    Returns:
        Copy of *pairwise* with an added ``disagreement_score`` column
        (float in [0, 1]).
    """
    if pairwise.empty:
        out = pairwise.copy()
        out["disagreement_score"] = pd.Series(dtype=float)
        return out

    required = {"classes_agree", "confidence_a", "confidence_b"}
    missing = required - set(pairwise.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = pairwise.copy()
    scores: List[float] = []
    for _, row in result.iterrows():
        conf_a = float(row["confidence_a"]) if pd.notna(row["confidence_a"]) else None
        conf_b = float(row["confidence_b"]) if pd.notna(row["confidence_b"]) else None
        scores.append(_score_single(
            classes_agree=bool(row["classes_agree"]),
            confidence_a=conf_a,
            confidence_b=conf_b,
        ))

    result["disagreement_score"] = scores
    logger.info(
        "Scores computed: min=%.4f, max=%.4f, mean=%.4f",
        min(scores), max(scores), np.mean(scores),
    )
    return result


def compute_score_summary(scored: pd.DataFrame) -> Dict:
    """
    Aggregate scoring statistics.

    Args:
        scored: Output of :func:`compute_disagreement_scores`.

    Returns:
        Dictionary with mean/max/min/median score and counts.
    """
    if scored.empty or "disagreement_score" not in scored.columns:
        return {
            "mean_score": 0.0, "max_score": 0.0,
            "min_score": 0.0, "median_score": 0.0,
            "total_scored": 0,
        }

    s = scored["disagreement_score"]
    return {
        "mean_score": round(float(s.mean()), 6),
        "max_score": round(float(s.max()), 6),
        "min_score": round(float(s.min()), 6),
        "median_score": round(float(s.median()), 6),
        "total_scored": int(len(s)),
    }


# ===========================================================================
# Export - Confidence analysis results
# ===========================================================================

_CONFIDENCE_OUTPUT_DIR = Path("results") / "disagreement" / "confidence_analysis"
_SEVERITY_OUTPUT_DIR = Path("results") / "disagreement" / "severity_analysis"


def _ensure_dir(d: Optional[Union[str, Path]], default: Path) -> Path:
    out = Path(d) if d is not None else default
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---- Confidence exports ----

def export_confidence_gaps_csv(
    gaps: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "confidence_gaps.csv",
) -> Path:
    """Export per-sample confidence gaps as CSV."""
    out = _ensure_dir(output_dir, _CONFIDENCE_OUTPUT_DIR)
    fp = out / filename
    gaps.to_csv(fp, index=False, float_format="%.6f")
    logger.info("Confidence gaps (CSV) saved to %s", fp)
    return fp


def export_confidence_gaps_json(
    gaps: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "confidence_gaps.json",
) -> Path:
    """Export per-sample confidence gaps as JSON."""
    out = _ensure_dir(output_dir, _CONFIDENCE_OUTPUT_DIR)
    fp = out / filename
    records = gaps.to_dict(orient="records")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(records), fh, indent=2, ensure_ascii=False)
    logger.info("Confidence gaps (JSON) saved to %s", fp)
    return fp


def export_pairwise_analysis_csv(
    pairwise: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "pairwise_confidence.csv",
) -> Path:
    """Export pairwise confidence analysis as CSV."""
    out = _ensure_dir(output_dir, _CONFIDENCE_OUTPUT_DIR)
    fp = out / filename
    pairwise.to_csv(fp, index=False, float_format="%.6f")
    logger.info("Pairwise confidence (CSV) saved to %s", fp)
    return fp


def export_pairwise_analysis_json(
    pairwise: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "pairwise_confidence.json",
) -> Path:
    """Export pairwise confidence analysis as JSON."""
    out = _ensure_dir(output_dir, _CONFIDENCE_OUTPUT_DIR)
    fp = out / filename
    records = pairwise.to_dict(orient="records")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(records), fh, indent=2, ensure_ascii=False)
    logger.info("Pairwise confidence (JSON) saved to %s", fp)
    return fp


def export_confidence_summary_json(
    summary: Dict,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "confidence_spread_summary.json",
) -> Path:
    """Export confidence spread summary as JSON."""
    out = _ensure_dir(output_dir, _CONFIDENCE_OUTPUT_DIR)
    fp = out / filename
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(summary), fh, indent=2, ensure_ascii=False)
    logger.info("Confidence summary saved to %s", fp)
    return fp


# ---- Severity exports ----

def export_severity_csv(
    severity_df: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "severity_classifications.csv",
) -> Path:
    """Export pairwise severity classifications as CSV."""
    out = _ensure_dir(output_dir, _SEVERITY_OUTPUT_DIR)
    fp = out / filename
    severity_df.to_csv(fp, index=False, float_format="%.6f")
    logger.info("Severity classifications (CSV) saved to %s", fp)
    return fp


def export_severity_json(
    severity_df: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "severity_classifications.json",
) -> Path:
    """Export pairwise severity classifications as JSON (grouped by sample)."""
    out = _ensure_dir(output_dir, _SEVERITY_OUTPUT_DIR)
    fp = out / filename

    grouped: Dict = {}
    for sample_id, grp in severity_df.groupby("sample_id"):
        entries = []
        for _, row in grp.iterrows():
            entry: Dict = {
                "model_a": str(row.get("model_a", "")),
                "model_b": str(row.get("model_b", "")),
                "class_a": str(row.get("class_a", "")),
                "class_b": str(row.get("class_b", "")),
                "severity": str(row.get("severity", "")),
            }
            for opt in ("confidence_a", "confidence_b", "confidence_diff", "disagreement_score"):
                if opt in row and pd.notna(row[opt]):
                    entry[opt] = round(float(row[opt]), 6)
            entries.append(entry)
        grouped[str(sample_id)] = entries

    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(grouped, fh, indent=2, ensure_ascii=False)
    logger.info("Severity classifications (JSON) saved to %s", fp)
    return fp


def export_severity_summary_json(
    summary: Dict,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "severity_summary.json",
) -> Path:
    """Export severity summary statistics as JSON."""
    out = _ensure_dir(output_dir, _SEVERITY_OUTPUT_DIR)
    fp = out / filename
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(summary), fh, indent=2, ensure_ascii=False)
    logger.info("Severity summary saved to %s", fp)
    return fp


def export_score_summary_json(
    summary: Dict,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "score_summary.json",
) -> Path:
    """Export disagreement score summary as JSON."""
    out = _ensure_dir(output_dir, _CONFIDENCE_OUTPUT_DIR)
    fp = out / filename
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(summary), fh, indent=2, ensure_ascii=False)
    logger.info("Score summary saved to %s", fp)
    return fp
