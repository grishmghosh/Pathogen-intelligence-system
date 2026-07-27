"""
Trust Analysis for Pathogen Intelligence System (Step 4).

Generates structured trust labels and provides CSV/JSON export for all
Step 4 consensus reliability analysis results.

Trust levels:
    very_high  – highly stable agreement
    high       – strong reliable consensus
    moderate   – partially stable
    low        – unstable agreement
    critical   – unreliable consensus

Architecture note:
    - Trust labelling is separated from reliability scoring
      (``consensus_reliability``) and false consensus detection
      (``false_consensus_detection``).
    - Export is co-located with trust labelling to keep Step 4's
      export surface in one place.
    - All logic is deterministic and rule-based.

Depends on:
    analysis.disagreement.consensus_reliability       (Step 4)
    analysis.disagreement.false_consensus_detection   (Step 4)
    analysis.disagreement.disagreement_export         (Step 1 – serialisation)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Reuse Step 1's serialisation helper
from analysis.disagreement.disagreement_export import _make_serialisable


# ---------------------------------------------------------------------------
# Trust level thresholds (deterministic, tunable)
# ---------------------------------------------------------------------------

# Consensus reliability score thresholds for trust labels
_TRUST_THRESHOLDS = {
    "very_high": 0.85,   # score >= 0.85
    "high": 0.70,        # score >= 0.70
    "moderate": 0.45,    # score >= 0.45
    "low": 0.25,         # score >= 0.25
    # Below 0.25 → "critical"
}


# ---------------------------------------------------------------------------
# Public API – trust classification
# ---------------------------------------------------------------------------

def classify_trust_level(reliability_score: float) -> str:
    """
    Map a consensus reliability score to a trust label.

    Args:
        reliability_score: Score in [0, 1] from
            :func:`consensus_reliability.compute_consensus_reliability`.

    Returns:
        One of ``"very_high"``, ``"high"``, ``"moderate"``,
        ``"low"``, ``"critical"``.
    """
    score = float(np.clip(reliability_score, 0.0, 1.0))

    if score >= _TRUST_THRESHOLDS["very_high"]:
        return "very_high"
    elif score >= _TRUST_THRESHOLDS["high"]:
        return "high"
    elif score >= _TRUST_THRESHOLDS["moderate"]:
        return "moderate"
    elif score >= _TRUST_THRESHOLDS["low"]:
        return "low"
    else:
        return "critical"


def assign_trust_labels(
    reliability: pd.DataFrame,
    false_consensus: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Assign trust labels to each sample based on consensus reliability
    scores and false consensus flags.

    The base trust level comes from the reliability score.  If a sample
    is also flagged as false consensus, its trust level may be downgraded:

    * ``severe`` false consensus → downgrade by 2 levels
    * ``moderate`` false consensus → downgrade by 1 level
    * ``mild`` false consensus → no downgrade (flag only)

    Args:
        reliability: Output of
            :func:`consensus_reliability.compute_consensus_reliability`.
        false_consensus: Output of
            :func:`false_consensus_detection.detect_false_consensus`.
            Optional.

    Returns:
        DataFrame with columns:

        * ``sample_id``
        * ``consensus_reliability_score``
        * ``base_trust_level``           – trust from score alone
        * ``adjusted_trust_level``       – trust after false consensus adjustment
        * ``false_consensus_flag``       – True if flagged
        * ``false_consensus_severity``   – severity or None
        * ``trust_downgrade_reason``     – reason string or None
    """
    if reliability.empty:
        logger.warning("Empty reliability data – no trust labels to assign.")
        return _empty_trust_df()

    if "consensus_reliability_score" not in reliability.columns:
        logger.warning("Missing 'consensus_reliability_score' column.")
        return _empty_trust_df()

    # Build false consensus lookup
    fc_map: Dict[str, Dict] = {}
    if false_consensus is not None and not false_consensus.empty:
        for _, row in false_consensus.iterrows():
            sid = str(row.get("sample_id", ""))
            if sid:
                fc_map[sid] = {
                    "severity": str(row.get("false_consensus_severity", "mild")),
                    "flags": row.get("flags", []),
                }

    _TRUST_LEVELS = ["critical", "low", "moderate", "high", "very_high"]

    records: List[Dict] = []
    for _, row in reliability.iterrows():
        sid = str(row["sample_id"])
        score = float(row["consensus_reliability_score"])

        base_trust = classify_trust_level(score)
        adjusted_trust = base_trust
        fc_flag = sid in fc_map
        fc_severity = fc_map[sid]["severity"] if fc_flag else None
        downgrade_reason = None

        if fc_flag:
            base_idx = _TRUST_LEVELS.index(base_trust)

            if fc_severity == "severe":
                new_idx = max(0, base_idx - 2)
                if new_idx < base_idx:
                    downgrade_reason = (
                        f"Severe false consensus (flags: "
                        f"{fc_map[sid].get('flags', [])})"
                    )
            elif fc_severity == "moderate":
                new_idx = max(0, base_idx - 1)
                if new_idx < base_idx:
                    downgrade_reason = (
                        f"Moderate false consensus (flags: "
                        f"{fc_map[sid].get('flags', [])})"
                    )
            else:
                new_idx = base_idx
                # Mild: no downgrade, but still flagged

            adjusted_trust = _TRUST_LEVELS[new_idx]

        records.append({
            "sample_id": sid,
            "consensus_reliability_score": round(score, 6),
            "base_trust_level": base_trust,
            "adjusted_trust_level": adjusted_trust,
            "false_consensus_flag": fc_flag,
            "false_consensus_severity": fc_severity,
            "trust_downgrade_reason": downgrade_reason,
        })

    result = pd.DataFrame(records)
    result = result.sort_values(
        "consensus_reliability_score", ascending=True
    ).reset_index(drop=True)

    # Log distribution
    dist = result["adjusted_trust_level"].value_counts().to_dict()
    logger.info("Trust label distribution: %s", dist)

    return result


def _empty_trust_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "sample_id", "consensus_reliability_score",
        "base_trust_level", "adjusted_trust_level",
        "false_consensus_flag", "false_consensus_severity",
        "trust_downgrade_reason",
    ])


# ---------------------------------------------------------------------------
# Public API – trust summary
# ---------------------------------------------------------------------------

def compute_trust_summary(
    trust_labels: pd.DataFrame,
) -> Dict:
    """
    Aggregate trust labels into a summary dictionary.

    Args:
        trust_labels: Output of :func:`assign_trust_labels`.

    Returns:
        Dictionary with:

        * ``total_samples``           – number of labelled samples
        * ``trust_distribution``      – {level: count}
        * ``trust_rates``             – {level: proportion}
        * ``downgraded_count``        – samples where trust was downgraded
        * ``critical_sample_ids``     – samples at critical trust
        * ``very_high_sample_ids``    – samples at very_high trust
        * ``mean_reliability_by_trust`` – average reliability per trust tier
    """
    if trust_labels.empty:
        return {
            "total_samples": 0,
            "trust_distribution": {},
            "trust_rates": {},
            "downgraded_count": 0,
            "critical_sample_ids": [],
            "very_high_sample_ids": [],
            "mean_reliability_by_trust": {},
        }

    total = len(trust_labels)

    # Distribution
    dist = trust_labels["adjusted_trust_level"].value_counts().to_dict()
    for level in ["very_high", "high", "moderate", "low", "critical"]:
        dist.setdefault(level, 0)

    rates = {k: round(v / total, 6) for k, v in dist.items()}

    # Downgraded
    downgraded = 0
    if ("base_trust_level" in trust_labels.columns
            and "adjusted_trust_level" in trust_labels.columns):
        downgraded = int(
            (trust_labels["base_trust_level"]
             != trust_labels["adjusted_trust_level"]).sum()
        )

    # Critical and very_high IDs
    critical_ids = sorted(
        trust_labels.loc[
            trust_labels["adjusted_trust_level"] == "critical", "sample_id"
        ].unique().tolist()
    )
    very_high_ids = sorted(
        trust_labels.loc[
            trust_labels["adjusted_trust_level"] == "very_high", "sample_id"
        ].unique().tolist()
    )

    # Mean reliability by trust level
    mean_by_trust: Dict[str, float] = {}
    if "consensus_reliability_score" in trust_labels.columns:
        for level, grp in trust_labels.groupby("adjusted_trust_level"):
            mean_by_trust[str(level)] = round(
                float(grp["consensus_reliability_score"].mean()), 6
            )

    return {
        "total_samples": total,
        "trust_distribution": dist,
        "trust_rates": rates,
        "downgraded_count": downgraded,
        "critical_sample_ids": critical_ids,
        "very_high_sample_ids": very_high_ids,
        "mean_reliability_by_trust": mean_by_trust,
    }


# ===========================================================================
# Export – Consensus Reliability
# ===========================================================================

_RELIABILITY_DIR = Path("results") / "disagreement" / "consensus_reliability"
_FALSE_CONSENSUS_DIR = Path("results") / "disagreement" / "false_consensus"
_TRUST_DIR = Path("results") / "disagreement" / "trust_analysis"


def _ensure_dir(d: Optional[Union[str, Path]], default: Path) -> Path:
    out = Path(d) if d is not None else default
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---- Consensus Reliability exports ----

def export_consensus_reliability_csv(
    reliability: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "consensus_reliability.csv",
) -> Path:
    """Export per-sample consensus reliability scores as CSV."""
    out = _ensure_dir(output_dir, _RELIABILITY_DIR)
    fp = out / filename
    reliability.to_csv(fp, index=False, float_format="%.6f")
    logger.info("Consensus reliability (CSV) saved to %s", fp)
    return fp


def export_consensus_reliability_json(
    reliability: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "consensus_reliability.json",
) -> Path:
    """Export per-sample consensus reliability scores as JSON."""
    out = _ensure_dir(output_dir, _RELIABILITY_DIR)
    fp = out / filename
    records = reliability.to_dict(orient="records")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(records), fh, indent=2, ensure_ascii=False)
    logger.info("Consensus reliability (JSON) saved to %s", fp)
    return fp


def export_reliability_summary_json(
    summary: Dict,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "reliability_summary.json",
) -> Path:
    """Export consensus reliability summary as JSON."""
    out = _ensure_dir(output_dir, _RELIABILITY_DIR)
    fp = out / filename
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(summary), fh, indent=2, ensure_ascii=False)
    logger.info("Reliability summary saved to %s", fp)
    return fp


def export_consensus_breakdown_csv(
    breakdown: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "consensus_breakdown.csv",
) -> Path:
    """Export consensus breakdown analysis as CSV."""
    out = _ensure_dir(output_dir, _RELIABILITY_DIR)
    fp = out / filename
    breakdown.to_csv(fp, index=False, float_format="%.6f")
    logger.info("Consensus breakdown (CSV) saved to %s", fp)
    return fp


def export_consensus_breakdown_json(
    breakdown: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "consensus_breakdown.json",
) -> Path:
    """Export consensus breakdown analysis as JSON."""
    out = _ensure_dir(output_dir, _RELIABILITY_DIR)
    fp = out / filename
    records = breakdown.to_dict(orient="records")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(records), fh, indent=2, ensure_ascii=False)
    logger.info("Consensus breakdown (JSON) saved to %s", fp)
    return fp


def export_consistency_metrics_json(
    metrics: Dict,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "consistency_metrics.json",
) -> Path:
    """Export consensus consistency metrics as JSON."""
    out = _ensure_dir(output_dir, _RELIABILITY_DIR)
    fp = out / filename
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(metrics), fh, indent=2, ensure_ascii=False)
    logger.info("Consistency metrics saved to %s", fp)
    return fp


def export_model_trust_contribution_csv(
    trust_contrib: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "model_trust_contribution.csv",
) -> Path:
    """Export model trust contributions as CSV."""
    out = _ensure_dir(output_dir, _TRUST_DIR)
    fp = out / filename
    trust_contrib.to_csv(fp, index=False, float_format="%.6f")
    logger.info("Model trust contribution (CSV) saved to %s", fp)
    return fp


def export_model_trust_contribution_json(
    trust_contrib: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "model_trust_contribution.json",
) -> Path:
    """Export model trust contributions as JSON."""
    out = _ensure_dir(output_dir, _TRUST_DIR)
    fp = out / filename
    records = trust_contrib.to_dict(orient="records")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(records), fh, indent=2, ensure_ascii=False)
    logger.info("Model trust contribution (JSON) saved to %s", fp)
    return fp


def export_model_trust_summary_json(
    summary: Dict,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "model_trust_summary.json",
) -> Path:
    """Export model trust summary as JSON."""
    out = _ensure_dir(output_dir, _TRUST_DIR)
    fp = out / filename
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(summary), fh, indent=2, ensure_ascii=False)
    logger.info("Model trust summary saved to %s", fp)
    return fp


# ---- False Consensus exports ----

def export_false_consensus_csv(
    false_consensus: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "false_consensus.csv",
) -> Path:
    """Export false consensus detections as CSV."""
    out = _ensure_dir(output_dir, _FALSE_CONSENSUS_DIR)
    fp = out / filename
    # Serialise the 'flags' list column for CSV
    export_df = false_consensus.copy()
    if "flags" in export_df.columns:
        export_df["flags"] = export_df["flags"].apply(
            lambda x: "|".join(x) if isinstance(x, list) else str(x)
        )
    export_df.to_csv(fp, index=False, float_format="%.6f")
    logger.info("False consensus (CSV) saved to %s", fp)
    return fp


def export_false_consensus_json(
    false_consensus: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "false_consensus.json",
) -> Path:
    """Export false consensus detections as JSON."""
    out = _ensure_dir(output_dir, _FALSE_CONSENSUS_DIR)
    fp = out / filename
    records = false_consensus.to_dict(orient="records")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(records), fh, indent=2, ensure_ascii=False)
    logger.info("False consensus (JSON) saved to %s", fp)
    return fp


def export_fragile_consensus_csv(
    fragile: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "fragile_consensus.csv",
) -> Path:
    """Export fragile consensus detections as CSV."""
    out = _ensure_dir(output_dir, _FALSE_CONSENSUS_DIR)
    fp = out / filename
    fragile.to_csv(fp, index=False)
    logger.info("Fragile consensus (CSV) saved to %s", fp)
    return fp


def export_fragile_consensus_json(
    fragile: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "fragile_consensus.json",
) -> Path:
    """Export fragile consensus detections as JSON."""
    out = _ensure_dir(output_dir, _FALSE_CONSENSUS_DIR)
    fp = out / filename
    records = fragile.to_dict(orient="records")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(records), fh, indent=2, ensure_ascii=False)
    logger.info("Fragile consensus (JSON) saved to %s", fp)
    return fp


def export_unstable_agreement_csv(
    unstable: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "unstable_agreement.csv",
) -> Path:
    """Export unstable agreement detections as CSV."""
    out = _ensure_dir(output_dir, _FALSE_CONSENSUS_DIR)
    fp = out / filename
    unstable.to_csv(fp, index=False, float_format="%.6f")
    logger.info("Unstable agreement (CSV) saved to %s", fp)
    return fp


def export_unstable_agreement_json(
    unstable: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "unstable_agreement.json",
) -> Path:
    """Export unstable agreement detections as JSON."""
    out = _ensure_dir(output_dir, _FALSE_CONSENSUS_DIR)
    fp = out / filename
    records = unstable.to_dict(orient="records")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(records), fh, indent=2, ensure_ascii=False)
    logger.info("Unstable agreement (JSON) saved to %s", fp)
    return fp


def export_false_consensus_summary_json(
    summary: Dict,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "false_consensus_summary.json",
) -> Path:
    """Export false consensus summary as JSON."""
    out = _ensure_dir(output_dir, _FALSE_CONSENSUS_DIR)
    fp = out / filename
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(summary), fh, indent=2, ensure_ascii=False)
    logger.info("False consensus summary saved to %s", fp)
    return fp


# ---- Trust Analysis exports ----

def export_trust_labels_csv(
    trust_labels: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "trust_labels.csv",
) -> Path:
    """Export trust labels as CSV."""
    out = _ensure_dir(output_dir, _TRUST_DIR)
    fp = out / filename
    trust_labels.to_csv(fp, index=False, float_format="%.6f")
    logger.info("Trust labels (CSV) saved to %s", fp)
    return fp


def export_trust_labels_json(
    trust_labels: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "trust_labels.json",
) -> Path:
    """Export trust labels as JSON (grouped by trust level)."""
    out = _ensure_dir(output_dir, _TRUST_DIR)
    fp = out / filename

    grouped: Dict = {}
    for _, row in trust_labels.iterrows():
        level = str(row.get("adjusted_trust_level", "unknown"))
        if level not in grouped:
            grouped[level] = []
        entry = {
            "sample_id": str(row["sample_id"]),
            "consensus_reliability_score": round(
                float(row.get("consensus_reliability_score", 0)), 6
            ),
            "base_trust_level": str(row.get("base_trust_level", "")),
            "false_consensus_flag": bool(row.get("false_consensus_flag", False)),
        }
        if row.get("trust_downgrade_reason"):
            entry["downgrade_reason"] = str(row["trust_downgrade_reason"])
        grouped[level].append(entry)

    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(grouped), fh, indent=2, ensure_ascii=False)
    logger.info("Trust labels (JSON) saved to %s", fp)
    return fp


def export_trust_summary_json(
    summary: Dict,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "trust_summary.json",
) -> Path:
    """Export trust summary as JSON."""
    out = _ensure_dir(output_dir, _TRUST_DIR)
    fp = out / filename
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(summary), fh, indent=2, ensure_ascii=False)
    logger.info("Trust summary saved to %s", fp)
    return fp
