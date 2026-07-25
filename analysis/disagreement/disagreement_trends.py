"""
Disagreement Trend Analysis and Step 3 Export for Pathogen Intelligence System.

Generates structured trend summaries from perturbation and instability
analysis results, and provides CSV/JSON export for all Step 3 outputs.

Architecture note:
    Trend generation is separated from perturbation detection
    (``perturbation_disagreement``) and instability scoring
    (``instability_analysis``).  Export is co-located with trends
    to keep Step 3's export surface in one place.

Depends on:
    analysis.disagreement.disagreement_export._make_serialisable  (Step 1)
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


# ===========================================================================
# Default output directories
# ===========================================================================

_PERTURBATION_DIR = Path("results") / "disagreement" / "perturbation_analysis"
_INSTABILITY_DIR = Path("results") / "disagreement" / "instability_analysis"
_TREND_DIR = Path("results") / "disagreement" / "trend_analysis"


def _ensure_dir(d: Optional[Union[str, Path]], default: Path) -> Path:
    out = Path(d) if d is not None else default
    out.mkdir(parents=True, exist_ok=True)
    return out


# ===========================================================================
# Trend generation
# ===========================================================================

def generate_escalation_trend(
    severity_rates: pd.DataFrame,
) -> Dict:
    """
    Generate a trend summary for disagreement escalation across severity.

    The trend captures whether disagreement *increases* as perturbation
    severity grows, and by how much.

    Args:
        severity_rates: Output of
            :func:`perturbation_disagreement.compute_severity_disagreement_rates`.

    Returns:
        Dictionary with:

        * ``trend_direction``  - ``"increasing"``, ``"decreasing"``,
          ``"stable"``, or ``"insufficient_data"``
        * ``severity_steps``   - ordered list of {severity, rate}
        * ``rate_delta``       - change from lowest to highest severity
        * ``max_rate``         - peak disagreement rate
        * ``max_rate_severity``- severity level at peak
        * ``monotonic``        - True if rates strictly increase with severity
    """
    if severity_rates.empty or "disagreement_rate" not in severity_rates.columns:
        return {
            "trend_direction": "insufficient_data",
            "severity_steps": [],
            "rate_delta": 0.0,
            "max_rate": 0.0,
            "max_rate_severity": None,
            "monotonic": False,
        }

    df = severity_rates.sort_values("severity_rank").reset_index(drop=True)
    rates = df["disagreement_rate"].tolist()
    labels = df["severity_level"].tolist()

    steps = [{"severity": labels[i], "disagreement_rate": round(rates[i], 6)}
             for i in range(len(rates))]

    rate_delta = rates[-1] - rates[0] if len(rates) >= 2 else 0.0
    max_rate = max(rates)
    max_idx = rates.index(max_rate)

    # Trend direction
    if len(rates) < 2:
        direction = "insufficient_data"
    elif rate_delta > 0.02:
        direction = "increasing"
    elif rate_delta < -0.02:
        direction = "decreasing"
    else:
        direction = "stable"

    # Monotonicity check
    monotonic = all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1)) if len(rates) >= 2 else False

    return {
        "trend_direction": direction,
        "severity_steps": steps,
        "rate_delta": round(rate_delta, 6),
        "max_rate": round(max_rate, 6),
        "max_rate_severity": labels[max_idx],
        "monotonic": monotonic,
    }


def generate_model_comparison_trend(
    model_instability: pd.DataFrame,
) -> Dict:
    """
    Generate a trend summary comparing model stability.

    Args:
        model_instability: Output of
            :func:`instability_analysis.compute_model_instability`.

    Returns:
        Dictionary with:

        * ``model_ranking``       - list of {model, instability_score, flip_rate}
        * ``most_stable``         - model name with lowest instability
        * ``least_stable``        - model name with highest instability
        * ``instability_spread``  - max - min instability score
        * ``summary``             - human-readable one-line summary
    """
    if model_instability.empty or "instability_score" not in model_instability.columns:
        return {
            "model_ranking": [],
            "most_stable": None,
            "least_stable": None,
            "instability_spread": 0.0,
            "summary": "Insufficient data for model comparison.",
        }

    df = model_instability.sort_values("instability_score", ascending=False).reset_index(drop=True)
    ranking = []
    for _, row in df.iterrows():
        ranking.append({
            "model": str(row["model_name"]),
            "instability_score": round(float(row["instability_score"]), 6),
            "flip_rate": round(float(row["flip_rate"]), 6),
        })

    least_stable = ranking[0]["model"]
    most_stable = ranking[-1]["model"]
    spread = ranking[0]["instability_score"] - ranking[-1]["instability_score"]

    if spread < 0.05:
        summary = f"All models show similar stability (spread={spread:.4f})."
    else:
        summary = (
            f"{least_stable} destabilises faster than {most_stable} "
            f"(instability spread={spread:.4f})."
        )

    return {
        "model_ranking": ranking,
        "most_stable": most_stable,
        "least_stable": least_stable,
        "instability_spread": round(spread, 6),
        "summary": summary,
    }


def generate_perturbation_ranking_trend(
    sensitivity: pd.DataFrame,
) -> Dict:
    """
    Generate a trend summary ranking perturbation types by impact.

    Args:
        sensitivity: Output of
            :func:`perturbation_disagreement.compute_perturbation_sensitivity`.

    Returns:
        Dictionary with:

        * ``ranking``                - sorted list of {perturbation, rate}
        * ``most_disruptive``        - perturbation type with highest rate
        * ``least_disruptive``       - perturbation type with lowest rate
        * ``disruption_spread``      - max - min disagreement rate
        * ``summary``                - human-readable one-line summary
    """
    if sensitivity.empty or "disagreement_rate" not in sensitivity.columns:
        return {
            "ranking": [],
            "most_disruptive": None,
            "least_disruptive": None,
            "disruption_spread": 0.0,
            "summary": "Insufficient data for perturbation ranking.",
        }

    df = sensitivity.sort_values("disagreement_rate", ascending=False).reset_index(drop=True)
    ranking = []
    for _, row in df.iterrows():
        ranking.append({
            "perturbation_type": str(row["perturbation_type"]),
            "disagreement_rate": round(float(row["disagreement_rate"]), 6),
            "total_observations": int(row["total_observations"]),
        })

    most = ranking[0]["perturbation_type"]
    least = ranking[-1]["perturbation_type"]
    spread = ranking[0]["disagreement_rate"] - ranking[-1]["disagreement_rate"]

    pct = round(ranking[0]["disagreement_rate"] * 100, 1)
    summary = f"{most} produces highest instability ({pct}% disagreement rate)."

    return {
        "ranking": ranking,
        "most_disruptive": most,
        "least_disruptive": least,
        "disruption_spread": round(spread, 6),
        "summary": summary,
    }


def generate_full_trend_report(
    severity_rates: pd.DataFrame,
    model_instability: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> Dict:
    """
    Assemble escalation, model-comparison, and perturbation ranking
    trends into a single structured report.
    """
    return {
        "escalation_trend": generate_escalation_trend(severity_rates),
        "model_comparison_trend": generate_model_comparison_trend(model_instability),
        "perturbation_ranking_trend": generate_perturbation_ranking_trend(sensitivity),
    }


# ===========================================================================
# Export - Perturbation analysis
# ===========================================================================

def export_induced_disagreements_csv(
    induced: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "induced_disagreements.csv",
) -> Path:
    """Export perturbation-induced disagreements as CSV."""
    out = _ensure_dir(output_dir, _PERTURBATION_DIR)
    fp = out / filename
    induced.to_csv(fp, index=False, float_format="%.6f")
    logger.info("Induced disagreements (CSV) saved to %s", fp)
    return fp


def export_induced_disagreements_json(
    induced: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "induced_disagreements.json",
) -> Path:
    """Export perturbation-induced disagreements as JSON (grouped by sample)."""
    out = _ensure_dir(output_dir, _PERTURBATION_DIR)
    fp = out / filename

    grouped: Dict = {}
    group_cols = ["sample_id", "perturbation_type", "severity_level"]
    for key, grp in induced.groupby(group_cols):
        sid, pt, sl = key
        k = f"{sid}|{pt}|{sl}"
        entries = []
        for _, row in grp.iterrows():
            entry = {
                "model_name": str(row["model_name"]),
                "predicted_class": str(row["predicted_class"]),
            }
            if "confidence" in row and pd.notna(row.get("confidence")):
                entry["confidence"] = round(float(row["confidence"]), 6)
            entries.append(entry)
        grouped[k] = {
            "sample_id": str(sid),
            "perturbation_type": str(pt),
            "severity_level": str(sl),
            "predictions": entries,
        }

    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(grouped), fh, indent=2, ensure_ascii=False)
    logger.info("Induced disagreements (JSON) saved to %s", fp)
    return fp


def export_perturbation_sensitivity_csv(
    sensitivity: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "perturbation_sensitivity.csv",
) -> Path:
    """Export perturbation sensitivity ranking as CSV."""
    out = _ensure_dir(output_dir, _PERTURBATION_DIR)
    fp = out / filename
    sensitivity.to_csv(fp, index=False, float_format="%.6f")
    logger.info("Perturbation sensitivity (CSV) saved to %s", fp)
    return fp


def export_perturbation_sensitivity_json(
    sensitivity: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "perturbation_sensitivity.json",
) -> Path:
    """Export perturbation sensitivity ranking as JSON."""
    out = _ensure_dir(output_dir, _PERTURBATION_DIR)
    fp = out / filename
    records = sensitivity.to_dict(orient="records")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(records), fh, indent=2, ensure_ascii=False)
    logger.info("Perturbation sensitivity (JSON) saved to %s", fp)
    return fp


def export_severity_rates_csv(
    rates: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "severity_disagreement_rates.csv",
) -> Path:
    """Export per-severity disagreement rates as CSV."""
    out = _ensure_dir(output_dir, _PERTURBATION_DIR)
    fp = out / filename
    rates.to_csv(fp, index=False, float_format="%.6f")
    logger.info("Severity rates (CSV) saved to %s", fp)
    return fp


def export_consensus_stability_csv(
    stability: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "consensus_stability.csv",
) -> Path:
    """Export consensus stability tracking as CSV."""
    out = _ensure_dir(output_dir, _PERTURBATION_DIR)
    fp = out / filename
    stability.to_csv(fp, index=False)
    logger.info("Consensus stability (CSV) saved to %s", fp)
    return fp


def export_consensus_stability_json(
    stability: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "consensus_stability.json",
) -> Path:
    """Export consensus stability tracking as JSON (grouped by sample)."""
    out = _ensure_dir(output_dir, _PERTURBATION_DIR)
    fp = out / filename

    grouped: Dict = {}
    for sample_id, grp in stability.groupby("sample_id"):
        steps = []
        for _, row in grp.iterrows():
            steps.append({
                "severity_level": str(row["severity_level"]),
                "severity_rank": int(row["severity_rank"]),
                "models_agree": bool(row["models_agree"]),
                "n_unique_classes": int(row["n_unique_classes"]),
                "stability_breakpoint": bool(row["stability_breakpoint"]),
            })
        grouped[str(sample_id)] = {
            "steps": steps,
            "collapse_severity": str(grp.iloc[0]["collapse_severity"]) if pd.notna(grp.iloc[0]["collapse_severity"]) else None,
        }

    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(grouped), fh, indent=2, ensure_ascii=False)
    logger.info("Consensus stability (JSON) saved to %s", fp)
    return fp


# ===========================================================================
# Export - Instability analysis
# ===========================================================================

def export_model_instability_csv(
    instability: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "model_instability.csv",
) -> Path:
    """Export per-model instability as CSV."""
    out = _ensure_dir(output_dir, _INSTABILITY_DIR)
    fp = out / filename
    instability.to_csv(fp, index=False, float_format="%.6f")
    logger.info("Model instability (CSV) saved to %s", fp)
    return fp


def export_model_instability_json(
    instability: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "model_instability.json",
) -> Path:
    """Export per-model instability as JSON."""
    out = _ensure_dir(output_dir, _INSTABILITY_DIR)
    fp = out / filename
    records = instability.to_dict(orient="records")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(records), fh, indent=2, ensure_ascii=False)
    logger.info("Model instability (JSON) saved to %s", fp)
    return fp


def export_sample_instability_csv(
    instability: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "sample_instability.csv",
) -> Path:
    """Export per-sample instability as CSV."""
    out = _ensure_dir(output_dir, _INSTABILITY_DIR)
    fp = out / filename
    instability.to_csv(fp, index=False, float_format="%.6f")
    logger.info("Sample instability (CSV) saved to %s", fp)
    return fp


def export_sample_instability_json(
    instability: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "sample_instability.json",
) -> Path:
    """Export per-sample instability as JSON."""
    out = _ensure_dir(output_dir, _INSTABILITY_DIR)
    fp = out / filename
    records = instability.to_dict(orient="records")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(records), fh, indent=2, ensure_ascii=False)
    logger.info("Sample instability (JSON) saved to %s", fp)
    return fp


def export_instability_summary_json(
    summary: Dict,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "instability_summary.json",
) -> Path:
    """Export instability summary as JSON."""
    out = _ensure_dir(output_dir, _INSTABILITY_DIR)
    fp = out / filename
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(summary), fh, indent=2, ensure_ascii=False)
    logger.info("Instability summary saved to %s", fp)
    return fp


# ===========================================================================
# Export - Trend analysis
# ===========================================================================

def export_trend_report_json(
    report: Dict,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "trend_report.json",
) -> Path:
    """Export the full trend report as JSON."""
    out = _ensure_dir(output_dir, _TREND_DIR)
    fp = out / filename
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(report), fh, indent=2, ensure_ascii=False)
    logger.info("Trend report saved to %s", fp)
    return fp
