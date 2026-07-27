"""
Disagreement Export for Pathogen Intelligence System.

Provides CSV and JSON export routines for agreement matrices,
disagreement tables, and summary statistics.

All outputs are saved under ``results/disagreement/``.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Default output directory (relative to project root)
_DEFAULT_OUTPUT_DIR = Path("results") / "disagreement"


def _ensure_output_dir(output_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve and create the output directory if it does not exist."""
    out = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Agreement matrix exports
# ---------------------------------------------------------------------------

def export_agreement_matrix_csv(
    agreement_pct: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "agreement_matrix.csv",
) -> Path:
    """
    Export the pairwise agreement-percentage matrix as CSV.

    Args:
        agreement_pct: Square DataFrame (models × models) of percentages.
        output_dir: Directory to save into (default: ``results/disagreement``).
        filename: Name of the CSV file.

    Returns:
        Path to the written file.
    """
    out = _ensure_output_dir(output_dir)
    filepath = out / filename
    agreement_pct.to_csv(filepath, float_format="%.2f")
    logger.info("Agreement matrix (CSV) saved to %s", filepath)
    return filepath


def export_agreement_matrix_json(
    agreement_pct: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "agreement_matrix.json",
) -> Path:
    """
    Export the pairwise agreement-percentage matrix as JSON.

    The JSON structure is a list of ``{model_a, model_b, agreement_pct}``
    records for every unique pair (plus self-pairs on the diagonal).

    Args:
        agreement_pct: Square DataFrame (models × models) of percentages.
        output_dir: Directory to save into.
        filename: Name of the JSON file.

    Returns:
        Path to the written file.
    """
    out = _ensure_output_dir(output_dir)
    filepath = out / filename

    records = []
    for model_a in agreement_pct.index:
        for model_b in agreement_pct.columns:
            val = agreement_pct.loc[model_a, model_b]
            records.append({
                "model_a": str(model_a),
                "model_b": str(model_b),
                "agreement_pct": round(float(val), 2) if pd.notna(val) else None,
            })

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    logger.info("Agreement matrix (JSON) saved to %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Disagreement table exports
# ---------------------------------------------------------------------------

def export_disagreements_csv(
    disagreements: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "disagreements.csv",
) -> Path:
    """
    Export the per-sample disagreement table as CSV.

    Args:
        disagreements: DataFrame of disagreement rows (one row per model
            per disagreement sample).
        output_dir: Directory to save into.
        filename: Name of the CSV file.

    Returns:
        Path to the written file.
    """
    out = _ensure_output_dir(output_dir)
    filepath = out / filename
    disagreements.to_csv(filepath, index=False, float_format="%.6f")
    logger.info("Disagreements (CSV) saved to %s", filepath)
    return filepath


def export_disagreements_json(
    disagreements: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "disagreements.json",
) -> Path:
    """
    Export the per-sample disagreement table as JSON.

    The JSON is organised by ``sample_id``, with each sample containing
    a list of per-model prediction records::

        {
          "sample_001": [
            {"model_name": "efficientnet_b0", "predicted_class": "s_aureus", "confidence": 0.92},
            {"model_name": "resnet50",        "predicted_class": "e_coli",   "confidence": 0.87}
          ],
          ...
        }

    Args:
        disagreements: DataFrame of disagreement rows.
        output_dir: Directory to save into.
        filename: Name of the JSON file.

    Returns:
        Path to the written file.
    """
    out = _ensure_output_dir(output_dir)
    filepath = out / filename

    grouped: Dict = {}
    for sample_id, group in disagreements.groupby("sample_id"):
        records = []
        for _, row in group.iterrows():
            entry = {
                "model_name": str(row["model_name"]),
                "predicted_class": str(row["predicted_class"]),
            }
            if "confidence" in row and pd.notna(row["confidence"]):
                entry["confidence"] = round(float(row["confidence"]), 6)
            records.append(entry)
        grouped[str(sample_id)] = records

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(grouped, fh, indent=2, ensure_ascii=False)

    logger.info("Disagreements (JSON) saved to %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Statistics export
# ---------------------------------------------------------------------------

def export_statistics_json(
    statistics: Dict,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "disagreement_statistics.json",
) -> Path:
    """
    Export disagreement summary statistics as JSON.

    Args:
        statistics: Dictionary returned by
            :func:`analysis.disagreement.agreement_metrics.compute_disagreement_statistics`.
        output_dir: Directory to save into.
        filename: Name of the JSON file.

    Returns:
        Path to the written file.
    """
    out = _ensure_output_dir(output_dir)
    filepath = out / filename

    # Ensure all values are JSON-serialisable
    serialisable = _make_serialisable(statistics)

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2, ensure_ascii=False)

    logger.info("Disagreement statistics saved to %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_serialisable(obj):
    """Recursively convert numpy / pandas types to native Python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj
