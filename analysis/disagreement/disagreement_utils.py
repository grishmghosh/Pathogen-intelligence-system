"""
Disagreement Utilities for Pathogen Intelligence System.

Handles input parsing, schema normalisation, validation, and per-sample
disagreement detection across an arbitrary number of models.

Supported input formats:
    - List of prediction dictionaries
    - CSV file path

Expected prediction fields (flexible):
    - sample_id       (required)
    - model_name      (required)
    - predicted_class (required)
    - confidence      (optional, defaults to NaN)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name normalisation map
# ---------------------------------------------------------------------------
_COLUMN_ALIASES: Dict[str, List[str]] = {
    "sample_id": ["sample_id", "id", "image_id", "filename", "sample"],
    "model_name": ["model_name", "model", "classifier"],
    "predicted_class": ["predicted_class", "prediction", "pred", "label", "class"],
    "confidence": ["confidence", "conf", "score", "probability", "prob"],
}

_REQUIRED_COLUMNS = ["sample_id", "model_name", "predicted_class"]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common alternative column names to the canonical schema."""
    rename_map: Dict[str, str] = {}
    existing = {c.lower().strip(): c for c in df.columns}

    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in existing:
            # Already present under canonical name
            if existing[canonical] != canonical:
                rename_map[existing[canonical]] = canonical
            continue
        for alias in aliases:
            if alias.lower() in existing:
                rename_map[existing[alias.lower()]] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.debug("Normalised columns: %s", rename_map)

    return df


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Raise *ValueError* if required columns are missing or data is empty."""
    if df.empty:
        raise ValueError("Predictions DataFrame is empty.")

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Public API – loading predictions
# ---------------------------------------------------------------------------

def load_predictions(
    source: Union[str, Path, List[dict], pd.DataFrame],
) -> pd.DataFrame:
    """
    Load and validate predictions from various input formats.

    Accepts:
        - ``list[dict]`` – each dict is one prediction row.
        - ``str | Path``  – path to a CSV file.
        - ``pd.DataFrame`` – used directly.

    Returns:
        A normalised ``pd.DataFrame`` with at least the columns
        ``sample_id``, ``model_name``, ``predicted_class``, and
        optionally ``confidence``.

    Raises:
        FileNotFoundError: If a CSV path does not exist.
        ValueError: If required columns are missing or data is empty.
    """
    # ------- Resolve source to DataFrame -------
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, list):
        if len(source) == 0:
            raise ValueError("Prediction list is empty.")
        df = pd.DataFrame(source)
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Failed to read CSV file '{path}': {exc}") from exc
    else:
        raise TypeError(
            f"Unsupported source type: {type(source).__name__}. "
            "Expected list[dict], str/Path (CSV), or pd.DataFrame."
        )

    # ------- Normalise & validate -------
    df = _normalise_columns(df)
    _validate_dataframe(df)

    # Ensure confidence column exists (fill NaN if absent)
    if "confidence" not in df.columns:
        df["confidence"] = np.nan
        logger.info("'confidence' column not found – filled with NaN.")

    # Coerce types
    df["sample_id"] = df["sample_id"].astype(str)
    df["model_name"] = df["model_name"].astype(str)
    df["predicted_class"] = df["predicted_class"].astype(str)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Public API – disagreement detection
# ---------------------------------------------------------------------------

def detect_disagreements(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect samples where at least two models predict different classes.

    For every sample that has a disagreement, one row per *model* is
    returned so the caller can inspect each model's prediction.

    Args:
        predictions: Normalised DataFrame (output of :func:`load_predictions`).

    Returns:
        A ``pd.DataFrame`` with columns ``sample_id``, ``model_name``,
        ``predicted_class``, ``confidence``.  Only rows belonging to
        disagreement samples are included.  An empty DataFrame is
        returned when there are no disagreements.

    Raises:
        ValueError: If fewer than two distinct models are present.
    """
    _validate_dataframe(predictions)

    model_names = predictions["model_name"].unique()
    if len(model_names) < 2:
        raise ValueError(
            f"At least 2 models are required for disagreement detection; "
            f"found {len(model_names)}: {list(model_names)}"
        )

    # Pivot: one column per model, values = predicted_class
    pivot = predictions.pivot_table(
        index="sample_id",
        columns="model_name",
        values="predicted_class",
        aggfunc="first",
    )

    # Drop samples that don't have predictions from all models (partial data)
    pivot_complete = pivot.dropna()

    if pivot_complete.empty:
        logger.warning("No samples have predictions from all models.")
        return pd.DataFrame(columns=["sample_id", "model_name", "predicted_class", "confidence"])

    # A sample is a disagreement if not all predictions in the row are equal
    is_disagreement = pivot_complete.apply(
        lambda row: row.nunique() > 1, axis=1
    )
    disagreement_ids = set(is_disagreement[is_disagreement].index)

    if not disagreement_ids:
        logger.info("No disagreements found – all models agree on every sample.")
        return pd.DataFrame(columns=["sample_id", "model_name", "predicted_class", "confidence"])

    # Return original rows for disagreement samples
    result = predictions[predictions["sample_id"].isin(disagreement_ids)].copy()
    result = result.sort_values(["sample_id", "model_name"]).reset_index(drop=True)

    logger.info(
        "Detected %d disagreement samples out of %d total.",
        len(disagreement_ids),
        len(pivot_complete),
    )

    return result
