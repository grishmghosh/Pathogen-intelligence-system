"""
Perturbation-Aware Disagreement Detection for Pathogen Intelligence System.

Analyses how perturbations affect model consensus.  Detects disagreements
that are *induced* by perturbations (i.e. models agree on the clean image
but diverge after perturbation), computes per-perturbation-type disagreement
rates, and tracks consensus stability across escalating severity levels.

Architecture Flow:
    Perturbation Predictions -> Perturbation Grouping -> Induced Detection
                             -> Sensitivity Ranking -> Consensus Tracking

Depends on:
    analysis.disagreement.disagreement_utils.load_predictions  (Step 1)
"""

import itertools
import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column alias map for perturbation-specific fields
# ---------------------------------------------------------------------------
_PERTURBATION_ALIASES: Dict[str, List[str]] = {
    "perturbation_type": [
        "perturbation_type", "perturbation", "pert_type", "type",
        "augmentation", "transform",
    ],
    "severity_level": [
        "severity_level", "severity", "level", "pert_severity",
        "perturbation_severity", "intensity",
    ],
}

# Canonical ordering for severity levels (lowest -> highest)
_SEVERITY_ORDER_MAP: Dict[str, int] = {
    "clean": 0, "none": 0, "original": 0,
    "mild": 1, "low": 1, "light": 1,
    "moderate": 2, "medium": 2, "mid": 2,
    "severe": 3, "high": 3, "heavy": 3,
    "extreme": 4, "very_high": 4, "critical": 4,
}

_CLEAN_LABELS = {"clean", "none", "original"}


def _normalise_perturbation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map alternative perturbation / severity column names to canonical."""
    rename_map: Dict[str, str] = {}
    existing = {c.lower().strip(): c for c in df.columns}

    for canonical, aliases in _PERTURBATION_ALIASES.items():
        if canonical in existing:
            if existing[canonical] != canonical:
                rename_map[existing[canonical]] = canonical
            continue
        for alias in aliases:
            if alias.lower() in existing:
                rename_map[existing[alias.lower()]] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.debug("Perturbation columns normalised: %s", rename_map)
    return df


def _severity_rank(label: str) -> int:
    """Map a severity label to a numeric rank.  Unknown labels get rank 99."""
    return _SEVERITY_ORDER_MAP.get(str(label).lower().strip(), 99)


def _is_clean(label: str) -> bool:
    return str(label).lower().strip() in _CLEAN_LABELS


# ---------------------------------------------------------------------------
# Public API - load perturbation predictions
# ---------------------------------------------------------------------------

def load_perturbation_predictions(
    source: Union[str, "pathlib.Path", List[dict], pd.DataFrame],
) -> pd.DataFrame:
    """
    Load and normalise predictions that include perturbation metadata.

    This wraps Step 1's ``load_predictions`` and additionally normalises
    ``perturbation_type`` and ``severity_level`` columns.  Missing
    perturbation columns are filled with sensible defaults:

    * ``perturbation_type`` -> ``"unknown"``
    * ``severity_level``    -> ``"unknown"``

    Args:
        source: Any input accepted by ``load_predictions`` (list, CSV, DF).

    Returns:
        Normalised DataFrame with columns: ``sample_id``, ``model_name``,
        ``predicted_class``, ``confidence``, ``perturbation_type``,
        ``severity_level``.
    """
    from analysis.disagreement.disagreement_utils import load_predictions

    df = load_predictions(source)
    df = _normalise_perturbation_columns(df)

    if "perturbation_type" not in df.columns:
        df["perturbation_type"] = "unknown"
        logger.info("'perturbation_type' not found - filled with 'unknown'.")

    if "severity_level" not in df.columns:
        df["severity_level"] = "unknown"
        logger.info("'severity_level' not found - filled with 'unknown'.")

    df["perturbation_type"] = df["perturbation_type"].astype(str).str.strip().str.lower()
    df["severity_level"] = df["severity_level"].astype(str).str.strip().str.lower()

    return df


# ---------------------------------------------------------------------------
# Public API - perturbation-induced disagreement detection
# ---------------------------------------------------------------------------

def detect_perturbation_induced_disagreements(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect disagreements that are *induced* by perturbation.

    A perturbation-induced disagreement is a sample where:
    1.  All models agree on the **clean** version.
    2.  At least two models disagree on a **perturbed** version.

    The *base sample* is identified by stripping perturbation metadata
    from ``sample_id`` (if it encodes the perturbation) **or** by
    grouping rows that share the same ``sample_id`` prefix before the
    perturbation suffix.

    In practice, each unique ``(sample_id, perturbation_type, severity_level)``
    triplet is treated as one observation.  The clean observation is the one
    whose ``severity_level`` is in ``{"clean", "none", "original"}``.

    Args:
        predictions: Output of :func:`load_perturbation_predictions`.

    Returns:
        DataFrame containing only the rows of induced disagreements,
        enriched with a boolean ``perturbation_induced`` column.
        An empty DataFrame is returned when there are no induced
        disagreements.
    """
    if predictions.empty:
        raise ValueError("Predictions DataFrame is empty.")

    model_names = sorted(predictions["model_name"].unique())
    if len(model_names) < 2:
        raise ValueError(
            f"At least 2 models required; found {len(model_names)}: {model_names}"
        )

    # Separate clean vs perturbed
    clean_mask = predictions["severity_level"].apply(_is_clean)
    clean_df = predictions[clean_mask]
    perturbed_df = predictions[~clean_mask]

    if clean_df.empty:
        logger.warning("No clean predictions found - cannot detect induced disagreements.")
        return _empty_induced_df()

    if perturbed_df.empty:
        logger.warning("No perturbed predictions found.")
        return _empty_induced_df()

    # --- Determine which sample_ids have full agreement on clean ---
    clean_pivot = clean_df.pivot_table(
        index="sample_id", columns="model_name",
        values="predicted_class", aggfunc="first",
    )
    clean_complete = clean_pivot.dropna()

    if clean_complete.empty:
        logger.warning("No clean samples with all models present.")
        return _empty_induced_df()

    clean_agrees = clean_complete.apply(lambda r: r.nunique() == 1, axis=1)
    agreed_clean_ids = set(clean_agrees[clean_agrees].index)

    if not agreed_clean_ids:
        logger.info("No clean samples with full agreement - no induced disagreements possible.")
        return _empty_induced_df()

    # --- Among perturbed rows, find disagreements for those same sample_ids ---
    perturbed_subset = perturbed_df[perturbed_df["sample_id"].isin(agreed_clean_ids)]
    if perturbed_subset.empty:
        return _empty_induced_df()

    # Group by (sample_id, perturbation_type, severity_level)
    group_cols = ["sample_id", "perturbation_type", "severity_level"]
    induced_records: List[pd.DataFrame] = []

    for key, grp in perturbed_subset.groupby(group_cols):
        if len(grp["model_name"].unique()) < 2:
            continue
        classes = grp["predicted_class"].unique()
        if len(classes) > 1:
            g = grp.copy()
            g["perturbation_induced"] = True
            induced_records.append(g)

    if not induced_records:
        logger.info("No perturbation-induced disagreements found.")
        return _empty_induced_df()

    result = pd.concat(induced_records, ignore_index=True)
    result = result.sort_values(group_cols + ["model_name"]).reset_index(drop=True)
    logger.info("Detected %d induced disagreement observation(s).",
                result.groupby(group_cols).ngroups)
    return result


def _empty_induced_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "sample_id", "model_name", "predicted_class", "confidence",
        "perturbation_type", "severity_level", "perturbation_induced",
    ])


# ---------------------------------------------------------------------------
# Public API - perturbation-type sensitivity ranking
# ---------------------------------------------------------------------------

def compute_perturbation_sensitivity(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank perturbation types by their disagreement-triggering frequency.

    For each ``perturbation_type`` the function computes:

    * ``total_observations``  - unique (sample, perturbation, severity) combos
    * ``disagreement_count``  - observations where models disagree
    * ``agreement_count``     - observations where models agree
    * ``disagreement_rate``   - disagreement_count / total (0-1)

    Clean/original observations are excluded from the ranking.

    Args:
        predictions: Output of :func:`load_perturbation_predictions`.

    Returns:
        DataFrame sorted by ``disagreement_rate`` descending.
    """
    if predictions.empty:
        raise ValueError("Predictions DataFrame is empty.")

    model_names = sorted(predictions["model_name"].unique())
    if len(model_names) < 2:
        raise ValueError(f"At least 2 models required; found {len(model_names)}")

    # Exclude clean
    perturbed = predictions[~predictions["severity_level"].apply(_is_clean)]
    if perturbed.empty:
        logger.warning("No perturbed predictions to analyse.")
        return pd.DataFrame(columns=[
            "perturbation_type", "total_observations", "disagreement_count",
            "agreement_count", "disagreement_rate",
        ])

    group_cols = ["sample_id", "perturbation_type", "severity_level"]
    records: List[Dict] = []

    for pert_type, pert_grp in perturbed.groupby("perturbation_type"):
        total = 0
        disagree = 0
        for _, obs_grp in pert_grp.groupby(["sample_id", "severity_level"]):
            if len(obs_grp["model_name"].unique()) < 2:
                continue
            total += 1
            if obs_grp["predicted_class"].nunique() > 1:
                disagree += 1

        if total == 0:
            continue

        records.append({
            "perturbation_type": str(pert_type),
            "total_observations": total,
            "disagreement_count": disagree,
            "agreement_count": total - disagree,
            "disagreement_rate": round(disagree / total, 6),
        })

    if not records:
        return pd.DataFrame(columns=[
            "perturbation_type", "total_observations", "disagreement_count",
            "agreement_count", "disagreement_rate",
        ])

    result = pd.DataFrame(records)
    result = result.sort_values("disagreement_rate", ascending=False).reset_index(drop=True)
    logger.info("Perturbation sensitivity computed for %d types.", len(result))
    return result


# ---------------------------------------------------------------------------
# Public API - consensus stability tracking
# ---------------------------------------------------------------------------

def track_consensus_stability(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Track how model agreement degrades as perturbation severity escalates.

    For each ``sample_id`` the function walks through severity levels in
    ascending order and records:

    * ``severity_level``     - the severity label
    * ``severity_rank``      - numeric rank (0 = clean)
    * ``models_agree``       - boolean, all models predict the same class
    * ``n_unique_classes``   - number of distinct predictions

    Additionally, a ``stability_breakpoint`` is identified: the *first*
    severity level at which consensus breaks (or ``None`` if it never does).

    Args:
        predictions: Output of :func:`load_perturbation_predictions`.

    Returns:
        DataFrame with one row per (sample_id, severity_level), sorted by
        sample then severity rank.  Extra columns:

        * ``stability_breakpoint`` - True on the row where consensus first
          breaks for that sample.
        * ``collapse_severity``    - the severity label of the breakpoint
          (same value on every row of the sample for easy filtering).
    """
    if predictions.empty:
        raise ValueError("Predictions DataFrame is empty.")

    model_names = sorted(predictions["model_name"].unique())
    if len(model_names) < 2:
        raise ValueError(f"At least 2 models required; found {len(model_names)}")

    records: List[Dict] = []

    for sample_id, sample_grp in predictions.groupby("sample_id"):
        severity_groups = sample_grp.groupby("severity_level")
        steps: List[Dict] = []

        for sev_label, sev_grp in severity_groups:
            if len(sev_grp["model_name"].unique()) < 2:
                continue
            n_classes = sev_grp["predicted_class"].nunique()
            steps.append({
                "sample_id": str(sample_id),
                "severity_level": str(sev_label),
                "severity_rank": _severity_rank(str(sev_label)),
                "models_agree": n_classes == 1,
                "n_unique_classes": int(n_classes),
            })

        if not steps:
            continue

        # Sort by severity rank
        steps.sort(key=lambda s: s["severity_rank"])

        # Find breakpoint
        collapse_sev = None
        for s in steps:
            if not s["models_agree"] and collapse_sev is None:
                collapse_sev = s["severity_level"]
                s["stability_breakpoint"] = True
            else:
                s["stability_breakpoint"] = False
            s["collapse_severity"] = collapse_sev

        records.extend(steps)

    if not records:
        return pd.DataFrame(columns=[
            "sample_id", "severity_level", "severity_rank", "models_agree",
            "n_unique_classes", "stability_breakpoint", "collapse_severity",
        ])

    result = pd.DataFrame(records)
    result = result.sort_values(["sample_id", "severity_rank"]).reset_index(drop=True)
    logger.info("Consensus stability tracked for %d samples.",
                result["sample_id"].nunique())
    return result


# ---------------------------------------------------------------------------
# Public API - severity-level disagreement rates
# ---------------------------------------------------------------------------

def compute_severity_disagreement_rates(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute disagreement rate at each perturbation severity level.

    Returns a DataFrame with one row per severity level:

    * ``severity_level``
    * ``severity_rank``
    * ``total_observations``
    * ``disagreement_count``
    * ``disagreement_rate``

    Sorted by ``severity_rank`` ascending (clean -> extreme).
    """
    if predictions.empty:
        raise ValueError("Predictions DataFrame is empty.")

    model_names = sorted(predictions["model_name"].unique())
    if len(model_names) < 2:
        raise ValueError(f"At least 2 models required; found {len(model_names)}")

    records: List[Dict] = []
    for sev_label, sev_grp in predictions.groupby("severity_level"):
        total = 0
        disagree = 0
        for _, obs_grp in sev_grp.groupby("sample_id"):
            if len(obs_grp["model_name"].unique()) < 2:
                continue
            total += 1
            if obs_grp["predicted_class"].nunique() > 1:
                disagree += 1

        if total == 0:
            continue
        records.append({
            "severity_level": str(sev_label),
            "severity_rank": _severity_rank(str(sev_label)),
            "total_observations": total,
            "disagreement_count": disagree,
            "disagreement_rate": round(disagree / total, 6),
        })

    if not records:
        return pd.DataFrame(columns=[
            "severity_level", "severity_rank", "total_observations",
            "disagreement_count", "disagreement_rate",
        ])

    result = pd.DataFrame(records)
    result = result.sort_values("severity_rank").reset_index(drop=True)
    return result
