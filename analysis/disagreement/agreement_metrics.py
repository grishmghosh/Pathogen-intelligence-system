"""
Agreement Metrics for Pathogen Intelligence System.

Computes the pairwise agreement matrix and basic disagreement statistics
for an arbitrary number of models.

All computation functions accept a normalised ``pd.DataFrame`` produced by
:func:`analysis.disagreement.disagreement_utils.load_predictions`.
"""

import itertools
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API – agreement matrix
# ---------------------------------------------------------------------------

def compute_agreement_matrix(
    predictions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise agreement percentages between every pair of models.

    Agreement is defined as: **both models predict the same class for
    a given sample**.  Only samples that have predictions from *both*
    models in a pair are considered.

    Args:
        predictions: Normalised DataFrame with at least columns
            ``sample_id``, ``model_name``, ``predicted_class``.

    Returns:
        A tuple ``(agreement_pct, agreement_counts)`` where

        * ``agreement_pct`` – symmetric ``DataFrame`` (models × models)
          with agreement percentages (0–100).  Diagonal is always 100.
        * ``agreement_counts`` – symmetric ``DataFrame`` with raw
          agreement counts.  Diagonal equals the number of samples that
          model was evaluated on.

    Raises:
        ValueError: If fewer than two distinct models are present.
    """
    model_names = sorted(predictions["model_name"].unique())
    n_models = len(model_names)

    if n_models < 2:
        raise ValueError(
            f"At least 2 models are required; found {n_models}: {model_names}"
        )

    # Pivot to sample_id × model_name → predicted_class
    pivot = predictions.pivot_table(
        index="sample_id",
        columns="model_name",
        values="predicted_class",
        aggfunc="first",
    )

    # Initialise matrices
    pct_matrix = np.full((n_models, n_models), np.nan)
    cnt_matrix = np.zeros((n_models, n_models), dtype=int)

    for i, model_a in enumerate(model_names):
        if model_a not in pivot.columns:
            continue
        for j, model_b in enumerate(model_names):
            if model_b not in pivot.columns:
                continue

            if i == j:
                # Self-agreement is trivially 100 %
                valid = pivot[model_a].dropna()
                pct_matrix[i, j] = 100.0
                cnt_matrix[i, j] = len(valid)
                continue

            # Restrict to samples present in both models
            mask = pivot[[model_a, model_b]].notna().all(axis=1)
            common = pivot.loc[mask]

            if common.empty:
                pct_matrix[i, j] = np.nan
                cnt_matrix[i, j] = 0
                logger.warning(
                    "No common samples between '%s' and '%s'.", model_a, model_b
                )
                continue

            agreements = (common[model_a] == common[model_b]).sum()
            total = len(common)
            pct_matrix[i, j] = (agreements / total) * 100.0
            cnt_matrix[i, j] = int(agreements)

    agreement_pct = pd.DataFrame(
        pct_matrix, index=model_names, columns=model_names
    )
    agreement_counts = pd.DataFrame(
        cnt_matrix, index=model_names, columns=model_names
    )

    logger.info("Agreement matrix computed for %d models.", n_models)
    return agreement_pct, agreement_counts


# ---------------------------------------------------------------------------
# Public API – disagreement statistics
# ---------------------------------------------------------------------------

def compute_disagreement_statistics(
    predictions: pd.DataFrame,
    disagreements: pd.DataFrame,
) -> Dict:
    """
    Compute basic summary statistics about model disagreements.

    Args:
        predictions:   Full normalised predictions DataFrame.
        disagreements: DataFrame of disagreement rows (output of
            :func:`analysis.disagreement.disagreement_utils.detect_disagreements`).

    Returns:
        Dictionary with keys:

        * ``total_samples``       – number of unique sample IDs evaluated
        * ``total_models``        – number of distinct models
        * ``model_names``         – list of model name strings
        * ``agreement_count``     – samples where all models agree
        * ``disagreement_count``  – samples where at least two models disagree
        * ``agreement_rate``      – agreement_count / total_samples (0–1)
        * ``disagreement_rate``   – disagreement_count / total_samples (0–1)
        * ``per_model_sample_count`` – dict mapping model → number of samples
    """
    model_names = sorted(predictions["model_name"].unique())

    # Pivot to find samples that have predictions from all models
    pivot = predictions.pivot_table(
        index="sample_id",
        columns="model_name",
        values="predicted_class",
        aggfunc="first",
    )
    # Only consider complete rows (all models present)
    complete = pivot.dropna()
    total_samples = len(complete)

    disagreement_ids = set(disagreements["sample_id"].unique()) if not disagreements.empty else set()
    # Only count disagreements that are within the complete set
    disagreement_count = len(disagreement_ids & set(complete.index))
    agreement_count = total_samples - disagreement_count

    agreement_rate = (agreement_count / total_samples) if total_samples > 0 else 0.0
    disagreement_rate = (disagreement_count / total_samples) if total_samples > 0 else 0.0

    per_model_count = (
        predictions.groupby("model_name")["sample_id"].nunique().to_dict()
    )

    stats = {
        "total_samples": total_samples,
        "total_models": len(model_names),
        "model_names": model_names,
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "agreement_rate": round(agreement_rate, 6),
        "disagreement_rate": round(disagreement_rate, 6),
        "per_model_sample_count": per_model_count,
    }

    logger.info(
        "Stats: %d samples, %d agreements, %d disagreements (%.2f%% agreement).",
        total_samples,
        agreement_count,
        disagreement_count,
        agreement_rate * 100,
    )

    return stats
