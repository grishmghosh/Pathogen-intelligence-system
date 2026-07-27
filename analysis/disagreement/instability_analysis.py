"""
Instability Analysis and Scoring for Pathogen Intelligence System.

Computes deterministic instability scores for individual models and
the ensemble, based on disagreement frequency, severity escalation,
and perturbation sensitivity.

Instability score range: 0 (stable) to 1 (highly unstable).

Architecture note:
    Scoring is separated from perturbation detection
    (``perturbation_disagreement``) and from trend generation
    (``disagreement_trends``).

Depends on:
    analysis.disagreement.perturbation_disagreement
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instability score weights (deterministic, tunable)
# ---------------------------------------------------------------------------

# Weight for disagreement frequency component
_W_FREQUENCY: float = 0.40

# Weight for severity escalation component
_W_ESCALATION: float = 0.35

# Weight for perturbation breadth component (how many pert types trigger it)
_W_BREADTH: float = 0.25


# ---------------------------------------------------------------------------
# Public API - per-model instability
# ---------------------------------------------------------------------------

def compute_model_instability(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute an instability profile for each model.

    For every model the function measures how often its predictions
    *change* relative to its own clean prediction when perturbations
    are applied.

    Returns a DataFrame with one row per model:

    * ``model_name``
    * ``total_observations``   - perturbed observations for this model
    * ``flip_count``           - times prediction changed from clean
    * ``flip_rate``            - flip_count / total_observations (0-1)
    * ``mean_severity_at_flip``- average severity rank when flips occur
    * ``instability_score``    - composite score (0-1)

    Args:
        predictions: Output of ``load_perturbation_predictions``.

    Returns:
        DataFrame sorted by ``instability_score`` descending.
    """
    from analysis.disagreement.perturbation_disagreement import _is_clean, _severity_rank

    if predictions.empty:
        raise ValueError("Predictions DataFrame is empty.")

    model_names = sorted(predictions["model_name"].unique())
    clean_mask = predictions["severity_level"].apply(_is_clean)
    clean_df = predictions[clean_mask]
    perturbed_df = predictions[~clean_mask]

    if clean_df.empty or perturbed_df.empty:
        logger.warning("Need both clean and perturbed predictions for instability analysis.")
        return pd.DataFrame(columns=[
            "model_name", "total_observations", "flip_count", "flip_rate",
            "mean_severity_at_flip", "instability_score",
        ])

    # Build clean reference: model -> sample -> predicted_class
    clean_ref: Dict[str, Dict[str, str]] = {}
    for _, row in clean_df.iterrows():
        mn = str(row["model_name"])
        sid = str(row["sample_id"])
        clean_ref.setdefault(mn, {})[sid] = str(row["predicted_class"])

    records: List[Dict] = []
    for model in model_names:
        model_clean = clean_ref.get(model, {})
        model_perturbed = perturbed_df[perturbed_df["model_name"] == model]

        total = 0
        flips = 0
        flip_severity_ranks: List[int] = []

        for _, row in model_perturbed.iterrows():
            sid = str(row["sample_id"])
            if sid not in model_clean:
                continue
            total += 1
            if str(row["predicted_class"]) != model_clean[sid]:
                flips += 1
                flip_severity_ranks.append(_severity_rank(str(row["severity_level"])))

        if total == 0:
            records.append({
                "model_name": model,
                "total_observations": 0,
                "flip_count": 0,
                "flip_rate": 0.0,
                "mean_severity_at_flip": None,
                "instability_score": 0.0,
            })
            continue

        flip_rate = flips / total
        mean_sev = float(np.mean(flip_severity_ranks)) if flip_severity_ranks else 0.0

        # Instability score: high flip_rate -> unstable,
        # flips at lower severity -> more unstable (inverted normalisation)
        max_sev_rank = max(_severity_rank(s) for s in predictions["severity_level"].unique()
                          if not _is_clean(s)) if not perturbed_df.empty else 1
        if max_sev_rank == 0:
            max_sev_rank = 1

        # Early-flip penalty: flips at low severity are worse
        early_flip_factor = 1.0 - (mean_sev / (max_sev_rank + 1)) if flip_severity_ranks else 0.0

        score = _W_FREQUENCY * flip_rate + _W_ESCALATION * early_flip_factor * flip_rate + _W_BREADTH * flip_rate
        score = float(np.clip(score, 0.0, 1.0))

        records.append({
            "model_name": model,
            "total_observations": total,
            "flip_count": flips,
            "flip_rate": round(flip_rate, 6),
            "mean_severity_at_flip": round(mean_sev, 4) if flip_severity_ranks else None,
            "instability_score": round(score, 6),
        })

    result = pd.DataFrame(records)
    result = result.sort_values("instability_score", ascending=False).reset_index(drop=True)
    logger.info("Model instability computed for %d models.", len(result))
    return result


# ---------------------------------------------------------------------------
# Public API - per-sample instability
# ---------------------------------------------------------------------------

def compute_sample_instability(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute instability for each sample across all models and perturbations.

    For each ``sample_id``:

    * ``total_perturbed_obs``  - perturbed (sample, pert, sev) observations
    * ``disagreement_obs``     - observations where models disagree
    * ``disagreement_rate``    - disagreement_obs / total_perturbed_obs
    * ``stability_breakpoint`` - earliest severity where disagreement occurs
    * ``instability_score``    - composite (0-1)

    Args:
        predictions: Output of ``load_perturbation_predictions``.

    Returns:
        DataFrame sorted by ``instability_score`` descending.
    """
    from analysis.disagreement.perturbation_disagreement import _is_clean, _severity_rank

    if predictions.empty:
        raise ValueError("Predictions DataFrame is empty.")

    model_names = sorted(predictions["model_name"].unique())
    if len(model_names) < 2:
        raise ValueError(f"At least 2 models required; found {len(model_names)}")

    perturbed = predictions[~predictions["severity_level"].apply(_is_clean)]
    if perturbed.empty:
        logger.warning("No perturbed data for sample instability.")
        return pd.DataFrame(columns=[
            "sample_id", "total_perturbed_obs", "disagreement_obs",
            "disagreement_rate", "stability_breakpoint", "instability_score",
        ])

    max_sev_rank = max(_severity_rank(s) for s in perturbed["severity_level"].unique()) or 1

    records: List[Dict] = []
    for sample_id, sample_grp in perturbed.groupby("sample_id"):
        total = 0
        disagree = 0
        earliest_break_rank = None

        for (pt, sl), obs_grp in sample_grp.groupby(["perturbation_type", "severity_level"]):
            if len(obs_grp["model_name"].unique()) < 2:
                continue
            total += 1
            if obs_grp["predicted_class"].nunique() > 1:
                disagree += 1
                rank = _severity_rank(str(sl))
                if earliest_break_rank is None or rank < earliest_break_rank:
                    earliest_break_rank = rank

        if total == 0:
            continue

        dis_rate = disagree / total

        # Early breakpoint penalty
        early_factor = (1.0 - earliest_break_rank / (max_sev_rank + 1)) if earliest_break_rank is not None else 0.0

        # Breadth: fraction of perturbation types causing disagreement
        pert_types = sample_grp["perturbation_type"].unique()
        dis_pert_types = set()
        for (pt, sl), obs_grp in sample_grp.groupby(["perturbation_type", "severity_level"]):
            if len(obs_grp["model_name"].unique()) < 2:
                continue
            if obs_grp["predicted_class"].nunique() > 1:
                dis_pert_types.add(pt)
        breadth = len(dis_pert_types) / len(pert_types) if len(pert_types) > 0 else 0.0

        score = _W_FREQUENCY * dis_rate + _W_ESCALATION * early_factor * dis_rate + _W_BREADTH * breadth
        score = float(np.clip(score, 0.0, 1.0))

        # Map breakpoint rank back to label
        break_label = None
        if earliest_break_rank is not None:
            for sev_label in sample_grp["severity_level"].unique():
                if _severity_rank(str(sev_label)) == earliest_break_rank:
                    break_label = str(sev_label)
                    break

        records.append({
            "sample_id": str(sample_id),
            "total_perturbed_obs": total,
            "disagreement_obs": disagree,
            "disagreement_rate": round(dis_rate, 6),
            "stability_breakpoint": break_label,
            "instability_score": round(score, 6),
        })

    if not records:
        return pd.DataFrame(columns=[
            "sample_id", "total_perturbed_obs", "disagreement_obs",
            "disagreement_rate", "stability_breakpoint", "instability_score",
        ])

    result = pd.DataFrame(records)
    result = result.sort_values("instability_score", ascending=False).reset_index(drop=True)
    logger.info("Sample instability computed for %d samples.", len(result))
    return result


# ---------------------------------------------------------------------------
# Public API - ensemble instability summary
# ---------------------------------------------------------------------------

def compute_instability_summary(
    model_instability: pd.DataFrame,
    sample_instability: pd.DataFrame,
    severity_rates: pd.DataFrame,
) -> Dict:
    """
    Aggregate instability metrics into a single summary dictionary.

    Args:
        model_instability:  Output of :func:`compute_model_instability`.
        sample_instability: Output of :func:`compute_sample_instability`.
        severity_rates:     Output of
            :func:`perturbation_disagreement.compute_severity_disagreement_rates`.

    Returns:
        Dictionary with:

        * ``mean_model_instability``   - average model instability score
        * ``max_model_instability``    - worst model instability score
        * ``most_unstable_model``      - name of most unstable model
        * ``mean_sample_instability``  - average sample instability score
        * ``most_unstable_samples``    - top-5 sample IDs by instability
        * ``severity_escalation``      - list of {severity, disagreement_rate}
        * ``overall_instability``      - single ensemble-level score (0-1)
    """
    # Model stats
    if model_instability.empty or "instability_score" not in model_instability.columns:
        mean_m = 0.0; max_m = 0.0; worst_model = None
    else:
        mean_m = float(model_instability["instability_score"].mean())
        max_m = float(model_instability["instability_score"].max())
        worst_model = str(model_instability.iloc[0]["model_name"])

    # Sample stats
    if sample_instability.empty or "instability_score" not in sample_instability.columns:
        mean_s = 0.0; top_samples = []
    else:
        mean_s = float(sample_instability["instability_score"].mean())
        top_samples = sample_instability.head(5)["sample_id"].tolist()

    # Severity escalation
    escalation = []
    if not severity_rates.empty and "disagreement_rate" in severity_rates.columns:
        for _, row in severity_rates.iterrows():
            escalation.append({
                "severity_level": str(row["severity_level"]),
                "disagreement_rate": round(float(row["disagreement_rate"]), 6),
            })

    overall = round((mean_m + mean_s) / 2.0, 6) if (mean_m + mean_s) > 0 else 0.0

    return {
        "mean_model_instability": round(mean_m, 6),
        "max_model_instability": round(max_m, 6),
        "most_unstable_model": worst_model,
        "mean_sample_instability": round(mean_s, 6),
        "most_unstable_samples": top_samples,
        "severity_escalation": escalation,
        "overall_instability": overall,
    }
