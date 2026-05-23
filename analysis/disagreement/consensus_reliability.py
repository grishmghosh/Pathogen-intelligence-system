"""
Consensus Reliability Scoring for Pathogen Intelligence System (Step 4).

Computes a normalised consensus reliability score that combines signals
from all prior analysis steps to determine how much a given model
consensus should be trusted.

Score components:
    - Agreement strength          (from Step 1 agreement metrics)
    - Confidence consistency      (from Step 2 confidence analysis)
    - Perturbation stability      (from Step 3 consensus stability)
    - Disagreement severity       (from Step 2 severity classification)
    - Instability score           (from Step 3 instability analysis)

Consensus reliability score range:
    0 → unreliable consensus
    1 → highly reliable consensus

Architecture note:
    - Scoring logic is separated from trust labelling (``trust_analysis``)
      and false consensus detection (``false_consensus_detection``).
    - All computation is deterministic and rule-based.
    - No probabilistic or Bayesian logic.

Depends on:
    analysis.disagreement.agreement_metrics          (Step 1)
    analysis.disagreement.confidence_disagreement    (Step 2)
    analysis.disagreement.disagreement_severity      (Step 2)
    analysis.disagreement.perturbation_disagreement  (Step 3)
    analysis.disagreement.instability_analysis       (Step 3)
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component weights (deterministic, tunable)
# ---------------------------------------------------------------------------

# Weight for agreement strength component
_W_AGREEMENT: float = 0.25

# Weight for confidence consistency component
_W_CONFIDENCE: float = 0.20

# Weight for perturbation stability component
_W_PERTURBATION: float = 0.25

# Weight for disagreement severity component (inverted: low severity = good)
_W_SEVERITY: float = 0.15

# Weight for instability component (inverted: low instability = good)
_W_INSTABILITY: float = 0.15

# Severity ordering for normalisation
_SEVERITY_ORDER = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
_MAX_SEVERITY_RANK = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(val, default: float = 0.0) -> float:
    """Coerce a value to a finite float, returning *default* on failure."""
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _normalise_agreement_strength(agreement_rate: float) -> float:
    """
    Map agreement rate (0–1) directly to a component score.

    1.0 = perfect agreement → component = 1.0
    0.0 = no agreement     → component = 0.0
    """
    return float(np.clip(agreement_rate, 0.0, 1.0))


def _normalise_confidence_consistency(
    confidence_gap: float,
    confidence_mean: float,
) -> float:
    """
    Compute a confidence consistency score.

    Low confidence gap AND high mean confidence → score near 1.0
    High confidence gap OR low mean confidence  → score near 0.0
    """
    # gap_component: small gap is good → invert
    gap_component = 1.0 - float(np.clip(confidence_gap, 0.0, 1.0))

    # mean_component: high mean confidence is good
    mean_component = float(np.clip(confidence_mean, 0.0, 1.0))

    return 0.6 * gap_component + 0.4 * mean_component


def _normalise_perturbation_stability(
    perturbation_survival_rate: float,
) -> float:
    """
    Map perturbation survival rate to a stability component.

    survival_rate = fraction of perturbations where consensus held.
    1.0 = survived all perturbations → component = 1.0
    0.0 = collapsed on every perturbation → component = 0.0
    """
    return float(np.clip(perturbation_survival_rate, 0.0, 1.0))


def _normalise_severity(severity_label: str) -> float:
    """
    Map severity label to a *reliability* component (inverted severity).

    "low"      → 1.0  (reliable)
    "moderate" → 0.67
    "high"     → 0.33
    "critical" → 0.0  (unreliable)
    """
    rank = _SEVERITY_ORDER.get(str(severity_label).lower().strip(), 1)
    if _MAX_SEVERITY_RANK == 0:
        return 1.0
    return 1.0 - (rank / _MAX_SEVERITY_RANK)


def _normalise_instability(instability_score: float) -> float:
    """
    Map instability score (0-1) to a *reliability* component (inverted).

    0.0 instability → 1.0 reliability
    1.0 instability → 0.0 reliability
    """
    return 1.0 - float(np.clip(instability_score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API – per-sample consensus reliability scoring
# ---------------------------------------------------------------------------

def compute_consensus_reliability(
    predictions: pd.DataFrame,
    confidence_gaps: Optional[pd.DataFrame] = None,
    sample_severity: Optional[pd.DataFrame] = None,
    consensus_stability: Optional[pd.DataFrame] = None,
    sample_instability: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute a per-sample consensus reliability score.

    The score combines five normalised components from Steps 1–3:

    1. **Agreement strength** – whether all models agree on the sample.
    2. **Confidence consistency** – how close confidence values are
       across agreeing models (low gap + high mean = good).
    3. **Perturbation stability** – fraction of perturbation conditions
       where consensus survived.
    4. **Severity** – worst-case severity classification (inverted).
    5. **Instability** – sample instability score (inverted).

    Missing components are gracefully handled by using neutral defaults
    (0.5) so the score remains valid even with partial data.

    Args:
        predictions: Normalised predictions DataFrame (Step 1).
        confidence_gaps: Per-sample confidence gaps (Step 2,
            output of ``compute_confidence_gaps``).  Optional.
        sample_severity: Per-sample severity classification (Step 2,
            output of ``classify_sample_severity``).  Optional.
        consensus_stability: Consensus stability tracking (Step 3,
            output of ``track_consensus_stability``).  Optional.
        sample_instability: Per-sample instability scores (Step 3,
            output of ``compute_sample_instability``).  Optional.

    Returns:
        DataFrame with columns:

        * ``sample_id``
        * ``agreement_strength``          – component score (0-1)
        * ``confidence_consistency``      – component score (0-1)
        * ``perturbation_stability``      – component score (0-1)
        * ``severity_reliability``        – component score (0-1)
        * ``instability_reliability``     – component score (0-1)
        * ``consensus_reliability_score`` – weighted composite (0-1)

    Raises:
        ValueError: If predictions is empty.
    """
    if predictions.empty:
        raise ValueError("Predictions DataFrame is empty.")

    model_names = sorted(predictions["model_name"].unique())
    if len(model_names) < 2:
        raise ValueError(
            f"At least 2 models required; found {len(model_names)}: {model_names}"
        )

    # --- Build per-sample agreement lookup ---
    pivot = predictions.pivot_table(
        index="sample_id",
        columns="model_name",
        values="predicted_class",
        aggfunc="first",
    )
    complete = pivot.dropna()

    if complete.empty:
        logger.warning("No samples with predictions from all models.")
        return _empty_reliability_df()

    all_agree = complete.apply(lambda r: r.nunique() == 1, axis=1)

    # --- Build lookup dictionaries for optional data ---
    conf_gap_map: Dict[str, Dict] = {}
    if confidence_gaps is not None and not confidence_gaps.empty:
        for _, row in confidence_gaps.iterrows():
            sid = str(row.get("sample_id", ""))
            if sid:
                conf_gap_map[sid] = {
                    "gap": _safe_float(row.get("confidence_gap"), 0.5),
                    "mean": _safe_float(row.get("confidence_mean"), 0.5),
                }

    severity_map: Dict[str, str] = {}
    if sample_severity is not None and not sample_severity.empty:
        if "severity" in sample_severity.columns:
            for _, row in sample_severity.iterrows():
                sid = str(row.get("sample_id", ""))
                if sid:
                    severity_map[sid] = str(row.get("severity", "moderate"))

    stability_map: Dict[str, float] = {}
    if consensus_stability is not None and not consensus_stability.empty:
        # Compute survival rate per sample: fraction of severity levels
        # where models_agree is True
        for sid, grp in consensus_stability.groupby("sample_id"):
            total = len(grp)
            agreed = grp["models_agree"].sum() if "models_agree" in grp.columns else 0
            survival = agreed / total if total > 0 else 0.5
            stability_map[str(sid)] = float(survival)

    instability_map: Dict[str, float] = {}
    if sample_instability is not None and not sample_instability.empty:
        if "instability_score" in sample_instability.columns:
            for _, row in sample_instability.iterrows():
                sid = str(row.get("sample_id", ""))
                if sid:
                    instability_map[sid] = _safe_float(
                        row.get("instability_score"), 0.5
                    )

    # --- Compute per-sample scores ---
    records: List[Dict] = []
    for sample_id in complete.index:
        sid = str(sample_id)

        # Component 1: Agreement strength
        agrees = bool(all_agree.loc[sample_id])
        agreement_component = 1.0 if agrees else 0.0

        # Component 2: Confidence consistency
        if sid in conf_gap_map:
            confidence_component = _normalise_confidence_consistency(
                conf_gap_map[sid]["gap"],
                conf_gap_map[sid]["mean"],
            )
        else:
            confidence_component = 0.5  # neutral default

        # Component 3: Perturbation stability
        if sid in stability_map:
            perturbation_component = _normalise_perturbation_stability(
                stability_map[sid]
            )
        else:
            perturbation_component = 0.5  # neutral default

        # Component 4: Severity (inverted)
        if sid in severity_map:
            severity_component = _normalise_severity(severity_map[sid])
        else:
            severity_component = 0.5 if not agrees else 1.0

        # Component 5: Instability (inverted)
        if sid in instability_map:
            instability_component = _normalise_instability(
                instability_map[sid]
            )
        else:
            instability_component = 0.5  # neutral default

        # Weighted composite
        score = (
            _W_AGREEMENT * agreement_component
            + _W_CONFIDENCE * confidence_component
            + _W_PERTURBATION * perturbation_component
            + _W_SEVERITY * severity_component
            + _W_INSTABILITY * instability_component
        )
        score = float(np.clip(score, 0.0, 1.0))

        records.append({
            "sample_id": sid,
            "agreement_strength": round(agreement_component, 6),
            "confidence_consistency": round(confidence_component, 6),
            "perturbation_stability": round(perturbation_component, 6),
            "severity_reliability": round(severity_component, 6),
            "instability_reliability": round(instability_component, 6),
            "consensus_reliability_score": round(score, 6),
        })

    if not records:
        return _empty_reliability_df()

    result = pd.DataFrame(records)
    result = result.sort_values(
        "consensus_reliability_score", ascending=True
    ).reset_index(drop=True)

    logger.info(
        "Consensus reliability computed for %d samples "
        "(min=%.4f, max=%.4f, mean=%.4f).",
        len(result),
        result["consensus_reliability_score"].min(),
        result["consensus_reliability_score"].max(),
        result["consensus_reliability_score"].mean(),
    )
    return result


def _empty_reliability_df() -> pd.DataFrame:
    """Return an empty DataFrame with the reliability schema."""
    return pd.DataFrame(columns=[
        "sample_id", "agreement_strength", "confidence_consistency",
        "perturbation_stability", "severity_reliability",
        "instability_reliability", "consensus_reliability_score",
    ])


# ---------------------------------------------------------------------------
# Public API – consensus reliability summary
# ---------------------------------------------------------------------------

def compute_reliability_summary(
    reliability: pd.DataFrame,
) -> Dict:
    """
    Aggregate consensus reliability scores into a summary dictionary.

    Args:
        reliability: Output of :func:`compute_consensus_reliability`.

    Returns:
        Dictionary with:

        * ``total_samples``              – number of scored samples
        * ``mean_reliability``           – average reliability score
        * ``min_reliability``            – lowest reliability score
        * ``max_reliability``            – highest reliability score
        * ``median_reliability``         – median reliability score
        * ``reliable_count``             – samples with score >= 0.7
        * ``unreliable_count``           – samples with score < 0.3
        * ``component_means``            – average of each component
        * ``weakest_component``          – component with lowest average
    """
    if reliability.empty or "consensus_reliability_score" not in reliability.columns:
        return {
            "total_samples": 0,
            "mean_reliability": 0.0,
            "min_reliability": 0.0,
            "max_reliability": 0.0,
            "median_reliability": 0.0,
            "reliable_count": 0,
            "unreliable_count": 0,
            "component_means": {},
            "weakest_component": None,
        }

    scores = reliability["consensus_reliability_score"]

    # Component means
    component_cols = [
        "agreement_strength", "confidence_consistency",
        "perturbation_stability", "severity_reliability",
        "instability_reliability",
    ]
    component_means: Dict[str, float] = {}
    for col in component_cols:
        if col in reliability.columns:
            component_means[col] = round(float(reliability[col].mean()), 6)

    weakest = min(component_means, key=component_means.get) if component_means else None

    return {
        "total_samples": int(len(reliability)),
        "mean_reliability": round(float(scores.mean()), 6),
        "min_reliability": round(float(scores.min()), 6),
        "max_reliability": round(float(scores.max()), 6),
        "median_reliability": round(float(scores.median()), 6),
        "reliable_count": int((scores >= 0.7).sum()),
        "unreliable_count": int((scores < 0.3).sum()),
        "component_means": component_means,
        "weakest_component": weakest,
    }


# ---------------------------------------------------------------------------
# Public API – consensus breakdown analysis
# ---------------------------------------------------------------------------

def compute_consensus_breakdown(
    consensus_stability: pd.DataFrame,
    sample_instability: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Analyse how and where consensus breaks down for each sample.

    For each sample that experiences a consensus collapse, this function
    identifies:

    * ``collapse_trigger``       – the perturbation type causing first collapse
    * ``collapse_severity``      – severity level of first collapse
    * ``stability_duration``     – number of severity steps before collapse
    * ``escalation_speed``       – 1 / stability_duration (how quickly collapse occurs)
    * ``instability_score``      – from Step 3 (if available)

    Args:
        consensus_stability: Output of
            :func:`perturbation_disagreement.track_consensus_stability`.
        sample_instability: Output of
            :func:`instability_analysis.compute_sample_instability`.
            Optional.

    Returns:
        DataFrame with one row per sample that experienced collapse,
        sorted by ``escalation_speed`` descending.
    """
    if consensus_stability.empty:
        logger.warning("Empty consensus stability data - no breakdown analysis possible.")
        return _empty_breakdown_df()

    # Ensure required columns exist
    required = {"sample_id", "severity_level", "models_agree"}
    missing = required - set(consensus_stability.columns)
    if missing:
        logger.warning("Missing columns for breakdown analysis: %s", missing)
        return _empty_breakdown_df()

    # Build instability lookup
    instability_map: Dict[str, float] = {}
    if sample_instability is not None and not sample_instability.empty:
        if "instability_score" in sample_instability.columns:
            for _, row in sample_instability.iterrows():
                sid = str(row.get("sample_id", ""))
                if sid:
                    instability_map[sid] = _safe_float(
                        row.get("instability_score"), 0.0
                    )

    records: List[Dict] = []
    for sample_id, grp in consensus_stability.groupby("sample_id"):
        sid = str(sample_id)

        # Sort by severity rank
        if "severity_rank" in grp.columns:
            grp_sorted = grp.sort_values("severity_rank")
        else:
            grp_sorted = grp

        # Find first collapse point
        collapse_row = None
        steps_before_collapse = 0

        for idx, (_, row) in enumerate(grp_sorted.iterrows()):
            if not bool(row.get("models_agree", True)):
                collapse_row = row
                steps_before_collapse = idx
                break

        if collapse_row is None:
            # No collapse for this sample
            continue

        # Extract collapse trigger from perturbation type if available
        collapse_trigger = "unknown"
        if "perturbation_type" in collapse_row.index:
            collapse_trigger = str(collapse_row["perturbation_type"])

        collapse_sev = str(collapse_row.get("severity_level", "unknown"))

        # Escalation speed: faster collapse = higher speed
        total_steps = len(grp_sorted)
        stability_duration = steps_before_collapse
        escalation_speed = (
            1.0 / (stability_duration + 1)
            if stability_duration >= 0
            else 1.0
        )

        records.append({
            "sample_id": sid,
            "collapse_trigger": collapse_trigger,
            "collapse_severity": collapse_sev,
            "stability_duration": stability_duration,
            "total_severity_steps": total_steps,
            "escalation_speed": round(escalation_speed, 6),
            "instability_score": round(
                instability_map.get(sid, 0.0), 6
            ),
        })

    if not records:
        logger.info("No consensus breakdowns detected.")
        return _empty_breakdown_df()

    result = pd.DataFrame(records)
    result = result.sort_values(
        "escalation_speed", ascending=False
    ).reset_index(drop=True)

    logger.info(
        "Consensus breakdown analysed for %d samples.", len(result)
    )
    return result


def _empty_breakdown_df() -> pd.DataFrame:
    """Return an empty DataFrame with the breakdown schema."""
    return pd.DataFrame(columns=[
        "sample_id", "collapse_trigger", "collapse_severity",
        "stability_duration", "total_severity_steps",
        "escalation_speed", "instability_score",
    ])


# ---------------------------------------------------------------------------
# Public API – consensus consistency metrics
# ---------------------------------------------------------------------------

def compute_consensus_consistency_metrics(
    predictions: pd.DataFrame,
    confidence_gaps: Optional[pd.DataFrame] = None,
    consensus_stability: Optional[pd.DataFrame] = None,
    model_instability: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Compute ensemble-level consensus consistency metrics.

    Metrics:
        * ``agreement_persistence``    – fraction of samples where all
          models agree (same as Step 1 agreement rate, cached here).
        * ``confidence_stability``     – 1 - mean(confidence_gap),
          how consistent confidence is across models.
        * ``perturbation_survival_rate`` – fraction of perturbation
          observations where consensus held.
        * ``cross_model_consistency``  – 1 - std(model_instability),
          how uniformly stable/unstable models are.
        * ``overall_consistency``      – weighted average of above.

    Args:
        predictions: Normalised predictions DataFrame.
        confidence_gaps: Per-sample confidence gaps (Step 2).  Optional.
        consensus_stability: Consensus stability tracking (Step 3).  Optional.
        model_instability: Per-model instability (Step 3).  Optional.

    Returns:
        Dictionary with all metrics.
    """
    result: Dict = {}

    # --- Agreement persistence ---
    if not predictions.empty:
        pivot = predictions.pivot_table(
            index="sample_id", columns="model_name",
            values="predicted_class", aggfunc="first",
        )
        complete = pivot.dropna()
        if not complete.empty:
            all_agree = complete.apply(lambda r: r.nunique() == 1, axis=1)
            result["agreement_persistence"] = round(
                float(all_agree.mean()), 6
            )
        else:
            result["agreement_persistence"] = 0.0
    else:
        result["agreement_persistence"] = 0.0

    # --- Confidence stability ---
    if confidence_gaps is not None and not confidence_gaps.empty:
        if "confidence_gap" in confidence_gaps.columns:
            mean_gap = float(confidence_gaps["confidence_gap"].mean())
            result["confidence_stability"] = round(
                1.0 - np.clip(mean_gap, 0.0, 1.0), 6
            )
        else:
            result["confidence_stability"] = 0.5
    else:
        result["confidence_stability"] = 0.5

    # --- Perturbation survival rate ---
    if consensus_stability is not None and not consensus_stability.empty:
        if "models_agree" in consensus_stability.columns:
            total_obs = len(consensus_stability)
            survived = int(consensus_stability["models_agree"].sum())
            result["perturbation_survival_rate"] = round(
                survived / total_obs if total_obs > 0 else 0.0, 6
            )
        else:
            result["perturbation_survival_rate"] = 0.5
    else:
        result["perturbation_survival_rate"] = 0.5

    # --- Cross-model consistency ---
    if model_instability is not None and not model_instability.empty:
        if "instability_score" in model_instability.columns:
            scores = model_instability["instability_score"].dropna()
            if len(scores) > 1:
                std = float(scores.std())
                result["cross_model_consistency"] = round(
                    1.0 - np.clip(std, 0.0, 1.0), 6
                )
            else:
                result["cross_model_consistency"] = 1.0
        else:
            result["cross_model_consistency"] = 0.5
    else:
        result["cross_model_consistency"] = 0.5

    # --- Overall consistency ---
    components = [
        result["agreement_persistence"],
        result["confidence_stability"],
        result["perturbation_survival_rate"],
        result["cross_model_consistency"],
    ]
    result["overall_consistency"] = round(float(np.mean(components)), 6)

    logger.info(
        "Consensus consistency metrics: overall=%.4f",
        result["overall_consistency"],
    )
    return result


# ---------------------------------------------------------------------------
# Public API – model trust contribution
# ---------------------------------------------------------------------------

def compute_model_trust_contribution(
    model_instability: pd.DataFrame,
    predictions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Analyse how much each model contributes to or undermines consensus
    trust.

    For each model:

    * ``model_name``
    * ``instability_score``      – from Step 3
    * ``flip_rate``              – from Step 3
    * ``trust_contribution``     – 1 - instability_score (0-1)
    * ``trust_rank``             – rank among models (1 = most trustworthy)
    * ``stability_label``        – "stabiliser", "neutral", or "destabiliser"

    A model is a *stabiliser* if its trust_contribution >= 0.7,
    a *destabiliser* if < 0.3, and *neutral* otherwise.

    Args:
        model_instability: Output of
            :func:`instability_analysis.compute_model_instability`.
        predictions: Normalised predictions (optional, for additional context).

    Returns:
        DataFrame sorted by ``trust_contribution`` descending.
    """
    if model_instability.empty:
        logger.warning("Empty model instability data - no trust contribution analysis.")
        return pd.DataFrame(columns=[
            "model_name", "instability_score", "flip_rate",
            "trust_contribution", "trust_rank", "stability_label",
        ])

    required = {"model_name", "instability_score"}
    missing = required - set(model_instability.columns)
    if missing:
        logger.warning("Missing columns for trust contribution: %s", missing)
        return pd.DataFrame(columns=[
            "model_name", "instability_score", "flip_rate",
            "trust_contribution", "trust_rank", "stability_label",
        ])

    records: List[Dict] = []
    for _, row in model_instability.iterrows():
        inst = _safe_float(row.get("instability_score"), 0.5)
        flip = _safe_float(row.get("flip_rate"), 0.0)
        contribution = 1.0 - inst

        if contribution >= 0.7:
            label = "stabiliser"
        elif contribution < 0.3:
            label = "destabiliser"
        else:
            label = "neutral"

        records.append({
            "model_name": str(row["model_name"]),
            "instability_score": round(inst, 6),
            "flip_rate": round(flip, 6),
            "trust_contribution": round(contribution, 6),
            "stability_label": label,
        })

    result = pd.DataFrame(records)
    result = result.sort_values(
        "trust_contribution", ascending=False
    ).reset_index(drop=True)

    # Add rank
    result["trust_rank"] = range(1, len(result) + 1)

    logger.info(
        "Model trust contribution: %d models analysed.",
        len(result),
    )
    return result


def compute_model_trust_summary(
    trust_contributions: pd.DataFrame,
) -> Dict:
    """
    Aggregate model trust contributions into a summary.

    Args:
        trust_contributions: Output of :func:`compute_model_trust_contribution`.

    Returns:
        Dictionary with:

        * ``most_trusted_model``    – model with highest trust contribution
        * ``least_trusted_model``   – model with lowest trust contribution
        * ``trust_spread``          – max - min trust contribution
        * ``stabiliser_count``      – number of stabiliser models
        * ``destabiliser_count``    – number of destabiliser models
        * ``neutral_count``         – number of neutral models
        * ``model_rankings``        – ordered list of {model, contribution, label}
    """
    if trust_contributions.empty:
        return {
            "most_trusted_model": None,
            "least_trusted_model": None,
            "trust_spread": 0.0,
            "stabiliser_count": 0,
            "destabiliser_count": 0,
            "neutral_count": 0,
            "model_rankings": [],
        }

    tc = trust_contributions.sort_values(
        "trust_contribution", ascending=False
    ).reset_index(drop=True)

    rankings = []
    for _, row in tc.iterrows():
        rankings.append({
            "model": str(row["model_name"]),
            "trust_contribution": round(float(row["trust_contribution"]), 6),
            "stability_label": str(row["stability_label"]),
        })

    labels = tc["stability_label"].value_counts().to_dict()

    return {
        "most_trusted_model": str(tc.iloc[0]["model_name"]),
        "least_trusted_model": str(tc.iloc[-1]["model_name"]),
        "trust_spread": round(
            float(tc["trust_contribution"].max() - tc["trust_contribution"].min()),
            6,
        ),
        "stabiliser_count": labels.get("stabiliser", 0),
        "destabiliser_count": labels.get("destabiliser", 0),
        "neutral_count": labels.get("neutral", 0),
        "model_rankings": rankings,
    }
