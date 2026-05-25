"""
False Consensus Detection for Pathogen Intelligence System (Step 4).

Detects cases where model consensus appears to exist but is unreliable:

1. **Fragile consensus** – models agree on clean input but collapse
   under mild perturbation.
2. **False consensus** – agreement exists but confidence patterns
   indicate shallow/unreliable agreement.
3. **Unstable agreement** – consensus oscillates across perturbation
   severity levels.

Architecture note:
    - Detection logic is separated from scoring (``consensus_reliability``)
      and trust labelling (``trust_analysis``).
    - Export is separated from detection.
    - All logic is deterministic and rule-based.

Depends on:
    analysis.disagreement.perturbation_disagreement  (Step 3)
    analysis.disagreement.instability_analysis       (Step 3)
    analysis.disagreement.confidence_disagreement    (Step 2)
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection thresholds (deterministic, tunable)
# ---------------------------------------------------------------------------

# Confidence gap above this within an agreeing group → suspicious
SUSPICIOUS_CONFIDENCE_GAP: float = 0.30

# Mean confidence below this within an agreeing group → low confidence agreement
LOW_CONFIDENCE_MEAN: float = 0.50

# If consensus collapses at severity rank <= this, it is fragile
FRAGILE_COLLAPSE_MAX_RANK: int = 1  # "mild" / "low"

# If consensus oscillates (agree → disagree → agree) more than this
# fraction of steps, it is unstable
UNSTABLE_OSCILLATION_THRESHOLD: float = 0.3

# Severity rank mapping (re-declared to avoid tight coupling)
_SEVERITY_RANK_MAP: Dict[str, int] = {
    "clean": 0, "none": 0, "original": 0,
    "mild": 1, "low": 1, "light": 1,
    "moderate": 2, "medium": 2, "mid": 2,
    "severe": 3, "high": 3, "heavy": 3,
    "extreme": 4, "very_high": 4, "critical": 4,
}

_CLEAN_LABELS = {"clean", "none", "original"}


def _is_clean(label: str) -> bool:
    return str(label).lower().strip() in _CLEAN_LABELS


def _severity_rank(label: str) -> int:
    return _SEVERITY_RANK_MAP.get(str(label).lower().strip(), 99)


def _safe_float(val, default: float = 0.0) -> float:
    """Coerce to finite float or return default."""
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Public API – fragile consensus detection
# ---------------------------------------------------------------------------

def detect_fragile_consensus(
    predictions: pd.DataFrame,
    consensus_stability: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Detect samples with fragile consensus: all models agree on clean
    input but consensus collapses under mild perturbation.

    A consensus is considered *fragile* when:
    1. All models agree on the clean version of the sample.
    2. At least one perturbation at severity_rank <= ``FRAGILE_COLLAPSE_MAX_RANK``
       causes disagreement.

    Args:
        predictions: Normalised predictions with perturbation metadata
            (output of ``load_perturbation_predictions``).
        consensus_stability: Output of ``track_consensus_stability``
            (Step 3).  If provided, used to accelerate detection via
            pre-computed breakpoints.  Optional.

    Returns:
        DataFrame with columns:

        * ``sample_id``
        * ``clean_agreement``           – True (by definition for fragile)
        * ``collapse_severity``         – severity label of first collapse
        * ``collapse_severity_rank``    – numeric rank of collapse severity
        * ``flag``                      – ``"fragile_consensus"``

        Only samples with fragile consensus are included.
    """
    if predictions.empty:
        raise ValueError("Predictions DataFrame is empty.")

    # --- Strategy 1: use pre-computed consensus stability ---
    if consensus_stability is not None and not consensus_stability.empty:
        return _detect_fragile_from_stability(consensus_stability)

    # --- Strategy 2: compute from raw predictions ---
    return _detect_fragile_from_predictions(predictions)


def _detect_fragile_from_stability(
    consensus_stability: pd.DataFrame,
) -> pd.DataFrame:
    """Detect fragile consensus using pre-computed stability tracking."""
    records: List[Dict] = []

    for sample_id, grp in consensus_stability.groupby("sample_id"):
        sid = str(sample_id)

        if "severity_rank" in grp.columns:
            grp_sorted = grp.sort_values("severity_rank")
        else:
            grp_sorted = grp

        # Check if clean agreement exists
        clean_rows = grp_sorted[
            grp_sorted["severity_level"].apply(_is_clean)
        ] if "severity_level" in grp_sorted.columns else pd.DataFrame()

        if not clean_rows.empty:
            clean_agrees = bool(clean_rows.iloc[0].get("models_agree", False))
        else:
            # If no explicit clean row, check if there's a breakpoint
            # after the first row
            clean_agrees = bool(grp_sorted.iloc[0].get("models_agree", False))

        if not clean_agrees:
            continue  # No clean agreement → not fragile consensus

        # Find first collapse
        for _, row in grp_sorted.iterrows():
            sev = str(row.get("severity_level", "unknown"))
            if _is_clean(sev):
                continue
            if not bool(row.get("models_agree", True)):
                sev_rank = _severity_rank(sev)
                if sev_rank <= FRAGILE_COLLAPSE_MAX_RANK:
                    records.append({
                        "sample_id": sid,
                        "clean_agreement": True,
                        "collapse_severity": sev,
                        "collapse_severity_rank": sev_rank,
                        "flag": "fragile_consensus",
                    })
                break  # Only need first collapse

    if not records:
        logger.info("No fragile consensus detected.")
        return _empty_fragile_df()

    result = pd.DataFrame(records)
    result = result.sort_values("sample_id").reset_index(drop=True)
    logger.info("Fragile consensus detected for %d samples.", len(result))
    return result


def _detect_fragile_from_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Detect fragile consensus from raw predictions."""
    model_names = sorted(predictions["model_name"].unique())
    if len(model_names) < 2:
        return _empty_fragile_df()

    # Separate clean vs perturbed
    clean_mask = predictions["severity_level"].apply(_is_clean) \
        if "severity_level" in predictions.columns else pd.Series(False, index=predictions.index)
    clean_df = predictions[clean_mask]
    perturbed_df = predictions[~clean_mask]

    if clean_df.empty or perturbed_df.empty:
        return _empty_fragile_df()

    # Find samples with clean agreement
    clean_pivot = clean_df.pivot_table(
        index="sample_id", columns="model_name",
        values="predicted_class", aggfunc="first",
    )
    clean_complete = clean_pivot.dropna()
    if clean_complete.empty:
        return _empty_fragile_df()

    clean_agrees = clean_complete.apply(lambda r: r.nunique() == 1, axis=1)
    agreed_ids = set(clean_agrees[clean_agrees].index)

    if not agreed_ids:
        return _empty_fragile_df()

    # Check perturbed for collapses at mild severity
    records: List[Dict] = []
    for sample_id in agreed_ids:
        sid = str(sample_id)
        sample_perturbed = perturbed_df[perturbed_df["sample_id"] == sample_id]

        if sample_perturbed.empty:
            continue

        # Group by severity level, check for disagreement at mild levels
        for sev_label, sev_grp in sample_perturbed.groupby("severity_level"):
            if len(sev_grp["model_name"].unique()) < 2:
                continue
            sev_rank = _severity_rank(str(sev_label))
            if sev_rank > FRAGILE_COLLAPSE_MAX_RANK:
                continue  # Only check mild levels

            if sev_grp["predicted_class"].nunique() > 1:
                records.append({
                    "sample_id": sid,
                    "clean_agreement": True,
                    "collapse_severity": str(sev_label),
                    "collapse_severity_rank": sev_rank,
                    "flag": "fragile_consensus",
                })
                break  # One collapse is enough

    if not records:
        logger.info("No fragile consensus detected.")
        return _empty_fragile_df()

    result = pd.DataFrame(records)
    result = result.sort_values("sample_id").reset_index(drop=True)
    logger.info("Fragile consensus detected for %d samples.", len(result))
    return result


def _empty_fragile_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "sample_id", "clean_agreement", "collapse_severity",
        "collapse_severity_rank", "flag",
    ])


# ---------------------------------------------------------------------------
# Public API – false consensus detection
# ---------------------------------------------------------------------------

def detect_false_consensus(
    predictions: pd.DataFrame,
    confidence_gaps: Optional[pd.DataFrame] = None,
    consensus_stability: Optional[pd.DataFrame] = None,
    sample_instability: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Detect samples where apparent agreement is misleading.

    A consensus is considered *false* when models agree on the
    prediction class BUT one or more of these red flags is present:

    1. **High confidence gap** – models agree on the class but their
       confidence values are widely spread (gap > ``SUSPICIOUS_CONFIDENCE_GAP``).
    2. **Low mean confidence** – all models agree but none is particularly
       confident (mean < ``LOW_CONFIDENCE_MEAN``).
    3. **Fragile under perturbation** – agreement collapses at mild
       perturbation (detected via fragile consensus).
    4. **High instability** – the sample has a high instability score
       despite clean agreement.

    Args:
        predictions: Normalised predictions DataFrame.
        confidence_gaps: Per-sample confidence gaps (Step 2).  Optional.
        consensus_stability: Consensus stability tracking (Step 3).  Optional.
        sample_instability: Per-sample instability (Step 3).  Optional.

    Returns:
        DataFrame with columns:

        * ``sample_id``
        * ``clean_agreement``           – True (by definition)
        * ``flags``                     – list of flag strings
        * ``flag_count``                – number of flags triggered
        * ``confidence_gap``            – gap value (or NaN)
        * ``confidence_mean``           – mean value (or NaN)
        * ``instability_score``         – from Step 3 (or NaN)
        * ``false_consensus_severity``  – "mild", "moderate", or "severe"
          based on flag_count
    """
    if predictions.empty:
        raise ValueError("Predictions DataFrame is empty.")

    model_names = sorted(predictions["model_name"].unique())
    if len(model_names) < 2:
        raise ValueError(
            f"At least 2 models required; found {len(model_names)}"
        )

    # --- Determine samples with clean agreement ---
    # Try to separate clean vs perturbed
    has_severity = "severity_level" in predictions.columns
    if has_severity:
        clean_mask = predictions["severity_level"].apply(_is_clean)
        clean_df = predictions[clean_mask]
    else:
        clean_df = predictions

    pivot = clean_df.pivot_table(
        index="sample_id", columns="model_name",
        values="predicted_class", aggfunc="first",
    )
    complete = pivot.dropna()
    if complete.empty:
        return _empty_false_consensus_df()

    all_agree = complete.apply(lambda r: r.nunique() == 1, axis=1)
    agreed_ids = set(all_agree[all_agree].index)

    if not agreed_ids:
        logger.info("No clean agreement samples → no false consensus possible.")
        return _empty_false_consensus_df()

    # --- Build lookups ---
    conf_map: Dict[str, Dict] = {}
    if confidence_gaps is not None and not confidence_gaps.empty:
        for _, row in confidence_gaps.iterrows():
            sid = str(row.get("sample_id", ""))
            if sid in agreed_ids:
                conf_map[sid] = {
                    "gap": _safe_float(row.get("confidence_gap"), np.nan),
                    "mean": _safe_float(row.get("confidence_mean"), np.nan),
                }

    # Fragile consensus detection
    fragile_ids: set = set()
    if consensus_stability is not None and not consensus_stability.empty:
        fragile_df = detect_fragile_consensus(
            predictions, consensus_stability
        )
        if not fragile_df.empty:
            fragile_ids = set(fragile_df["sample_id"].unique())
    elif has_severity:
        fragile_df = detect_fragile_consensus(predictions)
        if not fragile_df.empty:
            fragile_ids = set(fragile_df["sample_id"].unique())

    instab_map: Dict[str, float] = {}
    if sample_instability is not None and not sample_instability.empty:
        if "instability_score" in sample_instability.columns:
            for _, row in sample_instability.iterrows():
                sid = str(row.get("sample_id", ""))
                if sid in agreed_ids:
                    instab_map[sid] = _safe_float(
                        row.get("instability_score"), np.nan
                    )

    # --- Evaluate each agreeing sample ---
    records: List[Dict] = []
    for sid in sorted(agreed_ids):
        sid = str(sid)
        flags: List[str] = []

        conf_gap = np.nan
        conf_mean = np.nan
        inst_score = np.nan

        # Flag 1: High confidence gap
        if sid in conf_map:
            conf_gap = conf_map[sid]["gap"]
            conf_mean = conf_map[sid]["mean"]
            if np.isfinite(conf_gap) and conf_gap > SUSPICIOUS_CONFIDENCE_GAP:
                flags.append("high_confidence_gap")

        # Flag 2: Low mean confidence
        if sid in conf_map:
            if np.isfinite(conf_mean) and conf_mean < LOW_CONFIDENCE_MEAN:
                flags.append("low_mean_confidence")

        # Flag 3: Fragile under perturbation
        if sid in fragile_ids:
            flags.append("fragile_under_perturbation")

        # Flag 4: High instability despite agreement
        if sid in instab_map:
            inst_score = instab_map[sid]
            if np.isfinite(inst_score) and inst_score > 0.5:
                flags.append("high_instability")

        if not flags:
            continue  # No false consensus indicators

        # Classify severity based on flag count
        if len(flags) >= 3:
            severity = "severe"
        elif len(flags) >= 2:
            severity = "moderate"
        else:
            severity = "mild"

        records.append({
            "sample_id": sid,
            "clean_agreement": True,
            "flags": flags,
            "flag_count": len(flags),
            "confidence_gap": round(conf_gap, 6) if np.isfinite(conf_gap) else None,
            "confidence_mean": round(conf_mean, 6) if np.isfinite(conf_mean) else None,
            "instability_score": round(inst_score, 6) if np.isfinite(inst_score) else None,
            "false_consensus_severity": severity,
        })

    if not records:
        logger.info("No false consensus detected.")
        return _empty_false_consensus_df()

    result = pd.DataFrame(records)
    result = result.sort_values(
        "flag_count", ascending=False
    ).reset_index(drop=True)

    logger.info(
        "False consensus detected for %d samples (severe=%d, moderate=%d, mild=%d).",
        len(result),
        (result["false_consensus_severity"] == "severe").sum(),
        (result["false_consensus_severity"] == "moderate").sum(),
        (result["false_consensus_severity"] == "mild").sum(),
    )
    return result


def _empty_false_consensus_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "sample_id", "clean_agreement", "flags", "flag_count",
        "confidence_gap", "confidence_mean", "instability_score",
        "false_consensus_severity",
    ])


# ---------------------------------------------------------------------------
# Public API – unstable agreement detection
# ---------------------------------------------------------------------------

def detect_unstable_agreement(
    consensus_stability: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect samples with oscillating consensus across severity levels.

    A sample has *unstable agreement* when its consensus state oscillates
    (agree → disagree → agree) across severity levels more than
    ``UNSTABLE_OSCILLATION_THRESHOLD`` fraction of transitions.

    Args:
        consensus_stability: Output of ``track_consensus_stability`` (Step 3).

    Returns:
        DataFrame with columns:

        * ``sample_id``
        * ``total_transitions``     – number of severity-level transitions
        * ``oscillation_count``     – number of state changes
        * ``oscillation_rate``      – oscillation_count / total_transitions
        * ``flag``                  – ``"unstable_agreement"``

        Only samples with unstable agreement are included.
    """
    if consensus_stability.empty:
        logger.warning("Empty consensus stability data.")
        return _empty_unstable_df()

    if "models_agree" not in consensus_stability.columns:
        logger.warning("'models_agree' column missing from consensus stability.")
        return _empty_unstable_df()

    records: List[Dict] = []
    for sample_id, grp in consensus_stability.groupby("sample_id"):
        sid = str(sample_id)

        if "severity_rank" in grp.columns:
            grp_sorted = grp.sort_values("severity_rank")
        else:
            grp_sorted = grp

        states = grp_sorted["models_agree"].tolist()
        if len(states) < 2:
            continue

        # Count state transitions (oscillations)
        transitions = len(states) - 1
        oscillations = sum(
            1 for i in range(1, len(states))
            if states[i] != states[i - 1]
        )

        osc_rate = oscillations / transitions if transitions > 0 else 0.0

        if osc_rate > UNSTABLE_OSCILLATION_THRESHOLD:
            records.append({
                "sample_id": sid,
                "total_transitions": transitions,
                "oscillation_count": oscillations,
                "oscillation_rate": round(osc_rate, 6),
                "flag": "unstable_agreement",
            })

    if not records:
        logger.info("No unstable agreement detected.")
        return _empty_unstable_df()

    result = pd.DataFrame(records)
    result = result.sort_values(
        "oscillation_rate", ascending=False
    ).reset_index(drop=True)

    logger.info(
        "Unstable agreement detected for %d samples.", len(result)
    )
    return result


def _empty_unstable_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "sample_id", "total_transitions", "oscillation_count",
        "oscillation_rate", "flag",
    ])


# ---------------------------------------------------------------------------
# Public API – false consensus summary
# ---------------------------------------------------------------------------

def compute_false_consensus_summary(
    false_consensus: pd.DataFrame,
    fragile_consensus: pd.DataFrame,
    unstable_agreement: pd.DataFrame,
) -> Dict:
    """
    Aggregate all false consensus detection results into a summary.

    Args:
        false_consensus: Output of :func:`detect_false_consensus`.
        fragile_consensus: Output of :func:`detect_fragile_consensus`.
        unstable_agreement: Output of :func:`detect_unstable_agreement`.

    Returns:
        Dictionary with:

        * ``total_false_consensus``     – count of false consensus samples
        * ``total_fragile_consensus``   – count of fragile consensus samples
        * ``total_unstable_agreement``  – count of unstable agreement samples
        * ``all_flagged_sample_ids``    – union of all flagged sample IDs
        * ``total_unique_flagged``      – number of unique flagged samples
        * ``severity_distribution``     – {mild, moderate, severe} counts
        * ``flag_frequency``            – how often each flag appears
    """
    all_ids: set = set()

    # False consensus stats
    fc_count = 0
    severity_dist = {"mild": 0, "moderate": 0, "severe": 0}
    flag_freq: Dict[str, int] = {}

    if not false_consensus.empty:
        fc_count = len(false_consensus)
        all_ids.update(false_consensus["sample_id"].unique())

        if "false_consensus_severity" in false_consensus.columns:
            for sev in false_consensus["false_consensus_severity"]:
                severity_dist[str(sev)] = severity_dist.get(str(sev), 0) + 1

        if "flags" in false_consensus.columns:
            for flags in false_consensus["flags"]:
                if isinstance(flags, list):
                    for f in flags:
                        flag_freq[f] = flag_freq.get(f, 0) + 1

    # Fragile consensus stats
    frag_count = 0
    if not fragile_consensus.empty:
        frag_count = len(fragile_consensus)
        all_ids.update(fragile_consensus["sample_id"].unique())

    # Unstable agreement stats
    unstab_count = 0
    if not unstable_agreement.empty:
        unstab_count = len(unstable_agreement)
        all_ids.update(unstable_agreement["sample_id"].unique())

    return {
        "total_false_consensus": fc_count,
        "total_fragile_consensus": frag_count,
        "total_unstable_agreement": unstab_count,
        "all_flagged_sample_ids": sorted(all_ids),
        "total_unique_flagged": len(all_ids),
        "severity_distribution": severity_dist,
        "flag_frequency": flag_freq,
    }
