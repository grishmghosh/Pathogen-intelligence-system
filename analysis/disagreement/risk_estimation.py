"""
Intelligent Risk Estimation Layer for Pathogen Intelligence System (Step 5).

This module adds deterministic, rule-based reliability risk scoring on top of
Steps 1–4. It does not change earlier analyses or introduce probabilistic
estimation. The public API focuses on two tasks:

* sample-level reliability risk estimation
* model-level reliability risk profiling

Inputs are optional DataFrames produced by earlier steps. Missing or malformed
inputs are handled gracefully by falling back to neutral defaults and by
renormalising the remaining component weights.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Tunable deterministic weights
# ---------------------------------------------------------------------------

_SAMPLE_COMPONENT_WEIGHTS: Dict[str, float] = {
    "disagreement_risk": 0.22,
    "instability_risk": 0.20,
    "consensus_reliability_risk": 0.20,
    "fragility_risk": 0.16,
    "confidence_instability_risk": 0.12,
    "escalation_risk": 0.10,
}

_MODEL_COMPONENT_WEIGHTS: Dict[str, float] = {
    "disagreement_contribution": 0.40,
    "instability_risk": 0.40,
    "trust_deficit_risk": 0.20,
}

_RISK_FACTOR_LABELS: Dict[str, str] = {
    "disagreement_risk": "severe disagreement",
    "instability_risk": "prediction instability",
    "consensus_reliability_risk": "low consensus reliability",
    "fragility_risk": "perturbation collapse",
    "confidence_instability_risk": "confidence instability",
    "escalation_risk": "rapid escalation",
    "disagreement_contribution": "highest disagreement contribution",
    "trust_deficit_risk": "low trust contribution",
}

_RISK_LEVELS: Tuple[str, ...] = (
    "minimal",
    "low",
    "moderate",
    "high",
    "critical",
)

_RISK_LEVEL_RANGES: Tuple[Tuple[str, float], ...] = (
    ("minimal", 0.15),
    ("low", 0.35),
    ("moderate", 0.60),
    ("high", 0.80),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(value: object, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if not np.isfinite(numeric):
        numeric = default
    return float(np.clip(numeric, 0.0, 1.0))


def _safe_mean(values: Sequence[object], default: float = 0.5) -> float:
    cleaned = [float(v) for v in values if pd.notna(v) and np.isfinite(float(v))]
    if not cleaned:
        return default
    return float(np.mean(cleaned))


def _first_present_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _normalise_score_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    column = _first_present_column(frame, candidates)
    return column


def _default_empty_sample_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "sample_id",
        "disagreement_risk",
        "instability_risk",
        "consensus_reliability_risk",
        "fragility_risk",
        "confidence_instability_risk",
        "escalation_risk",
        "reliability_risk_score",
        "available_component_count",
        "dominant_risk_factor",
        "secondary_risk_factors",
        "risk_factor_scores",
    ])


def _default_empty_model_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "model_name",
        "disagreement_contribution",
        "instability_risk",
        "trust_deficit_risk",
        "reliability_risk_score",
        "available_component_count",
        "dominant_risk_factor",
        "secondary_risk_factors",
        "risk_factor_scores",
    ])


def _group_numeric_metric(
    frame: Optional[pd.DataFrame],
    key_column: str,
    value_columns: Sequence[str],
    output_column: str,
    reducer: str = "mean",
) -> pd.DataFrame:
    if frame is None or frame.empty or key_column not in frame.columns:
        return pd.DataFrame(columns=[key_column, output_column])

    value_column = _first_present_column(frame, value_columns)
    if value_column is None:
        return pd.DataFrame(columns=[key_column, output_column])

    records: List[Dict[str, object]] = []
    for key_value, group in frame.groupby(key_column):
        numeric = pd.to_numeric(group[value_column], errors="coerce").dropna()
        if numeric.empty:
            continue
        if reducer == "max":
            metric = float(numeric.max())
        elif reducer == "min":
            metric = float(numeric.min())
        else:
            metric = float(numeric.mean())
        records.append({
            key_column: str(key_value),
            output_column: float(np.clip(metric, 0.0, 1.0)),
        })

    if not records:
        return pd.DataFrame(columns=[key_column, output_column])

    return pd.DataFrame(records)


def _build_sample_union(*frames: Optional[pd.DataFrame]) -> List[str]:
    sample_ids: set = set()
    for frame in frames:
        if frame is None or frame.empty or "sample_id" not in frame.columns:
            continue
        sample_ids.update(str(v) for v in frame["sample_id"].dropna().unique())
    return sorted(sample_ids)


def _resolve_sample_columns(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
    if frame is None or frame.empty or "sample_id" not in frame.columns:
        return pd.DataFrame(columns=["sample_id"])
    result = frame.copy()
    result["sample_id"] = result["sample_id"].astype(str)
    return result


def _aggregate_pairwise_disagreement(
    pairwise_scores: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if pairwise_scores is None or pairwise_scores.empty or "sample_id" not in pairwise_scores.columns:
        return pd.DataFrame(columns=[
            "sample_id",
            "disagreement_risk",
            "pairwise_disagreement_mean",
            "pairwise_disagreement_max",
            "pairwise_observation_count",
        ])

    score_column = _first_present_column(
        pairwise_scores,
        ["disagreement_score", "confidence_gap", "confidence_diff"],
    )
    if score_column is None:
        if "classes_agree" not in pairwise_scores.columns:
            return pd.DataFrame(columns=[
                "sample_id",
                "disagreement_risk",
                "pairwise_disagreement_mean",
                "pairwise_disagreement_max",
                "pairwise_observation_count",
            ])

    records: List[Dict[str, object]] = []
    for sample_id, group in pairwise_scores.groupby("sample_id"):
        sample = str(sample_id)
        values: List[float] = []
        if score_column is not None:
            numeric = pd.to_numeric(group[score_column], errors="coerce").dropna()
            values.extend(float(v) for v in numeric.tolist())

        if not values and "classes_agree" in group.columns:
            values = [0.0 if bool(v) else 1.0 for v in group["classes_agree"].tolist()]

        if not values:
            continue

        mean_value = float(np.mean(values))
        max_value = float(np.max(values))
        disagreement_risk = float(np.clip(0.6 * max_value + 0.4 * mean_value, 0.0, 1.0))

        records.append({
            "sample_id": sample,
            "disagreement_risk": disagreement_risk,
            "pairwise_disagreement_mean": float(np.clip(mean_value, 0.0, 1.0)),
            "pairwise_disagreement_max": float(np.clip(max_value, 0.0, 1.0)),
            "pairwise_observation_count": int(len(values)),
        })

    if not records:
        return pd.DataFrame(columns=[
            "sample_id",
            "disagreement_risk",
            "pairwise_disagreement_mean",
            "pairwise_disagreement_max",
            "pairwise_observation_count",
        ])

    return pd.DataFrame(records)


def _aggregate_confidence_instability(
    confidence_gaps: Optional[pd.DataFrame],
    pairwise_scores: Optional[pd.DataFrame],
) -> pd.DataFrame:
    source = confidence_gaps
    if source is None or source.empty:
        source = pairwise_scores
    if source is None or source.empty or "sample_id" not in source.columns:
        return pd.DataFrame(columns=[
            "sample_id",
            "confidence_instability_risk",
            "confidence_gap_mean",
            "confidence_mean_mean",
            "confidence_observation_count",
        ])

    records: List[Dict[str, object]] = []
    for sample_id, group in source.groupby("sample_id"):
        sample = str(sample_id)
        gap_values: List[float] = []
        mean_values: List[float] = []

        for _, row in group.iterrows():
            gap_value = None
            mean_value = None

            if "confidence_gap" in row and pd.notna(row["confidence_gap"]):
                gap_value = _clamp01(row["confidence_gap"])
            elif "confidence_diff" in row and pd.notna(row["confidence_diff"]):
                gap_value = _clamp01(row["confidence_diff"])
            elif {"confidence_a", "confidence_b"}.issubset(group.columns):
                a = row.get("confidence_a")
                b = row.get("confidence_b")
                if pd.notna(a) and pd.notna(b):
                    gap_value = _clamp01(abs(float(a) - float(b)))
                    mean_value = _clamp01((float(a) + float(b)) / 2.0)

            if "confidence_mean" in row and pd.notna(row["confidence_mean"]):
                mean_value = _clamp01(row["confidence_mean"])
            elif mean_value is None and {"confidence_a", "confidence_b"}.issubset(group.columns):
                a = row.get("confidence_a")
                b = row.get("confidence_b")
                if pd.notna(a) and pd.notna(b):
                    mean_value = _clamp01((float(a) + float(b)) / 2.0)

            if gap_value is not None:
                gap_values.append(gap_value)
            if mean_value is not None:
                mean_values.append(mean_value)

        if not gap_values and not mean_values:
            continue

        gap_mean = float(np.mean(gap_values)) if gap_values else None
        mean_mean = float(np.mean(mean_values)) if mean_values else None

        if gap_mean is not None and mean_mean is not None:
            confidence_instability = 0.65 * gap_mean + 0.35 * (1.0 - mean_mean)
        elif gap_mean is not None:
            confidence_instability = gap_mean
        elif mean_mean is not None:
            confidence_instability = 1.0 - mean_mean
        else:
            confidence_instability = 0.5

        records.append({
            "sample_id": sample,
            "confidence_instability_risk": float(np.clip(confidence_instability, 0.0, 1.0)),
            "confidence_gap_mean": round(gap_mean, 6) if gap_mean is not None else None,
            "confidence_mean_mean": round(mean_mean, 6) if mean_mean is not None else None,
            "confidence_observation_count": int(max(len(gap_values), len(mean_values))),
        })

    if not records:
        return pd.DataFrame(columns=[
            "sample_id",
            "confidence_instability_risk",
            "confidence_gap_mean",
            "confidence_mean_mean",
            "confidence_observation_count",
        ])

    return pd.DataFrame(records)


def _aggregate_consensus_reliability(
    consensus_reliability: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if consensus_reliability is None or consensus_reliability.empty or "sample_id" not in consensus_reliability.columns:
        return pd.DataFrame(columns=[
            "sample_id",
            "consensus_reliability_risk",
            "consensus_reliability_score",
        ])

    score_column = _first_present_column(
        consensus_reliability,
        ["consensus_reliability_score", "reliability_score"],
    )
    if score_column is None:
        if "adjusted_trust_level" not in consensus_reliability.columns:
            return pd.DataFrame(columns=[
                "sample_id",
                "consensus_reliability_risk",
                "consensus_reliability_score",
            ])

    trust_map = {
        "very_high": 0.10,
        "high": 0.25,
        "moderate": 0.50,
        "low": 0.75,
        "critical": 0.95,
    }

    records: List[Dict[str, object]] = []
    for sample_id, group in consensus_reliability.groupby("sample_id"):
        sample = str(sample_id)
        if score_column is not None:
            numeric = pd.to_numeric(group[score_column], errors="coerce").dropna()
            if numeric.empty:
                continue
            reliability_score = float(np.clip(numeric.mean(), 0.0, 1.0))
            risk_score = 1.0 - reliability_score
        else:
            labels = [str(v).lower() for v in group["adjusted_trust_level"].tolist()]
            mapped = [trust_map.get(label, 0.50) for label in labels]
            if not mapped:
                continue
            risk_score = float(np.clip(np.mean(mapped), 0.0, 1.0))
            reliability_score = 1.0 - risk_score

        records.append({
            "sample_id": sample,
            "consensus_reliability_risk": float(np.clip(risk_score, 0.0, 1.0)),
            "consensus_reliability_score": float(np.clip(reliability_score, 0.0, 1.0)),
        })

    if not records:
        return pd.DataFrame(columns=[
            "sample_id",
            "consensus_reliability_risk",
            "consensus_reliability_score",
        ])

    return pd.DataFrame(records)


def _aggregate_fragility(
    fragile_consensus: Optional[pd.DataFrame],
    false_consensus: Optional[pd.DataFrame],
    consensus_breakdown: Optional[pd.DataFrame],
    consensus_stability: Optional[pd.DataFrame],
) -> pd.DataFrame:
    records_by_sample: Dict[str, float] = {}

    if fragile_consensus is not None and not fragile_consensus.empty and "sample_id" in fragile_consensus.columns:
        for _, row in fragile_consensus.iterrows():
            sample = str(row.get("sample_id", ""))
            if not sample:
                continue
            rank = row.get("collapse_severity_rank")
            if pd.notna(rank):
                rank_value = float(rank)
                fragility = float(np.clip(1.0 - min(rank_value, 5.0) / 5.0, 0.0, 1.0))
            else:
                fragility = 0.90
            records_by_sample[sample] = max(records_by_sample.get(sample, 0.0), fragility)

    if false_consensus is not None and not false_consensus.empty and "sample_id" in false_consensus.columns:
        severity_map = {"mild": 0.50, "moderate": 0.70, "severe": 0.90}
        for _, row in false_consensus.iterrows():
            sample = str(row.get("sample_id", ""))
            if not sample:
                continue
            severity = str(row.get("false_consensus_severity", "mild")).lower()
            flags = row.get("flags", [])
            if isinstance(flags, list):
                bonus = min(0.15, 0.05 * max(0, len(flags) - 1))
            else:
                bonus = 0.0
            fragility = float(np.clip(severity_map.get(severity, 0.50) + bonus, 0.0, 1.0))
            records_by_sample[sample] = max(records_by_sample.get(sample, 0.0), fragility)

    if consensus_breakdown is not None and not consensus_breakdown.empty and "sample_id" in consensus_breakdown.columns:
        for _, row in consensus_breakdown.iterrows():
            sample = str(row.get("sample_id", ""))
            if not sample:
                continue
            if "escalation_speed" in consensus_breakdown.columns and pd.notna(row.get("escalation_speed")):
                fragility = float(np.clip(float(row.get("escalation_speed")), 0.0, 1.0))
            elif "stability_duration" in consensus_breakdown.columns and pd.notna(row.get("stability_duration")):
                duration = max(float(row.get("stability_duration")), 1.0)
                fragility = float(np.clip(1.0 / duration, 0.0, 1.0))
            else:
                fragility = 0.50
            records_by_sample[sample] = max(records_by_sample.get(sample, 0.0), fragility)

    if consensus_stability is not None and not consensus_stability.empty and {"sample_id", "severity_rank", "models_agree"}.issubset(consensus_stability.columns):
        for sample_id, group in consensus_stability.groupby("sample_id"):
            sample = str(sample_id)
            group = group.sort_values("severity_rank")
            states = [bool(v) for v in group["models_agree"].tolist()]
            ranks = pd.to_numeric(group["severity_rank"], errors="coerce").dropna().tolist()
            if not states or not ranks:
                continue
            transitions = max(len(states) - 1, 1)
            oscillations = sum(1 for index in range(1, len(states)) if states[index] != states[index - 1])
            oscillation_risk = oscillations / transitions
            collapse_rank = None
            max_rank = max(float(v) for v in ranks)
            for rank_value, state in zip(ranks, states):
                if not state:
                    collapse_rank = float(rank_value)
                    break
            if collapse_rank is None:
                breakpoint_risk = 0.0
            else:
                breakpoint_risk = float(np.clip(1.0 - (collapse_rank - 1.0) / max(max_rank, 1.0), 0.0, 1.0))
            fragility = float(np.clip(0.6 * oscillation_risk + 0.4 * breakpoint_risk, 0.0, 1.0))
            records_by_sample[sample] = max(records_by_sample.get(sample, 0.0), fragility)

    if not records_by_sample:
        return pd.DataFrame(columns=[
            "sample_id",
            "fragility_risk",
        ])

    return pd.DataFrame([
        {"sample_id": sample, "fragility_risk": float(np.clip(score, 0.0, 1.0))}
        for sample, score in sorted(records_by_sample.items())
    ])


def _aggregate_escalation(
    consensus_breakdown: Optional[pd.DataFrame],
    consensus_stability: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if consensus_breakdown is not None and not consensus_breakdown.empty and "sample_id" in consensus_breakdown.columns:
        records: List[Dict[str, object]] = []
        for sample_id, group in consensus_breakdown.groupby("sample_id"):
            sample = str(sample_id)
            values: List[float] = []
            if "escalation_speed" in group.columns:
                values.extend(float(v) for v in pd.to_numeric(group["escalation_speed"], errors="coerce").dropna().tolist())
            if "stability_duration" in group.columns:
                durations = pd.to_numeric(group["stability_duration"], errors="coerce").dropna().tolist()
                values.extend(float(np.clip(1.0 / max(v, 1.0), 0.0, 1.0)) for v in durations)
            if not values:
                continue
            records.append({
                "sample_id": sample,
                "escalation_risk": float(np.clip(np.mean(values), 0.0, 1.0)),
                "escalation_observation_count": int(len(values)),
            })
        if records:
            return pd.DataFrame(records)

    if consensus_stability is None or consensus_stability.empty or not {"sample_id", "severity_rank", "models_agree"}.issubset(consensus_stability.columns):
        return pd.DataFrame(columns=[
            "sample_id",
            "escalation_risk",
            "escalation_observation_count",
        ])

    records: List[Dict[str, object]] = []
    for sample_id, group in consensus_stability.groupby("sample_id"):
        sample = str(sample_id)
        group = group.sort_values("severity_rank")
        states = [bool(v) for v in group["models_agree"].tolist()]
        ranks = pd.to_numeric(group["severity_rank"], errors="coerce").dropna().tolist()
        if not states or not ranks:
            continue
        transitions = max(len(states) - 1, 1)
        oscillations = sum(1 for index in range(1, len(states)) if states[index] != states[index - 1])
        oscillation_risk = oscillations / transitions
        collapse_rank = None
        max_rank = max(float(v) for v in ranks)
        for rank_value, state in zip(ranks, states):
            if not state:
                collapse_rank = float(rank_value)
                break
        if collapse_rank is None:
            collapse_component = 0.0
        else:
            collapse_component = float(np.clip(1.0 - (collapse_rank - 1.0) / max(max_rank, 1.0), 0.0, 1.0))
        escalation_risk = float(np.clip(0.6 * oscillation_risk + 0.4 * collapse_component, 0.0, 1.0))
        records.append({
            "sample_id": sample,
            "escalation_risk": escalation_risk,
            "escalation_observation_count": int(len(states)),
        })

    if not records:
        return pd.DataFrame(columns=[
            "sample_id",
            "escalation_risk",
            "escalation_observation_count",
        ])

    return pd.DataFrame(records)


def _combine_component_scores(row: pd.Series, component_weights: Dict[str, float]) -> Tuple[float, int, Dict[str, float]]:
    available: Dict[str, float] = {}
    for component, _ in component_weights.items():
        value = row.get(component)
        if pd.notna(value):
            available[component] = _clamp01(value)

    if not available:
        return 0.5, 0, {}

    total_weight = sum(component_weights[component] for component in available)
    if total_weight <= 0:
        return 0.5, 0, available

    weighted = sum(component_weights[component] * available[component] for component in available)
    return float(np.clip(weighted / total_weight, 0.0, 1.0)), len(available), available


def _dominant_and_secondary_factors(component_scores: Dict[str, float]) -> Tuple[Optional[str], List[str]]:
    if not component_scores:
        return None, []

    ranked = sorted(
        component_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    dominant_component = ranked[0][0]
    dominant_label = _RISK_FACTOR_LABELS.get(dominant_component, dominant_component)
    secondary_labels = [
        _RISK_FACTOR_LABELS.get(component, component)
        for component, _ in ranked[1:3]
    ]
    return dominant_label, secondary_labels


# ---------------------------------------------------------------------------
# Public API - sample-level risk scoring
# ---------------------------------------------------------------------------


def compute_reliability_risk_scores(
    pairwise_disagreement: Optional[pd.DataFrame] = None,
    confidence_gaps: Optional[pd.DataFrame] = None,
    consensus_reliability: Optional[pd.DataFrame] = None,
    sample_instability: Optional[pd.DataFrame] = None,
    fragile_consensus: Optional[pd.DataFrame] = None,
    false_consensus: Optional[pd.DataFrame] = None,
    consensus_breakdown: Optional[pd.DataFrame] = None,
    consensus_stability: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute a normalized reliability risk score for each sample.

    The score is a deterministic weighted average of the available
    components. Missing inputs are ignored rather than crashing, which
    keeps the layer compatible with partially populated pipelines.

    Returns a DataFrame with one row per sample and the following core
    columns:

    * ``sample_id``
    * ``disagreement_risk``
    * ``instability_risk``
    * ``consensus_reliability_risk``
    * ``fragility_risk``
    * ``confidence_instability_risk``
    * ``escalation_risk``
    * ``reliability_risk_score``
    * ``dominant_risk_factor``
    * ``secondary_risk_factors``
    * ``risk_factor_scores``
    """
    pairwise_sample = _aggregate_pairwise_disagreement(pairwise_disagreement)
    confidence_sample = _aggregate_confidence_instability(confidence_gaps, pairwise_disagreement)
    reliability_sample = _aggregate_consensus_reliability(consensus_reliability)
    instability_sample = _group_numeric_metric(
        sample_instability,
        "sample_id",
        ["instability_score", "disagreement_rate"],
        "instability_risk",
        reducer="mean",
    )
    fragility_sample = _aggregate_fragility(
        fragile_consensus,
        false_consensus,
        consensus_breakdown,
        consensus_stability,
    )
    escalation_sample = _aggregate_escalation(consensus_breakdown, consensus_stability)

    sample_ids = _build_sample_union(
        pairwise_sample,
        confidence_sample,
        reliability_sample,
        instability_sample,
        fragility_sample,
        escalation_sample,
    )
    if not sample_ids:
        return _default_empty_sample_frame()

    result = pd.DataFrame({"sample_id": sample_ids})
    for frame in (
        pairwise_sample,
        confidence_sample,
        reliability_sample,
        instability_sample,
        fragility_sample,
        escalation_sample,
    ):
        if frame is not None and not frame.empty:
            result = result.merge(frame, on="sample_id", how="left")

    records: List[Dict[str, object]] = []
    for _, row in result.iterrows():
        score, count, available = _combine_component_scores(row, _SAMPLE_COMPONENT_WEIGHTS)
        dominant, secondary = _dominant_and_secondary_factors(available)
        records.append({
            "sample_id": str(row["sample_id"]),
            "disagreement_risk": row.get("disagreement_risk"),
            "instability_risk": row.get("instability_risk"),
            "consensus_reliability_risk": row.get("consensus_reliability_risk"),
            "fragility_risk": row.get("fragility_risk"),
            "confidence_instability_risk": row.get("confidence_instability_risk"),
            "escalation_risk": row.get("escalation_risk"),
            "reliability_risk_score": round(float(score), 6),
            "available_component_count": int(count),
            "dominant_risk_factor": dominant,
            "secondary_risk_factors": secondary,
            "risk_factor_scores": available,
            **{
                key: row.get(key)
                for key in [
                    "pairwise_disagreement_mean",
                    "pairwise_disagreement_max",
                    "pairwise_observation_count",
                    "confidence_gap_mean",
                    "confidence_mean_mean",
                    "confidence_observation_count",
                    "escalation_observation_count",
                ]
                if key in result.columns
            },
        })

    risk = pd.DataFrame(records)
    risk = risk.sort_values(
        ["reliability_risk_score", "sample_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    risk["risk_rank"] = range(1, len(risk) + 1)
    return risk


# ---------------------------------------------------------------------------
# Public API - model-level risk profiling
# ---------------------------------------------------------------------------


def compute_model_reliability_risk_profiles(
    pairwise_disagreement: Optional[pd.DataFrame] = None,
    model_instability: Optional[pd.DataFrame] = None,
    model_trust_contribution: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute a deterministic reliability-risk profile for each model.

    The model profile blends disagreement contribution, instability, and
    trust deficit. If one or more inputs are missing, the remaining
    components are renormalised automatically.
    """
    disagreement_records: List[Dict[str, object]] = []
    if pairwise_disagreement is not None and not pairwise_disagreement.empty and {"model_a", "model_b"}.issubset(pairwise_disagreement.columns):
        score_column = _first_present_column(
            pairwise_disagreement,
            ["disagreement_score", "confidence_gap", "confidence_diff"],
        )
        for model_name in sorted(
            set(pairwise_disagreement["model_a"].dropna().astype(str).unique()).union(
                set(pairwise_disagreement["model_b"].dropna().astype(str).unique())
            )
        ):
            mask = (
                pairwise_disagreement["model_a"].astype(str) == model_name
            ) | (
                pairwise_disagreement["model_b"].astype(str) == model_name
            )
            subset = pairwise_disagreement.loc[mask]
            if subset.empty:
                continue
            if score_column is not None:
                values = pd.to_numeric(subset[score_column], errors="coerce").dropna()
                conflict_values = [float(v) for v in values.tolist()]
            elif "classes_agree" in subset.columns:
                conflict_values = [0.0 if bool(v) else 1.0 for v in subset["classes_agree"].tolist()]
            else:
                conflict_values = []
            if not conflict_values:
                continue
            disagreement_records.append({
                "model_name": model_name,
                "disagreement_contribution": float(np.clip(np.mean(conflict_values), 0.0, 1.0)),
                "disagreement_peak": float(np.clip(np.max(conflict_values), 0.0, 1.0)),
                "pairwise_observation_count": int(len(conflict_values)),
            })

    disagreement_frame = pd.DataFrame(disagreement_records) if disagreement_records else pd.DataFrame(columns=[
        "model_name",
        "disagreement_contribution",
        "disagreement_peak",
        "pairwise_observation_count",
    ])

    instability_frame = _group_numeric_metric(
        model_instability,
        "model_name",
        ["instability_score", "flip_rate"],
        "instability_risk",
        reducer="mean",
    )

    trust_frame = pd.DataFrame(columns=["model_name", "trust_deficit_risk", "trust_contribution"])
    if model_trust_contribution is not None and not model_trust_contribution.empty and "model_name" in model_trust_contribution.columns:
        score_column = _first_present_column(
            model_trust_contribution,
            ["trust_contribution", "trust_score"],
        )
        if score_column is not None:
            rows: List[Dict[str, object]] = []
            for _, row in model_trust_contribution.iterrows():
                score = _clamp01(row.get(score_column))
                rows.append({
                    "model_name": str(row.get("model_name", "")),
                    "trust_contribution": score,
                    "trust_deficit_risk": float(np.clip(1.0 - score, 0.0, 1.0)),
                })
            trust_frame = pd.DataFrame(rows)

    sample_ids = sorted(set())
    for frame in (disagreement_frame, instability_frame, trust_frame):
        if not frame.empty and "model_name" in frame.columns:
            sample_ids.extend(frame["model_name"].dropna().astype(str).tolist())
    model_names = sorted(set(sample_ids))
    if not model_names:
        return _default_empty_model_frame()

    result = pd.DataFrame({"model_name": model_names})
    for frame in (disagreement_frame, instability_frame, trust_frame):
        if not frame.empty:
            result = result.merge(frame, on="model_name", how="left")

    records: List[Dict[str, object]] = []
    for _, row in result.iterrows():
        score, count, available = _combine_component_scores(row, _MODEL_COMPONENT_WEIGHTS)
        dominant, secondary = _dominant_and_secondary_factors(available)
        records.append({
            "model_name": str(row["model_name"]),
            "disagreement_contribution": row.get("disagreement_contribution"),
            "disagreement_peak": row.get("disagreement_peak"),
            "instability_risk": row.get("instability_risk"),
            "trust_deficit_risk": row.get("trust_deficit_risk"),
            "trust_contribution": row.get("trust_contribution"),
            "reliability_risk_score": round(float(score), 6),
            "available_component_count": int(count),
            "dominant_risk_factor": dominant,
            "secondary_risk_factors": secondary,
            "risk_factor_scores": available,
            **{
                key: row.get(key)
                for key in [
                    "pairwise_observation_count",
                    "flip_rate",
                    "total_observations",
                    "mean_severity_at_flip",
                ]
                if key in result.columns
            },
        })

    profile = pd.DataFrame(records)
    profile = profile.sort_values(
        ["reliability_risk_score", "model_name"],
        ascending=[False, True],
    ).reset_index(drop=True)
    profile["risk_rank"] = range(1, len(profile) + 1)
    return profile


# ---------------------------------------------------------------------------
# Public API - compatibility helpers
# ---------------------------------------------------------------------------


def compute_reliability_risk_summary(
    risk_scores: pd.DataFrame,
) -> Dict:
    """Lightweight summary retained here for direct programmatic use."""
    if risk_scores is None or risk_scores.empty or "reliability_risk_score" not in risk_scores.columns:
        return {
            "total_samples": 0,
            "mean_risk": 0.0,
            "max_risk": 0.0,
            "min_risk": 0.0,
            "median_risk": 0.0,
            "high_risk_count": 0,
            "critical_risk_count": 0,
        }

    s = pd.to_numeric(risk_scores["reliability_risk_score"], errors="coerce").dropna()
    if s.empty:
        return {
            "total_samples": 0,
            "mean_risk": 0.0,
            "max_risk": 0.0,
            "min_risk": 0.0,
            "median_risk": 0.0,
            "high_risk_count": 0,
            "critical_risk_count": 0,
        }

    return {
        "total_samples": int(len(s)),
        "mean_risk": round(float(s.mean()), 6),
        "max_risk": round(float(s.max()), 6),
        "min_risk": round(float(s.min()), 6),
        "median_risk": round(float(s.median()), 6),
        "high_risk_count": int((s >= 0.60).sum()),
        "critical_risk_count": int((s >= 0.80).sum()),
    }
