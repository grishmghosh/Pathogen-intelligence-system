"""
Uncertainty Summary, Profiling, Trend Analysis, and Export for Step 6.

This module separates classification from entropy computation and keeps all
exports out of the analytical modules. It consumes the deterministic outputs
from the uncertainty metrics, entropy analysis, and confidence dispersion
layers, and connects them to disagreement / instability summaries from
Steps 1–5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from analysis.uncertainty.confidence_dispersion import aggregate_confidence_dispersion
from analysis.uncertainty.entropy_analysis import (
    aggregate_sample_uncertainty,
    compute_uncertainty_severity_curve,
    detect_confidence_collapse,
    rank_perturbation_uncertainty,
)
from analysis.uncertainty.uncertainty_metrics import compute_entropy_summary


_UNCERTAINTY_LEVELS: Sequence[str] = (
    "very_low",
    "low",
    "moderate",
    "high",
    "critical",
)

_UNCERTAINTY_THRESHOLDS: Dict[str, float] = {
    "very_low": 0.15,
    "low": 0.35,
    "moderate": 0.60,
    "high": 0.80,
}

_ENTROPY_DIR = Path("results") / "uncertainty" / "entropy_analysis"
_DISPERSION_DIR = Path("results") / "uncertainty" / "confidence_dispersion"
_TREND_DIR = Path("results") / "uncertainty" / "uncertainty_trends"
_PROFILE_DIR = Path("results") / "uncertainty" / "uncertainty_profiles"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(output_dir: Optional[Union[str, Path]], default_dir: Path) -> Path:
    directory = Path(output_dir) if output_dir is not None else default_dir
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _serialise(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _serialise(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_serialise(value) for value in obj]
    if isinstance(obj, tuple):
        return [_serialise(value) for value in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def _flatten_mapping(obj: Dict[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, value in obj.items():
        path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            rows.extend(_flatten_mapping(value, path))
        elif isinstance(value, list):
            if not value:
                rows.append({"path": path, "value": []})
            else:
                for index, item in enumerate(value):
                    item_path = f"{path}[{index}]"
                    if isinstance(item, dict):
                        rows.extend(_flatten_mapping(item, item_path))
                    else:
                        rows.append({"path": item_path, "value": _serialise(item)})
        else:
            rows.append({"path": path, "value": _serialise(value)})
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        return default
    return float(numeric)


def _first_present_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _severity_rank(value: Any) -> int:
    label = str(value).strip().lower()
    if label in {"clean", "none", "original", "baseline"}:
        return 0
    if label == "very_low":
        return 1
    if label == "low":
        return 2
    if label == "moderate":
        return 3
    if label == "high":
        return 4
    if label == "critical":
        return 5
    return 99


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if pair.empty or len(pair) < 2:
        return 0.0
    corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    if pd.isna(corr):
        return 0.0
    return float(corr)


def _coerce_summary_frame(data: Any) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data)


def _empty_uncertainty_frame(sample_column: str = "sample_id") -> pd.DataFrame:
    return pd.DataFrame(columns=[
        sample_column,
        "mean_uncertainty_score",
        "max_uncertainty_score",
        "min_uncertainty_score",
        "mean_entropy",
        "mean_confidence",
        "observation_count",
    ])


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_uncertainty_level(uncertainty_score: float) -> str:
    """Map a normalized uncertainty score to a deterministic label."""
    score = float(np.clip(_safe_float(uncertainty_score, 0.5), 0.0, 1.0))
    if score >= _UNCERTAINTY_THRESHOLDS["high"]:
        return "critical"
    if score >= _UNCERTAINTY_THRESHOLDS["moderate"]:
        return "high"
    if score >= _UNCERTAINTY_THRESHOLDS["low"]:
        return "moderate"
    if score >= _UNCERTAINTY_THRESHOLDS["very_low"]:
        return "low"
    return "very_low"


def assign_uncertainty_labels(
    entropy_analysis: pd.DataFrame,
    score_column: str = "uncertainty_score",
) -> pd.DataFrame:
    """Add uncertainty labels to an entropy analysis table."""
    if entropy_analysis is None or entropy_analysis.empty:
        result = entropy_analysis.copy() if entropy_analysis is not None else pd.DataFrame()
        result["uncertainty_level"] = pd.Series(dtype=str)
        result["uncertainty_level_rank"] = pd.Series(dtype=int)
        return result

    if score_column not in entropy_analysis.columns:
        result = entropy_analysis.copy()
        result["uncertainty_level"] = pd.Series(dtype=str)
        result["uncertainty_level_rank"] = pd.Series(dtype=int)
        return result

    ranks = {level: index + 1 for index, level in enumerate(_UNCERTAINTY_LEVELS)}
    result = entropy_analysis.copy()
    labels: List[str] = []
    label_ranks: List[int] = []
    for _, row in result.iterrows():
        label = classify_uncertainty_level(row.get(score_column))
        labels.append(label)
        label_ranks.append(ranks[label])
    result["uncertainty_level"] = labels
    result["uncertainty_level_rank"] = label_ranks
    return result


# ---------------------------------------------------------------------------
# Summary and relationship analysis
# ---------------------------------------------------------------------------


def compute_uncertainty_summary(entropy_analysis: pd.DataFrame) -> Dict[str, Any]:
    """Summarise entropy uncertainty outputs."""
    if entropy_analysis is None or entropy_analysis.empty or "uncertainty_score" not in entropy_analysis.columns:
        return {
            "total_observations": 0,
            "mean_uncertainty": 0.0,
            "max_uncertainty": 0.0,
            "min_uncertainty": 0.0,
            "median_uncertainty": 0.0,
            "uncertainty_distribution": {level: 0 for level in _UNCERTAINTY_LEVELS},
            "most_uncertain_sample_ids": [],
        }

    labelled = assign_uncertainty_labels(entropy_analysis)
    scores = pd.to_numeric(labelled["uncertainty_score"], errors="coerce").dropna()
    if scores.empty:
        return {
            "total_observations": 0,
            "mean_uncertainty": 0.0,
            "max_uncertainty": 0.0,
            "min_uncertainty": 0.0,
            "median_uncertainty": 0.0,
            "uncertainty_distribution": {level: 0 for level in _UNCERTAINTY_LEVELS},
            "most_uncertain_sample_ids": [],
        }

    distribution = labelled["uncertainty_level"].astype(str).value_counts().to_dict() if "uncertainty_level" in labelled.columns else {}
    for level in _UNCERTAINTY_LEVELS:
        distribution.setdefault(level, 0)

    most_uncertain: List[str] = []
    if "sample_id" in labelled.columns:
        sample_scores = aggregate_sample_uncertainty(labelled)
        if not sample_scores.empty:
            most_uncertain = [str(value) for value in sample_scores.head(5)["sample_id"].tolist()]

    return {
        "total_observations": int(len(scores)),
        "mean_uncertainty": round(float(scores.mean()), 6),
        "max_uncertainty": round(float(scores.max()), 6),
        "min_uncertainty": round(float(scores.min()), 6),
        "median_uncertainty": round(float(scores.median()), 6),
        "uncertainty_distribution": distribution,
        "most_uncertain_sample_ids": most_uncertain,
        "entropy_summary": compute_entropy_summary(labelled),
    }


def detect_uncertainty_conflicts(
    entropy_analysis: pd.DataFrame,
    disagreement_scores: Optional[pd.DataFrame] = None,
    sample_instability: Optional[pd.DataFrame] = None,
    sample_column: str = "sample_id",
    uncertainty_threshold: float = 0.60,
    conflict_threshold: float = 0.65,
) -> pd.DataFrame:
    """Detect samples where uncertainty aligns with disagreement or instability."""
    sample_uncertainty = aggregate_sample_uncertainty(entropy_analysis, sample_column=sample_column)
    if sample_uncertainty.empty:
        return pd.DataFrame(columns=[
            sample_column,
            "mean_uncertainty_score",
            "disagreement_score",
            "instability_score",
            "conflict_score",
            "conflict_reason",
        ])

    working = sample_uncertainty.copy()

    def _aggregate_by_sample(frame: Optional[pd.DataFrame], value_candidates: Sequence[str]) -> pd.DataFrame:
        if frame is None or frame.empty or sample_column not in frame.columns:
            return pd.DataFrame(columns=[sample_column])
        value_column = _first_present_column(frame, value_candidates)
        if value_column is None:
            return pd.DataFrame(columns=[sample_column])
        rows: List[Dict[str, Any]] = []
        for sample_id, group in frame.groupby(sample_column):
            values = pd.to_numeric(group[value_column], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({sample_column: str(sample_id), value_column: float(values.mean())})
        return pd.DataFrame(rows)

    disagreement = _aggregate_by_sample(disagreement_scores, ["disagreement_score", "uncertainty_score", "score"])
    instability = _aggregate_by_sample(sample_instability, ["instability_score", "score"])

    if not disagreement.empty:
        working = working.merge(disagreement, on=sample_column, how="left")
        if "score" in working.columns and "disagreement_score" not in working.columns:
            working = working.rename(columns={"score": "disagreement_score"})
    else:
        working["disagreement_score"] = np.nan

    if not instability.empty:
        working = working.merge(instability, on=sample_column, how="left")
        if "score" in working.columns and "instability_score" not in working.columns:
            working = working.rename(columns={"score": "instability_score"})
    else:
        working["instability_score"] = np.nan

    if "disagreement_score" not in working.columns:
        working["disagreement_score"] = np.nan
    if "instability_score" not in working.columns:
        working["instability_score"] = np.nan

    rows: List[Dict[str, Any]] = []
    for _, row in working.iterrows():
        uncertainty = _safe_float(row.get("mean_uncertainty_score"), 0.0)
        disagreement_score = _safe_float(row.get("disagreement_score"), np.nan)
        instability_score = _safe_float(row.get("instability_score"), np.nan)

        components: List[float] = [uncertainty]
        reasons: List[str] = []

        if pd.notna(row.get("disagreement_score")):
            components.append(float(np.clip(disagreement_score, 0.0, 1.0)))
            if disagreement_score >= uncertainty_threshold:
                reasons.append("disagreement")
        if pd.notna(row.get("instability_score")):
            components.append(float(np.clip(instability_score, 0.0, 1.0)))
            if instability_score >= uncertainty_threshold:
                reasons.append("instability")
        if uncertainty >= uncertainty_threshold:
            reasons.append("uncertainty")

        if not reasons:
            continue

        conflict_score = float(np.clip(np.mean(components), 0.0, 1.0))
        if conflict_score < conflict_threshold and len(reasons) < 2:
            continue

        rows.append({
            sample_column: str(row[sample_column]),
            "mean_uncertainty_score": round(uncertainty, 6),
            "disagreement_score": round(disagreement_score, 6) if pd.notna(row.get("disagreement_score")) else None,
            "instability_score": round(instability_score, 6) if pd.notna(row.get("instability_score")) else None,
            "conflict_score": round(conflict_score, 6),
            "conflict_reason": "|".join(sorted(set(reasons))),
        })

    if not rows:
        return pd.DataFrame(columns=[
            sample_column,
            "mean_uncertainty_score",
            "disagreement_score",
            "instability_score",
            "conflict_score",
            "conflict_reason",
        ])

    return pd.DataFrame(rows).sort_values(
        ["conflict_score", "mean_uncertainty_score"], ascending=[False, False]
    ).reset_index(drop=True)


def compute_disagreement_uncertainty_relationship(
    entropy_analysis: pd.DataFrame,
    disagreement_scores: Optional[pd.DataFrame] = None,
    sample_instability: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Relate uncertainty to disagreement and instability."""
    sample_uncertainty = aggregate_sample_uncertainty(entropy_analysis)
    conflicts = detect_uncertainty_conflicts(
        entropy_analysis,
        disagreement_scores=disagreement_scores,
        sample_instability=sample_instability,
    )

    if sample_uncertainty.empty:
        return {
            "sample_correlations": {
                "uncertainty_vs_disagreement": 0.0,
                "uncertainty_vs_instability": 0.0,
                "disagreement_vs_instability": 0.0,
            },
            "conflict_count": 0,
            "conflict_sample_ids": [],
            "conflict_records": [],
            "mean_conflict_score": 0.0,
        }

    working = sample_uncertainty.copy()

    def _merge_metric(frame: Optional[pd.DataFrame], candidates: Sequence[str], target: str) -> None:
        nonlocal working
        if frame is None or frame.empty or "sample_id" not in frame.columns:
            working[target] = np.nan
            return
        value_column = _first_present_column(frame, candidates)
        if value_column is None:
            working[target] = np.nan
            return
        rows: List[Dict[str, Any]] = []
        for sample_id, group in frame.groupby("sample_id"):
            values = pd.to_numeric(group[value_column], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({"sample_id": str(sample_id), target: float(values.mean())})
        if not rows:
            working[target] = np.nan
            return
        working = working.merge(pd.DataFrame(rows), on="sample_id", how="left")

    _merge_metric(disagreement_scores, ["disagreement_score", "score"], "disagreement_score")
    _merge_metric(sample_instability, ["instability_score", "score"], "instability_score")

    sample_correlations = {
        "uncertainty_vs_disagreement": _safe_corr(
            pd.to_numeric(working["mean_uncertainty_score"], errors="coerce"),
            pd.to_numeric(working["disagreement_score"], errors="coerce"),
        ),
        "uncertainty_vs_instability": _safe_corr(
            pd.to_numeric(working["mean_uncertainty_score"], errors="coerce"),
            pd.to_numeric(working["instability_score"], errors="coerce"),
        ),
        "disagreement_vs_instability": _safe_corr(
            pd.to_numeric(working["disagreement_score"], errors="coerce"),
            pd.to_numeric(working["instability_score"], errors="coerce"),
        ),
    }

    conflict_records = conflicts.to_dict(orient="records") if not conflicts.empty else []
    conflict_sample_ids = [str(value) for value in conflicts["sample_id"].tolist()] if not conflicts.empty and "sample_id" in conflicts.columns else []
    mean_conflict_score = 0.0
    if not conflicts.empty and "conflict_score" in conflicts.columns:
        mean_conflict_score = round(float(pd.to_numeric(conflicts["conflict_score"], errors="coerce").dropna().mean()), 6)

    return {
        "sample_correlations": sample_correlations,
        "conflict_count": int(len(conflicts)),
        "conflict_sample_ids": conflict_sample_ids,
        "conflict_records": conflict_records,
        "mean_conflict_score": mean_conflict_score,
    }


# ---------------------------------------------------------------------------
# Model profiling
# ---------------------------------------------------------------------------


def compute_model_uncertainty_profiles(
    entropy_analysis: pd.DataFrame,
    dispersion_analysis: Optional[pd.DataFrame] = None,
    collapse_events: Optional[pd.DataFrame] = None,
    model_column: str = "model_name",
    severity_column: str = "severity_level",
) -> pd.DataFrame:
    """Generate per-model uncertainty summaries."""
    if entropy_analysis is None or entropy_analysis.empty or model_column not in entropy_analysis.columns:
        return pd.DataFrame(columns=[
            model_column,
            "total_observations",
            "mean_uncertainty_score",
            "max_uncertainty_score",
            "min_uncertainty_score",
            "uncertainty_std",
            "clean_mean_uncertainty",
            "perturbed_mean_uncertainty",
            "uncertainty_delta",
            "mean_confidence_variance",
            "mean_confidence_std",
            "mean_confidence_spread",
            "mean_confidence_concentration",
            "collapse_count",
            "mean_entropy_shift",
            "perturbation_sensitivity_score",
            "stability_label",
        ])

    dispersion_by_model = aggregate_confidence_dispersion(dispersion_analysis, group_column=model_column) if dispersion_analysis is not None else pd.DataFrame()

    collapse_summary = pd.DataFrame(columns=[model_column, "collapse_count", "mean_entropy_shift"])
    if collapse_events is not None and not collapse_events.empty and model_column in collapse_events.columns:
        rows: List[Dict[str, Any]] = []
        for model_name, group in collapse_events.groupby(model_column):
            entropy_shift = pd.to_numeric(group["entropy_shift"], errors="coerce").dropna() if "entropy_shift" in group.columns else pd.Series(dtype=float)
            rows.append({
                model_column: str(model_name),
                "collapse_count": int(len(group)),
                "mean_entropy_shift": round(float(entropy_shift.mean()), 6) if not entropy_shift.empty else 0.0,
            })
        collapse_summary = pd.DataFrame(rows)

    rows: List[Dict[str, Any]] = []
    for model_name, group in entropy_analysis.groupby(model_column):
        uncertainty = pd.to_numeric(group["uncertainty_score"], errors="coerce").dropna()
        if uncertainty.empty:
            continue

        model_label = str(model_name)
        mean_uncertainty = float(uncertainty.mean())
        max_uncertainty = float(uncertainty.max())
        min_uncertainty = float(uncertainty.min())
        uncertainty_std = float(uncertainty.std(ddof=0)) if len(uncertainty) > 1 else 0.0

        if severity_column in group.columns:
            severity_labels = group[severity_column].astype(str).str.lower()
            clean_mask = severity_labels.isin({"clean", "none", "original", "baseline"})
            clean_uncertainty = pd.to_numeric(group.loc[clean_mask, "uncertainty_score"], errors="coerce").dropna()
            perturbed_uncertainty = pd.to_numeric(group.loc[~clean_mask, "uncertainty_score"], errors="coerce").dropna()
        else:
            clean_uncertainty = pd.Series(dtype=float)
            perturbed_uncertainty = pd.Series(dtype=float)

        clean_mean = float(clean_uncertainty.mean()) if not clean_uncertainty.empty else None
        perturbed_mean = float(perturbed_uncertainty.mean()) if not perturbed_uncertainty.empty else None
        if clean_mean is not None and perturbed_mean is not None:
            uncertainty_delta = perturbed_mean - clean_mean
        elif perturbed_mean is not None:
            uncertainty_delta = perturbed_mean - mean_uncertainty
        else:
            uncertainty_delta = 0.0

        dispersion_row = dispersion_by_model[dispersion_by_model[model_column].astype(str) == model_label] if not dispersion_by_model.empty and model_column in dispersion_by_model.columns else pd.DataFrame()
        mean_confidence_variance = float(pd.to_numeric(dispersion_row["mean_confidence_variance"], errors="coerce").dropna().mean()) if not dispersion_row.empty and "mean_confidence_variance" in dispersion_row.columns else None
        mean_confidence_std = float(pd.to_numeric(dispersion_row["mean_confidence_std"], errors="coerce").dropna().mean()) if not dispersion_row.empty and "mean_confidence_std" in dispersion_row.columns else None
        mean_confidence_spread = float(pd.to_numeric(dispersion_row["mean_confidence_spread"], errors="coerce").dropna().mean()) if not dispersion_row.empty and "mean_confidence_spread" in dispersion_row.columns else None
        mean_confidence_concentration = float(pd.to_numeric(dispersion_row["mean_confidence_concentration"], errors="coerce").dropna().mean()) if not dispersion_row.empty and "mean_confidence_concentration" in dispersion_row.columns else None

        collapse_row = collapse_summary[collapse_summary[model_column].astype(str) == model_label] if not collapse_summary.empty else pd.DataFrame()
        collapse_count = int(collapse_row["collapse_count"].iloc[0]) if not collapse_row.empty else 0
        mean_entropy_shift = float(collapse_row["mean_entropy_shift"].iloc[0]) if not collapse_row.empty else 0.0

        spread_component = 0.0 if mean_confidence_spread is None or pd.isna(mean_confidence_spread) else float(mean_confidence_spread)
        sensitivity_score = float(np.clip(0.7 * max(uncertainty_delta, 0.0) + 0.3 * spread_component, 0.0, 1.0))
        if mean_uncertainty >= 0.80 or uncertainty_delta >= 0.30 or collapse_count > 0:
            stability_label = "highly_sensitive"
        elif mean_uncertainty >= 0.60 or uncertainty_delta >= 0.15:
            stability_label = "sensitive"
        elif mean_uncertainty >= 0.35 or uncertainty_delta >= 0.08:
            stability_label = "moderate"
        elif mean_uncertainty >= 0.15:
            stability_label = "stable"
        else:
            stability_label = "very_stable"

        rows.append({
            model_column: model_label,
            "total_observations": int(len(uncertainty)),
            "mean_uncertainty_score": round(mean_uncertainty, 6),
            "max_uncertainty_score": round(max_uncertainty, 6),
            "min_uncertainty_score": round(min_uncertainty, 6),
            "uncertainty_std": round(uncertainty_std, 6),
            "clean_mean_uncertainty": round(clean_mean, 6) if clean_mean is not None else None,
            "perturbed_mean_uncertainty": round(perturbed_mean, 6) if perturbed_mean is not None else None,
            "uncertainty_delta": round(float(uncertainty_delta), 6),
            "mean_confidence_variance": round(mean_confidence_variance, 6) if mean_confidence_variance is not None else None,
            "mean_confidence_std": round(mean_confidence_std, 6) if mean_confidence_std is not None else None,
            "mean_confidence_spread": round(mean_confidence_spread, 6) if mean_confidence_spread is not None else None,
            "mean_confidence_concentration": round(mean_confidence_concentration, 6) if mean_confidence_concentration is not None else None,
            "collapse_count": collapse_count,
            "mean_entropy_shift": round(mean_entropy_shift, 6),
            "perturbation_sensitivity_score": round(sensitivity_score, 6),
            "stability_label": stability_label,
        })

    if not rows:
        return pd.DataFrame(columns=[
            model_column,
            "total_observations",
            "mean_uncertainty_score",
            "max_uncertainty_score",
            "min_uncertainty_score",
            "uncertainty_std",
            "clean_mean_uncertainty",
            "perturbed_mean_uncertainty",
            "uncertainty_delta",
            "mean_confidence_variance",
            "mean_confidence_std",
            "mean_confidence_spread",
            "mean_confidence_concentration",
            "collapse_count",
            "mean_entropy_shift",
            "perturbation_sensitivity_score",
            "stability_label",
        ])

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["mean_uncertainty_score", "uncertainty_delta", model_column],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def compute_model_uncertainty_summary(model_profiles: pd.DataFrame) -> Dict[str, Any]:
    """Summarise per-model uncertainty profiles."""
    if model_profiles is None or model_profiles.empty or "mean_uncertainty_score" not in model_profiles.columns:
        return {
            "total_models": 0,
            "mean_uncertainty": 0.0,
            "most_stable_model": None,
            "highest_uncertainty_contributor": None,
            "fastest_deteriorating_model": None,
            "perturbation_sensitive_models": [],
            "model_rankings": [],
        }

    ordered = model_profiles.sort_values(
        ["mean_uncertainty_score", "uncertainty_delta"], ascending=[False, False]
    ).reset_index(drop=True)

    model_rankings: List[Dict[str, Any]] = []
    for _, row in ordered.iterrows():
        model_rankings.append({
            "model_name": str(row.get("model_name", "")),
            "mean_uncertainty_score": round(_safe_float(row.get("mean_uncertainty_score"), 0.0), 6),
            "uncertainty_delta": round(_safe_float(row.get("uncertainty_delta"), 0.0), 6),
            "stability_label": str(row.get("stability_label", "")),
        })

    most_stable_model = str(ordered.iloc[-1]["model_name"]) if not ordered.empty else None
    highest_uncertainty_contributor = str(ordered.iloc[0]["model_name"]) if not ordered.empty else None

    delta_sorted = model_profiles.sort_values(
        ["uncertainty_delta", "mean_uncertainty_score"], ascending=[False, False]
    ).reset_index(drop=True)
    fastest_deteriorating_model = str(delta_sorted.iloc[0]["model_name"]) if not delta_sorted.empty else None

    perturbation_sensitive_models = [
        str(value)
        for value in model_profiles.loc[
            model_profiles.get("stability_label", pd.Series(dtype=str)).astype(str).isin({"sensitive", "highly_sensitive"}),
            "model_name",
        ].tolist()
    ] if "stability_label" in model_profiles.columns else []

    return {
        "total_models": int(len(model_profiles)),
        "mean_uncertainty": round(float(pd.to_numeric(model_profiles["mean_uncertainty_score"], errors="coerce").dropna().mean()), 6),
        "most_stable_model": most_stable_model,
        "highest_uncertainty_contributor": highest_uncertainty_contributor,
        "fastest_deteriorating_model": fastest_deteriorating_model,
        "perturbation_sensitive_models": perturbation_sensitive_models,
        "model_rankings": model_rankings,
    }


def generate_uncertainty_trend_summary(
    entropy_analysis: pd.DataFrame,
    dispersion_analysis: Optional[pd.DataFrame] = None,
    collapse_events: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Generate trend summaries for uncertainty growth and collapse."""
    severity_curve = compute_uncertainty_severity_curve(entropy_analysis)
    perturbation_ranking = rank_perturbation_uncertainty(entropy_analysis)
    collapse_summary = compute_confidence_collapse_summary(collapse_events)

    if severity_curve is not None and not severity_curve.empty:
        scores = pd.to_numeric(severity_curve["mean_uncertainty_score"], errors="coerce").dropna()
        labels = severity_curve["severity_level"].astype(str).tolist() if "severity_level" in severity_curve.columns else []
        if len(scores) >= 2:
            delta = float(scores.iloc[-1] - scores.iloc[0])
            if delta > 0.02:
                trend_direction = "increasing"
            elif delta < -0.02:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
            max_index = int(scores.idxmax())
            max_label = str(severity_curve.loc[max_index, "severity_level"]) if "severity_level" in severity_curve.columns else None
            monotonic = bool(all(scores.iloc[index] <= scores.iloc[index + 1] for index in range(len(scores) - 1)))
        else:
            delta = 0.0
            trend_direction = "insufficient_data"
            max_label = labels[0] if labels else None
            monotonic = False
    else:
        delta = 0.0
        trend_direction = "insufficient_data"
        max_label = None
        monotonic = False

    if perturbation_ranking is not None and not perturbation_ranking.empty:
        highest_perturbation = str(perturbation_ranking.iloc[0]["perturbation_type"])
        lowest_perturbation = str(perturbation_ranking.iloc[-1]["perturbation_type"])
        spread = float(pd.to_numeric(perturbation_ranking["mean_uncertainty_score"], errors="coerce").dropna().iloc[0] - pd.to_numeric(perturbation_ranking["mean_uncertainty_score"], errors="coerce").dropna().iloc[-1]) if len(perturbation_ranking) > 1 else 0.0
    else:
        highest_perturbation = None
        lowest_perturbation = None
        spread = 0.0

    summary_line = "Uncertainty remains stable across observed perturbations."
    if trend_direction == "increasing" and highest_perturbation is not None:
        summary_line = f"Uncertainty increases with perturbation severity and {highest_perturbation} produces the highest uncertainty."
    elif highest_perturbation is not None:
        summary_line = f"{highest_perturbation} produces the highest uncertainty contribution."

    return {
        "severity_curve": [] if severity_curve is None else _serialise(severity_curve.to_dict(orient="records")),
        "trend_direction": trend_direction,
        "uncertainty_delta": round(delta, 6),
        "max_uncertainty_severity": max_label,
        "monotonic": monotonic,
        "perturbation_ranking": [] if perturbation_ranking is None else _serialise(perturbation_ranking.to_dict(orient="records")),
        "highest_risk_perturbation": highest_perturbation,
        "lowest_risk_perturbation": lowest_perturbation,
        "perturbation_spread": round(spread, 6),
        "collapse_summary": collapse_summary,
        "summary": summary_line,
    }


def compute_confidence_collapse_summary(collapse_events: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Summarise confidence collapse detections."""
    if collapse_events is None or collapse_events.empty:
        return {
            "total_collapses": 0,
            "unique_samples": 0,
            "unique_models": 0,
            "collapse_triggers": {},
            "collapse_severities": {},
            "mean_entropy_shift": 0.0,
            "mean_confidence_drop": 0.0,
            "collapse_sample_ids": [],
        }

    trigger_counts = collapse_events["collapse_trigger"].astype(str).value_counts().to_dict() if "collapse_trigger" in collapse_events.columns else {}
    severity_counts = collapse_events["collapse_severity"].astype(str).value_counts().to_dict() if "collapse_severity" in collapse_events.columns else {}
    entropy_shift = pd.to_numeric(collapse_events["entropy_shift"], errors="coerce").dropna() if "entropy_shift" in collapse_events.columns else pd.Series(dtype=float)
    confidence_drop = pd.to_numeric(collapse_events["confidence_drop"], errors="coerce").dropna() if "confidence_drop" in collapse_events.columns else pd.Series(dtype=float)

    return {
        "total_collapses": int(len(collapse_events)),
        "unique_samples": int(collapse_events["sample_id"].nunique()) if "sample_id" in collapse_events.columns else int(len(collapse_events)),
        "unique_models": int(collapse_events["model_name"].nunique()) if "model_name" in collapse_events.columns and collapse_events["model_name"].notna().any() else 0,
        "collapse_triggers": trigger_counts,
        "collapse_severities": severity_counts,
        "mean_entropy_shift": round(float(entropy_shift.mean()), 6) if not entropy_shift.empty else 0.0,
        "mean_confidence_drop": round(float(confidence_drop.mean()), 6) if not confidence_drop.empty else 0.0,
        "collapse_sample_ids": [str(value) for value in collapse_events["sample_id"].tolist()] if "sample_id" in collapse_events.columns else [],
    }


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _export_dataframe_csv(frame: pd.DataFrame, output_dir: Optional[Union[str, Path]], default_dir: Path, filename: str) -> Path:
    directory = _ensure_dir(output_dir, default_dir)
    filepath = directory / filename
    export_frame = frame.copy() if frame is not None else pd.DataFrame()
    if not export_frame.empty:
        export_frame.to_csv(filepath, index=False, float_format="%.6f")
    else:
        export_frame.to_csv(filepath, index=False)
    return filepath


def _export_dataframe_json(frame: pd.DataFrame, output_dir: Optional[Union[str, Path]], default_dir: Path, filename: str) -> Path:
    directory = _ensure_dir(output_dir, default_dir)
    filepath = directory / filename
    records = frame.to_dict(orient="records") if frame is not None else []
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(_serialise(records), handle, indent=2, ensure_ascii=False)
    return filepath


def _export_mapping_json(mapping: Dict[str, Any], output_dir: Optional[Union[str, Path]], default_dir: Path, filename: str) -> Path:
    directory = _ensure_dir(output_dir, default_dir)
    filepath = directory / filename
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(_serialise(mapping), handle, indent=2, ensure_ascii=False)
    return filepath


def _export_mapping_csv(mapping: Dict[str, Any], output_dir: Optional[Union[str, Path]], default_dir: Path, filename: str) -> Path:
    directory = _ensure_dir(output_dir, default_dir)
    filepath = directory / filename
    rows = _flatten_mapping(mapping)
    pd.DataFrame(rows).to_csv(filepath, index=False, float_format="%.6f")
    return filepath


def export_entropy_analysis_csv(
    entropy_analysis: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "entropy_analysis.csv",
) -> Path:
    return _export_dataframe_csv(entropy_analysis, output_dir, _ENTROPY_DIR, filename)


def export_entropy_analysis_json(
    entropy_analysis: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "entropy_analysis.json",
) -> Path:
    return _export_dataframe_json(entropy_analysis, output_dir, _ENTROPY_DIR, filename)


def export_confidence_dispersion_csv(
    dispersion: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "confidence_dispersion.csv",
) -> Path:
    return _export_dataframe_csv(dispersion, output_dir, _DISPERSION_DIR, filename)


def export_confidence_dispersion_json(
    dispersion: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "confidence_dispersion.json",
) -> Path:
    return _export_dataframe_json(dispersion, output_dir, _DISPERSION_DIR, filename)


def export_confidence_collapse_csv(
    collapse_events: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "confidence_collapse.csv",
) -> Path:
    return _export_dataframe_csv(collapse_events, output_dir, _ENTROPY_DIR, filename)


def export_confidence_collapse_json(
    collapse_events: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "confidence_collapse.json",
) -> Path:
    return _export_dataframe_json(collapse_events, output_dir, _ENTROPY_DIR, filename)


def export_uncertainty_trends_csv(
    trends: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "uncertainty_trends.csv",
) -> Path:
    return _export_mapping_csv(trends, output_dir, _TREND_DIR, filename)


def export_uncertainty_trends_json(
    trends: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "uncertainty_trends.json",
) -> Path:
    return _export_mapping_json(trends, output_dir, _TREND_DIR, filename)


def export_uncertainty_summary_json(
    summary: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "uncertainty_summary.json",
) -> Path:
    return _export_mapping_json(summary, output_dir, _TREND_DIR, filename)


def export_relationship_summary_json(
    summary: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "uncertainty_relationship_summary.json",
) -> Path:
    return _export_mapping_json(summary, output_dir, _TREND_DIR, filename)


def export_uncertainty_conflicts_csv(
    conflicts: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "uncertainty_conflicts.csv",
) -> Path:
    return _export_dataframe_csv(conflicts, output_dir, _TREND_DIR, filename)


def export_uncertainty_conflicts_json(
    conflicts: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "uncertainty_conflicts.json",
) -> Path:
    return _export_dataframe_json(conflicts, output_dir, _TREND_DIR, filename)


def export_model_uncertainty_profiles_csv(
    profiles: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "model_uncertainty_profiles.csv",
) -> Path:
    return _export_dataframe_csv(profiles, output_dir, _PROFILE_DIR, filename)


def export_model_uncertainty_profiles_json(
    profiles: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "model_uncertainty_profiles.json",
) -> Path:
    return _export_dataframe_json(profiles, output_dir, _PROFILE_DIR, filename)


def export_model_uncertainty_summary_json(
    summary: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "model_uncertainty_summary.json",
) -> Path:
    return _export_mapping_json(summary, output_dir, _PROFILE_DIR, filename)
