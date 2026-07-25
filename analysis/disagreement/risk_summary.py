"""
Risk Summary, Fragility Analysis, Trend Analysis, and Export for Step 5.

This module stays separate from scoring and classification. It consumes the
DataFrames produced by Steps 1–4 plus the Step 5 risk tables and provides:

* aggregated sample-risk summaries
* contributor summaries
* perturbation fragility rankings
* model-risk summaries
* deterministic risk trend summaries
* CSV / JSON export helpers

All exports are written under results/disagreement/.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from analysis.disagreement.disagreement_export import _make_serialisable


_RISK_ANALYSIS_DIR = Path("results") / "disagreement" / "risk_analysis"
_FRAGILITY_DIR = Path("results") / "disagreement" / "fragility_analysis"
_TREND_DIR = Path("results") / "disagreement" / "risk_trends"


def _ensure_dir(output_dir: Optional[Union[str, Path]], default_dir: Path) -> Path:
    directory = Path(output_dir) if output_dir is not None else default_dir
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _clamp01(value: object, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if not np.isfinite(numeric):
        numeric = default
    return float(np.clip(numeric, 0.0, 1.0))


def _first_present_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _empty_dataframe(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _severity_sort_key(labels: Sequence[str]) -> List[str]:
    order = {
        "clean": 0,
        "none": 0,
        "original": 0,
        "low": 1,
        "moderate": 2,
        "high": 3,
        "critical": 4,
    }
    return sorted([str(label) for label in labels], key=lambda label: (order.get(label, 99), label))


def _collapse_nested_report(report: Dict, prefix: str = "") -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key, value in report.items():
        path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            rows.extend(_collapse_nested_report(value, path))
        elif isinstance(value, list):
            if not value:
                rows.append({"path": path, "value": []})
            else:
                for index, item in enumerate(value):
                    item_path = f"{path}[{index}]"
                    if isinstance(item, dict):
                        rows.extend(_collapse_nested_report(item, item_path))
                    else:
                        rows.append({"path": item_path, "value": item})
        else:
            rows.append({"path": path, "value": value})
    return rows


# ---------------------------------------------------------------------------
# Sample-risk summary and contributor analysis
# ---------------------------------------------------------------------------


def compute_risk_summary(risk_scores: pd.DataFrame) -> Dict:
    """
    Summarise sample-level reliability risk results.
    """
    if risk_scores is None or risk_scores.empty or "reliability_risk_score" not in risk_scores.columns:
        return {
            "total_samples": 0,
            "mean_risk": 0.0,
            "max_risk": 0.0,
            "min_risk": 0.0,
            "median_risk": 0.0,
            "high_risk_count": 0,
            "critical_risk_count": 0,
            "component_means": {},
            "risk_level_distribution": {},
            "top_risk_sample_ids": [],
        }

    scores = pd.to_numeric(risk_scores["reliability_risk_score"], errors="coerce").dropna()
    if scores.empty:
        return {
            "total_samples": 0,
            "mean_risk": 0.0,
            "max_risk": 0.0,
            "min_risk": 0.0,
            "median_risk": 0.0,
            "high_risk_count": 0,
            "critical_risk_count": 0,
            "component_means": {},
            "risk_level_distribution": {},
            "top_risk_sample_ids": [],
        }

    component_means: Dict[str, float] = {}
    for column in [
        "disagreement_risk",
        "instability_risk",
        "consensus_reliability_risk",
        "fragility_risk",
        "confidence_instability_risk",
        "escalation_risk",
    ]:
        if column in risk_scores.columns:
            numeric = pd.to_numeric(risk_scores[column], errors="coerce").dropna()
            if not numeric.empty:
                component_means[column] = round(float(numeric.mean()), 6)

    level_counts: Dict[str, int] = {}
    if "risk_level" in risk_scores.columns:
        level_counts = risk_scores["risk_level"].astype(str).value_counts().to_dict()

    top_ids = []
    if "sample_id" in risk_scores.columns:
        top_ids = [str(v) for v in risk_scores.sort_values(
            "reliability_risk_score", ascending=False
        ).head(5)["sample_id"].tolist()]

    return {
        "total_samples": int(len(scores)),
        "mean_risk": round(float(scores.mean()), 6),
        "max_risk": round(float(scores.max()), 6),
        "min_risk": round(float(scores.min()), 6),
        "median_risk": round(float(scores.median()), 6),
        "high_risk_count": int((scores >= 0.60).sum()),
        "critical_risk_count": int((scores >= 0.80).sum()),
        "component_means": component_means,
        "risk_level_distribution": level_counts,
        "top_risk_sample_ids": top_ids,
    }


def compute_risk_contributor_summary(risk_scores: pd.DataFrame) -> Dict:
    """
    Summarise which factors contributed most often to high reliability risk.
    """
    if risk_scores is None or risk_scores.empty:
        return {
            "dominant_factor_counts": {},
            "secondary_factor_counts": {},
            "most_common_dominant_factor": None,
            "high_risk_sample_ids": [],
        }

    dominant_counts: Dict[str, int] = {}
    secondary_counts: Dict[str, int] = {}

    if "dominant_risk_factor" in risk_scores.columns:
        dominant_counts = risk_scores["dominant_risk_factor"].dropna().astype(str).value_counts().to_dict()
    if "secondary_risk_factors" in risk_scores.columns:
        for factors in risk_scores["secondary_risk_factors"]:
            if isinstance(factors, list):
                for factor in factors:
                    secondary_counts[str(factor)] = secondary_counts.get(str(factor), 0) + 1

    most_common = None
    if dominant_counts:
        most_common = max(dominant_counts, key=lambda key: dominant_counts[key])

    high_risk_ids: List[str] = []
    if {"sample_id", "reliability_risk_score"}.issubset(risk_scores.columns):
        subset = risk_scores.sort_values("reliability_risk_score", ascending=False).head(5)
        high_risk_ids = [str(v) for v in subset["sample_id"].tolist()]

    return {
        "dominant_factor_counts": dominant_counts,
        "secondary_factor_counts": secondary_counts,
        "most_common_dominant_factor": most_common,
        "high_risk_sample_ids": high_risk_ids,
    }


# ---------------------------------------------------------------------------
# Fragility analysis
# ---------------------------------------------------------------------------


def compute_perturbation_fragility_analysis(
    risk_scores: Optional[pd.DataFrame] = None,
    perturbation_sensitivity: Optional[pd.DataFrame] = None,
    severity_rates: Optional[pd.DataFrame] = None,
    consensus_stability: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a deterministic fragility ranking for perturbations and severities.

    The function prefers direct risk-bearing columns if present. If those are
    unavailable, it falls back to Step 3 disagreement rates as a proxy so the
    analysis remains usable without redesigning earlier steps.
    """
    rows: List[Dict[str, object]] = []

    if risk_scores is not None and not risk_scores.empty:
        if {"perturbation_type", "reliability_risk_score"}.issubset(risk_scores.columns):
            for perturbation_type, group in risk_scores.groupby("perturbation_type"):
                risk_values = pd.to_numeric(group["reliability_risk_score"], errors="coerce").dropna()
                if risk_values.empty:
                    continue
                rows.append({
                    "analysis_type": "perturbation",
                    "factor_name": str(perturbation_type),
                    "mean_risk": round(float(risk_values.mean()), 6),
                    "max_risk": round(float(risk_values.max()), 6),
                    "min_risk": round(float(risk_values.min()), 6),
                    "risk_delta": round(float(risk_values.max() - risk_values.min()), 6),
                    "observation_count": int(len(risk_values)),
                })
        if {"severity_level", "reliability_risk_score"}.issubset(risk_scores.columns):
            severity_order = _severity_sort_key(risk_scores["severity_level"].dropna().astype(str).unique())
            for severity_level in severity_order:
                group = risk_scores[risk_scores["severity_level"].astype(str) == severity_level]
                risk_values = pd.to_numeric(group["reliability_risk_score"], errors="coerce").dropna()
                if risk_values.empty:
                    continue
                rows.append({
                    "analysis_type": "severity",
                    "factor_name": str(severity_level),
                    "mean_risk": round(float(risk_values.mean()), 6),
                    "max_risk": round(float(risk_values.max()), 6),
                    "min_risk": round(float(risk_values.min()), 6),
                    "risk_delta": round(float(risk_values.max() - risk_values.min()), 6),
                    "observation_count": int(len(risk_values)),
                })

    if not rows and perturbation_sensitivity is not None and not perturbation_sensitivity.empty:
        if "perturbation_type" in perturbation_sensitivity.columns:
            for _, row in perturbation_sensitivity.iterrows():
                risk_value = _clamp01(row.get("disagreement_rate"), 0.5)
                rows.append({
                    "analysis_type": "perturbation",
                    "factor_name": str(row.get("perturbation_type", "unknown")),
                    "mean_risk": round(risk_value, 6),
                    "max_risk": round(risk_value, 6),
                    "min_risk": round(risk_value, 6),
                    "risk_delta": 0.0,
                    "observation_count": int(row.get("total_observations", 0) or 0),
                })

    if not rows and severity_rates is not None and not severity_rates.empty:
        if "severity_level" in severity_rates.columns:
            for _, row in severity_rates.iterrows():
                risk_value = _clamp01(row.get("disagreement_rate"), 0.5)
                rows.append({
                    "analysis_type": "severity",
                    "factor_name": str(row.get("severity_level", "unknown")),
                    "mean_risk": round(risk_value, 6),
                    "max_risk": round(risk_value, 6),
                    "min_risk": round(risk_value, 6),
                    "risk_delta": 0.0,
                    "observation_count": int(row.get("total_observations", 0) or 0),
                })

    if not rows and consensus_stability is not None and not consensus_stability.empty:
        if {"severity_level", "severity_rank", "models_agree"}.issubset(consensus_stability.columns):
            for severity_level, group in consensus_stability.groupby("severity_level"):
                disagree_rate = 1.0 - float(pd.Series(group["models_agree"]).astype(bool).mean())
                rows.append({
                    "analysis_type": "severity",
                    "factor_name": str(severity_level),
                    "mean_risk": round(float(np.clip(disagree_rate, 0.0, 1.0)), 6),
                    "max_risk": round(float(np.clip(disagree_rate, 0.0, 1.0)), 6),
                    "min_risk": round(float(np.clip(disagree_rate, 0.0, 1.0)), 6),
                    "risk_delta": 0.0,
                    "observation_count": int(len(group)),
                })

    if not rows:
        return _empty_dataframe([
            "analysis_type",
            "factor_name",
            "mean_risk",
            "max_risk",
            "min_risk",
            "risk_delta",
            "observation_count",
            "risk_rank",
            "summary",
        ])

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["analysis_type", "mean_risk", "factor_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    result["risk_rank"] = result.groupby("analysis_type").cumcount() + 1
    result["summary"] = result.apply(_fragility_summary_line, axis=1)
    return result


# ---------------------------------------------------------------------------
# Model-risk summary
# ---------------------------------------------------------------------------


def compute_model_risk_summary(model_risk_profiles: pd.DataFrame) -> Dict:
    """
    Summarise the model-risk profile table.
    """
    if model_risk_profiles is None or model_risk_profiles.empty or "reliability_risk_score" not in model_risk_profiles.columns:
        return {
            "total_models": 0,
            "mean_risk": 0.0,
            "most_stable_model": None,
            "highest_risk_model": None,
            "fastest_destabilizing_model": None,
            "highest_disagreement_contributor": None,
            "high_risk_model_names": [],
            "model_rankings": [],
        }

    risk_values = pd.to_numeric(model_risk_profiles["reliability_risk_score"], errors="coerce").dropna()
    if risk_values.empty:
        return {
            "total_models": 0,
            "mean_risk": 0.0,
            "most_stable_model": None,
            "highest_risk_model": None,
            "fastest_destabilizing_model": None,
            "highest_disagreement_contributor": None,
            "high_risk_model_names": [],
            "model_rankings": [],
        }

    ordered = model_risk_profiles.sort_values("reliability_risk_score", ascending=False).reset_index(drop=True)
    most_stable = str(ordered.iloc[-1]["model_name"]) if not ordered.empty else None
    highest_risk = str(ordered.iloc[0]["model_name"]) if not ordered.empty else None

    fastest_destabilizing = None
    if "instability_risk" in ordered.columns:
        instability_scores = pd.to_numeric(ordered["instability_risk"], errors="coerce").fillna(-1.0)
        if not instability_scores.empty:
            fastest_destabilizing = str(ordered.loc[instability_scores.idxmax(), "model_name"])

    highest_disagreement = None
    if "disagreement_contribution" in ordered.columns:
        disagreement_scores = pd.to_numeric(ordered["disagreement_contribution"], errors="coerce").fillna(-1.0)
        if not disagreement_scores.empty:
            highest_disagreement = str(ordered.loc[disagreement_scores.idxmax(), "model_name"])

    model_rankings = []
    for _, row in ordered.iterrows():
        model_rankings.append({
            "model_name": str(row["model_name"]),
            "risk_score": round(float(row["reliability_risk_score"]), 6),
            "risk_level": str(row.get("risk_level", "")),
            "dominant_risk_factor": str(row.get("dominant_risk_factor", "")),
        })

    high_risk_names = []
    if "risk_level" in ordered.columns:
        high_risk_names = [str(v) for v in ordered.loc[ordered["risk_level"].isin(["high", "critical"]), "model_name"].tolist()]

    return {
        "total_models": int(len(risk_values)),
        "mean_risk": round(float(risk_values.mean()), 6),
        "highest_risk_model": highest_risk,
        "most_stable_model": most_stable,
        "fastest_destabilizing_model": fastest_destabilizing,
        "highest_disagreement_contributor": highest_disagreement,
        "high_risk_model_names": high_risk_names,
        "model_rankings": model_rankings,
    }


# ---------------------------------------------------------------------------
# Risk trend analysis
# ---------------------------------------------------------------------------


def generate_risk_trend_summary(
    risk_scores: Optional[pd.DataFrame] = None,
    severity_rates: Optional[pd.DataFrame] = None,
    perturbation_sensitivity: Optional[pd.DataFrame] = None,
    model_risk_profiles: Optional[pd.DataFrame] = None,
    consensus_stability: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Generate deterministic trend summaries for reliability risk.
    """
    severity_trend = _generate_severity_trend(risk_scores, severity_rates, consensus_stability)
    perturbation_trend = _generate_perturbation_trend(risk_scores, perturbation_sensitivity)
    model_trend = _generate_model_trend(model_risk_profiles)
    return {
        "severity_trend": severity_trend,
        "perturbation_trend": perturbation_trend,
        "model_trend": model_trend,
    }


def _generate_severity_trend(
    risk_scores: Optional[pd.DataFrame],
    severity_rates: Optional[pd.DataFrame],
    consensus_stability: Optional[pd.DataFrame],
) -> Dict:
    if risk_scores is not None and not risk_scores.empty and {"severity_level", "reliability_risk_score"}.issubset(risk_scores.columns):
        order = _severity_sort_key(risk_scores["severity_level"].dropna().astype(str).unique())
        steps: List[Dict[str, object]] = []
        values: List[float] = []
        labels: List[str] = []
        for label in order:
            group = risk_scores[risk_scores["severity_level"].astype(str) == label]
            scores = pd.to_numeric(group["reliability_risk_score"], errors="coerce").dropna()
            if scores.empty:
                continue
            score = float(scores.mean())
            steps.append({"severity": str(label), "risk_score": round(score, 6)})
            values.append(score)
            labels.append(str(label))
        if not values:
            return {"trend_direction": "insufficient_data", "severity_steps": []}
        delta = values[-1] - values[0]
        direction = "stable"
        if delta > 0.02:
            direction = "increasing"
        elif delta < -0.02:
            direction = "decreasing"
        return {
            "trend_direction": direction,
            "severity_steps": steps,
            "risk_delta": round(float(delta), 6),
            "max_risk": round(float(max(values)), 6),
            "max_risk_severity": labels[int(np.argmax(values))],
            "monotonic": bool(all(values[index] <= values[index + 1] for index in range(len(values) - 1))),
        }

    if severity_rates is not None and not severity_rates.empty and {"severity_level", "disagreement_rate"}.issubset(severity_rates.columns):
        ordered = _severity_sort_key(severity_rates["severity_level"].dropna().astype(str).unique())
        steps = []
        values = []
        labels = []
        for label in ordered:
            group = severity_rates[severity_rates["severity_level"].astype(str) == label]
            rates = pd.to_numeric(group["disagreement_rate"], errors="coerce").dropna()
            if rates.empty:
                continue
            score = float(rates.mean())
            steps.append({"severity": str(label), "risk_score": round(score, 6)})
            values.append(score)
            labels.append(str(label))
        if not values:
            return {"trend_direction": "insufficient_data", "severity_steps": []}
        delta = values[-1] - values[0]
        direction = "stable"
        if delta > 0.02:
            direction = "increasing"
        elif delta < -0.02:
            direction = "decreasing"
        return {
            "trend_direction": direction,
            "severity_steps": steps,
            "risk_delta": round(float(delta), 6),
            "max_risk": round(float(max(values)), 6),
            "max_risk_severity": labels[int(np.argmax(values))],
            "monotonic": bool(all(values[index] <= values[index + 1] for index in range(len(values) - 1))),
        }

    if consensus_stability is not None and not consensus_stability.empty and {"severity_level", "models_agree"}.issubset(consensus_stability.columns):
        ordered = _severity_sort_key(consensus_stability["severity_level"].dropna().astype(str).unique())
        steps = []
        values = []
        labels = []
        for label in ordered:
            group = consensus_stability[consensus_stability["severity_level"].astype(str) == label]
            if group.empty:
                continue
            score = 1.0 - float(pd.Series(group["models_agree"]).astype(bool).mean())
            steps.append({"severity": str(label), "risk_score": round(score, 6)})
            values.append(score)
            labels.append(str(label))
        if not values:
            return {"trend_direction": "insufficient_data", "severity_steps": []}
        delta = values[-1] - values[0]
        direction = "stable"
        if delta > 0.02:
            direction = "increasing"
        elif delta < -0.02:
            direction = "decreasing"
        return {
            "trend_direction": direction,
            "severity_steps": steps,
            "risk_delta": round(float(delta), 6),
            "max_risk": round(float(max(values)), 6),
            "max_risk_severity": labels[int(np.argmax(values))],
            "monotonic": bool(all(values[index] <= values[index + 1] for index in range(len(values) - 1))),
        }

    return {"trend_direction": "insufficient_data", "severity_steps": []}


def _generate_perturbation_trend(
    risk_scores: Optional[pd.DataFrame],
    perturbation_sensitivity: Optional[pd.DataFrame],
) -> Dict:
    if risk_scores is not None and not risk_scores.empty and {"perturbation_type", "reliability_risk_score"}.issubset(risk_scores.columns):
        rows = []
        for perturbation_type, group in risk_scores.groupby("perturbation_type"):
            scores = pd.to_numeric(group["reliability_risk_score"], errors="coerce").dropna()
            if scores.empty:
                continue
            rows.append({
                "perturbation_type": str(perturbation_type),
                "risk_score": round(float(scores.mean()), 6),
                "observation_count": int(len(scores)),
            })
        if rows:
            ordered = sorted(rows, key=lambda row: row["risk_score"], reverse=True)
            spread = ordered[0]["risk_score"] - ordered[-1]["risk_score"] if len(ordered) > 1 else 0.0
            summary = f"{ordered[0]['perturbation_type']} produces the highest reliability risk."
            return {
                "ranking": ordered,
                "highest_risk_perturbation": ordered[0]["perturbation_type"],
                "lowest_risk_perturbation": ordered[-1]["perturbation_type"],
                "risk_spread": round(float(spread), 6),
                "summary": summary,
            }

    if perturbation_sensitivity is not None and not perturbation_sensitivity.empty and {"perturbation_type", "disagreement_rate"}.issubset(perturbation_sensitivity.columns):
        ordered = []
        for _, row in perturbation_sensitivity.iterrows():
            ordered.append({
                "perturbation_type": str(row.get("perturbation_type", "unknown")),
                "risk_score": round(_clamp01(row.get("disagreement_rate"), 0.5), 6),
                "observation_count": int(row.get("total_observations", 0) or 0),
            })
        ordered = sorted(ordered, key=lambda row: row["risk_score"], reverse=True)
        spread = ordered[0]["risk_score"] - ordered[-1]["risk_score"] if len(ordered) > 1 else 0.0
        summary = f"{ordered[0]['perturbation_type']} produces the highest disagreement-driven risk."
        return {
            "ranking": ordered,
            "highest_risk_perturbation": ordered[0]["perturbation_type"],
            "lowest_risk_perturbation": ordered[-1]["perturbation_type"],
            "risk_spread": round(float(spread), 6),
            "summary": summary,
        }

    return {
        "ranking": [],
        "highest_risk_perturbation": None,
        "lowest_risk_perturbation": None,
        "risk_spread": 0.0,
        "summary": "Insufficient data for perturbation risk trend analysis.",
    }


def _generate_model_trend(model_risk_profiles: Optional[pd.DataFrame]) -> Dict:
    if model_risk_profiles is None or model_risk_profiles.empty or "reliability_risk_score" not in model_risk_profiles.columns:
        return {
            "ranking": [],
            "most_stable_model": None,
            "highest_risk_model": None,
            "summary": "Insufficient data for model risk trend analysis.",
        }

    ordered = model_risk_profiles.sort_values("reliability_risk_score", ascending=False).reset_index(drop=True)
    ranking = []
    for _, row in ordered.iterrows():
        ranking.append({
            "model_name": str(row["model_name"]),
            "risk_score": round(float(row["reliability_risk_score"]), 6),
            "dominant_risk_factor": str(row.get("dominant_risk_factor", "")),
        })

    highest = ranking[0]["model_name"]
    stable = ranking[-1]["model_name"]
    summary = f"{highest} contributes disproportionately to reliability collapse relative to {stable}."
    return {
        "ranking": ranking,
        "most_stable_model": stable,
        "highest_risk_model": highest,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def export_risk_analysis_csv(
    risk_scores: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "reliability_risk_scores.csv",
) -> Path:
    directory = _ensure_dir(output_dir, _RISK_ANALYSIS_DIR)
    filepath = directory / filename
    export_df = risk_scores.copy() if risk_scores is not None else _empty_dataframe([])
    if not export_df.empty and "secondary_risk_factors" in export_df.columns:
        export_df = export_df.copy()
        export_df["secondary_risk_factors"] = export_df["secondary_risk_factors"].apply(
            lambda value: "|".join(value) if isinstance(value, list) else str(value)
        )
    if not export_df.empty and "risk_factor_scores" in export_df.columns:
        export_df["risk_factor_scores"] = export_df["risk_factor_scores"].apply(
            lambda value: json.dumps(_make_serialisable(value), ensure_ascii=False) if isinstance(value, dict) else str(value)
        )
    export_df.to_csv(filepath, index=False, float_format="%.6f")
    return filepath


def export_risk_analysis_json(
    risk_scores: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "reliability_risk_scores.json",
) -> Path:
    directory = _ensure_dir(output_dir, _RISK_ANALYSIS_DIR)
    filepath = directory / filename
    records = risk_scores.to_dict(orient="records") if risk_scores is not None else []
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(_make_serialisable(records), handle, indent=2, ensure_ascii=False)
    return filepath


def export_model_risk_profile_csv(
    model_risk_profiles: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "model_risk_profiles.csv",
) -> Path:
    directory = _ensure_dir(output_dir, _RISK_ANALYSIS_DIR)
    filepath = directory / filename
    export_df = model_risk_profiles.copy() if model_risk_profiles is not None else _empty_dataframe([])
    if not export_df.empty and "secondary_risk_factors" in export_df.columns:
        export_df["secondary_risk_factors"] = export_df["secondary_risk_factors"].apply(
            lambda value: "|".join(value) if isinstance(value, list) else str(value)
        )
    if not export_df.empty and "risk_factor_scores" in export_df.columns:
        export_df["risk_factor_scores"] = export_df["risk_factor_scores"].apply(
            lambda value: json.dumps(_make_serialisable(value), ensure_ascii=False) if isinstance(value, dict) else str(value)
        )
    export_df.to_csv(filepath, index=False, float_format="%.6f")
    return filepath


def export_model_risk_profile_json(
    model_risk_profiles: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "model_risk_profiles.json",
) -> Path:
    directory = _ensure_dir(output_dir, _RISK_ANALYSIS_DIR)
    filepath = directory / filename
    records = model_risk_profiles.to_dict(orient="records") if model_risk_profiles is not None else []
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(_make_serialisable(records), handle, indent=2, ensure_ascii=False)
    return filepath


def export_fragility_analysis_csv(
    fragility: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "fragility_analysis.csv",
) -> Path:
    directory = _ensure_dir(output_dir, _FRAGILITY_DIR)
    filepath = directory / filename
    export_df = fragility.copy() if fragility is not None else _empty_dataframe([])
    export_df.to_csv(filepath, index=False, float_format="%.6f")
    return filepath


def export_fragility_analysis_json(
    fragility: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "fragility_analysis.json",
) -> Path:
    directory = _ensure_dir(output_dir, _FRAGILITY_DIR)
    filepath = directory / filename
    records = fragility.to_dict(orient="records") if fragility is not None else []
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(_make_serialisable(records), handle, indent=2, ensure_ascii=False)
    return filepath


def export_risk_trends_csv(
    trends: Dict,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "risk_trends.csv",
) -> Path:
    directory = _ensure_dir(output_dir, _TREND_DIR)
    filepath = directory / filename
    rows = _collapse_nested_report(trends if trends is not None else {})
    pd.DataFrame(rows).to_csv(filepath, index=False, float_format="%.6f")
    return filepath


def export_risk_trends_json(
    trends: Dict,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "risk_trends.json",
) -> Path:
    directory = _ensure_dir(output_dir, _TREND_DIR)
    filepath = directory / filename
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(_make_serialisable(trends if trends is not None else {}), handle, indent=2, ensure_ascii=False)
    return filepath


# ---------------------------------------------------------------------------
# Internal trend helpers
# ---------------------------------------------------------------------------


def _severity_sort_key(labels: Sequence[str]) -> List[str]:
    order = {"clean": 0, "none": 0, "original": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}
    return sorted([str(label) for label in labels], key=lambda label: (order.get(label, 99), label))


def _fragility_summary_line(row: pd.Series) -> str:
    scope = str(row.get("analysis_type", "analysis"))
    factor = str(row.get("factor_name", "unknown"))
    risk = float(row.get("mean_risk", 0.0) or 0.0)
    if scope == "perturbation":
        return f"{factor} drives reliability risk at {risk:.3f}."
    return f"{factor} shows risk level {risk:.3f}."
