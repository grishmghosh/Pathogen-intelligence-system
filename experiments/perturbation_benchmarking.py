"""
Perturbation benchmarking for deterministic comparison of degradation patterns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.experiment_statistics import compute_group_statistics, compute_severity_escalation, export_csv, export_json


PERTURBATION_METRICS = [
    "robustness_score",
    "instability_score",
    "mean_sample_instability",
    "overall_instability",
    "uncertainty_score",
    "mean_uncertainty_score",
    "risk_score",
    "reliability_risk_score",
    "attention_drift_score",
    "mean_attention_drift_score",
    "consensus_reliability_score",
    "consensus_reliability",
    "agreement_rate",
    "disagreement_rate",
]


def _to_frame(records):
    if records is None:
        return pd.DataFrame()
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, list):
        return pd.DataFrame(records)
    if isinstance(records, dict):
        return pd.DataFrame([records])
    return pd.DataFrame([records])


def _normalise_frame(frame):
    if frame.empty:
        return frame
    working = frame.copy()
    if "perturbation" in working.columns and "perturbation_type" not in working.columns:
        working = working.rename(columns={"perturbation": "perturbation_type"})
    if "mean_attention_drift_score" in working.columns and "attention_drift_score" not in working.columns:
        working["attention_drift_score"] = working["mean_attention_drift_score"]
    if "mean_uncertainty_score" in working.columns and "uncertainty_score" not in working.columns:
        working["uncertainty_score"] = working["mean_uncertainty_score"]
    if "reliability_risk_score" in working.columns and "risk_score" not in working.columns:
        working["risk_score"] = working["reliability_risk_score"]
    if "consensus_reliability" in working.columns and "consensus_reliability_score" not in working.columns:
        working["consensus_reliability_score"] = working["consensus_reliability"]
    return working


def build_perturbation_benchmark_frame(records):
    frame = _normalise_frame(_to_frame(records))
    if frame.empty:
        return frame
    perturbation_column = "perturbation_type" if "perturbation_type" in frame.columns else "perturbation"
    if perturbation_column not in frame.columns:
        return pd.DataFrame()
    available_metrics = [column for column in PERTURBATION_METRICS if column in frame.columns]
    if not available_metrics:
        return pd.DataFrame()
    baseline_mask = frame[perturbation_column].astype(str).str.lower().isin({"clean", "original", "baseline"})
    baseline_frame = frame[baseline_mask].copy()
    if baseline_frame.empty:
        baseline_frame = frame.copy()
    rows = []
    for perturbation_type, group in frame.groupby(perturbation_column, dropna=False):
        row = {
            "perturbation_type": str(perturbation_type),
            "record_count": int(len(group)),
        }
        for metric in available_metrics:
            series = pd.to_numeric(group[metric], errors="coerce").dropna()
            if series.empty:
                continue
            row[f"mean_{metric}"] = round(float(series.mean()), 6)
            row[f"median_{metric}"] = round(float(series.median()), 6)
            row[f"std_{metric}"] = round(float(series.std(ddof=0)), 6)
            row[f"min_{metric}"] = round(float(series.min()), 6)
            row[f"max_{metric}"] = round(float(series.max()), 6)
            baseline_series = pd.to_numeric(baseline_frame[metric], errors="coerce").dropna()
            if baseline_series.empty:
                row[f"delta_{metric}"] = None
            else:
                row[f"delta_{metric}"] = round(float(series.mean() - baseline_series.mean()), 6)
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    for column in ["delta_robustness_score", "delta_uncertainty_score", "delta_attention_drift_score", "delta_consensus_reliability_score", "delta_risk_score", "delta_disagreement_rate"]:
        if column not in result.columns:
            result[column] = 0.0
    result["degradation_impact"] = result[[column for column in ["delta_robustness_score"] if column in result.columns]].fillna(0.0).apply(lambda row: -float(row.iloc[0]) if len(row) else 0.0, axis=1)
    result["disagreement_escalation"] = result[[column for column in ["delta_disagreement_rate"] if column in result.columns]].fillna(0.0).apply(lambda row: float(row.iloc[0]) if len(row) else 0.0, axis=1)
    result["uncertainty_escalation"] = result[[column for column in ["delta_uncertainty_score"] if column in result.columns]].fillna(0.0).apply(lambda row: float(row.iloc[0]) if len(row) else 0.0, axis=1)
    result["attention_drift_impact"] = result[[column for column in ["delta_attention_drift_score"] if column in result.columns]].fillna(0.0).apply(lambda row: float(row.iloc[0]) if len(row) else 0.0, axis=1)
    result["collapse_pressure"] = result[[column for column in ["disagreement_escalation", "uncertainty_escalation", "attention_drift_impact"] if column in result.columns]].fillna(0.0).mean(axis=1)
    return result.sort_values(["collapse_pressure", "degradation_impact"], ascending=[False, False]).reset_index(drop=True)


def benchmark_severity_levels(records, severity_order=None):
    frame = _normalise_frame(_to_frame(records))
    if frame.empty or "severity_level" not in frame.columns:
        return pd.DataFrame(), {}
    order = severity_order or ["mild", "moderate", "severe"]
    severity_frame, escalation = compute_severity_escalation(frame, "severity_level", [column for column in PERTURBATION_METRICS if column in frame.columns], order)
    if severity_frame.empty:
        return severity_frame, escalation
    if "mean_robustness_score" in severity_frame.columns:
        severity_frame["degradation_curve"] = -pd.to_numeric(severity_frame["mean_robustness_score"], errors="coerce").fillna(0.0)
    if "mean_uncertainty_score" in severity_frame.columns:
        severity_frame["uncertainty_growth"] = pd.to_numeric(severity_frame["mean_uncertainty_score"], errors="coerce").fillna(0.0)
    if "mean_attention_drift_score" in severity_frame.columns:
        severity_frame["attention_drift_growth"] = pd.to_numeric(severity_frame["mean_attention_drift_score"], errors="coerce").fillna(0.0)
    if "mean_consensus_reliability_score" in severity_frame.columns:
        severity_frame["consensus_collapse_behavior"] = 1.0 - pd.to_numeric(severity_frame["mean_consensus_reliability_score"], errors="coerce").fillna(0.0)
    severity_frame["escalation_rank"] = np.arange(1, len(severity_frame) + 1)
    severity_summary = {
        "severity_levels": severity_frame["severity_level"].astype(str).tolist(),
        "trend_direction": "increasing" if escalation and any(value > 0 for value in escalation.values()) else "stable",
        "escalation_rankings": severity_frame[[column for column in severity_frame.columns if column in {"severity_level", "escalation_rank", "degradation_curve", "uncertainty_growth", "attention_drift_growth", "consensus_collapse_behavior"}]].to_dict(orient="records"),
        "escalation_deltas": escalation,
    }
    return severity_frame, severity_summary


def benchmark_perturbations(records):
    frame = build_perturbation_benchmark_frame(records)
    severity_frame, severity_summary = benchmark_severity_levels(records)
    if frame.empty:
        return {
            "perturbation_frame": frame,
            "severity_frame": severity_frame,
            "rankings": pd.DataFrame(),
            "summary": {
                "total_perturbations": 0,
                "most_disruptive_perturbation": None,
                "most_uncertain_perturbation": None,
                "most_fragile_perturbation": None,
            },
            "severity_summary": severity_summary,
        }
    ranking_frame = frame.copy()
    ranking_frame = ranking_frame.sort_values(["collapse_pressure", "degradation_impact"], ascending=[False, False]).reset_index(drop=True)
    summary = {
        "total_perturbations": int(len(ranking_frame)),
        "most_disruptive_perturbation": str(ranking_frame.iloc[0]["perturbation_type"]),
        "most_uncertain_perturbation": str(ranking_frame.sort_values("uncertainty_escalation", ascending=False).iloc[0]["perturbation_type"]) if "uncertainty_escalation" in ranking_frame.columns else str(ranking_frame.iloc[0]["perturbation_type"]),
        "most_fragile_perturbation": str(ranking_frame.sort_values("degradation_impact", ascending=False).iloc[0]["perturbation_type"]),
    }
    summary["fragility_summary"] = {
        "mean_degradation_impact": round(float(pd.to_numeric(ranking_frame["degradation_impact"], errors="coerce").fillna(0.0).mean()), 6),
        "mean_disagreement_escalation": round(float(pd.to_numeric(ranking_frame["disagreement_escalation"], errors="coerce").fillna(0.0).mean()), 6),
        "mean_uncertainty_escalation": round(float(pd.to_numeric(ranking_frame["uncertainty_escalation"], errors="coerce").fillna(0.0).mean()), 6),
        "mean_attention_drift_impact": round(float(pd.to_numeric(ranking_frame["attention_drift_impact"], errors="coerce").fillna(0.0).mean()), 6),
    }
    return {
        "perturbation_frame": frame,
        "severity_frame": severity_frame,
        "rankings": ranking_frame,
        "summary": summary,
        "severity_summary": severity_summary,
        "statistics": compute_group_statistics(frame, ["perturbation_type"], [column for column in PERTURBATION_METRICS if column in frame.columns]),
    }


def export_perturbation_benchmark_outputs(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_csv(output_dir / "perturbation_benchmark_rankings.csv", result.get("rankings", pd.DataFrame()))
    export_json(output_dir / "perturbation_benchmark_rankings.json", result.get("rankings", pd.DataFrame()).to_dict(orient="records") if isinstance(result.get("rankings"), pd.DataFrame) else result.get("rankings", []))
    export_csv(output_dir / "severity_benchmark.csv", result.get("severity_frame", pd.DataFrame()))
    export_json(output_dir / "severity_benchmark.json", result.get("severity_frame", pd.DataFrame()).to_dict(orient="records") if isinstance(result.get("severity_frame"), pd.DataFrame) else result.get("severity_frame", []))
    export_json(output_dir / "perturbation_benchmark_summary.json", result.get("summary", {}))
    export_json(output_dir / "severity_benchmark_summary.json", result.get("severity_summary", {}))
    export_csv(output_dir / "perturbation_statistical_summary.csv", result.get("statistics", pd.DataFrame()))
    return output_dir
