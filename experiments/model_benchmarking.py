"""
Model benchmarking for deterministic experiment comparisons.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.experiment_statistics import compute_scalar_statistics, export_csv, export_json, to_json_ready


MODEL_METRICS = [
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
    "attention_stability_score",
    "consensus_reliability_score",
    "consensus_reliability",
    "agreement_rate",
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


def _choose_column(frame, candidates):
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _normalise_model_frame(frame):
    if frame.empty:
        return frame
    working = frame.copy()
    if "model" in working.columns and "model_name" not in working.columns:
        working = working.rename(columns={"model": "model_name"})
    alias_map = {
        "risk_score": "reliability_risk_score",
        "uncertainty_score": "mean_uncertainty_score",
        "attention_drift_score": "mean_attention_drift_score",
        "consensus_reliability_score": "consensus_reliability",
    }
    for target, source in alias_map.items():
        if target not in working.columns and source in working.columns:
            working[target] = working[source]
    return working


def build_model_benchmark_frame(records):
    frame = _normalise_model_frame(_to_frame(records))
    if frame.empty:
        return frame
    model_column = _choose_column(frame, ["model_name", "model"])
    if model_column is None:
        return pd.DataFrame()
    available_metrics = [column for column in MODEL_METRICS if column in frame.columns]
    if not available_metrics:
        return pd.DataFrame()
    grouped = frame.groupby(model_column, dropna=False)
    rows = []
    for model_name, group in grouped:
        row = {"model_name": str(model_name), "record_count": int(len(group))}
        for metric in available_metrics:
            series = pd.to_numeric(group[metric], errors="coerce").dropna()
            if series.empty:
                continue
            row[f"mean_{metric}"] = round(float(series.mean()), 6)
            row[f"median_{metric}"] = round(float(series.median()), 6)
            row[f"std_{metric}"] = round(float(series.std(ddof=0)), 6)
            row[f"min_{metric}"] = round(float(series.min()), 6)
            row[f"max_{metric}"] = round(float(series.max()), 6)
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    if "mean_robustness_score" not in result.columns:
        result["mean_robustness_score"] = 0.0
    if "mean_reliability_risk_score" not in result.columns:
        result["mean_reliability_risk_score"] = 0.0
    if "mean_attention_drift_score" not in result.columns:
        result["mean_attention_drift_score"] = 0.0
    if "mean_uncertainty_score" not in result.columns:
        result["mean_uncertainty_score"] = 0.0
    if "mean_consensus_reliability" not in result.columns:
        result["mean_consensus_reliability"] = 0.0
    if "mean_attention_stability_score" not in result.columns and "attention_stability_score" in frame.columns:
        stability_frame = frame.groupby(model_column, dropna=False)["attention_stability_score"].mean().reset_index()
        stability_frame = stability_frame.rename(columns={model_column: "model_name", "attention_stability_score": "mean_attention_stability_score"})
        result = result.merge(stability_frame, on="model_name", how="left")
    if "mean_attention_stability_score" not in result.columns:
        result["mean_attention_stability_score"] = 0.0
    numeric_columns = [column for column in result.columns if column.startswith("mean_")]
    if numeric_columns:
        working = result.copy()
        for column in numeric_columns:
            working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)
        beneficial = [
            "mean_robustness_score",
            "mean_attention_stability_score",
            "mean_consensus_reliability",
        ]
        harmful = [
            "mean_reliability_risk_score",
            "mean_attention_drift_score",
            "mean_uncertainty_score",
            "mean_instability_score",
            "mean_mean_sample_instability",
            "mean_overall_instability",
        ]
        score = np.zeros(len(working), dtype=float)
        count = 0
        for column in beneficial:
            if column in working.columns:
                values = working[column].to_numpy(dtype=float)
                maximum = np.nanmax(values) if np.isfinite(values).any() else 0.0
                minimum = np.nanmin(values) if np.isfinite(values).any() else 0.0
                scale = maximum - minimum
                if scale == 0:
                    normalised = np.ones(len(values))
                else:
                    normalised = (values - minimum) / scale
                score += np.nan_to_num(normalised, nan=0.0)
                count += 1
        for column in harmful:
            if column in working.columns:
                values = working[column].to_numpy(dtype=float)
                maximum = np.nanmax(values) if np.isfinite(values).any() else 0.0
                minimum = np.nanmin(values) if np.isfinite(values).any() else 0.0
                scale = maximum - minimum
                if scale == 0:
                    normalised = np.ones(len(values))
                else:
                    normalised = 1.0 - ((values - minimum) / scale)
                score += np.nan_to_num(normalised, nan=0.0)
                count += 1
        if count > 0:
            working["benchmark_score"] = score / count
        else:
            working["benchmark_score"] = 0.0
        result = working
    return result.sort_values(["benchmark_score", "mean_robustness_score"], ascending=[False, False]).reset_index(drop=True)


def benchmark_models(records):
    model_frame = build_model_benchmark_frame(records)
    if model_frame.empty:
        return {
            "model_frame": model_frame,
            "rankings": pd.DataFrame(),
            "leaderboards": {},
            "summary": {
                "total_models": 0,
                "best_robustness_model": None,
                "most_stable_model": None,
                "lowest_risk_model": None,
                "best_attention_stability_model": None,
                "strongest_perturbation_resilience": None,
                "reliability_leader": None,
            },
        }

    ranking_frame = model_frame.copy()
    ranking_frame["benchmark_score"] = pd.to_numeric(ranking_frame["benchmark_score"], errors="coerce").fillna(0.0)
    ranking_frame = ranking_frame.sort_values(["benchmark_score", "mean_robustness_score"], ascending=[False, False]).reset_index(drop=True)

    leaderboards = {}
    metric_mappings = {
        "robustness": ("mean_robustness_score", False),
        "stability": ("mean_attention_stability_score", False),
        "reliability": ("mean_consensus_reliability", False),
        "risk": ("mean_reliability_risk_score", True),
        "uncertainty": ("mean_uncertainty_score", True),
        "instability": ("mean_overall_instability", True),
        "attention_drift": ("mean_attention_drift_score", True),
    }
    for leaderboard_name, (metric_column, ascending) in metric_mappings.items():
        if metric_column in ranking_frame.columns:
            leaderboard = ranking_frame[["model_name", metric_column]].copy()
            leaderboard = leaderboard.sort_values(metric_column, ascending=ascending).reset_index(drop=True)
            leaderboard["rank"] = np.arange(1, len(leaderboard) + 1)
            leaderboards[leaderboard_name] = leaderboard

    summary = {
        "total_models": int(len(ranking_frame)),
        "best_robustness_model": str(ranking_frame.iloc[0]["model_name"]),
        "most_stable_model": str(ranking_frame.sort_values("mean_overall_instability", ascending=True).iloc[0]["model_name"]) if "mean_overall_instability" in ranking_frame.columns else str(ranking_frame.iloc[0]["model_name"]),
        "lowest_risk_model": str(ranking_frame.sort_values("mean_reliability_risk_score", ascending=True).iloc[0]["model_name"]) if "mean_reliability_risk_score" in ranking_frame.columns else str(ranking_frame.iloc[0]["model_name"]),
        "best_attention_stability_model": str(ranking_frame.sort_values("mean_attention_stability_score", ascending=False).iloc[0]["model_name"]) if "mean_attention_stability_score" in ranking_frame.columns else str(ranking_frame.iloc[0]["model_name"]),
        "strongest_perturbation_resilience": str(ranking_frame.sort_values("mean_robustness_score", ascending=False).iloc[0]["model_name"]),
        "reliability_leader": str(ranking_frame.sort_values("mean_consensus_reliability", ascending=False).iloc[0]["model_name"]) if "mean_consensus_reliability" in ranking_frame.columns else str(ranking_frame.iloc[0]["model_name"]),
    }

    return {
        "model_frame": model_frame,
        "rankings": ranking_frame,
        "leaderboards": leaderboards,
        "summary": summary,
        "statistics": compute_scalar_statistics(model_frame, [column for column in model_frame.columns if column.startswith("mean_") or column.startswith("median_")]),
    }


def export_model_benchmark_outputs(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_csv(output_dir / "model_benchmark_rankings.csv", result.get("rankings", pd.DataFrame()))
    export_json(output_dir / "model_benchmark_rankings.json", to_json_ready(result.get("rankings", pd.DataFrame()).to_dict(orient="records") if isinstance(result.get("rankings"), pd.DataFrame) else result.get("rankings", [])))
    for leaderboard_name, leaderboard in result.get("leaderboards", {}).items():
        export_csv(output_dir / f"{leaderboard_name}_leaderboard.csv", leaderboard)
        export_json(output_dir / f"{leaderboard_name}_leaderboard.json", leaderboard.to_dict(orient="records"))
    export_json(output_dir / "model_benchmark_summary.json", result.get("summary", {}))
    export_json(output_dir / "model_benchmark_statistics.json", result.get("statistics", {}))
    return output_dir
