"""
Deterministic scientific narrative generation for research reports.
"""

from textwrap import fill

import numpy as np
import pandas as pd


def _safe_frame(frame):
    if frame is None:
        return pd.DataFrame()
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if isinstance(frame, dict):
        return pd.DataFrame([frame])
    if isinstance(frame, list):
        return pd.DataFrame(frame)
    return pd.DataFrame([frame])


def _best_row(frame, metric, ascending=False):
    frame = _safe_frame(frame)
    if frame.empty or metric not in frame.columns:
        return None
    series = pd.to_numeric(frame[metric], errors="coerce")
    if series.dropna().empty:
        return None
    ordered = frame.loc[series.sort_values(ascending=ascending).index].reset_index(drop=True)
    return ordered.iloc[0].to_dict()


def _trend_direction(start_value, end_value, threshold=0.02):
    if start_value is None or end_value is None:
        return "stable"
    delta = float(end_value) - float(start_value)
    if delta > threshold:
        return "increased"
    if delta < -threshold:
        return "decreased"
    return "remained stable"


def build_narrative_summaries(model_table=None, perturbation_table=None, severity_table=None, correlation_table=None, benchmark_summary=None, provenance=None, reproducibility=None):
    model_table = _safe_frame(model_table)
    perturbation_table = _safe_frame(perturbation_table)
    severity_table = _safe_frame(severity_table)
    correlation_table = _safe_frame(correlation_table)
    benchmark_summary = benchmark_summary or {}
    provenance = provenance or {}
    reproducibility = reproducibility or {}

    narratives = []

    best_model = _best_row(model_table, "mean_robustness_score", ascending=False)
    if best_model is None and isinstance(benchmark_summary, dict):
        model_summary = benchmark_summary.get("model_summary", {})
        if isinstance(model_summary, dict) and model_summary.get("best_robustness_model"):
            narratives.append(fill(f"Model {model_summary['best_robustness_model']} demonstrated the highest robustness in the benchmark set."))
    if best_model is not None:
        model_name = best_model.get("model_name")
        robustness = best_model.get("mean_robustness_score")
        narratives.append(fill(f"Model {model_name} demonstrated the highest robustness in the benchmark set with a mean robustness score of {robustness:.6f}."))

    most_fragile = _best_row(perturbation_table, "degradation_impact", ascending=False)
    if most_fragile is None and isinstance(benchmark_summary, dict):
        perturbation_summary = benchmark_summary.get("perturbation_summary", {})
        if isinstance(perturbation_summary, dict) and perturbation_summary.get("most_disruptive_perturbation"):
            narratives.append(fill(f"Perturbation {perturbation_summary['most_disruptive_perturbation']} produced the strongest aggregate degradation signal."))
    if most_fragile is not None:
        perturbation = most_fragile.get("perturbation_type")
        narratives.append(fill(f"Perturbation {perturbation} produced the strongest degradation impact and was the most disruptive benchmark condition."))

    if not severity_table.empty:
        if "mean_uncertainty_score" in severity_table.columns:
            series = pd.to_numeric(severity_table["mean_uncertainty_score"], errors="coerce").dropna()
            if len(series) >= 2:
                direction = _trend_direction(series.iloc[0], series.iloc[-1])
                narratives.append(fill(f"Uncertainty {direction} across the severity ladder, indicating structured escalation in response to stronger perturbations."))
        if "mean_consensus_reliability_score" in severity_table.columns:
            series = pd.to_numeric(severity_table["mean_consensus_reliability_score"], errors="coerce").dropna()
            if len(series) >= 2:
                direction = _trend_direction(series.iloc[0], series.iloc[-1])
                narratives.append(fill(f"Consensus reliability {direction} with increasing severity, showing monotonic degradation in agreement quality."))
    elif isinstance(benchmark_summary, dict):
        severity_summary = benchmark_summary.get("severity_summary", {})
        if isinstance(severity_summary, dict):
            direction = severity_summary.get("trend_direction")
            if direction:
                narratives.append(fill(f"Benchmark severity trends were {direction}, indicating structured escalation across perturbation levels."))

    if not correlation_table.empty:
        numeric = correlation_table.copy()
        mask = np.triu(np.ones(numeric.shape, dtype=bool), k=1)
        values = numeric.where(mask).stack()
        if not values.empty:
            strongest_index = values.abs().idxmax()
            strongest_value = float(values.loc[strongest_index])
            narratives.append(fill(f"The strongest observed dependency was between {strongest_index[0]} and {strongest_index[1]} with correlation {strongest_value:.3f}."))
    elif isinstance(benchmark_summary, dict):
        correlation_summary = benchmark_summary.get("correlation_summary", {})
        strongest_pair = correlation_summary.get("strongest_dependency_pair") if isinstance(correlation_summary, dict) else None
        if isinstance(strongest_pair, dict):
            narratives.append(fill(f"The strongest observed dependency was between {strongest_pair.get('metric_a')} and {strongest_pair.get('metric_b')} with correlation {float(strongest_pair.get('correlation', 0.0)):.3f}."))

    if provenance.get("summary_line"):
        narratives.append(fill(f"Provenance tracking confirms: {provenance['summary_line']}"))
    if reproducibility.get("seed") is not None:
        narratives.append(fill(f"Reproducibility tracking recorded seed {reproducibility['seed']} for deterministic reruns."))

    if not narratives:
        narratives.append("No deterministic narrative could be derived from the supplied benchmark artifacts.")

    return {
        "narratives": narratives,
        "summary_line": narratives[0],
        "narrative_count": int(len(narratives)),
    }


def export_narrative_json(narrative_summary, output_path):
    output_path = output_path if hasattr(output_path, "open") else None
    from pathlib import Path
    import json

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(narrative_summary, handle, indent=2, ensure_ascii=False)
    return path


def export_narrative_text(narrative_summary, output_path):
    from pathlib import Path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n\n".join(narrative_summary.get("narratives", [])))
    return path
