"""
Cross-subsystem consistency checking for scientific stabilization.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


def _safe_frame(payload):
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    return pd.DataFrame([payload])


def _to_float(series, column):
    if series.empty or column not in series.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(series[column], errors="coerce").dropna()


def check_cross_subsystem_consistency(experiment_summary=None, benchmark_summary=None, report_summary=None, trust_summary=None):
    experiment_summary = experiment_summary or {}
    benchmark_summary = benchmark_summary or {}
    report_summary = report_summary or {}
    trust_summary = trust_summary or {}

    warnings = []
    anomalies = []
    evidence = []

    per_model = experiment_summary.get("per_model_stats", {}) if isinstance(experiment_summary, dict) else {}
    degradation = experiment_summary.get("degradation_summary", {}) if isinstance(experiment_summary, dict) else {}

    if isinstance(per_model, dict) and per_model:
        model_frame = pd.DataFrame([
            {"model_name": name, **(stats if isinstance(stats, dict) else {})}
            for name, stats in per_model.items()
            if isinstance(stats, dict)
        ])
        if not model_frame.empty and "confidence_drop" in model_frame.columns and "robustness_score" in model_frame.columns:
            confidence = pd.to_numeric(model_frame["confidence_drop"], errors="coerce")
            robustness = pd.to_numeric(model_frame["robustness_score"], errors="coerce")
            if confidence.notna().any() and robustness.notna().any():
                corr = confidence.corr(robustness)
                evidence.append({"pair": "confidence_drop_vs_robustness", "correlation": round(float(corr), 6) if pd.notna(corr) else None})
                if pd.notna(corr) and corr > -0.1:
                    warnings.append("Robustness does not decrease clearly with confidence drop.")

    if isinstance(benchmark_summary, dict):
        benchmark_model = benchmark_summary.get("model_summary", {})
        perturbation_summary = benchmark_summary.get("perturbation_summary", {})
        severity_summary = benchmark_summary.get("severity_summary", {})
        if isinstance(benchmark_model, dict) and isinstance(perturbation_summary, dict):
            if benchmark_model.get("best_robustness_model") is None and perturbation_summary.get("most_disruptive_perturbation") is None:
                warnings.append("Benchmark summary lacks both model leader and perturbation leader.")
        if isinstance(severity_summary, dict) and severity_summary.get("trend_direction") == "stable":
            if degradation:
                warnings.append("Severity trend is stable while experiment degradation exists; verify summary alignment.")

    if isinstance(trust_summary, dict):
        if trust_summary.get("downgraded_count", 0) and not trust_summary.get("critical_sample_ids"):
            anomalies.append("Downgraded trust count exists without critical sample identifiers.")

    if report_summary:
        summary_line = report_summary.get("summary_line")
        if isinstance(summary_line, str) and "No deterministic narrative" in summary_line:
            warnings.append("Reporting layer produced only a placeholder narrative.")

    consistency_score = 100.0 - (len(warnings) * 8.0) - (len(anomalies) * 15.0)
    consistency_score = max(0.0, round(consistency_score, 6))
    return {
        "consistency_score": consistency_score,
        "warnings": warnings,
        "anomalies": anomalies,
        "evidence": evidence,
        "warning_count": int(len(warnings)),
        "anomaly_count": int(len(anomalies)),
    }


def build_consistency_summary(result):
    result = result or {}
    lines = []
    lines.append(f"Consistency score: {result.get('consistency_score', 0.0)}")
    if result.get("warnings"):
        lines.append(f"Warnings: {len(result['warnings'])}")
    if result.get("anomalies"):
        lines.append(f"Anomalies: {len(result['anomalies'])}")
    return {"lines": lines, "summary_line": lines[0] if lines else "Consistency score unavailable."}


def export_consistency_json(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return output_path


def export_consistency_csv(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{k: v for k, v in result.items() if not isinstance(v, (list, dict))}])
    frame.to_csv(output_path, index=False, float_format="%.6f")
    return output_path


def export_consistency_txt(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"consistency_score: {result.get('consistency_score', 0.0)}"]
    for warning in result.get("warnings", []):
        lines.append(f"warning: {warning}")
    for anomaly in result.get("anomalies", []):
        lines.append(f"anomaly: {anomaly}")
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return output_path
