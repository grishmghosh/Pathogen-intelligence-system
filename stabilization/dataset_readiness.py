"""
Dataset readiness analysis for expansion preparation.
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


def analyze_dataset_readiness(config=None, experiment_summary=None, benchmark_summary=None):
    config = config or {}
    experiment_summary = experiment_summary or {}
    benchmark_summary = benchmark_summary or {}
    dataset = config.get("dataset", {}) if isinstance(config, dict) else {}
    samples = dataset.get("samples", []) or []
    class_names = sorted({str(sample.get("true_label")) for sample in samples if isinstance(sample, dict) and sample.get("true_label") is not None})

    sample_frame = pd.DataFrame(samples) if samples else pd.DataFrame()
    class_counts = sample_frame["true_label"].astype(str).value_counts().sort_index().to_dict() if not sample_frame.empty and "true_label" in sample_frame.columns else {}
    class_balance = 0.0
    if class_counts:
        values = np.array(list(class_counts.values()), dtype=float)
        if values.sum() > 0:
            class_balance = float(values.min() / values.max()) if values.max() > 0 else 0.0

    perturbations = config.get("perturbations", {}) if isinstance(config, dict) else {}
    profiles = perturbations.get("profiles", []) or []
    severity_summary = benchmark_summary.get("severity_summary", {}) if isinstance(benchmark_summary, dict) else {}
    severity_levels = severity_summary.get("severity_levels", []) if isinstance(severity_summary, dict) else []

    validation_coverage = 0.0
    if class_names:
        validation_coverage = round(float(len(class_counts) / len(class_names)), 6) if class_names else 0.0
    perturbation_coverage = round(float(len(profiles) / max(len(severity_levels), 1)), 6)
    sample_sufficiency = round(float(min(len(samples) / 50.0, 1.0)), 6)
    calibration_sufficiency = 0.0
    calibration_summary = experiment_summary.get("calibration_summary", {}) if isinstance(experiment_summary, dict) else {}
    if isinstance(calibration_summary, dict) and calibration_summary:
        calibration_sufficiency = round(float(min(len(calibration_summary) / max(len(class_names), 1), 1.0)), 6)

    readiness_score = np.mean([
        class_balance,
        validation_coverage,
        perturbation_coverage,
        sample_sufficiency,
        calibration_sufficiency,
    ]) if class_names or samples else 0.0
    readiness_score = round(float(readiness_score * 100.0), 6)

    warnings = []
    recommendations = []
    if class_balance < 0.5 and class_counts:
        warnings.append("Class balance is weak for expansion.")
        recommendations.append("Collect additional samples for minority classes.")
    if perturbation_coverage < 1.0:
        warnings.append("Perturbation coverage is incomplete relative to severity structure.")
        recommendations.append("Expand perturbation families before scaling experiments.")
    if sample_sufficiency < 0.6:
        warnings.append("Current sample count is small for large-scale experiments.")
        recommendations.append("Increase validation sample coverage before expansion.")
    if calibration_sufficiency < 0.5:
        warnings.append("Calibration evidence is limited.")
        recommendations.append("Run calibration sweeps across larger validation subsets.")

    readiness = {
        "dataset_name": dataset.get("name"),
        "dataset_subset": dataset.get("subset"),
        "class_names": class_names,
        "class_counts": class_counts,
        "class_balance": round(float(class_balance), 6),
        "validation_coverage": validation_coverage,
        "perturbation_coverage": perturbation_coverage,
        "sample_sufficiency": sample_sufficiency,
        "calibration_sufficiency": calibration_sufficiency,
        "readiness_score": readiness_score,
        "warnings": warnings,
        "recommendations": recommendations,
        "calibration_sufficiency_warning": calibration_sufficiency < 0.5,
    }
    return readiness


def build_expansion_recommendations(readiness):
    readiness = readiness or {}
    lines = []
    for recommendation in readiness.get("recommendations", []) or []:
        lines.append(str(recommendation))
    if not lines:
        lines.append("Dataset is ready for controlled expansion with current coverage.")
    return {"lines": lines, "summary_line": lines[0], "count": int(len(lines))}


def export_readiness_json(readiness, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(readiness, handle, indent=2, ensure_ascii=False)
    return output_path


def export_readiness_csv(readiness, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in readiness.items() if not isinstance(v, (list, dict))}]).to_csv(output_path, index=False, float_format="%.6f")
    return output_path


def export_readiness_txt(readiness, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"readiness_score: {readiness.get('readiness_score', 0.0)}",
        f"class_balance: {readiness.get('class_balance', 0.0)}",
        f"validation_coverage: {readiness.get('validation_coverage', 0.0)}",
        f"perturbation_coverage: {readiness.get('perturbation_coverage', 0.0)}",
    ]
    for recommendation in readiness.get("recommendations", []):
        lines.append(f"recommendation: {recommendation}")
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return output_path
