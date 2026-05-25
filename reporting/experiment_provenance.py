"""
Experiment provenance tracking for research reports.
"""

from pathlib import Path
import json

import pandas as pd


def _safe_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return {str(key): _safe_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def build_provenance_record(experiment_id=None, config=None, manifest=None, reproducibility=None, benchmark_summary=None):
    config = config or {}
    manifest = manifest or {}
    reproducibility = reproducibility or {}
    benchmark_summary = benchmark_summary or {}

    dataset = config.get("dataset", {}) if isinstance(config, dict) else {}
    perturbations = config.get("perturbations", {}) if isinstance(config, dict) else {}

    record = {
        "experiment_id": experiment_id or manifest.get("experiment_id") or config.get("experiment_id"),
        "experiment_name": config.get("experiment_name") if isinstance(config, dict) else manifest.get("experiment_name"),
        "created_at": manifest.get("created_at"),
        "output_root": manifest.get("output_root"),
        "dataset_name": dataset.get("name"),
        "dataset_subset": dataset.get("subset"),
        "dataset_sample_count": len(dataset.get("samples", []) or []),
        "enabled_stages": list(config.get("enabled_stages", []) or []) if isinstance(config, dict) else [],
        "enabled_analyses": _safe_value(config.get("enabled_analyses", {})) if isinstance(config, dict) else {},
        "models": list(config.get("models", []) or []) if isinstance(config, dict) else [],
        "model_versions": _safe_value(config.get("model_versions", {})) if isinstance(config, dict) else {},
        "perturbation_profiles": list(perturbations.get("profiles", []) or []) if isinstance(config, dict) else [],
        "perturbation_parameters": _safe_value(perturbations.get("parameters", {})) if isinstance(config, dict) else {},
        "seed": config.get("seed") if isinstance(config, dict) else None,
        "registry_path": reproducibility.get("registry_path") if isinstance(reproducibility, dict) else None,
        "summary_line": benchmark_summary.get("summary_line") if isinstance(benchmark_summary, dict) else None,
    }
    return record


def build_traceability_summary(provenance_record):
    record = provenance_record or {}
    lines = []
    if record.get("experiment_name"):
        lines.append(f"Experiment {record['experiment_name']} was executed for reporting.")
    if record.get("dataset_name"):
        subset = record.get("dataset_subset") or "unspecified subset"
        lines.append(f"Dataset {record['dataset_name']} ({subset}) was used for the report.")
    if record.get("models"):
        lines.append(f"Models included: {', '.join([str(model) for model in record['models']])}.")
    if record.get("perturbation_profiles"):
        lines.append(f"Perturbation profiles tracked: {', '.join([str(profile) for profile in record['perturbation_profiles']])}.")
    if record.get("seed") is not None:
        lines.append(f"Random seed tracked: {record['seed']}.")
    if not lines:
        lines.append("No provenance metadata was available for traceability.")
    return {
        "traceability_lines": lines,
        "traceability_line": lines[0],
        "field_count": int(len(record)),
    }


def export_provenance_manifest_json(provenance_record, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_safe_value(provenance_record), handle, indent=2, ensure_ascii=False)
    return output_path


def export_traceability_summary_json(traceability_summary, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_safe_value(traceability_summary), handle, indent=2, ensure_ascii=False)
    return output_path
