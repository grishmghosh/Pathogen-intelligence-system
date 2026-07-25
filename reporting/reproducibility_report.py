"""
Reproducibility reporting for publication-ready experiment artifacts.
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


def build_reproducibility_report(config=None, manifest=None, provenance=None, benchmark_summary=None, report_id=None):
    config = config or {}
    manifest = manifest or {}
    provenance = provenance or {}
    benchmark_summary = benchmark_summary or {}

    dataset = config.get("dataset", {}) if isinstance(config, dict) else {}
    perturbations = config.get("perturbations", {}) if isinstance(config, dict) else {}

    report = {
        "report_id": report_id,
        "experiment_id": manifest.get("experiment_id") if isinstance(manifest, dict) else None,
        "experiment_name": config.get("experiment_name") if isinstance(config, dict) else None,
        "created_at": manifest.get("created_at") if isinstance(manifest, dict) else None,
        "seed": config.get("seed") if isinstance(config, dict) else None,
        "config_snapshot": _safe_value(config),
        "manifest_snapshot": _safe_value(manifest),
        "provenance_snapshot": _safe_value(provenance),
        "dataset_metadata": _safe_value(dataset),
        "model_metadata": _safe_value(config.get("models", {})) if isinstance(config, dict) else {},
        "perturbation_metadata": _safe_value(perturbations),
        "benchmark_snapshot": _safe_value(benchmark_summary),
    }
    return report


def build_seed_tracking_summary(report):
    report = report or {}
    seed = report.get("seed")
    lines = []
    if seed is not None:
        lines.append(f"Seed {seed} was recorded for deterministic reproduction.")
    else:
        lines.append("No explicit random seed was captured in the report.")
    if report.get("experiment_id"):
        lines.append(f"Experiment identifier: {report['experiment_id']}.")
    return {
        "seed": seed,
        "summary_line": lines[0],
        "details": lines,
    }


def build_configuration_snapshot(config):
    return _safe_value(config or {})


def build_dataset_metadata_summary(config):
    dataset = (config or {}).get("dataset", {}) if isinstance(config, dict) else {}
    return {
        "dataset_name": dataset.get("name"),
        "dataset_subset": dataset.get("subset"),
        "sample_count": len(dataset.get("samples", []) or []),
        "metadata": _safe_value(dataset.get("metadata", {})),
    }


def build_model_version_summary(config):
    model_versions = (config or {}).get("model_versions", {}) if isinstance(config, dict) else {}
    models = (config or {}).get("models", []) if isinstance(config, dict) else []
    return {
        "model_ids": list(models or []),
        "model_versions": _safe_value(model_versions),
        "model_count": len(models or []),
    }


def export_reproducibility_report_json(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_safe_value(report), handle, indent=2, ensure_ascii=False)
    return output_path


def export_configuration_snapshot_json(snapshot, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_safe_value(snapshot), handle, indent=2, ensure_ascii=False)
    return output_path
