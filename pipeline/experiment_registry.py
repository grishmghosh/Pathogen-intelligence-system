"""
Experiment registry helpers for the integrated experimental evaluation
pipeline (Step 8).

The registry keeps lightweight, file-based metadata about experiments so the
pipeline can track runs without introducing any database or orchestration
framework.
"""

from pathlib import Path
import json

from pipeline.pipeline_utils import current_timestamp, ensure_directory, safe_json_dump, safe_json_load, _serialise_value


DEFAULT_REGISTRY_PATH = Path("results") / "experiments" / "experiment_registry.json"


def _compact_config(config):
    if not isinstance(config, dict):
        return {}
    dataset = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    perturbations = config.get("perturbations") if isinstance(config.get("perturbations"), dict) else {}
    return {
        "experiment_name": config.get("experiment_name"),
        "description": config.get("description"),
        "seed": config.get("seed"),
        "enabled_stages": list(config.get("enabled_stages") or []),
        "stage_order": list(config.get("stage_order") or []),
        "models": list((config.get("models") or {}).keys()) if isinstance(config.get("models"), dict) else [],
        "dataset": {
            "name": dataset.get("name"),
            "subset": dataset.get("subset"),
            "sample_count": len(dataset.get("sample_paths") or []),
        },
        "perturbations": {
            "profile_count": len(perturbations.get("profiles") or []),
        },
        "export": config.get("export", {}),
    }


def create_experiment_manifest(config, experiment_id, output_root, validation=None):
    compact = _compact_config(config)
    dataset = config.get("dataset", {}) if isinstance(config, dict) else {}
    perturbations = config.get("perturbations", {}) if isinstance(config, dict) else {}
    manifest = {
        "manifest_version": "1.0",
        "experiment_id": experiment_id,
        "experiment_name": config.get("experiment_name") if isinstance(config, dict) else None,
        "created_at": current_timestamp(),
        "output_root": str(output_root),
        "config": compact,
        "dataset_info": {
            "name": dataset.get("name"),
            "subset": dataset.get("subset"),
            "sample_paths": [str(path) for path in dataset.get("sample_paths", [])],
            "samples": _serialise_value(dataset.get("samples", [])),
            "metadata": _serialise_value(dataset.get("metadata", {})),
        },
        "model_info": _serialise_value(config.get("models", {}) if isinstance(config, dict) else {}),
        "perturbation_profiles": _serialise_value(perturbations.get("profiles", [])),
        "perturbation_parameters": _serialise_value(perturbations.get("parameters", {})),
        "validation": _serialise_value(validation or {}),
    }
    return manifest


def load_registry(registry_path=None):
    path = Path(registry_path) if registry_path is not None else DEFAULT_REGISTRY_PATH
    payload = safe_json_load(path, default=[])
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("experiments"), list):
        return payload["experiments"]
    return []


def build_registry_summary(records):
    records = records or []
    summary = {
        "total_experiments": int(len(records)),
        "experiment_ids": [],
        "experiment_names": [],
        "stage_usage": {},
        "model_usage": {},
    }
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("experiment_id") is not None:
            summary["experiment_ids"].append(str(record.get("experiment_id")))
        if record.get("experiment_name") is not None:
            summary["experiment_names"].append(str(record.get("experiment_name")))
        for stage in record.get("enabled_stages", []) or []:
            stage = str(stage)
            summary["stage_usage"][stage] = summary["stage_usage"].get(stage, 0) + 1
        for model in record.get("models", []) or []:
            model = str(model)
            summary["model_usage"][model] = summary["model_usage"].get(model, 0) + 1
    summary["experiment_ids"] = list(dict.fromkeys(summary["experiment_ids"]))
    summary["experiment_names"] = list(dict.fromkeys(summary["experiment_names"]))
    return summary


def register_experiment(manifest, registry_path=None):
    path = Path(registry_path) if registry_path is not None else DEFAULT_REGISTRY_PATH
    ensure_directory(path.parent)
    records = load_registry(path)
    record = {
        "experiment_id": manifest.get("experiment_id"),
        "experiment_name": manifest.get("experiment_name"),
        "created_at": manifest.get("created_at"),
        "output_root": manifest.get("output_root"),
        "enabled_stages": manifest.get("config", {}).get("enabled_stages", []),
        "models": manifest.get("config", {}).get("models", []),
        "dataset": manifest.get("dataset_info", {}),
        "perturbation_profiles": manifest.get("perturbation_profiles", []),
        "perturbation_parameters": manifest.get("perturbation_parameters", {}),
    }
    records.append(record)
    payload = {
        "registry_version": "1.0",
        "updated_at": current_timestamp(),
        "experiments": records,
        "summary": build_registry_summary(records),
    }
    safe_json_dump(path, payload)
    return {
        "registry_path": str(path),
        "record": record,
        "summary": payload["summary"],
    }


def export_manifest_json(manifest, output_path):
    return safe_json_dump(output_path, manifest)


def export_registry_json(records, output_path):
    payload = {
        "registry_version": "1.0",
        "updated_at": current_timestamp(),
        "experiments": records,
        "summary": build_registry_summary(records),
    }
    return safe_json_dump(output_path, payload)
