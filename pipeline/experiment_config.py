"""
Experiment configuration helpers for the integrated experimental evaluation
pipeline (Step 8).

Supports both JSON-backed configs and plain Python dictionaries.
The normalised schema is intentionally simple so the runner can remain
orchestration-focused and architecture-safe.
"""

from pathlib import Path
import json

from pipeline.pipeline_utils import current_timestamp, safe_json_dump, safe_json_load


DEFAULT_STAGE_ORDER = [
    "inference",
    "perturbation_generation",
    "robustness",
    "calibration",
    "disagreement",
    "consensus",
    "risk",
    "uncertainty",
    "explainability",
]

DEFAULT_EXPORT_OPTIONS = {
    "csv": True,
    "json": True,
    "save_stage_outputs": True,
    "save_manifests": True,
    "save_summary": True,
}

DEFAULT_CONFIG = {
    "experiment_name": "pathogen_intelligence_experiment",
    "description": "Integrated experimental evaluation pipeline",
    "seed": 42,
    "stage_order": DEFAULT_STAGE_ORDER,
    "enabled_stages": DEFAULT_STAGE_ORDER,
    "models": {},
    "dataset": {
        "name": None,
        "subset": None,
        "sample_paths": [],
        "samples": [],
        "metadata": {},
    },
    "perturbations": {
        "profiles": [],
        "parameters": {},
    },
    "enabled_analyses": {
        "inference": True,
        "perturbation_generation": True,
        "robustness": True,
        "calibration": True,
        "disagreement": True,
        "consensus": True,
        "risk": True,
        "uncertainty": True,
        "explainability": True,
    },
    "export": DEFAULT_EXPORT_OPTIONS,
    "metadata": {},
}


def _deep_copy(value):
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    if isinstance(value, tuple):
        return [_deep_copy(item) for item in value]
    return value


def build_experiment_config(
    experiment_name=None,
    description=None,
    seed=None,
    models=None,
    dataset=None,
    perturbations=None,
    enabled_stages=None,
    enabled_analyses=None,
    export=None,
    metadata=None,
):
    config = _deep_copy(DEFAULT_CONFIG)
    if experiment_name is not None:
        config["experiment_name"] = experiment_name
    if description is not None:
        config["description"] = description
    if seed is not None:
        config["seed"] = seed
    if models is not None:
        config["models"] = _deep_copy(models)
    if dataset is not None:
        config["dataset"] = _merge_dict(config["dataset"], dataset)
    if perturbations is not None:
        config["perturbations"] = _merge_dict(config["perturbations"], perturbations)
    if enabled_stages is not None:
        config["enabled_stages"] = list(enabled_stages)
    if enabled_analyses is not None:
        config["enabled_analyses"] = _merge_dict(config["enabled_analyses"], enabled_analyses)
    if export is not None:
        config["export"] = _merge_dict(config["export"], export)
    if metadata is not None:
        config["metadata"] = _merge_dict(config["metadata"], metadata)
    config["created_at"] = current_timestamp()
    return normalise_experiment_config(config)


def _merge_dict(base, extra):
    merged = _deep_copy(base) if isinstance(base, dict) else {}
    if not isinstance(extra, dict):
        return merged
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = _deep_copy(value)
    return merged


def load_experiment_config(source):
    if source is None:
        return normalise_experiment_config(_deep_copy(DEFAULT_CONFIG))
    if isinstance(source, dict):
        return normalise_experiment_config(source)
    if isinstance(source, (str, Path)):
        path = Path(source)
        loaded = safe_json_load(path, default=None)
        if loaded is None:
            return normalise_experiment_config(_deep_copy(DEFAULT_CONFIG))
        return normalise_experiment_config(loaded)
    return normalise_experiment_config(_deep_copy(DEFAULT_CONFIG))


def load_experiment_config_json(path):
    return load_experiment_config(path)


def save_experiment_config_json(config, path):
    return safe_json_dump(path, normalise_experiment_config(config))


def _normalise_sample_entry(sample, index):
    if isinstance(sample, dict):
        entry = _deep_copy(sample)
    else:
        entry = {"image_path": sample}
    entry.setdefault("sample_id", entry.get("id") or f"sample_{index + 1}")
    if "image_path" in entry and entry["image_path"] is not None:
        entry["image_path"] = str(entry["image_path"])
    return entry


def _normalise_enabled_stages(config):
    stage_order = list(config.get("stage_order") or DEFAULT_STAGE_ORDER)
    enabled = config.get("enabled_stages")
    if enabled is None:
        enabled = stage_order
    if isinstance(enabled, dict):
        enabled = [name for name, flag in enabled.items() if bool(flag)]
    enabled = [str(stage) for stage in enabled]
    if not enabled:
        enabled = list(stage_order)
    config["stage_order"] = [str(stage) for stage in stage_order]
    config["enabled_stages"] = enabled
    config["stage_flags"] = {stage: stage in enabled for stage in config["stage_order"]}


def normalise_experiment_config(config):
    if config is None:
        config = _deep_copy(DEFAULT_CONFIG)
    elif not isinstance(config, dict):
        config = _deep_copy(DEFAULT_CONFIG)
    else:
        config = _merge_dict(_deep_copy(DEFAULT_CONFIG), config)

    config.setdefault("experiment_name", DEFAULT_CONFIG["experiment_name"])
    config.setdefault("description", DEFAULT_CONFIG["description"])
    config.setdefault("seed", DEFAULT_CONFIG["seed"])
    config.setdefault("models", {})
    config.setdefault("dataset", {})
    config.setdefault("perturbations", {})
    config.setdefault("enabled_analyses", {})
    config.setdefault("export", {})
    config.setdefault("metadata", {})

    dataset = config.get("dataset") or {}
    if not isinstance(dataset, dict):
        dataset = {"name": None, "subset": None, "sample_paths": [], "samples": [], "metadata": {}}
    dataset.setdefault("name", None)
    dataset.setdefault("subset", None)
    dataset.setdefault("sample_paths", [])
    dataset.setdefault("samples", [])
    dataset.setdefault("metadata", {})
    dataset["sample_paths"] = [str(path) for path in dataset.get("sample_paths", []) if path is not None]
    dataset["samples"] = [_normalise_sample_entry(sample, index) for index, sample in enumerate(dataset.get("samples", []))]
    config["dataset"] = dataset

    perturbations = config.get("perturbations") or {}
    if not isinstance(perturbations, dict):
        perturbations = {"profiles": [], "parameters": {}}
    perturbations.setdefault("profiles", [])
    perturbations.setdefault("parameters", {})
    config["perturbations"] = perturbations

    config["models"] = config.get("models") if isinstance(config.get("models"), dict) else {}
    config["enabled_analyses"] = _merge_dict(DEFAULT_CONFIG["enabled_analyses"], config.get("enabled_analyses", {}))
    config["export"] = _merge_dict(DEFAULT_EXPORT_OPTIONS, config.get("export", {}))
    config["metadata"] = config.get("metadata") if isinstance(config.get("metadata"), dict) else {}
    _normalise_enabled_stages(config)
    config["normalised_at"] = current_timestamp()
    return config


def validate_experiment_config(config):
    normalised = normalise_experiment_config(config)
    errors = []
    warnings = []

    if not normalised.get("experiment_name"):
        errors.append("experiment_name is required")
    if not normalised.get("enabled_stages"):
        warnings.append("No enabled stages were supplied; default stage order was applied")
    if not normalised.get("dataset", {}).get("sample_paths") and not normalised.get("dataset", {}).get("samples"):
        warnings.append("No dataset samples were supplied; the runner will rely on injected stage handlers or skip inference")
    if not normalised.get("models"):
        warnings.append("No models were configured; real inference will be skipped unless a custom handler is injected")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "config": normalised,
    }
