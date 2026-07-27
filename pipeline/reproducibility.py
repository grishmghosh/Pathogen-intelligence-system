"""
Reproducibility helpers for the integrated experimental evaluation pipeline
(Step 8).

The goal is to preserve the experiment configuration, deterministic settings,
and stage-level snapshot data so that a run can be reconstructed later without
introducing any external orchestration or metadata service.
"""

from pathlib import Path

from pipeline.pipeline_utils import current_timestamp, dataframe_metrics, experiment_snapshot, safe_json_dump, _serialise_value


def build_reproducibility_manifest(config, manifest, stage_results, seed=None, model_versions=None, dataset_metadata=None, perturbation_metadata=None):
    dataset = config.get("dataset", {}) if isinstance(config, dict) else {}
    perturbations = config.get("perturbations", {}) if isinstance(config, dict) else {}
    runtime = config.get("runtime", {}) if isinstance(config, dict) else {}
    reproducibility_seed = seed if seed is not None else config.get("seed") if isinstance(config, dict) else None
    manifest_payload = {
        "reproducibility_version": "1.0",
        "experiment_id": manifest.get("experiment_id") if isinstance(manifest, dict) else None,
        "experiment_name": manifest.get("experiment_name") if isinstance(manifest, dict) else None,
        "created_at": current_timestamp(),
        "seed": reproducibility_seed,
        "config_snapshot": _serialise_value(config),
        "dataset_metadata": _serialise_value(dataset_metadata if dataset_metadata is not None else dataset),
        "model_versions": _serialise_value(model_versions if model_versions is not None else config.get("models", {}) if isinstance(config, dict) else {}),
        "perturbation_metadata": _serialise_value(perturbation_metadata if perturbation_metadata is not None else perturbations),
        "runtime_metadata": _serialise_value(runtime),
        "stage_execution": _serialise_value(stage_results),
        "experiment_manifest": _serialise_value(manifest),
    }
    return manifest_payload


def build_experiment_snapshot(stage_outputs):
    snapshot = experiment_snapshot(stage_outputs)
    snapshot["captured_at"] = current_timestamp()
    return snapshot


def export_reproducibility_manifest_json(manifest, output_path):
    return safe_json_dump(output_path, manifest)


def export_experiment_snapshot_json(snapshot, output_path):
    return safe_json_dump(output_path, snapshot)
