"""
Shared utilities for the integrated experimental evaluation pipeline (Step 8).

This module intentionally stays lightweight and dependency-free beyond the
required standard library and pandas / numpy. It centralises:
    - directory management
    - JSON serialisation
    - dataframe flattening and summarisation
    - execution tracking helpers
    - reproducibility-friendly snapshots
"""

from pathlib import Path
import json
import time
import traceback
import uuid

import numpy as np
import pandas as pd


DEFAULT_EXPERIMENT_ROOT = Path("results") / "experiments"
DEFAULT_STAGE_NAMES = [
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


def current_timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def generate_experiment_id():
    return uuid.uuid4().hex[:12]


def ensure_directory(path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_experiment_root(experiment_id, root_dir=None):
    base = Path(root_dir) if root_dir is not None else DEFAULT_EXPERIMENT_ROOT
    return ensure_directory(base / f"experiment_{experiment_id}")


def ensure_experiment_directories(experiment_root):
    experiment_root = ensure_directory(experiment_root)
    directories = {
        "root": experiment_root,
        "configs": ensure_directory(experiment_root / "configs"),
        "manifests": ensure_directory(experiment_root / "manifests"),
        "summaries": ensure_directory(experiment_root / "summaries"),
        "robustness": ensure_directory(experiment_root / "robustness"),
        "disagreement": ensure_directory(experiment_root / "disagreement"),
        "uncertainty": ensure_directory(experiment_root / "uncertainty"),
        "explainability": ensure_directory(experiment_root / "explainability"),
        "logs": ensure_directory(experiment_root / "logs"),
        "perturbations": ensure_directory(experiment_root / "perturbations"),
    }
    return directories


def safe_json_load(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default
    except Exception:
        return default


def _serialise_value(value, max_records=50):
    if isinstance(value, dict):
        return {str(key): _serialise_value(item, max_records=max_records) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialise_value(item, max_records=max_records) for item in value]
    if isinstance(value, tuple):
        return [_serialise_value(item, max_records=max_records) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        preview = value.head(max_records).to_dict(orient="records")
        return {
            "type": "dataframe",
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns": [str(column) for column in value.columns.tolist()],
            "records": _serialise_value(preview, max_records=max_records),
        }
    if isinstance(value, pd.Series):
        return _serialise_value(value.to_dict(), max_records=max_records)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.ndim == 0:
            return _serialise_value(array.item(), max_records=max_records)
        if array.size > 200:
            return {
                "type": "ndarray",
                "shape": [int(dim) for dim in array.shape],
                "preview": array.reshape(-1)[:200].tolist(),
            }
        return array.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return None if not np.isfinite(numeric) else numeric
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None:
        return None
    return {
        "type": type(value).__name__,
        "repr": repr(value),
    }


def safe_json_dump(path, payload):
    path = Path(path)
    ensure_directory(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_serialise_value(payload), handle, indent=2, ensure_ascii=False)
    return path


def flatten_mapping(mapping, prefix=""):
    rows = []
    if mapping is None:
        return rows
    if not isinstance(mapping, dict):
        rows.append({"path": prefix or "value", "value": _serialise_value(mapping)})
        return rows
    for key, value in mapping.items():
        path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            rows.extend(flatten_mapping(value, path))
        elif isinstance(value, list):
            if not value:
                rows.append({"path": path, "value": []})
            else:
                for index, item in enumerate(value):
                    item_path = f"{path}[{index}]"
                    if isinstance(item, dict):
                        rows.extend(flatten_mapping(item, item_path))
                    else:
                        rows.append({"path": item_path, "value": _serialise_value(item)})
        else:
            rows.append({"path": path, "value": _serialise_value(value)})
    return rows


def dataframe_metrics(frame, prefix):
    if frame is None or not isinstance(frame, pd.DataFrame):
        return {
            f"{prefix}_row_count": 0,
            f"{prefix}_column_count": 0,
            f"{prefix}_columns": [],
            f"{prefix}_preview": [],
        }
    preview = frame.head(20).to_dict(orient="records")
    metrics = {
        f"{prefix}_row_count": int(frame.shape[0]),
        f"{prefix}_column_count": int(frame.shape[1]),
        f"{prefix}_columns": [str(column) for column in frame.columns.tolist()],
        f"{prefix}_preview": _serialise_value(preview),
    }
    numeric = frame.select_dtypes(include=[np.number])
    if not numeric.empty:
        metrics[f"{prefix}_numeric_means"] = {str(column): float(numeric[column].mean()) for column in numeric.columns}
        metrics[f"{prefix}_numeric_mins"] = {str(column): float(numeric[column].min()) for column in numeric.columns}
        metrics[f"{prefix}_numeric_maxs"] = {str(column): float(numeric[column].max()) for column in numeric.columns}
    return metrics


def dataframe_to_records(frame, max_records=50):
    if frame is None or not isinstance(frame, pd.DataFrame):
        return []
    return _serialise_value(frame.head(max_records).to_dict(orient="records"), max_records=max_records)


def coerce_dataframe(value):
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        if not value:
            return pd.DataFrame()
        return pd.DataFrame(value)
    if isinstance(value, dict):
        return pd.DataFrame([value])
    return pd.DataFrame()


def stage_record(stage_name, status, started_at, finished_at, output=None, error=None, skipped_reason=None):
    duration = max(0.0, float(finished_at) - float(started_at))
    record = {
        "stage": stage_name,
        "status": status,
        "started_at": round(float(started_at), 6),
        "finished_at": round(float(finished_at), 6),
        "duration_seconds": round(duration, 6),
    }
    if skipped_reason is not None:
        record["skipped_reason"] = skipped_reason
    if error is not None:
        record["error"] = error
    if output is not None:
        record["output"] = output
    return record


def collect_execution_report(stage_results):
    report = {
        "total_stages": int(len(stage_results)),
        "completed_stages": [],
        "failed_stages": [],
        "skipped_stages": [],
        "stage_order": [],
        "total_duration_seconds": 0.0,
    }
    total_duration = 0.0
    for name, result in stage_results.items():
        report["stage_order"].append(name)
        total_duration += float(result.get("duration_seconds", 0.0))
        status = result.get("status", "unknown")
        if status == "completed":
            report["completed_stages"].append(name)
        elif status == "failed":
            report["failed_stages"].append(name)
        elif status == "skipped":
            report["skipped_stages"].append(name)
    report["total_duration_seconds"] = round(total_duration, 6)
    report["completed_count"] = int(len(report["completed_stages"]))
    report["failed_count"] = int(len(report["failed_stages"]))
    report["skipped_count"] = int(len(report["skipped_stages"]))
    return report


def flatten_stage_outputs(stage_outputs):
    rows = []
    for stage_name, payload in stage_outputs.items():
        if stage_name in {"execution_report", "experiment_manifest", "reproducibility_manifest", "registry_record", "summary"}:
            continue
        if isinstance(payload, dict):
            rows.extend([{"stage": stage_name, **row} for row in flatten_mapping(payload)])
        elif isinstance(payload, pd.DataFrame):
            rows.append({
                "stage": stage_name,
                "path": "dataframe",
                "value": _serialise_value(dataframe_metrics(payload, stage_name)),
            })
        else:
            rows.append({"stage": stage_name, "path": "value", "value": _serialise_value(payload)})
    return rows


def experiment_snapshot(stage_outputs):
    snapshot = {}
    for stage_name, payload in stage_outputs.items():
        if stage_name in {"execution_report", "experiment_manifest", "reproducibility_manifest", "registry_record", "summary"}:
            continue
        if isinstance(payload, pd.DataFrame):
            snapshot[stage_name] = dataframe_metrics(payload, stage_name)
        elif isinstance(payload, dict):
            snapshot[stage_name] = {
                "type": "dict",
                "keys": sorted([str(key) for key in payload.keys()]),
                "serialised": _serialise_value(payload, max_records=20),
            }
        else:
            snapshot[stage_name] = {"type": type(payload).__name__, "value": _serialise_value(payload)}
    return snapshot


def traceback_string():
    return traceback.format_exc()
