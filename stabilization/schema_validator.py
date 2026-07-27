"""
Schema validation for framework artifacts and research outputs.
"""

from pathlib import Path
import json
import traceback

import numpy as np
import pandas as pd


def _safe_load_json(path):
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


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


ARTIFACT_SCHEMAS = {
    "experiment_summary": {
        "required_keys": ["per_model_stats", "best_robustness", "worst_robustness", "calibration_summary", "degradation_summary", "overall_statistics"],
        "optional_keys": ["summary_line"],
    },
    "benchmark_summary": {
        "required_keys": ["summary_line", "leaderboards"],
        "optional_keys": ["benchmark", "model_summary", "perturbation_summary", "severity_summary", "correlation_summary", "statistics_summary", "publication_summary"],
    },
    "report_manifest": {
        "required_keys": ["report_id", "report_root", "artifacts", "summary_line"],
        "optional_keys": ["traceability", "seed_tracking", "dataset_summary", "model_summary"],
    },
    "reproducibility_report": {
        "required_keys": ["report_id", "config_snapshot", "manifest_snapshot", "provenance_snapshot"],
        "optional_keys": ["seed", "dataset_metadata", "model_metadata", "perturbation_metadata", "benchmark_snapshot"],
    },
    "provenance_manifest": {
        "required_keys": ["experiment_id", "experiment_name", "dataset_name", "dataset_subset", "enabled_stages", "models", "perturbation_profiles"],
        "optional_keys": ["seed", "summary_line", "registry_path"],
    },
    "trust_summary": {
        "required_keys": ["total_samples", "trust_distribution", "trust_rates"],
        "optional_keys": ["downgraded_count", "critical_sample_ids", "very_high_sample_ids", "mean_reliability_by_trust"],
    },
}


def validate_schema(payload, schema_name, strict=False):
    schema = ARTIFACT_SCHEMAS.get(schema_name, {})
    payload = payload or {}
    errors = []
    warnings = []

    if not isinstance(payload, dict):
        return {
            "schema_name": schema_name,
            "valid": False,
            "errors": [f"Payload for {schema_name} is not a dictionary."],
            "warnings": [],
            "missing_keys": schema.get("required_keys", []),
            "extra_keys": [],
        }

    required_keys = schema.get("required_keys", [])
    optional_keys = schema.get("optional_keys", [])
    missing_keys = [key for key in required_keys if key not in payload]
    extra_keys = [key for key in payload if key not in required_keys + optional_keys]

    if missing_keys:
        errors.append(f"Missing required keys: {', '.join(missing_keys)}")
    if strict and extra_keys:
        warnings.append(f"Unexpected keys: {', '.join(extra_keys)}")

    if schema_name == "report_manifest":
        artifacts = payload.get("artifacts", {})
        if not isinstance(artifacts, dict):
            errors.append("report_manifest.artifacts must be a dictionary.")
        elif "tables" not in artifacts:
            errors.append("report_manifest.artifacts.tables is required.")

    if schema_name in {"experiment_summary", "benchmark_summary"} and "summary_line" in payload:
        if not isinstance(payload["summary_line"], str):
            errors.append("summary_line must be a string.")

    valid = len(errors) == 0
    return {
        "schema_name": schema_name,
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "missing_keys": missing_keys,
        "extra_keys": extra_keys,
    }


def build_compatibility_summary(validation_reports):
    reports = validation_reports or []
    rows = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        rows.append({
            "schema_name": report.get("schema_name"),
            "valid": bool(report.get("valid", False)),
            "missing_key_count": int(len(report.get("missing_keys", []) or [])),
            "error_count": int(len(report.get("errors", []) or [])),
            "warning_count": int(len(report.get("warnings", []) or [])),
        })
    frame = pd.DataFrame(rows)
    summary = {
        "total_schemas": int(len(rows)),
        "valid_schemas": int(frame["valid"].sum()) if not frame.empty else 0,
        "invalid_schemas": int((~frame["valid"]).sum()) if not frame.empty else 0,
        "compatibility_score": round(float(frame["valid"].mean() * 100.0), 6) if not frame.empty else 0.0,
    }
    return {"frame": frame, "summary": summary}


def validate_artifact_schemas(artifact_map):
    artifact_map = artifact_map or {}
    reports = []
    for schema_name, payload in artifact_map.items():
        reports.append(validate_schema(payload, schema_name))
    compatibility = build_compatibility_summary(reports)
    return {"reports": reports, "compatibility": compatibility}


def export_schema_validation_json(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    def _to_json_ready(obj):
        if isinstance(obj, dict):
            return {k: _to_json_ready(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_json_ready(v) for v in obj]
        try:
            import pandas as _pd

            if isinstance(obj, _pd.DataFrame):
                return obj.to_dict(orient="records")
        except Exception:
            pass
        return obj

    serial = _to_json_ready(result)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(serial, handle, indent=2, ensure_ascii=False)
    return output_path


def export_schema_validation_csv(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = result.get("compatibility", {}).get("frame", pd.DataFrame())
    frame.to_csv(output_path, index=False, float_format="%.6f")
    return output_path


def export_schema_validation_txt(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for report in result.get("reports", []):
        lines.append(f"{report.get('schema_name')}: {'valid' if report.get('valid') else 'invalid'}")
        if report.get("errors"):
            for error in report["errors"]:
                lines.append(f"  error: {error}")
        if report.get("warnings"):
            for warning in report["warnings"]:
                lines.append(f"  warning: {warning}")
    summary = result.get("compatibility", {}).get("summary", {})
    lines.append(f"compatibility_score: {summary.get('compatibility_score', 0.0)}")
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return output_path
