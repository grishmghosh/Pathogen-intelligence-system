"""
Consolidated stabilization summary and exports.
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
    return value


def build_stabilization_summary(schema_result=None, integrity_result=None, audit_result=None, consistency_result=None, readiness_result=None, campaign_result=None, health_result=None):
    summary = {
        "schema_validation": _safe_value(schema_result or {}),
        "artifact_integrity": _safe_value(integrity_result or {}),
        "experiment_audit": _safe_value(audit_result or {}),
        "consistency": _safe_value(consistency_result or {}),
        "dataset_readiness": _safe_value(readiness_result or {}),
        "validation_campaigns": _safe_value(campaign_result or {}),
        "framework_health": _safe_value(health_result or {}),
    }
    health_score = (health_result or {}).get("overall_health_score", 0.0) if isinstance(health_result, dict) else 0.0
    summary["summary_line"] = f"Framework health score: {health_score:.2f}"
    summary["health_label"] = (health_result or {}).get("health_label", "unknown") if isinstance(health_result, dict) else "unknown"
    return summary


def build_stabilization_frame(summary):
    rows = []
    for section_name, payload in (summary or {}).items():
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        rows.append({"section": section_name, "metric": f"{key}.{nested_key}", "value": nested_value})
                else:
                    rows.append({"section": section_name, "metric": key, "value": value})
        else:
            rows.append({"section": section_name, "metric": "value", "value": payload})
    return pd.DataFrame(rows)


def export_stabilization_json(summary, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_safe_value(summary), handle, indent=2, ensure_ascii=False)
    return output_path


def export_stabilization_csv(summary, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_stabilization_frame(summary).to_csv(output_path, index=False, float_format="%.6f")
    return output_path


def export_stabilization_txt(summary, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(summary.get("summary_line", "Framework stabilization summary unavailable."))]
    if summary.get("health_label"):
        lines.append(f"health_label: {summary['health_label']}")
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return output_path
