"""
Consolidated experiment summary generation for Step 8.

This module merges the outputs of the executed experiment stages into a single
research-oriented summary and provides CSV / JSON exports.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from pipeline.pipeline_utils import flatten_mapping, dataframe_metrics, safe_json_dump, ensure_directory, _serialise_value


SUMMARY_DIR_NAME = "summaries"


def _merge_section(summary, section_name, payload):
    summary[section_name] = _serialise_value(payload)
    return summary


def _stage_summary(stage_name, stage_result):
    if stage_result is None:
        return {"stage": stage_name, "status": "missing"}
    if isinstance(stage_result, pd.DataFrame):
        result = {"stage": stage_name, "status": "completed"}
        result.update(dataframe_metrics(stage_result, stage_name))
        return result
    if isinstance(stage_result, dict):
        result = {
            "stage": stage_name,
            "status": stage_result.get("status", "completed"),
            "duration_seconds": stage_result.get("duration_seconds"),
        }
        if "error" in stage_result:
            result["error"] = stage_result.get("error")
        if "summary" in stage_result and isinstance(stage_result["summary"], dict):
            for row in flatten_mapping(stage_result["summary"], prefix=f"{stage_name}.summary"):
                result[row["path"]] = row["value"]
        if "data" in stage_result:
            data = stage_result["data"]
            if isinstance(data, pd.DataFrame):
                result.update(dataframe_metrics(data, stage_name))
        if "metrics" in stage_result and isinstance(stage_result["metrics"], dict):
            for row in flatten_mapping(stage_result["metrics"], prefix=f"{stage_name}.metrics"):
                result[row["path"]] = row["value"]
        return result
    return {"stage": stage_name, "status": "completed", "value": _serialise_value(stage_result)}


def build_consolidated_summary(stage_results, manifest=None, registry_record=None, reproducibility_manifest=None, config=None):
    summary = {
        "experiment": {
            "experiment_id": manifest.get("experiment_id") if isinstance(manifest, dict) else None,
            "experiment_name": manifest.get("experiment_name") if isinstance(manifest, dict) else None,
            "created_at": manifest.get("created_at") if isinstance(manifest, dict) else None,
            "output_root": manifest.get("output_root") if isinstance(manifest, dict) else None,
        },
        "execution": stage_results.get("execution_report", {}) if isinstance(stage_results, dict) else {},
        "manifest": manifest if isinstance(manifest, dict) else {},
        "registry": registry_record if isinstance(registry_record, dict) else {},
        "reproducibility": reproducibility_manifest if isinstance(reproducibility_manifest, dict) else {},
        "stage_summaries": {},
        "sections": {},
    }

    stage_order = []
    for stage_name, stage_result in stage_results.items():
        if stage_name in {"execution_report", "experiment_manifest", "reproducibility_manifest", "registry_record", "summary", "summary_frame"}:
            continue
        stage_order.append(stage_name)
        summary["stage_summaries"][stage_name] = _stage_summary(stage_name, stage_result)

        if isinstance(stage_result, dict):
            for key in ["summary", "robustness_report", "calibration_summary", "disagreement_summary", "consensus_summary", "risk_summary", "uncertainty_summary", "explainability_summary"]:
                if key in stage_result:
                    section_name = f"{stage_name}.{key}"
                    summary["sections"][section_name] = _serialise_value(stage_result[key])

    summary["stage_order"] = stage_order
    summary["section_count"] = int(len(summary["sections"]))
    summary["stage_count"] = int(len(summary["stage_summaries"]))
    summary["summary_line"] = _build_summary_line(summary)
    return summary


def _build_summary_line(summary):
    execution = summary.get("execution", {}) if isinstance(summary, dict) else {}
    if execution.get("failed_count", 0) > 0:
        return f"{execution.get('failed_count')} stage(s) failed and the pipeline continued with graceful degradation."
    if execution.get("skipped_count", 0) > 0:
        return f"Pipeline completed with {execution.get('skipped_count')} skipped stage(s)."
    return "All enabled stages completed successfully."


def build_summary_frame(summary):
    rows = []
    for section_name, payload in (summary or {}).items():
        if isinstance(payload, dict):
            rows.extend([{"section": section_name, **row} for row in flatten_mapping(payload)])
        elif isinstance(payload, list):
            for index, item in enumerate(payload):
                if isinstance(item, dict):
                    rows.extend([{"section": f"{section_name}[{index}]", **row} for row in flatten_mapping(item)])
                else:
                    rows.append({"section": section_name, "path": f"[{index}]", "value": _serialise_value(item)})
        else:
            rows.append({"section": section_name, "path": "value", "value": _serialise_value(payload)})
    return pd.DataFrame(rows)


def export_consolidated_summary_json(summary, output_path):
    return safe_json_dump(output_path, summary)


def export_consolidated_summary_csv(summary, output_path):
    frame = build_summary_frame(summary)
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    frame.to_csv(output_path, index=False, float_format="%.6f")
    return output_path
