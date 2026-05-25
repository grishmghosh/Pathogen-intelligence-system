"""
Experiment audit utilities for reproducibility and execution integrity.
"""

from pathlib import Path
import json

import pandas as pd


def _safe_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _audit_stage_execution(experiment_summary):
    execution = (experiment_summary or {}).get("execution", {}) if isinstance(experiment_summary, dict) else {}
    stage_summaries = (experiment_summary or {}).get("stage_summaries", {}) if isinstance(experiment_summary, dict) else {}
    completed = set(execution.get("completed_stages", []) or [])
    failed = set(execution.get("failed_stages", []) or [])
    skipped = set(execution.get("skipped_stages", []) or [])
    total = int(execution.get("total_stages", len(stage_summaries)))
    execution_integrity = 0.0
    if total > 0:
        execution_integrity = round(float(len(completed) / total), 6)
    return {
        "total_stages": total,
        "completed_stages": sorted(list(completed)),
        "failed_stages": sorted(list(failed)),
        "skipped_stages": sorted(list(skipped)),
        "execution_integrity_score": execution_integrity,
        "partial_run": len(failed) > 0 or len(skipped) > 0,
    }


def audit_experiment(experiment_summary=None, manifest=None, reproducibility=None, report_manifest=None):
    experiment_summary = experiment_summary or {}
    manifest = manifest or {}
    reproducibility = reproducibility or {}
    report_manifest = report_manifest or {}

    stage_audit = _audit_stage_execution(experiment_summary)
    missing_metadata = []
    for key in ["experiment_id", "created_at", "output_root"]:
        if not manifest.get(key):
            missing_metadata.append(key)
    if not reproducibility.get("seed"):
        missing_metadata.append("seed")

    config = manifest.get("config", {}) if isinstance(manifest, dict) else {}
    dataset = config.get("dataset", {}) if isinstance(config, dict) else {}
    dataset_coverage = len(dataset.get("samples", []) or [])

    audit_rows = [{
        "audit_scope": "experiment",
        "execution_integrity_score": stage_audit["execution_integrity_score"],
        "missing_metadata_count": int(len(missing_metadata)),
        "partial_run": stage_audit["partial_run"],
        "dataset_sample_count": dataset_coverage,
        "report_manifest_present": bool(report_manifest),
    }]
    frame = pd.DataFrame(audit_rows)
    summary = {
        "execution_integrity_score": stage_audit["execution_integrity_score"],
        "missing_metadata": missing_metadata,
        "missing_metadata_count": int(len(missing_metadata)),
        "partial_run": stage_audit["partial_run"],
        "dataset_sample_count": dataset_coverage,
        "reproducibility_status": "complete" if not missing_metadata else "partial",
    }
    return {"frame": frame, "stage_audit": stage_audit, "summary": summary}


def build_reproducibility_report(audit_result):
    audit_result = audit_result or {}
    summary = audit_result.get("summary", {}) if isinstance(audit_result, dict) else {}
    lines = []
    lines.append(f"Execution integrity score: {summary.get('execution_integrity_score', 0.0)}")
    if summary.get("missing_metadata"):
        lines.append(f"Missing metadata: {', '.join(summary['missing_metadata'])}")
    else:
        lines.append("No missing metadata detected.")
    lines.append(f"Reproducibility status: {summary.get('reproducibility_status', 'unknown')}")
    return {"lines": lines, "summary_line": lines[0], "line_count": int(len(lines))}


def export_audit_json(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": result.get("summary", {}), "stage_audit": result.get("stage_audit", {})}, handle, indent=2, ensure_ascii=False)
    return output_path


def export_audit_csv(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = result.get("frame", pd.DataFrame())
    frame.to_csv(output_path, index=False, float_format="%.6f")
    return output_path


def export_audit_txt(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = result.get("summary", {})
    lines = [
        f"execution_integrity_score: {summary.get('execution_integrity_score', 0.0)}",
        f"reproducibility_status: {summary.get('reproducibility_status', 'unknown')}",
        f"missing_metadata_count: {summary.get('missing_metadata_count', 0)}",
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return output_path
