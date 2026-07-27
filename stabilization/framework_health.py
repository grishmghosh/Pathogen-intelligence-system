"""
Framework health scoring for stabilization reporting.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


def _score_from_validity(value):
    return 100.0 if value else 0.0


def assess_framework_health(schema_result=None, integrity_result=None, audit_result=None, consistency_result=None, readiness_result=None, campaign_result=None):
    schema_result = schema_result or {}
    integrity_result = integrity_result or {}
    audit_result = audit_result or {}
    consistency_result = consistency_result or {}
    readiness_result = readiness_result or {}
    campaign_result = campaign_result or {}

    schema_score = schema_result.get("compatibility", {}).get("summary", {}).get("compatibility_score", 0.0) if isinstance(schema_result, dict) else 0.0
    integrity_score = integrity_result.get("summary", {}).get("integrity_rate", 0.0) * 100.0 if isinstance(integrity_result, dict) else 0.0
    audit_score = audit_result.get("summary", {}).get("execution_integrity_score", 0.0) if isinstance(audit_result, dict) else 0.0
    consistency_score = consistency_result.get("consistency_score", 0.0) if isinstance(consistency_result, dict) else 0.0
    readiness_score = readiness_result.get("readiness_score", 0.0) if isinstance(readiness_result, dict) else 0.0
    campaign_score = campaign_result.get("summary", {}).get("validation_campaign_score", 0.0) if isinstance(campaign_result, dict) else 0.0

    overall = np.mean([schema_score, integrity_score, audit_score, consistency_score, readiness_score, campaign_score]) if any([schema_score, integrity_score, audit_score, consistency_score, readiness_score, campaign_score]) else 0.0
    overall = round(float(overall), 6)

    subsystem_scores = {
        "schema_validation": round(float(schema_score), 6),
        "artifact_integrity": round(float(integrity_score), 6),
        "experiment_audit": round(float(audit_score), 6),
        "cross_subsystem_consistency": round(float(consistency_score), 6),
        "dataset_readiness": round(float(readiness_score), 6),
        "validation_campaigns": round(float(campaign_score), 6),
    }
    ranking_frame = pd.DataFrame([
        {"subsystem": subsystem, "health_score": score}
        for subsystem, score in subsystem_scores.items()
    ]).sort_values("health_score", ascending=False).reset_index(drop=True)

    warnings = []
    if integrity_score < 80:
        warnings.append("Artifact integrity is below the preferred stability threshold.")
    if consistency_score < 80:
        warnings.append("Cross-subsystem consistency requires review.")
    if readiness_score < 60:
        warnings.append("Dataset readiness is limited for expansion.")

    return {
        "overall_health_score": overall,
        "subsystem_scores": subsystem_scores,
        "subsystem_rankings": ranking_frame,
        "warnings": warnings,
        "warning_count": int(len(warnings)),
        "health_label": "healthy" if overall >= 80 else "degraded" if overall >= 60 else "at_risk",
    }


def build_health_summary(health_result):
    health_result = health_result or {}
    lines = [f"overall_health_score: {health_result.get('overall_health_score', 0.0)}"]
    lines.append(f"health_label: {health_result.get('health_label', 'unknown')}")
    for warning in health_result.get("warnings", []):
        lines.append(f"warning: {warning}")
    return {"lines": lines, "summary_line": lines[0], "warning_count": int(len(health_result.get('warnings', []))) if isinstance(health_result, dict) else 0}


def export_health_json(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = dict(result)
    if isinstance(serialisable.get("subsystem_rankings"), pd.DataFrame):
        serialisable["subsystem_rankings"] = serialisable["subsystem_rankings"].to_dict(orient="records")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(serialisable, handle, indent=2, ensure_ascii=False)
    return output_path


def export_health_csv(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = result.get("subsystem_rankings", pd.DataFrame())
    frame.to_csv(output_path, index=False, float_format="%.6f")
    return output_path


def export_health_txt(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"overall_health_score: {result.get('overall_health_score', 0.0)}", f"health_label: {result.get('health_label', 'unknown')}"]
    for warning in result.get("warnings", []):
        lines.append(f"warning: {warning}")
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return output_path
