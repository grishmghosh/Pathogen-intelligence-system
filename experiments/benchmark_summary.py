"""
Consolidated benchmark summary generation for Step 9.
"""

from pathlib import Path

import pandas as pd

from experiments.experiment_statistics import export_csv, export_json, publication_sentences, to_json_ready


def build_benchmark_summary(model_result=None, perturbation_result=None, reliability_result=None, statistics_result=None):
    model_result = model_result or {}
    perturbation_result = perturbation_result or {}
    reliability_result = reliability_result or {}
    statistics_result = statistics_result or {}

    model_frame = model_result.get("rankings", pd.DataFrame())
    perturbation_frame = perturbation_result.get("rankings", pd.DataFrame())
    severity_frame = perturbation_result.get("severity_frame", pd.DataFrame())
    correlation_frame = reliability_result.get("correlation_frame", pd.DataFrame())

    summary = {
        "benchmark": {
            "total_models": int(len(model_frame)) if isinstance(model_frame, pd.DataFrame) else 0,
            "total_perturbations": int(len(perturbation_frame)) if isinstance(perturbation_frame, pd.DataFrame) else 0,
            "total_severity_levels": int(len(severity_frame)) if isinstance(severity_frame, pd.DataFrame) else 0,
            "metric_count": int(len(correlation_frame.columns)) if isinstance(correlation_frame, pd.DataFrame) and not correlation_frame.empty else 0,
        },
        "model_summary": model_result.get("summary", {}),
        "perturbation_summary": perturbation_result.get("summary", {}),
        "severity_summary": perturbation_result.get("severity_summary", {}),
        "correlation_summary": reliability_result.get("summary", {}),
        "statistics_summary": statistics_result,
    }
    sentences = publication_sentences(
        model_summary=summary["model_summary"],
        perturbation_summary=summary["perturbation_summary"],
        severity_summary=summary["severity_summary"],
        correlation_pairs=reliability_result.get("strongest_pairs", []),
    )
    summary["publication_summary"] = sentences
    summary["leaderboards"] = {
        "best_robustness_model": summary["model_summary"].get("best_robustness_model"),
        "most_stable_model": summary["model_summary"].get("most_stable_model"),
        "lowest_risk_model": summary["model_summary"].get("lowest_risk_model"),
        "best_attention_stability_model": summary["model_summary"].get("best_attention_stability_model"),
        "strongest_perturbation_resilience": summary["model_summary"].get("strongest_perturbation_resilience"),
        "most_disruptive_perturbation": summary["perturbation_summary"].get("most_disruptive_perturbation"),
        "strongest_dependency_pair": reliability_result.get("summary", {}).get("strongest_dependency_pair"),
    }
    summary["summary_line"] = sentences[0] if sentences else "No benchmark summary was available."
    return summary


def build_benchmark_summary_frame(summary):
    rows = []
    for section_name, payload in (summary or {}).items():
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        rows.append({"section": section_name, "metric": f"{key}.{nested_key}", "value": nested_value})
                elif isinstance(value, list):
                    rows.append({"section": section_name, "metric": key, "value": len(value)})
                else:
                    rows.append({"section": section_name, "metric": key, "value": value})
        elif isinstance(payload, list):
            rows.append({"section": section_name, "metric": "items", "value": len(payload)})
        else:
            rows.append({"section": section_name, "metric": "value", "value": payload})
    return pd.DataFrame(rows)


def export_benchmark_summary_json(summary, output_path):
    return export_json(output_path, summary)


def export_benchmark_summary_csv(summary, output_path):
    return export_csv(output_path, build_benchmark_summary_frame(summary))
