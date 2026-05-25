"""
Cross-metric reliability benchmarking and correlation analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.experiment_statistics import compute_correlation_table, export_csv, export_json, strongest_pairs


CORRELATION_METRICS = [
    "uncertainty_score",
    "mean_uncertainty_score",
    "disagreement_rate",
    "disagreement_score",
    "instability_score",
    "mean_sample_instability",
    "overall_instability",
    "attention_drift_score",
    "mean_attention_drift_score",
    "consensus_reliability_score",
    "consensus_reliability",
    "risk_score",
    "reliability_risk_score",
]


def _to_frame(records):
    if records is None:
        return pd.DataFrame()
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, list):
        return pd.DataFrame(records)
    if isinstance(records, dict):
        return pd.DataFrame([records])
    return pd.DataFrame([records])


def _normalise_frame(frame):
    if frame.empty:
        return frame
    working = frame.copy()
    aliases = {
        "mean_uncertainty_score": ["uncertainty_score"],
        "mean_attention_drift_score": ["attention_drift_score"],
        "consensus_reliability_score": ["consensus_reliability"],
        "reliability_risk_score": ["risk_score"],
    }
    for target, sources in aliases.items():
        if target in working.columns:
            continue
        for source in sources:
            if source in working.columns:
                working[target] = working[source]
                break
    return working


def benchmark_reliability_relationships(records):
    frame = _normalise_frame(_to_frame(records))
    correlation_frame = compute_correlation_table(frame, CORRELATION_METRICS)
    if correlation_frame.empty:
        return {
            "correlation_frame": correlation_frame,
            "strongest_pairs": [],
            "summary": {
                "total_metrics": 0,
                "strongest_dependency_pair": None,
                "relationship_line": "No correlation structure was available for benchmarking.",
            },
        }
    pairs = strongest_pairs(correlation_frame, top_n=10)
    top_pair = pairs[0] if pairs else None
    summary = {
        "total_metrics": int(len(correlation_frame.columns)),
        "strongest_dependency_pair": top_pair,
        "relationship_line": (
            f"{top_pair['metric_a']} and {top_pair['metric_b']} showed the strongest dependence at r={top_pair['correlation']:.3f}."
            if top_pair is not None else "No dominant dependency pair could be identified."
        ),
        "correlation_strength": {
            "mean_absolute_correlation": round(float(np.abs(correlation_frame.values[np.triu_indices(len(correlation_frame), k=1)]).mean()), 6) if len(correlation_frame) > 1 else 0.0,
        },
    }
    return {
        "correlation_frame": correlation_frame,
        "strongest_pairs": pairs,
        "summary": summary,
    }


def export_reliability_benchmark_outputs(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_csv(output_dir / "correlation_matrix.csv", result.get("correlation_frame", pd.DataFrame()))
    export_json(output_dir / "correlation_matrix.json", result.get("correlation_frame", pd.DataFrame()).to_dict() if isinstance(result.get("correlation_frame"), pd.DataFrame) else result.get("correlation_frame", {}))
    export_json(output_dir / "correlation_pairs.json", result.get("strongest_pairs", []))
    export_json(output_dir / "correlation_summary.json", result.get("summary", {}))
    return output_dir
