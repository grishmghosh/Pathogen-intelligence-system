"""
Consolidated reporting summaries for publication-ready artifacts.
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


def build_reporting_summary(report_id=None, provenance=None, reproducibility=None, tables=None, figures=None, narratives=None, benchmark_summary=None):
    tables = tables or {}
    figures = figures or {}
    narratives = narratives or {}
    benchmark_summary = benchmark_summary or {}

    summary = {
        "report_id": report_id,
        "provenance": _safe_value(provenance or {}),
        "reproducibility": _safe_value(reproducibility or {}),
        "tables": {str(key): {"row_count": int(value.get("row_count", 0)) if isinstance(value, dict) else None, "column_count": int(value.get("column_count", 0)) if isinstance(value, dict) else None} for key, value in tables.items()},
        "figures": _safe_value(figures),
        "narratives": _safe_value(narratives),
        "benchmark_summary": _safe_value(benchmark_summary),
    }
    summary["summary_line"] = (
        narratives.get("summary_line")
        or benchmark_summary.get("summary_line")
        or "Report package assembled successfully."
    )
    summary["artifact_counts"] = {
        "tables": int(len(tables)),
        "figure_groups": int(len((figures or {}).get("figure_groups", {}) if isinstance(figures, dict) else {})),
        "narratives": int(narratives.get("narrative_count", 0) if isinstance(narratives, dict) else 0),
    }
    return summary


def build_reporting_summary_frame(summary):
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


def export_reporting_summary_json(summary, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_safe_value(summary), handle, indent=2, ensure_ascii=False)
    return output_path


def export_reporting_summary_csv(summary, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_reporting_summary_frame(summary).to_csv(output_path, index=False, float_format="%.6f")
    return output_path
