"""
Publication-style table generation for reporting outputs.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


def _safe_frame(frame):
    if frame is None:
        return pd.DataFrame()
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if isinstance(frame, dict):
        return pd.DataFrame([frame])
    if isinstance(frame, list):
        return pd.DataFrame(frame)
    return pd.DataFrame([frame])


def _to_markdown_table(frame):
    frame = _safe_frame(frame)
    if frame.empty:
        return "| No data |\n| --- |\n| No rows available |"
    display = frame.copy()
    display = display.replace({np.nan: None})
    columns = list(display.columns)
    header = "| " + " | ".join(str(column) for column in columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in display.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator] + rows)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def _write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _safe_frame(frame).to_csv(path, index=False, float_format="%.6f")
    return path


def build_publication_table(frame, name=None):
    frame = _safe_frame(frame)
    return {
        "name": name,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns],
        "dataframe": frame,
        "csv_text": frame.to_csv(index=False, float_format="%.6f"),
        "markdown": _to_markdown_table(frame),
    }


def build_publication_tables(table_map):
    table_map = table_map or {}
    result = {}
    for name, frame in table_map.items():
        result[str(name)] = build_publication_table(frame, name=str(name))
    return result


def export_publication_tables(table_map, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = {}
    for name, payload in (table_map or {}).items():
        frame = payload.get("dataframe") if isinstance(payload, dict) else payload
        csv_path = _write_csv(output_dir / f"{name}.csv", frame)
        json_path = _write_json(output_dir / f"{name}.json", _safe_frame(frame).to_dict(orient="records"))
        md_path = output_dir / f"{name}.md"
        with md_path.open("w", encoding="utf-8") as handle:
            handle.write(payload.get("markdown") if isinstance(payload, dict) else _to_markdown_table(frame))
        exported[str(name)] = {
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
        }
    return exported


def build_model_ranking_table(model_summary):
    model_summary = model_summary or {}
    rows = []
    for key in ["best_robustness_model", "most_stable_model", "lowest_risk_model", "best_attention_stability_model", "strongest_perturbation_resilience", "reliability_leader"]:
        value = model_summary.get(key)
        if value is not None:
            rows.append({"ranking": key, "model_name": value})
    return pd.DataFrame(rows)


def build_perturbation_sensitivity_table(perturbation_summary):
    perturbation_summary = perturbation_summary or {}
    rows = []
    for key in ["most_disruptive_perturbation", "most_uncertain_perturbation", "most_fragile_perturbation"]:
        value = perturbation_summary.get(key)
        if value is not None:
            rows.append({"ranking": key, "perturbation_type": value})
    fragility_summary = perturbation_summary.get("fragility_summary", {})
    for key, value in fragility_summary.items():
        rows.append({"ranking": f"fragility.{key}", "value": value})
    return pd.DataFrame(rows)


def build_instability_table(statistics_summary):
    statistics_summary = statistics_summary or {}
    entries = statistics_summary.get("instability") if isinstance(statistics_summary, dict) else None
    if isinstance(entries, pd.DataFrame):
        return entries
    if isinstance(entries, list):
        return pd.DataFrame(entries)
    return pd.DataFrame([{"metric": "instability", "value": statistics_summary.get("mean_instability") if isinstance(statistics_summary, dict) else None}])


def build_uncertainty_table(statistics_summary):
    statistics_summary = statistics_summary or {}
    entries = statistics_summary.get("uncertainty") if isinstance(statistics_summary, dict) else None
    if isinstance(entries, pd.DataFrame):
        return entries
    if isinstance(entries, list):
        return pd.DataFrame(entries)
    return pd.DataFrame([{"metric": "uncertainty", "value": statistics_summary.get("mean_uncertainty") if isinstance(statistics_summary, dict) else None}])


def build_attention_table(statistics_summary):
    statistics_summary = statistics_summary or {}
    entries = statistics_summary.get("attention") if isinstance(statistics_summary, dict) else None
    if isinstance(entries, pd.DataFrame):
        return entries
    if isinstance(entries, list):
        return pd.DataFrame(entries)
    return pd.DataFrame([{"metric": "attention_stability", "value": statistics_summary.get("mean_attention_stability") if isinstance(statistics_summary, dict) else None}])


def build_consensus_table(statistics_summary):
    statistics_summary = statistics_summary or {}
    entries = statistics_summary.get("consensus") if isinstance(statistics_summary, dict) else None
    if isinstance(entries, pd.DataFrame):
        return entries
    if isinstance(entries, list):
        return pd.DataFrame(entries)
    return pd.DataFrame([{"metric": "consensus_reliability", "value": statistics_summary.get("mean_consensus_reliability") if isinstance(statistics_summary, dict) else None}])


def build_risk_table(statistics_summary):
    statistics_summary = statistics_summary or {}
    entries = statistics_summary.get("risk") if isinstance(statistics_summary, dict) else None
    if isinstance(entries, pd.DataFrame):
        return entries
    if isinstance(entries, list):
        return pd.DataFrame(entries)
    return pd.DataFrame([{"metric": "risk", "value": statistics_summary.get("mean_risk") if isinstance(statistics_summary, dict) else None}])
