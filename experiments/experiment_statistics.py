"""
Deterministic statistical helpers for large-scale benchmarking.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


def _to_frame(records):
    if records is None:
        return pd.DataFrame()
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, dict):
        return pd.DataFrame([records])
    if isinstance(records, list):
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)
    return pd.DataFrame([records])


def _numeric_series(frame, column):
    if frame is None or frame.empty or column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def _safe_scalar(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _safe_scalar(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_safe_scalar(item) for item in value]
    return value


def compute_scalar_statistics(frame, metric_columns):
    frame = _to_frame(frame)
    summary = {}
    for metric in metric_columns:
        series = _numeric_series(frame, metric)
        if series.empty:
            summary[metric] = {
                "count": 0,
                "mean": None,
                "median": None,
                "variance": None,
                "std": None,
                "min": None,
                "max": None,
                "escalation_delta": None,
            }
            continue
        summary[metric] = {
            "count": int(series.shape[0]),
            "mean": round(float(series.mean()), 6),
            "median": round(float(series.median()), 6),
            "variance": round(float(series.var(ddof=0)), 6),
            "std": round(float(series.std(ddof=0)), 6),
            "min": round(float(series.min()), 6),
            "max": round(float(series.max()), 6),
            "escalation_delta": round(float(series.max() - series.min()), 6),
        }
    return summary


def compute_group_statistics(frame, group_columns, metric_columns):
    frame = _to_frame(frame)
    if frame.empty or not group_columns:
        return pd.DataFrame()
    present_groups = [column for column in group_columns if column in frame.columns]
    if not present_groups:
        return pd.DataFrame()
    available_metrics = [column for column in metric_columns if column in frame.columns]
    if not available_metrics:
        return pd.DataFrame()
    grouped = frame.groupby(present_groups, dropna=False)
    rows = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {present_groups[index]: keys[index] for index in range(len(present_groups))}
        for metric in available_metrics:
            series = _numeric_series(group, metric)
            if series.empty:
                continue
            row[f"{metric}_mean"] = round(float(series.mean()), 6)
            row[f"{metric}_median"] = round(float(series.median()), 6)
            row[f"{metric}_std"] = round(float(series.std(ddof=0)), 6)
            row[f"{metric}_min"] = round(float(series.min()), 6)
            row[f"{metric}_max"] = round(float(series.max()), 6)
        row["record_count"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_severity_escalation(frame, severity_column, metric_columns, severity_order=None):
    frame = _to_frame(frame)
    if frame.empty or severity_column not in frame.columns:
        return pd.DataFrame(), {}
    order = severity_order or ["mild", "moderate", "severe"]
    severity_rank = {str(level): index for index, level in enumerate(order)}
    working = frame.copy()
    working["_severity_rank"] = working[severity_column].astype(str).map(severity_rank)
    working = working.dropna(subset=["_severity_rank"]).sort_values("_severity_rank")
    rows = []
    for severity_level, group in working.groupby(severity_column, dropna=False):
        row = {
            "severity_level": str(severity_level),
            "severity_rank": int(severity_rank.get(str(severity_level), -1)),
            "record_count": int(len(group)),
        }
        for metric in metric_columns:
            series = _numeric_series(group, metric)
            if series.empty:
                continue
            row[f"{metric}_mean"] = round(float(series.mean()), 6)
            row[f"{metric}_min"] = round(float(series.min()), 6)
            row[f"{metric}_max"] = round(float(series.max()), 6)
        rows.append(row)
    severity_frame = pd.DataFrame(rows).sort_values("severity_rank").reset_index(drop=True) if rows else pd.DataFrame()
    escalation = {}
    if not severity_frame.empty:
        for metric in metric_columns:
            metric_column = f"{metric}_mean"
            if metric_column in severity_frame.columns and len(severity_frame[metric_column].dropna()) >= 2:
                values = pd.to_numeric(severity_frame[metric_column], errors="coerce").dropna()
                if not values.empty:
                    escalation[metric] = round(float(values.iloc[-1] - values.iloc[0]), 6)
    return severity_frame, escalation


def compute_correlation_table(frame, metric_columns):
    frame = _to_frame(frame)
    available = [column for column in metric_columns if column in frame.columns]
    if frame.empty or not available:
        return pd.DataFrame()
    numeric = frame[available].apply(pd.to_numeric, errors="coerce")
    return numeric.corr().fillna(0.0)


def strongest_pairs(correlation_frame, top_n=5):
    if correlation_frame is None or correlation_frame.empty:
        return []
    pairs = []
    columns = list(correlation_frame.columns)
    for index, left in enumerate(columns):
        for right in columns[index + 1:]:
            value = correlation_frame.loc[left, right]
            if pd.isna(value):
                continue
            pairs.append({
                "metric_a": str(left),
                "metric_b": str(right),
                "correlation": round(float(value), 6),
                "absolute_correlation": round(abs(float(value)), 6),
            })
    pairs.sort(key=lambda item: item["absolute_correlation"], reverse=True)
    return pairs[:top_n]


def publication_sentences(model_summary=None, perturbation_summary=None, severity_summary=None, correlation_pairs=None):
    sentences = []
    if isinstance(model_summary, dict):
        stable_model = model_summary.get("most_stable_model") or model_summary.get("best_robustness_model")
        if stable_model is not None:
            sentences.append(f"Model {stable_model} demonstrated the strongest deterministic stability profile in the benchmark set.")
    if isinstance(perturbation_summary, dict):
        dominant = perturbation_summary.get("most_disruptive_perturbation")
        if dominant is not None:
            sentences.append(f"{dominant} produced the largest aggregate degradation signal across models.")
    if isinstance(severity_summary, dict):
        trend = severity_summary.get("trend_direction")
        if trend is not None:
            sentences.append(f"Benchmark severity trends were {trend}, indicating structured escalation across perturbation levels.")
    if correlation_pairs:
        top = correlation_pairs[0]
        sentences.append(
            f"{top['metric_a']} and {top['metric_b']} showed the strongest dependency (r={top['correlation']:.3f})."
        )
    if not sentences:
        sentences.append("No benchmark-level structure could be extracted from the provided records.")
    return sentences


def to_json_ready(payload):
    return _safe_scalar(payload)


def export_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_json_ready(payload), handle, indent=2, ensure_ascii=False)
    return path


def export_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame is None:
        frame = pd.DataFrame()
    frame.to_csv(path, index=False, float_format="%.6f")
    return path
