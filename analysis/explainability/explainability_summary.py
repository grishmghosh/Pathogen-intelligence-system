"""
Explainability summary, relationships, and exports for the Pathogen
Intelligence System (Step 7).

This module ties together Grad-CAM generation, attention drift, and
cross-model attention comparison with Step 6 uncertainty metrics and the
previous disagreement / instability layers. Visual exports are kept separate
from the metric computations to preserve modularity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from analysis.explainability.attention_analysis import (
    aggregate_attention_summary,
    compute_attention_analysis,
    normalize_attention_map,
)
from analysis.explainability.attention_comparison import (
    compute_attention_consistency_summary,
    compute_attention_divergence_ranking,
    compute_cross_model_attention_comparison,
)
from analysis.explainability.attention_drift import (
    compute_attention_collapse_report,
    compute_attention_drift_analysis,
    compute_attention_stability_curve,
)
from analysis.explainability.gradcam_generator import (
    create_attention_overlay,
    generate_gradcam,
    save_heatmap_png,
    save_overlay_png,
)


_HEATMAP_DIR = Path("results") / "explainability" / "heatmaps"
_OVERLAY_DIR = Path("results") / "explainability" / "overlays"
_DRIFT_DIR = Path("results") / "explainability" / "drift_analysis"
_COMPARISON_DIR = Path("results") / "explainability" / "comparisons"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(output_dir: Optional[Union[str, Path]], default_dir: Path) -> Path:
    directory = Path(output_dir) if output_dir is not None else default_dir
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _serialise(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _serialise(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_serialise(value) for value in obj]
    if isinstance(obj, tuple):
        return [_serialise(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Image.Image):
        return {"mode": obj.mode, "size": obj.size}
    return obj


def _flatten_mapping(obj: Dict[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, value in obj.items():
        path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            rows.extend(_flatten_mapping(value, path))
        elif isinstance(value, list):
            if not value:
                rows.append({"path": path, "value": []})
            else:
                for index, item in enumerate(value):
                    item_path = f"{path}[{index}]"
                    if isinstance(item, dict):
                        rows.extend(_flatten_mapping(item, item_path))
                    else:
                        rows.append({"path": item_path, "value": _serialise(item)})
        else:
            rows.append({"path": path, "value": _serialise(value)})
    return rows


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if pair.empty or len(pair) < 2:
        return 0.0
    left_values = pd.to_numeric(pair.iloc[:, 0], errors="coerce")
    right_values = pd.to_numeric(pair.iloc[:, 1], errors="coerce")
    if left_values.nunique(dropna=True) < 2 or right_values.nunique(dropna=True) < 2:
        return 0.0
    corr = left_values.corr(right_values)
    if pd.isna(corr):
        return 0.0
    return float(corr)


def _first_present_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        return default
    return float(numeric)


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _save_figure(fig: plt.Figure, output_path: Union[str, Path]) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return output


def _save_image(image: Image.Image, output_path: Union[str, Path]) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def _extract_heatmap(item: Any) -> Optional[np.ndarray]:
    if isinstance(item, dict):
        if item.get("status") == "ok":
            heatmap = item.get("normalized_heatmap")
            if heatmap is None:
                heatmap = item.get("raw_heatmap")
            if heatmap is not None:
                return normalize_attention_map(heatmap)
        return None
    return normalize_attention_map(item)


# ---------------------------------------------------------------------------
# Core summaries
# ---------------------------------------------------------------------------


def compute_explainability_summary(
    attention_analysis: pd.DataFrame,
    drift_analysis: Optional[pd.DataFrame] = None,
    comparison_analysis: Optional[pd.DataFrame] = None,
    uncertainty_data: Optional[pd.DataFrame] = None,
    disagreement_data: Optional[pd.DataFrame] = None,
    instability_data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Generate a structured summary of the explainability layer."""
    attention_summary = aggregate_attention_summary(attention_analysis)
    drift_summary = compute_explainability_drift_summary(drift_analysis)
    consistency_summary = compute_attention_consistency_summary(comparison_analysis) if comparison_analysis is not None else {
        "total_comparisons": 0,
        "mean_attention_similarity": 0.0,
        "mean_attention_divergence": 0.0,
        "high_consistency_count": 0,
        "low_consistency_count": 0,
        "most_divergent_pair": None,
        "model_pair_rankings": [],
    }
    relationship_summary = compute_explainability_relationship_summary(
        drift_analysis,
        uncertainty_data=uncertainty_data,
        disagreement_data=disagreement_data,
        instability_data=instability_data,
    )

    return {
        "attention_summary": attention_summary,
        "drift_summary": drift_summary,
        "consistency_summary": consistency_summary,
        "relationship_summary": relationship_summary,
        "summary_line": _build_summary_line(attention_summary, drift_summary, consistency_summary),
    }


def compute_explainability_drift_summary(drift_analysis: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Summarise attention drift and collapse behavior."""
    if drift_analysis is None or drift_analysis.empty or "attention_drift_score" not in drift_analysis.columns:
        return {
            "total_observations": 0,
            "mean_attention_drift_score": 0.0,
            "max_attention_drift_score": 0.0,
            "collapse_count": 0,
            "most_sensitive_model": None,
            "most_sensitive_perturbation": None,
            "collapse_report": [],
            "severity_curve": [],
        }

    drift_scores = pd.to_numeric(drift_analysis["attention_drift_score"], errors="coerce").dropna()
    collapse_report = compute_attention_collapse_report(drift_analysis)
    severity_curve = compute_attention_stability_curve(drift_analysis)

    most_sensitive_model = None
    if "model_name" in drift_analysis.columns:
        model_means = drift_analysis.groupby("model_name")["attention_drift_score"].mean().sort_values(ascending=False)
        if not model_means.empty:
            most_sensitive_model = str(model_means.index[0])

    most_sensitive_perturbation = None
    if "perturbation_type" in drift_analysis.columns:
        pert_means = drift_analysis.groupby("perturbation_type")["attention_drift_score"].mean().sort_values(ascending=False)
        if not pert_means.empty:
            most_sensitive_perturbation = str(pert_means.index[0])

    return {
        "total_observations": int(len(drift_scores)),
        "mean_attention_drift_score": round(float(drift_scores.mean()), 6),
        "max_attention_drift_score": round(float(drift_scores.max()), 6),
        "collapse_count": int(len(collapse_report)),
        "most_sensitive_model": most_sensitive_model,
        "most_sensitive_perturbation": most_sensitive_perturbation,
        "collapse_report": collapse_report.to_dict(orient="records") if not collapse_report.empty else [],
        "severity_curve": severity_curve.to_dict(orient="records") if not severity_curve.empty else [],
    }


def compute_explainability_relationship_summary(
    drift_analysis: Optional[pd.DataFrame],
    uncertainty_data: Optional[pd.DataFrame] = None,
    disagreement_data: Optional[pd.DataFrame] = None,
    instability_data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Relate attention drift to uncertainty, disagreement, and instability."""
    if drift_analysis is None or drift_analysis.empty:
        return {
            "sample_correlations": {
                "attention_drift_vs_uncertainty": 0.0,
                "attention_drift_vs_disagreement": 0.0,
                "attention_drift_vs_instability": 0.0,
            },
            "high_drift_count": 0,
            "high_drift_sample_ids": [],
            "instability_attention_count": 0,
            "uncertainty_attention_count": 0,
            "summary_line": "Insufficient data for explainability relationship analysis.",
        }

    drift_frame = drift_analysis.copy()
    if "sample_id" not in drift_frame.columns:
        return {
            "sample_correlations": {
                "attention_drift_vs_uncertainty": 0.0,
                "attention_drift_vs_disagreement": 0.0,
                "attention_drift_vs_instability": 0.0,
            },
            "high_drift_count": 0,
            "high_drift_sample_ids": [],
            "instability_attention_count": 0,
            "uncertainty_attention_count": 0,
            "summary_line": "Missing sample identifiers for explainability relationship analysis.",
        }

    def _merge_metric(frame: Optional[pd.DataFrame], candidates: Sequence[str], target: str) -> None:
        nonlocal drift_frame
        if frame is None or frame.empty or "sample_id" not in frame.columns:
            drift_frame[target] = np.nan
            return
        value_column = _first_present_column(frame, candidates)
        if value_column is None:
            drift_frame[target] = np.nan
            return
        rows: List[Dict[str, Any]] = []
        for sample_id, group in frame.groupby("sample_id"):
            values = pd.to_numeric(group[value_column], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({"sample_id": str(sample_id), target: float(values.mean())})
        if not rows:
            drift_frame[target] = np.nan
            return
        drift_frame = drift_frame.merge(pd.DataFrame(rows), on="sample_id", how="left")

    _merge_metric(uncertainty_data, ["mean_uncertainty_score", "uncertainty_score"], "mean_uncertainty_score")
    _merge_metric(disagreement_data, ["disagreement_score", "mean_disagreement_score", "score"], "mean_disagreement_score")
    _merge_metric(instability_data, ["instability_score", "score"], "mean_instability_score")

    correlations = {
        "attention_drift_vs_uncertainty": _safe_corr(
            pd.to_numeric(drift_frame["attention_drift_score"], errors="coerce"),
            pd.to_numeric(drift_frame["mean_uncertainty_score"], errors="coerce"),
        ),
        "attention_drift_vs_disagreement": _safe_corr(
            pd.to_numeric(drift_frame["attention_drift_score"], errors="coerce"),
            pd.to_numeric(drift_frame["mean_disagreement_score"], errors="coerce"),
        ),
        "attention_drift_vs_instability": _safe_corr(
            pd.to_numeric(drift_frame["attention_drift_score"], errors="coerce"),
            pd.to_numeric(drift_frame["mean_instability_score"], errors="coerce"),
        ),
    }

    high_drift = drift_frame[pd.to_numeric(drift_frame["attention_drift_score"], errors="coerce") >= 0.65]
    high_drift_ids = [str(value) for value in high_drift["sample_id"].dropna().astype(str).unique().tolist()]
    instability_attention_count = int((pd.to_numeric(drift_frame.get("mean_instability_score", pd.Series(dtype=float)), errors="coerce") >= 0.60).sum()) if "mean_instability_score" in drift_frame.columns else 0
    uncertainty_attention_count = int((pd.to_numeric(drift_frame.get("mean_uncertainty_score", pd.Series(dtype=float)), errors="coerce") >= 0.60).sum()) if "mean_uncertainty_score" in drift_frame.columns else 0

    summary_line = "Attention drift increases alongside uncertainty and disagreement." if correlations["attention_drift_vs_uncertainty"] >= 0.25 else "Attention drift remains only weakly related to uncertainty in the observed data."
    return {
        "sample_correlations": correlations,
        "high_drift_count": int(len(high_drift)),
        "high_drift_sample_ids": high_drift_ids,
        "instability_attention_count": instability_attention_count,
        "uncertainty_attention_count": uncertainty_attention_count,
        "summary_line": summary_line,
    }


def _build_summary_line(attention_summary: Dict[str, Any], drift_summary: Dict[str, Any], consistency_summary: Dict[str, Any]) -> str:
    if drift_summary.get("collapse_count", 0) > 0:
        return f"{drift_summary.get('collapse_count')} attention collapse events were detected, with {drift_summary.get('most_sensitive_model') or 'the leading model'} showing the highest drift."
    if consistency_summary.get("mean_attention_similarity", 0.0) >= 0.70:
        return "Attention patterns are broadly consistent across models."
    return "Attention patterns vary across perturbations and model pairs, but no dominant collapse signal was found."


# ---------------------------------------------------------------------------
# Visual exports
# ---------------------------------------------------------------------------


def export_attention_heatmap_png(
    gradcam_output: Any,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "attention_heatmap.png",
) -> Optional[Path]:
    """Export a raw Grad-CAM heatmap as a PNG file."""
    heatmap = None
    if isinstance(gradcam_output, dict):
        heatmap = gradcam_output.get("normalized_heatmap")
        if heatmap is None:
            heatmap = gradcam_output.get("raw_heatmap")
    else:
        heatmap = gradcam_output
    if heatmap is None:
        return None
    directory = _ensure_dir(output_dir, _HEATMAP_DIR)
    path = directory / filename
    return save_heatmap_png(heatmap, path)


def export_attention_overlay_png(
    gradcam_output: Any,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "attention_overlay.png",
) -> Optional[Path]:
    """Export an overlay image produced by Grad-CAM."""
    if not isinstance(gradcam_output, dict) or gradcam_output.get("overlay_image") is None:
        return None
    directory = _ensure_dir(output_dir, _OVERLAY_DIR)
    return save_overlay_png(gradcam_output["overlay_image"], directory / filename)


def export_attention_drift_png(
    drift_analysis: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "attention_drift.png",
) -> Path:
    """Export a simple attention-drift bar chart."""
    directory = _ensure_dir(output_dir, _DRIFT_DIR)
    output = directory / filename
    fig, ax = plt.subplots(figsize=(10, 5))
    if drift_analysis is not None and not drift_analysis.empty and "attention_drift_score" in drift_analysis.columns:
        plot_df = drift_analysis.copy()
        plot_df["label"] = plot_df.apply(
            lambda row: f"{row.get('sample_id', '')}::{row.get('model_name', '')}::{row.get('severity_level', '')}",
            axis=1,
        )
        plot_df = plot_df.sort_values("attention_drift_score", ascending=False).head(20)
        ax.barh(plot_df["label"], plot_df["attention_drift_score"], color="#3F51B5")
        ax.invert_yaxis()
        ax.set_xlabel("Attention Drift Score")
        ax.set_title("Attention Drift Ranking")
    else:
        ax.text(0.5, 0.5, "No drift data available", ha="center", va="center")
        ax.axis("off")
    return _save_figure(fig, output)


def export_attention_comparison_png(
    comparison_analysis: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "attention_comparison.png",
) -> Path:
    """Export a comparison chart for cross-model attention divergence."""
    directory = _ensure_dir(output_dir, _COMPARISON_DIR)
    output = directory / filename
    fig, ax = plt.subplots(figsize=(10, 5))
    if comparison_analysis is not None and not comparison_analysis.empty and "attention_divergence" in comparison_analysis.columns:
        ranking = compute_attention_divergence_ranking(comparison_analysis).head(20)
        if not ranking.empty:
            labels = ranking.apply(lambda row: f"{row['model_a']} vs {row['model_b']}", axis=1)
            ax.barh(labels, ranking["mean_attention_divergence"], color="#E91E63")
            ax.invert_yaxis()
            ax.set_xlabel("Mean Attention Divergence")
            ax.set_title("Cross-Model Attention Divergence")
        else:
            ax.text(0.5, 0.5, "No comparison data available", ha="center", va="center")
            ax.axis("off")
    else:
        ax.text(0.5, 0.5, "No comparison data available", ha="center", va="center")
        ax.axis("off")
    return _save_figure(fig, output)


# ---------------------------------------------------------------------------
# Structured exports
# ---------------------------------------------------------------------------


def _export_dataframe_csv(frame: pd.DataFrame, output_dir: Optional[Union[str, Path]], default_dir: Path, filename: str) -> Path:
    directory = _ensure_dir(output_dir, default_dir)
    path = directory / filename
    export_frame = frame.copy() if frame is not None else pd.DataFrame()
    if not export_frame.empty:
        for column in export_frame.columns:
            if export_frame[column].apply(lambda value: isinstance(value, (np.ndarray, list, tuple, dict, Image.Image))).any():
                export_frame[column] = export_frame[column].apply(lambda value: json.dumps(_serialise(value), ensure_ascii=False))
        export_frame.to_csv(path, index=False, float_format="%.6f")
    else:
        export_frame.to_csv(path, index=False)
    return path


def _export_dataframe_json(frame: pd.DataFrame, output_dir: Optional[Union[str, Path]], default_dir: Path, filename: str) -> Path:
    directory = _ensure_dir(output_dir, default_dir)
    path = directory / filename
    records = frame.to_dict(orient="records") if frame is not None else []
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_serialise(records), handle, indent=2, ensure_ascii=False)
    return path


def _export_mapping_json(mapping: Dict[str, Any], output_dir: Optional[Union[str, Path]], default_dir: Path, filename: str) -> Path:
    directory = _ensure_dir(output_dir, default_dir)
    path = directory / filename
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_serialise(mapping), handle, indent=2, ensure_ascii=False)
    return path


def _export_mapping_csv(mapping: Dict[str, Any], output_dir: Optional[Union[str, Path]], default_dir: Path, filename: str) -> Path:
    directory = _ensure_dir(output_dir, default_dir)
    path = directory / filename
    pd.DataFrame(_flatten_mapping(mapping)).to_csv(path, index=False, float_format="%.6f")
    return path


def export_attention_analysis_csv(attention_analysis: pd.DataFrame, output_dir: Optional[Union[str, Path]] = None, filename: str = "attention_analysis.csv") -> Path:
    return _export_dataframe_csv(attention_analysis, output_dir, _HEATMAP_DIR, filename)


def export_attention_analysis_json(attention_analysis: pd.DataFrame, output_dir: Optional[Union[str, Path]] = None, filename: str = "attention_analysis.json") -> Path:
    return _export_dataframe_json(attention_analysis, output_dir, _HEATMAP_DIR, filename)


def export_attention_drift_csv(drift_analysis: pd.DataFrame, output_dir: Optional[Union[str, Path]] = None, filename: str = "attention_drift.csv") -> Path:
    return _export_dataframe_csv(drift_analysis, output_dir, _DRIFT_DIR, filename)


def export_attention_drift_json(drift_analysis: pd.DataFrame, output_dir: Optional[Union[str, Path]] = None, filename: str = "attention_drift.json") -> Path:
    return _export_dataframe_json(drift_analysis, output_dir, _DRIFT_DIR, filename)


def export_attention_comparison_csv(comparison_analysis: pd.DataFrame, output_dir: Optional[Union[str, Path]] = None, filename: str = "attention_comparison.csv") -> Path:
    return _export_dataframe_csv(comparison_analysis, output_dir, _COMPARISON_DIR, filename)


def export_attention_comparison_json(comparison_analysis: pd.DataFrame, output_dir: Optional[Union[str, Path]] = None, filename: str = "attention_comparison.json") -> Path:
    return _export_dataframe_json(comparison_analysis, output_dir, _COMPARISON_DIR, filename)


def export_explainability_summary_json(summary: Dict[str, Any], output_dir: Optional[Union[str, Path]] = None, filename: str = "explainability_summary.json") -> Path:
    return _export_mapping_json(summary, output_dir, _DRIFT_DIR, filename)


def export_explainability_summary_csv(summary: Dict[str, Any], output_dir: Optional[Union[str, Path]] = None, filename: str = "explainability_summary.csv") -> Path:
    return _export_mapping_csv(summary, output_dir, _DRIFT_DIR, filename)


def export_drift_summary_json(summary: Dict[str, Any], output_dir: Optional[Union[str, Path]] = None, filename: str = "drift_summary.json") -> Path:
    return _export_mapping_json(summary, output_dir, _DRIFT_DIR, filename)


def export_consistency_summary_json(summary: Dict[str, Any], output_dir: Optional[Union[str, Path]] = None, filename: str = "attention_consistency_summary.json") -> Path:
    return _export_mapping_json(summary, output_dir, _COMPARISON_DIR, filename)


def export_relationship_summary_json(summary: Dict[str, Any], output_dir: Optional[Union[str, Path]] = None, filename: str = "explainability_relationship_summary.json") -> Path:
    return _export_mapping_json(summary, output_dir, _COMPARISON_DIR, filename)
