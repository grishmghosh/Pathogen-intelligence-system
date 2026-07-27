"""
Figure packaging utilities for publication-ready report bundles.
"""

from pathlib import Path
import shutil
import json

import pandas as pd


FIGURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".pdf"}


def _iter_figure_files(source_dir):
    source_dir = Path(source_dir)
    if not source_dir.exists():
        return []
    figure_files = []
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in FIGURE_EXTENSIONS:
            figure_files.append(path)
    return figure_files


def package_figures(source_dirs, destination_dir, include_patterns=None):
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    include_patterns = include_patterns or []
    grouped = {}
    copied = []

    for source_dir in source_dirs or []:
        source_path = Path(source_dir)
        if not source_path.exists():
            continue
        for figure_path in _iter_figure_files(source_path):
            relative = figure_path.name
            if include_patterns and not any(pattern in str(figure_path) for pattern in include_patterns):
                continue
            target = destination_dir / source_path.name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(figure_path, target)
            copied.append(target)
            grouped.setdefault(source_path.name, []).append({
                "source": str(figure_path),
                "copied_to": str(target),
                "figure_name": figure_path.stem,
                "extension": figure_path.suffix.lower(),
                "size_bytes": figure_path.stat().st_size,
            })

    return {
        "destination_dir": destination_dir,
        "copied_files": copied,
        "figure_groups": grouped,
        "figure_count": int(len(copied)),
    }


def build_figure_manifest(package_result):
    package_result = package_result or {}
    groups = package_result.get("figure_groups", {}) if isinstance(package_result, dict) else {}
    rows = []
    for group_name, entries in groups.items():
        for entry in entries or []:
            row = {"group": group_name}
            row.update(entry)
            rows.append(row)
    return pd.DataFrame(rows)


def build_figure_bundle(package_result):
    package_result = package_result or {}
    manifest = build_figure_manifest(package_result)
    summary = {
        "figure_count": int(package_result.get("figure_count", 0)) if isinstance(package_result, dict) else 0,
        "groups": sorted(list((package_result.get("figure_groups", {}) or {}).keys())) if isinstance(package_result, dict) else [],
        "destination_dir": str(package_result.get("destination_dir")) if isinstance(package_result, dict) else None,
    }
    return {"manifest": manifest, "summary": summary}


def export_figure_manifest_json(package_result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_figure_manifest(package_result)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(orient="records"), handle, indent=2, ensure_ascii=False)
    return output_path


def export_figure_manifest_csv(package_result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_figure_manifest(package_result)
    manifest.to_csv(output_path, index=False, float_format="%.6f")
    return output_path
