"""
Artifact integrity checks for files and report bundles.
"""

from pathlib import Path
import json

import pandas as pd


def _check_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            json.load(handle)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _check_csv(path):
    try:
        pd.read_csv(path)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _check_png(path):
    try:
        file_path = Path(path)
        if not file_path.exists():
            return False, "missing file"
        data = file_path.read_bytes()
        if len(data) < 8:
            return False, "file too small"
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            return False, "invalid PNG signature"
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _check_text(path):
    try:
        Path(path).read_text(encoding="utf-8")
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


CHECKERS = {
    ".json": _check_json,
    ".csv": _check_csv,
    ".png": _check_png,
    ".jpg": _check_png,
    ".jpeg": _check_png,
    ".txt": _check_text,
    ".md": _check_text,
}


def check_artifact(path):
    path = Path(path)
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "integrity": False,
            "error": "missing file",
            "size_bytes": 0,
        }
    checker = CHECKERS.get(path.suffix.lower())
    if checker is None:
        return {
            "path": str(path),
            "exists": True,
            "integrity": True,
            "error": None,
            "size_bytes": path.stat().st_size,
        }
    integrity, error = checker(path)
    return {
        "path": str(path),
        "exists": True,
        "integrity": bool(integrity),
        "error": error,
        "size_bytes": path.stat().st_size,
    }


def check_report_bundle(bundle_root):
    bundle_root = Path(bundle_root)
    expected_subdirs = ["tables", "figures", "narratives", "manifests", "provenance", "summaries"]
    checks = []
    for name in expected_subdirs:
        path = bundle_root / name
        checks.append({
            "path": str(path),
            "exists": path.exists(),
            "is_directory": path.is_dir(),
        })
    required_files = [
        bundle_root / "manifests" / "report_manifest.json",
        bundle_root / "manifests" / "reproducibility_report.json",
        bundle_root / "provenance" / "provenance_manifest.json",
        bundle_root / "summaries" / "report_summary.json",
        bundle_root / "summaries" / "report_summary.csv",
    ]
    file_checks = [check_artifact(path) for path in required_files]
    completeness = sum(1 for check in file_checks if check.get("integrity")) / float(len(file_checks)) if file_checks else 0.0
    return {
        "bundle_root": str(bundle_root),
        "directory_checks": checks,
        "file_checks": file_checks,
        "bundle_completeness": round(float(completeness), 6),
    }


def check_artifacts(paths):
    results = [check_artifact(path) for path in paths or []]
    summary = {
        "total_artifacts": int(len(results)),
        "integrity_passed": int(sum(1 for item in results if item.get("integrity"))),
        "integrity_failed": int(sum(1 for item in results if not item.get("integrity"))),
        "integrity_rate": round(float(sum(1 for item in results if item.get("integrity")) / len(results)), 6) if results else 0.0,
    }
    return {"results": results, "summary": summary}


def export_integrity_json(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return output_path


def export_integrity_csv(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.get("results", [])).to_csv(output_path, index=False, float_format="%.6f")
    return output_path


def export_integrity_txt(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for item in result.get("results", []):
        state = "pass" if item.get("integrity") else "fail"
        lines.append(f"{state}: {item.get('path')} ({item.get('error')})")
    summary = result.get("summary", {})
    lines.append(f"integrity_rate: {summary.get('integrity_rate', 0.0)}")
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return output_path
