"""
Validation campaign runners for stabilization and recovery checks.
"""

from pathlib import Path
import json
import traceback

import pandas as pd


def _safe_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def run_roundtrip_validation(paths):
    rows = []
    for path in paths or []:
        file_path = Path(path)
        original = _safe_json(file_path)
        if original is None:
            rows.append({"path": str(file_path), "roundtrip_ok": False, "reason": "unreadable json"})
            continue
        temp_path = file_path.with_suffix(file_path.suffix + ".roundtrip.tmp.json")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(original, handle, indent=2, ensure_ascii=False)
            reread = _safe_json(temp_path)
            rows.append({"path": str(file_path), "roundtrip_ok": reread == original, "reason": None})
        except Exception as exc:
            rows.append({"path": str(file_path), "roundtrip_ok": False, "reason": f"{type(exc).__name__}: {exc}"})
        finally:
            if temp_path.exists():
                temp_path.unlink()
    frame = pd.DataFrame(rows)
    return {
        "frame": frame,
        "summary": {
            "total_files": int(len(frame)),
            "roundtrip_passed": int(frame["roundtrip_ok"].sum()) if not frame.empty else 0,
            "roundtrip_failed": int((~frame["roundtrip_ok"]).sum()) if not frame.empty else 0,
        },
    }


def run_fallback_validation(primary_payload, fallback_loader):
    try:
        payload = primary_payload
        if payload is None and callable(fallback_loader):
            payload = fallback_loader()
        if payload is None:
            return {"fallback_ok": False, "reason": "no payload available"}
        return {"fallback_ok": True, "reason": None, "payload_type": type(payload).__name__}
    except Exception as exc:
        return {"fallback_ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def run_partial_artifact_recovery(paths):
    rows = []
    for path in paths or []:
        file_path = Path(path)
        if file_path.exists():
            rows.append({"path": str(file_path), "recoverable": True, "reason": None})
        else:
            rows.append({"path": str(file_path), "recoverable": False, "reason": "missing file"})
    frame = pd.DataFrame(rows)
    return {
        "frame": frame,
        "summary": {
            "total_paths": int(len(frame)),
            "recoverable_count": int(frame["recoverable"].sum()) if not frame.empty else 0,
            "missing_count": int((~frame["recoverable"]).sum()) if not frame.empty else 0,
        },
    }


def run_validation_campaigns(paths=None, primary_payload=None, fallback_loader=None):
    paths = paths or []
    roundtrip = run_roundtrip_validation(paths)
    fallback = run_fallback_validation(primary_payload, fallback_loader)
    recovery = run_partial_artifact_recovery(paths)
    campaign_frame = pd.DataFrame([
        {"campaign": "roundtrip_validation", **roundtrip.get("summary", {})},
        {"campaign": "fallback_validation", **fallback},
        {"campaign": "artifact_recovery", **recovery.get("summary", {})},
    ])
    score = 100.0
    score -= roundtrip.get("summary", {}).get("roundtrip_failed", 0) * 10.0
    score -= 20.0 if not fallback.get("fallback_ok") else 0.0
    score -= recovery.get("summary", {}).get("missing_count", 0) * 5.0
    score = max(0.0, round(float(score), 6))
    return {
        "roundtrip": roundtrip,
        "fallback": fallback,
        "recovery": recovery,
        "frame": campaign_frame,
        "summary": {
            "validation_campaign_score": score,
            "roundtrip_passed": roundtrip.get("summary", {}).get("roundtrip_passed", 0),
            "fallback_ok": fallback.get("fallback_ok", False),
            "recoverable_count": recovery.get("summary", {}).get("recoverable_count", 0),
        },
    }


def export_campaign_json(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    def _to_json_ready(obj):
        if isinstance(obj, dict):
            return {k: _to_json_ready(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_json_ready(v) for v in obj]
        try:
            import pandas as _pd

            if isinstance(obj, _pd.DataFrame):
                return obj.to_dict(orient="records")
        except Exception:
            pass
        return obj

    serial = _to_json_ready(result)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(serial, handle, indent=2, ensure_ascii=False)
    return output_path


def export_campaign_csv(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.get("frame", pd.DataFrame()).to_csv(output_path, index=False, float_format="%.6f")
    return output_path


def export_campaign_txt(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = result.get("summary", {})
    lines = [
        f"validation_campaign_score: {summary.get('validation_campaign_score', 0.0)}",
        f"roundtrip_passed: {summary.get('roundtrip_passed', 0)}",
        f"fallback_ok: {summary.get('fallback_ok', False)}",
        f"recoverable_count: {summary.get('recoverable_count', 0)}",
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return output_path
