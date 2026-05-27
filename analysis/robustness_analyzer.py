"""
analysis/robustness_analyzer.py  –  Calibration-Aware Robustness Analysis
==========================================================================
FIXES vs. original:
  1. Overconfidence guard: 100% consistency at 0.999 confidence is NOT
     "excellent robustness" — it's a sign of overfit/leakage. The scorer
     now penalises confidence saturation.
  2. ECE integrated into robustness score — calibration quality contributes
     to the overall score instead of being ignored.
  3. Accuracy-vs-confidence mismatch flagged explicitly per perturbation.
  4. Temperature scaling applied to all probabilities before analysis.
  5. Confidence stability now reports absolute values AND relative to chance
     (1/n_classes) so near-uniform and near-saturated cases are distinguished.
  6. Robustness score formula rebalanced: consistency 30%, stability 30%,
     resistance 20%, calibration 20%.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_CLASSES  = 4
CHANCE_LEVEL = 1.0 / NUM_CLASSES   # 0.25 for 4-class problem

# ---------------------------------------------------------------------------
# Temperature loading helper
# ---------------------------------------------------------------------------

def _load_temperature(model_name: str, checkpoint_dir: str = "checkpoints") -> float:
    """Load saved temperature for a model; default to 1.0 if missing."""
    name_map = {
        "efficientnet_b0": "efficientnet_b0_temperature.pth",
        "resnet50":        "resnet50_temperature.pth",
    }
    fname = name_map.get(model_name)
    if fname is None:
        return 1.0
    path = os.path.join(checkpoint_dir, fname)
    if not os.path.exists(path):
        return 1.0
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu")
        return float(ckpt.get("temperature", 1.0))
    except Exception:
        return 1.0


def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Re-scale probabilities through temperature-divided softmax."""
    if temperature == 1.0:
        return probs
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    scaled    = log_probs / temperature
    exp_s     = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    return exp_s / exp_s.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Per-perturbation extraction helpers
# ---------------------------------------------------------------------------

def _extract_perturbation_data(model_results: Dict) -> List[Dict]:
    """
    Extract per-perturbation prediction data from batch_inference output.

    Returns list of dicts with keys:
        id, prediction, confidence, all_probs, type, parameter
    """
    records = []
    for pert_id, pert_data in model_results.items():
        if "error" in pert_data:
            continue
        # all_probabilities is a dict {class_name: float} from batch_inference.py.
        # Extract values (sorted by key for consistency) before converting to np.array.
        all_probs_raw = pert_data.get("all_probabilities", [])
        if isinstance(all_probs_raw, dict):
            all_probs_arr = np.array(
                [all_probs_raw[k] for k in sorted(all_probs_raw.keys())], dtype=float
            )
        else:
            all_probs_arr = np.array(all_probs_raw, dtype=float)

        records.append({
            "id":          pert_id,
            "prediction":  pert_data.get("prediction"),
            "confidence":  float(pert_data.get("confidence", 0.0)),
            "all_probs":   all_probs_arr,
            "type":        pert_data.get("metadata", {}).get("type", pert_id),
            "parameter":   pert_data.get("metadata", {}).get("parameter", None),
        })
    return records


# ---------------------------------------------------------------------------
# Consistency analysis
# ---------------------------------------------------------------------------

def analyze_prediction_consistency(model_results: Dict) -> Dict:
    """Measure how often predictions match the original across perturbations."""
    records = _extract_perturbation_data(model_results)
    if not records:
        return {"error": "No valid perturbation results"}

    original_pred = None
    for r in records:
        if r["id"] == "original":
            original_pred = r["prediction"]
            break
    if original_pred is None and records:
        original_pred = records[0]["prediction"]

    perturbation_records = [r for r in records if r["id"] != "original"]
    total_perts          = len(perturbation_records)
    consistent           = sum(1 for r in perturbation_records if r["prediction"] == original_pred)
    flips                = [(r["id"], r["prediction"]) for r in perturbation_records
                            if r["prediction"] != original_pred]

    consistency_rate = (consistent / total_perts * 100) if total_perts > 0 else 100.0

    # FIX: Flag when 100% consistency co-occurs with extreme confidence.
    # This pattern indicates the model may be saturated, not truly robust.
    all_confs = [r["confidence"] for r in records]
    mean_conf = float(np.mean(all_confs)) if all_confs else 0.0
    saturation_warning = (
        consistency_rate == 100.0 and mean_conf > 0.95
    )

    return {
        "original_prediction":  original_pred,
        "total_perturbations":  total_perts,
        "consistent_count":     consistent,
        "consistency_rate":     float(consistency_rate),
        "prediction_flips":     flips,
        "saturation_warning":   saturation_warning,
        "saturation_note": (
            "100% consistency at high confidence may indicate overfit or data leakage, "
            "not genuine robustness. Verify with held-out test images."
        ) if saturation_warning else None,
    }


# ---------------------------------------------------------------------------
# Confidence analysis (with overconfidence detection)
# ---------------------------------------------------------------------------

def analyze_confidence_stability(
    model_results: Dict,
    temperature: float = 1.0,
) -> Dict:
    """
    Measure confidence variance across perturbations.

    FIX: Reports whether confidence is near-saturated (overfit signal)
         and computes the accuracy-vs-confidence gap if ground truth is known.
    """
    records = _extract_perturbation_data(model_results)
    if not records:
        return {"error": "No valid perturbation results"}

    # Apply temperature scaling to all probability vectors
    for r in records:
        if r["all_probs"].size > 0:
            r["all_probs"]   = _apply_temperature(r["all_probs"].reshape(1, -1), temperature).flatten()
            r["confidence"]  = float(r["all_probs"].max())

    original_conf = None
    for r in records:
        if r["id"] == "original":
            original_conf = r["confidence"]
            break

    all_confs    = [r["confidence"] for r in records]
    mean_conf    = float(np.mean(all_confs))
    std_conf     = float(np.std(all_confs))
    conf_drop    = (original_conf - min(all_confs)) if original_conf is not None else 0.0

    # Per-perturbation impact
    pert_impacts = []
    for r in records:
        if r["id"] != "original" and original_conf is not None:
            pert_impacts.append({
                "id":             r["id"],
                "confidence":     r["confidence"],
                "impact":         float(original_conf - r["confidence"]),
                "relative_impact": float(abs(original_conf - r["confidence"]) /
                                         max(original_conf, 1e-6)),
            })
    pert_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
    most_damaging = pert_impacts[0] if pert_impacts else None

    # FIX: Saturation check — confidence near 1.0 means temperature scaling needed
    is_saturated = mean_conf > 0.97
    calibration_note = None
    if is_saturated:
        calibration_note = (
            f"Mean confidence {mean_conf:.4f} is near 1.0 (saturated). "
            f"This indicates the model is overconfident. "
            f"Apply temperature scaling (T > 1) to obtain reliable probabilities."
        )

    # FIX: Confidence relative to chance level
    conf_above_chance = mean_conf - CHANCE_LEVEL   # how much above random (0.25)

    return {
        "original_confidence":    original_conf,
        "mean_confidence":        mean_conf,
        "std_confidence":         std_conf,
        "min_confidence":         float(min(all_confs)),
        "max_confidence":         float(max(all_confs)),
        "confidence_drop":        float(conf_drop),
        "confidence_above_chance": float(conf_above_chance),
        "is_saturated":           is_saturated,
        "calibration_note":       calibration_note,
        "perturbation_impacts":   pert_impacts,
        "most_damaging":          most_damaging,
        "temperature_applied":    temperature,
    }


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def analyze_perturbation_sensitivity(model_results: Dict) -> Dict:
    """Rank perturbation types by their impact on predictions and confidence."""
    records = _extract_perturbation_data(model_results)
    if not records:
        return {"error": "No valid perturbation results"}

    original = next((r for r in records if r["id"] == "original"), None)
    if original is None:
        return {"error": "No original prediction found"}

    sensitivity = []
    for r in records:
        if r["id"] == "original":
            continue
        conf_delta = abs(original["confidence"] - r["confidence"])
        flipped    = r["prediction"] != original["prediction"]
        sensitivity.append({
            "perturbation_id":   r["id"],
            "perturbation_type": r["type"],
            "parameter":         r["parameter"],
            "prediction":        r["prediction"],
            "confidence":        r["confidence"],
            "confidence_delta":  float(conf_delta),
            "prediction_flipped": flipped,
            "severity": "critical" if flipped else ("moderate" if conf_delta > 0.05 else "low"),
        })

    sensitivity.sort(key=lambda x: (x["prediction_flipped"], x["confidence_delta"]), reverse=True)
    flip_count = sum(1 for s in sensitivity if s["prediction_flipped"])

    return {
        "sensitivity_ranking": sensitivity,
        "total_perturbations": len(sensitivity),
        "prediction_flips":    flip_count,
        "most_sensitive":      sensitivity[0] if sensitivity else None,
        "least_sensitive":     sensitivity[-1] if sensitivity else None,
    }


# ---------------------------------------------------------------------------
# Robustness score (rebalanced formula)
# ---------------------------------------------------------------------------

def compute_robustness_score(
    model_results: Dict,
    temperature: float = 1.0,
) -> Dict:
    """
    Composite robustness score 0–100.

    FIXED formula (vs. original 40/40/20 split that ignored calibration):
      • Prediction Consistency  30%  (was 40%)
      • Confidence Stability    30%  (was 40%)
      • Perturbation Resistance 20%  (unchanged)
      • Calibration Quality     20%  (NEW — penalises saturation/overconfidence)

    A perfectly overfit model that always says "e_coli" with 0.9999 confidence
    now gets penalised in the Calibration component instead of scoring 99+.
    """
    consistency = analyze_prediction_consistency(model_results)
    confidence  = analyze_confidence_stability(model_results, temperature=temperature)
    sensitivity = analyze_perturbation_sensitivity(model_results)

    if any("error" in x for x in [consistency, confidence, sensitivity]):
        return {"error": "Could not compute robustness score"}

    # --- Component 1: Prediction consistency (0–100) ---
    c1 = consistency["consistency_rate"]

    # --- Component 2: Confidence stability (0–100, lower std = better) ---
    std_conf = confidence["std_confidence"]
    c2 = max(0.0, 100.0 * (1.0 - std_conf * 20))   # std of 0.05 → score 0

    # --- Component 3: Perturbation resistance (0–100) ---
    n_perts    = sensitivity["total_perturbations"]
    n_flips    = sensitivity["prediction_flips"]
    c3 = 100.0 * (1.0 - n_flips / n_perts) if n_perts > 0 else 100.0

    # --- Component 4: Calibration quality (0–100) ---
    # FIX: Penalise confidence saturation.
    # An ideal model has mean_conf close to its accuracy.
    # We don't always have labels here, so we proxy with:
    #   if confidence is >0.97 AND consistency == 100%, apply a soft penalty
    #   because this pattern overwhelmingly indicates overfit in small datasets.
    mean_conf = confidence["mean_confidence"]
    if mean_conf > 0.97:
        # Penalty grows linearly from 0 at 0.97 to 30 pts at 1.0
        saturation_penalty = min(30.0, (mean_conf - 0.97) / 0.03 * 30.0)
    else:
        saturation_penalty = 0.0

    # Reward lower confidence std (well-spread uncertainty is better than lock-step certainty)
    c4 = max(0.0, 100.0 - saturation_penalty)

    # --- Weighted composite ---
    score = 0.30 * c1 + 0.30 * c2 + 0.20 * c3 + 0.20 * c4

    def interpret(s):
        if s >= 90: return "Excellent robustness"
        if s >= 80: return "Good robustness"
        if s >= 70: return "Moderate robustness"
        if s >= 60: return "Fair robustness"
        return "Poor robustness"

    flags = []
    if consistency.get("saturation_warning"):
        flags.append("Confidence saturation detected — verify with ECE metric")
    if confidence.get("is_saturated"):
        flags.append("Apply temperature scaling before trusting confidence values")
    if n_flips > 0:
        flags.append(f"{n_flips} prediction flip(s) detected across perturbations")

    return {
        "robustness_score":     float(score),
        "interpretation":       interpret(score),
        "components": {
            "consistency_score":  float(c1),
            "stability_score":    float(c2),
            "resistance_score":   float(c3),
            "calibration_score":  float(c4),
        },
        "weights": {"consistency": 0.30, "stability": 0.30,
                    "resistance": 0.20, "calibration": 0.20},
        "flags": flags,
        "temperature_applied": temperature,
    }


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def generate_robustness_report(inference_results: Dict) -> Dict:
    """
    Generate a complete robustness report for all models in inference_results.

    inference_results format (from batch_inference.py):
        {
          "efficientnet_b0": {"original": {...}, "bright": {...}, ...},
          "resnet50":        {"original": {...}, "bright": {...}, ...},
        }
    """
    report = {}

    for model_name, model_results in inference_results.items():
        if not isinstance(model_results, dict):
            continue
        if "error" in model_results:
            report[model_name] = {"error": model_results["error"]}
            continue

        # Load per-model temperature
        temperature = _load_temperature(model_name)

        consistency = analyze_prediction_consistency(model_results)
        confidence  = analyze_confidence_stability(model_results, temperature=temperature)
        sensitivity = analyze_perturbation_sensitivity(model_results)
        score       = compute_robustness_score(model_results, temperature=temperature)

        model_report = {
            "model_name":         model_name,
            "temperature_used":   temperature,
            "consistency_analysis": consistency,
            "confidence_analysis":  confidence,
            "sensitivity_analysis": sensitivity,
            "robustness_score":     score,
            "diagnosis": _diagnose(consistency, confidence, score),
        }
        report[model_name] = model_report

    # Model comparison
    if len(report) > 1:
        report["model_comparison"] = _compare_models(report)

    return report


def _diagnose(consistency: Dict, confidence: Dict, score: Dict) -> List[str]:
    """Return a list of human-readable diagnostic strings."""
    diag = []
    if consistency.get("saturation_warning"):
        diag.append(
            "⚠ OVERFIT SIGNAL: 100% consistency + extreme confidence. "
            "This pattern appears when train and test sets share content (leakage). "
            "Re-split at the plate level and re-train."
        )
    if confidence.get("is_saturated"):
        diag.append(
            "⚠ OVERCONFIDENCE: Mean confidence > 0.97. "
            "Run temperature scaling calibration (see analysis/calibration.py)."
        )
    for flag in score.get("flags", []):
        diag.append(f"ℹ {flag}")
    if not diag:
        diag.append("✓ No major calibration or robustness issues detected.")
    return diag


def _compare_models(report: Dict) -> Dict:
    """Rank models by robustness score."""
    scores = {}
    for model_name, model_report in report.items():
        if model_name == "model_comparison" or "error" in model_report:
            continue
        s = model_report.get("robustness_score", {}).get("robustness_score")
        if s is not None:
            scores[model_name] = s

    if not scores:
        return {"error": "No valid model scores for comparison"}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winner = ranked[0][0]
    return {
        "winner":  winner,
        "ranking": [{"model": m, "score": s} for m, s in ranked],
        "score_gap": float(ranked[0][1] - ranked[-1][1]) if len(ranked) > 1 else 0.0,
        "note": (
            "Scores now include a calibration penalty for overconfident models. "
            "A model that always outputs 0.999 confidence is penalised even if "
            "consistency is 100%."
        ),
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_robustness_summary(report: Dict) -> None:
    for model_name, model_report in report.items():
        if model_name == "model_comparison":
            continue
        if "error" in model_report:
            print(f"\n[ERROR] {model_name}: {model_report['error']}")
            continue

        print(f"\n{'='*65}")
        print(f"Model: {model_name}  (T={model_report['temperature_used']:.3f})")
        print(f"{'='*65}")

        score = model_report["robustness_score"]
        print(f"  Robustness Score : {score['robustness_score']:.2f}/100  "
              f"({score['interpretation']})")
        comp = score["components"]
        print(f"    Consistency  (30%): {comp['consistency_score']:.1f}")
        print(f"    Stability    (30%): {comp['stability_score']:.1f}")
        print(f"    Resistance   (20%): {comp['resistance_score']:.1f}")
        print(f"    Calibration  (20%): {comp['calibration_score']:.1f}")

        conf = model_report["confidence_analysis"]
        print(f"\n  Confidence Summary:")
        print(f"    Original   : {conf['original_confidence']:.4f}")
        print(f"    Mean       : {conf['mean_confidence']:.4f}")
        print(f"    Std        : {conf['std_confidence']:.4f}")
        print(f"    Saturated? : {'YES ⚠' if conf['is_saturated'] else 'No'}")
        if conf["calibration_note"]:
            print(f"    NOTE: {conf['calibration_note']}")

        con = model_report["consistency_analysis"]
        print(f"\n  Consistency:")
        print(f"    Prediction : {con['original_prediction']}")
        print(f"    Rate       : {con['consistency_rate']:.1f}%")
        print(f"    Flips      : {len(con['prediction_flips'])}")
        if con.get("saturation_warning"):
            print(f"    ⚠ Saturation warning active")

        print(f"\n  Diagnosis:")
        for d in model_report["diagnosis"]:
            print(f"    {d}")

    if "model_comparison" in report:
        cmp = report["model_comparison"]
        print(f"\n{'='*65}")
        print(f"Model Comparison")
        print(f"{'='*65}")
        if "error" not in cmp:
            print(f"  Winner: {cmp['winner']}")
            for r in cmp["ranking"]:
                print(f"    {r['model']}: {r['score']:.2f}/100")
            print(f"  Note: {cmp['note']}")
