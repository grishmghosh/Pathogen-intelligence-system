"""
analysis/calibration.py  –  Confidence Calibration & Reliability Analysis
==========================================================================
This module was missing entirely from the original system.

Problems it fixes:
  • "Confidence reliability / overconfidence" — models output 0.999+ always
  • "Calibration quality" — no ECE or reliability diagram existed
  • "Accuracy-vs-confidence mismatch" — high confidence ≠ high accuracy
  • "Confidence stability" treated as robustness — it isn't: a broken model
    confidently wrong on every perturbation also scores 100% consistency

Provides:
  - expected_calibration_error()   ECE metric (lower = better calibrated)
  - overconfidence_ratio()         fraction of predictions where conf > acc
  - reliability_diagram_data()     bins for plotting calibration curve
  - calibration_summary()          human-readable report dict
  - apply_temperature_scaling()    post-hoc calibration correction
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Core calibration metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Dict:
    """
    Expected Calibration Error (ECE).

    Args:
        probs:   (N, C) softmax probabilities from the model.
        labels:  (N,) integer ground-truth class indices.
        n_bins:  number of confidence bins.

    Returns:
        dict with 'ece', 'mce', 'bins' keys.
    """
    confidences = probs.max(axis=1)          # shape (N,)
    predictions = probs.argmax(axis=1)       # shape (N,)
    accuracies  = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece  = 0.0
    mce  = 0.0   # maximum calibration error

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc  = accuracies[mask].mean()
        bin_frac = mask.sum() / len(labels)
        gap      = abs(bin_conf - bin_acc)

        ece += bin_frac * gap
        mce  = max(mce, gap)
        bins.append({
            "lo": lo, "hi": hi,
            "mean_conf": float(bin_conf),
            "mean_acc":  float(bin_acc),
            "count":     int(mask.sum()),
            "gap":       float(gap),
        })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bins": bins,
        "interpretation": _ece_interpretation(ece),
    }


def _ece_interpretation(ece: float) -> str:
    if ece < 0.02:
        return "Well-calibrated"
    if ece < 0.05:
        return "Acceptable calibration"
    if ece < 0.10:
        return "Moderate overconfidence — consider temperature scaling"
    return "Severe overconfidence — model predictions are unreliable"


def overconfidence_ratio(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.90,
) -> Dict:
    """
    Fraction of samples where the model is highly confident but wrong.

    Returns:
        dict with 'high_conf_wrong_frac', 'mean_conf_wrong', etc.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels)

    high_conf   = confidences >= threshold
    high_conf_wrong = high_conf & ~correct

    return {
        "threshold": threshold,
        "mean_confidence":        float(confidences.mean()),
        "mean_confidence_correct":   float(confidences[correct].mean()) if correct.sum() > 0 else None,
        "mean_confidence_wrong":     float(confidences[~correct].mean()) if (~correct).sum() > 0 else None,
        "high_conf_fraction":        float(high_conf.sum() / len(labels)),
        "high_conf_wrong_fraction":  float(high_conf_wrong.sum() / len(labels)),
        "overconfident_count":       int(high_conf_wrong.sum()),
        "total_samples":             int(len(labels)),
    }


def reliability_diagram_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict:
    """
    Data for a reliability (calibration) diagram.

    A perfectly calibrated model has mean_conf == mean_acc in every bin.
    Points below the diagonal = overconfident.
    Points above = underconfident.
    """
    ece_result = expected_calibration_error(probs, labels, n_bins=n_bins)
    bin_confs  = [b["mean_conf"] for b in ece_result["bins"]]
    bin_accs   = [b["mean_acc"]  for b in ece_result["bins"]]

    return {
        "bin_confidences": bin_confs,
        "bin_accuracies":  bin_accs,
        "perfect_line":    bin_confs,      # y=x reference
        "ece":             ece_result["ece"],
    }


# ---------------------------------------------------------------------------
# Temperature scaling (inference-time fix)
# ---------------------------------------------------------------------------

def apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Divide logits by temperature before softmax.
    Temperature > 1 → less confident (moves distribution toward uniform).
    Temperature < 1 → more confident.

    Load the saved temperature from checkpoints/efficientnet_b0_temperature.pth
    and apply here at inference time.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}")
    scaled = logits / temperature
    exp    = np.exp(scaled - scaled.max(axis=1, keepdims=True))   # numerical stability
    return exp / exp.sum(axis=1, keepdims=True)


def load_temperature(checkpoint_path: str) -> float:
    """Load the calibration temperature from a .pth file."""
    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        return float(ckpt["temperature"])
    except Exception as e:
        print(f"[WARN] Could not load temperature from {checkpoint_path}: {e}")
        return 1.0   # no-op fallback


# ---------------------------------------------------------------------------
# High-level summary
# ---------------------------------------------------------------------------

def calibration_summary(
    probs: np.ndarray,
    labels: np.ndarray,
    model_name: str = "model",
    temperature: Optional[float] = None,
) -> Dict:
    """
    Full calibration report for one model.

    Args:
        probs:        (N, C) softmax probabilities.
        labels:       (N,) ground-truth class indices.
        model_name:   display name.
        temperature:  if provided, also computes post-calibration metrics.
    """
    raw_ece   = expected_calibration_error(probs, labels)
    overconf  = overconfidence_ratio(probs, labels)
    rel_diag  = reliability_diagram_data(probs, labels)

    result = {
        "model":               model_name,
        "n_samples":           len(labels),
        "overall_accuracy":    float((probs.argmax(1) == labels).mean()),
        "mean_confidence":     float(probs.max(1).mean()),
        "ece":                 raw_ece,
        "overconfidence":      overconf,
        "reliability_diagram": rel_diag,
        "flags": [],
    }

    # --- Diagnostic flags ---
    acc  = result["overall_accuracy"]
    conf = result["mean_confidence"]

    if conf - acc > 0.10:
        result["flags"].append(
            f"OVERCONFIDENCE: mean confidence ({conf:.3f}) exceeds accuracy "
            f"({acc:.3f}) by {conf - acc:.3f}. Apply temperature scaling."
        )
    if raw_ece["ece"] > 0.05:
        result["flags"].append(
            f"POOR CALIBRATION: ECE = {raw_ece['ece']:.4f} "
            f"({raw_ece['interpretation']})"
        )
    if conf > 0.97 and acc < 0.95:
        result["flags"].append(
            "POSSIBLE OVERFIT/LEAKAGE: Model is extremely confident "
            "but accuracy doesn't justify it. Check train/val/test split integrity."
        )

    # Post-calibration (temperature scaled)
    if temperature is not None and temperature != 1.0:
        cal_probs = apply_temperature_scaling(
            np.log(probs + 1e-12),   # convert back to logits approximately
            temperature,
        )
        cal_ece = expected_calibration_error(cal_probs, labels)
        result["calibrated_ece"] = cal_ece
        result["temperature_used"] = temperature
        improvement = raw_ece["ece"] - cal_ece["ece"]
        result["ece_improvement"] = float(improvement)
        if improvement > 0:
            result["flags"].append(
                f"Temperature scaling improved ECE by {improvement:.4f} "
                f"(T={temperature:.3f})"
            )

    return result


def print_calibration_report(report: Dict) -> None:
    """Pretty-print the calibration summary."""
    m = report["model"]
    n = report["n_samples"]
    print(f"\n{'='*60}")
    print(f"Calibration Report: {m}  ({n} samples)")
    print(f"{'='*60}")
    print(f"  Accuracy:         {report['overall_accuracy']:.4f}")
    print(f"  Mean Confidence:  {report['mean_confidence']:.4f}")
    ece = report["ece"]
    print(f"  ECE:              {ece['ece']:.4f}  ({ece['interpretation']})")
    print(f"  MCE:              {ece['mce']:.4f}")

    oc = report["overconfidence"]
    print(f"  High-conf wrong:  {oc['overconfident_count']} / {oc['total_samples']} "
          f"({oc['high_conf_wrong_fraction']*100:.1f}%)")

    if "calibrated_ece" in report:
        print(f"\n  After Temperature Scaling (T={report['temperature_used']:.3f}):")
        print(f"    ECE:            {report['calibrated_ece']['ece']:.4f}")
        print(f"    ECE improvement:{report['ece_improvement']:+.4f}")

    if report["flags"]:
        print(f"\n  ⚠ Flags:")
        for flag in report["flags"]:
            print(f"    • {flag}")
    else:
        print("\n  ✓ No calibration issues detected.")
    print(f"{'='*60}")


def fit_vector_temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 4,
) -> np.ndarray:
    """
    Fits class-wise vector temperature parameters T = [T_1, T_2, ..., T_K]
    using L-BFGS optimization on validation logits without modifying model weights.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    # Initialize per-class log-temperatures (log(1.0) = 0)
    log_temps = nn.Parameter(torch.zeros(num_classes, dtype=torch.float32))
    optimizer = optim.LBFGS([log_temps], lr=0.01, max_iter=50)

    criterion = nn.CrossEntropyLoss()

    def eval_loss():
        optimizer.zero_grad()
        temps = torch.exp(log_temps)
        scaled_logits = logits_t / temps
        loss = criterion(scaled_logits, labels_t)
        loss.backward()
        return loss

    optimizer.step(eval_loss)

    best_temps = torch.exp(log_temps).detach().cpu().numpy()
    return best_temps

