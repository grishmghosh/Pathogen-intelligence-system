"""
inference/batch_inference.py  –  Calibrated Batch Inference
============================================================
FIXES vs. original:
  1. Temperature scaling is applied BEFORE returning probabilities.
     Previously raw softmax outputs (0.9999) were returned and treated
     as reliable confidence values — they are not.
  2. All-probabilities dict returned so calibration.py can compute ECE.
  3. Confidence ceiling warning emitted when any result is > 0.99.
  4. s_aureus added to CLASS_NAMES (was silently missing).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = r"C:\Pathogen-intelligence-system\checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["e_coli", "k_pneumoniae", "p_aeruginosa", "s_aureus"]   # FIX: s_aureus added

MODEL_CONFIGS = {
    "efficientnet_b0": {
        "checkpoint": os.path.join(CHECKPOINT_DIR, "efficientnet_b0_best.pth"),
        "temperature_file": os.path.join(CHECKPOINT_DIR, "efficientnet_b0_temperature.pth"),
    },
    "resnet50": {
        "checkpoint": os.path.join(CHECKPOINT_DIR, "resnet50_best.pth"),
        "temperature_file": os.path.join(CHECKPOINT_DIR, "resnet50_temperature.pth"),
    },
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_temperature(temperature_file: str) -> float:
    """Load temperature from checkpoint; default 1.0 (no scaling)."""
    if not os.path.exists(temperature_file):
        print(f"  [WARN] No temperature file at {temperature_file}. "
              f"Using T=1.0 (uncalibrated). Run training first.")
        return 1.0
    try:
        ckpt = torch.load(temperature_file, map_location="cpu")
        t    = float(ckpt.get("temperature", 1.0))
        print(f"  [INFO] Temperature loaded: {t:.4f}")
        return t
    except Exception as e:
        print(f"  [WARN] Could not load temperature: {e}. Using T=1.0.")
        return 1.0


def load_model(model_name: str) -> Tuple[Optional[nn.Module], Optional[float]]:
    """
    Load a trained model and its calibration temperature.

    Returns:
        (model, temperature) or (None, None) on failure.
    """
    cfg = MODEL_CONFIGS.get(model_name)
    if cfg is None:
        print(f"[ERROR] Unknown model: {model_name}")
        return None, None

    if not os.path.exists(cfg["checkpoint"]):
        print(f"[ERROR] Checkpoint not found: {cfg['checkpoint']}")
        return None, None

    try:
        from torchvision import models as tvm

        ckpt = torch.load(cfg["checkpoint"], map_location=DEVICE)
        num_classes = len(CLASS_NAMES)

        if model_name == "efficientnet_b0":
            model = tvm.efficientnet_b0(weights=None)
            in_feat = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_feat, num_classes),
            )
        elif model_name == "resnet50":
            model = tvm.resnet50(weights=None)
            in_feat = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_feat, num_classes),
            )
        else:
            print(f"[ERROR] No architecture defined for {model_name}")
            return None, None

        model.load_state_dict(ckpt["model_state"])
        model.to(DEVICE).eval()

        temperature = _load_temperature(cfg["temperature_file"])
        return model, temperature

    except Exception as e:
        print(f"[ERROR] Failed to load {model_name}: {e}")
        return None, None


def preprocess_image(image_array: np.ndarray) -> torch.Tensor:
    """Convert numpy uint8 image to preprocessed tensor."""
    img = Image.fromarray(image_array.astype(np.uint8))
    return preprocess(img).unsqueeze(0)   # (1, 3, 224, 224)


def predict_single(
    model: nn.Module,
    image_array: np.ndarray,
    temperature: float = 1.0,
) -> Dict:
    """
    Run inference on one image.

    FIX: Applies temperature scaling so returned confidence values are
         calibrated, not the raw overconfident softmax outputs.

    Returns:
        {
          "prediction":        str,
          "confidence":        float  (calibrated),
          "all_probabilities": dict[class_name -> float],
          "raw_confidence":    float  (pre-calibration, for comparison),
          "temperature":       float,
        }
    """
    tensor = preprocess_image(image_array).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)

    # Raw softmax (what the original code returned)
    raw_probs  = torch.softmax(logits, dim=1).cpu().numpy().flatten()

    # FIX: Temperature-scaled softmax (calibrated)
    cal_logits = logits / temperature
    cal_probs  = torch.softmax(cal_logits, dim=1).cpu().numpy().flatten()

    pred_idx   = int(cal_probs.argmax())
    prediction = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"
    confidence = float(cal_probs[pred_idx])
    raw_conf   = float(raw_probs[pred_idx])

    # Warn if still saturated after scaling (indicates very strong overfit)
    if confidence > 0.99:
        print(f"  [WARN] Post-calibration confidence {confidence:.4f} is still > 0.99. "
              f"Check for data leakage (plate-level split issue).")

    all_probs_dict = {
        CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}": float(cal_probs[i])
        for i in range(len(cal_probs))
    }

    return {
        "prediction":        prediction,
        "confidence":        confidence,
        "raw_confidence":    raw_conf,
        "all_probabilities": all_probs_dict,
        "temperature":       temperature,
        "class_index":       pred_idx,
    }


# ---------------------------------------------------------------------------
# Batch inference (all perturbations × all models)
# ---------------------------------------------------------------------------

def run_batch_inference(image_path: str) -> Dict:
    """
    Run inference on one image across all perturbations and all models.

    Returns nested dict:
        {
          "efficientnet_b0": {
              "original": {prediction, confidence, all_probabilities, ...},
              "bright":   {...},
              ...
          },
          "resnet50": {...},
        }
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found: {image_path}")

    import cv2
    img_bgr  = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Generate perturbations
    from perturbations.perturbation_engine import generate_all_perturbations
    perturbations = generate_all_perturbations(img_rgb)

    results = {}

    for model_name in MODEL_CONFIGS:
        print(f"\nLoading {model_name}...")
        model, temperature = load_model(model_name)

        if model is None:
            results[model_name] = {"error": f"Could not load {model_name}"}
            continue

        model_results = {}
        for pert in perturbations:
            pert_id = pert["id"]
            try:
                pred = predict_single(model, pert["image"], temperature=temperature)
                pred["metadata"] = {
                    "type":      pert.get("type", pert_id),
                    "parameter": pert.get("parameter"),
                    "id":        pert_id,
                }
                model_results[pert_id] = pred
            except Exception as e:
                model_results[pert_id] = {"error": str(e)}

        results[model_name] = model_results

    return results


def print_inference_summary(results: Dict) -> None:
    """Print a formatted summary of batch inference results."""
    print("\n" + "="*65)
    print("BATCH INFERENCE SUMMARY")
    print("="*65)

    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}")
        if "error" in model_results:
            print(f"  ERROR: {model_results['error']}")
            continue

        confs     = []
        raw_confs = []
        for pert_id, pred in model_results.items():
            if "error" in pred:
                print(f"  {pert_id:20s}: ERROR – {pred['error']}")
                continue
            c   = pred.get("confidence", 0)
            rc  = pred.get("raw_confidence", c)
            confs.append(c)
            raw_confs.append(rc)
            print(f"  {pert_id:20s}: {pred['prediction']:15s} "
                  f"conf={c:.4f} (raw={rc:.4f})  T={pred.get('temperature', 1.0):.3f}")

        if confs:
            mean_raw = float(np.mean(raw_confs)) if raw_confs else 0
            mean_cal = float(np.mean(confs))
            print(f"\n  Mean confidence — raw: {mean_raw:.4f} | calibrated: {mean_cal:.4f}")
            if mean_cal > 0.95:
                print("  ⚠ Still high after calibration — verify split integrity")
