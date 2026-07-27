"""
inference/batch_inference.py  –  Calibrated Batch Inference
============================================================
This module loads trained CNN checkpoints, applies temperature scaling,
and evaluates a single image under a suite of perturbations.

It is designed to be portable across CPU-only and GPU environments and
supports common image formats including HEIC when the Pillow plugin is
available.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = os.environ.get("PATHOGEN_CHECKPOINT_DIR", str(ROOT_DIR / "checkpoints"))
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["e_coli", "k_pneumoniae", "p_aeruginosa", "s_aureus"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _resolve_path(path: Optional[str], default: Optional[str] = None) -> str:
    if path is None:
        path = default
    if path is None:
        return str(ROOT_DIR)
    return str(Path(path).expanduser())


def _resolve_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = os.environ.get("PATHOGEN_DEVICE", DEFAULT_DEVICE)
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device(DEFAULT_DEVICE)
    if str(device).lower() == "auto":
        return torch.device(DEFAULT_DEVICE)
    return torch.device(str(device))


def _load_temperature(temperature_file: str) -> Union[float, np.ndarray]:
    """Load temperature scalar or class-wise vector from checkpoint; default 1.0."""
    if not os.path.exists(temperature_file):
        print(f"  [WARN] No temperature file at {temperature_file}. Using T=1.0 (uncalibrated).")
        return 1.0
    if _is_git_lfs_pointer(Path(temperature_file)):
        print(f"  [WARN] {temperature_file} is a Git LFS pointer. Using T=1.0.")
        return 1.0
    try:
        ckpt = torch.load(temperature_file, map_location="cpu", weights_only=False)
        temp = ckpt.get("temperature", 1.0)
        if isinstance(temp, (torch.Tensor, np.ndarray, list)):
            temp = np.array(temp, dtype=np.float32)
            print(f"  [INFO] Vector temperature loaded: {np.round(temp, 4).tolist()}")
        else:
            temp = float(temp)
            print(f"  [INFO] Temperature loaded: {temp:.4f}")
    except Exception as exc:
        print(f"  [WARN] Could not load temperature: {exc}. Using T=1.0.")
        return 1.0


def _load_image_array(image_path: str) -> np.ndarray:
    """Load an image from disk into RGB uint8 numpy format."""
    image_path = _resolve_path(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        import cv2

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is not None:
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    try:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except (ImportError, Exception):
            pass

        with Image.open(image_path) as image_handle:
            if getattr(image_handle, "n_frames", 1) > 1:
                image_handle.seek(0)
            rgb_image = image_handle.convert("RGB")
            return np.asarray(rgb_image, dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"Could not read image: {image_path}: {exc}") from exc


def _is_git_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            text = handle.read(200)
        return "git-lfs.github.com" in text
    except Exception:
        return False


def _build_pretrained_model(model_name: str) -> nn.Module:
    num_classes = len(CLASS_NAMES)
    if model_name == "efficientnet_b0":
        from models.efficientnet_setup import build_efficientnet_b0
        return build_efficientnet_b0(num_classes=num_classes)

    if model_name == "swin_t":
        from models.swin_setup import build_swin_t
        return build_swin_t(num_classes=num_classes)

    if model_name == "convnext_tiny":
        from models.convnext_setup import build_convnext_tiny
        return build_convnext_tiny(num_classes=num_classes)

    if model_name == "resnet50":
        from models.resnet_setup import build_resnet50
        return build_resnet50(num_classes=num_classes)

    raise ValueError(f"Unknown model name: {model_name}")


def load_model(model_name: str, checkpoint_dir: Optional[str] = None, device: Optional[str] = None) -> Tuple[Optional[nn.Module], Optional[float]]:
    """Load a trained model and its calibration temperature."""
    resolved_device = _resolve_device(device)
    checkpoint_dir_path = Path(_resolve_path(checkpoint_dir, CHECKPOINT_DIR))

    if model_name == "efficientnet_b0":
        checkpoint = checkpoint_dir_path / "efficientnet_b0_best.pth"
        temperature_file = checkpoint_dir_path / "efficientnet_b0_temperature.pth"
    elif model_name == "resnet50":
        checkpoint = checkpoint_dir_path / "resnet50_best.pth"
        temperature_file = checkpoint_dir_path / "resnet50_temperature.pth"
    elif model_name == "swin_t":
        checkpoint = checkpoint_dir_path / "swin_t_best.pth"
        temperature_file = checkpoint_dir_path / "swin_t_temperature.pth"
    elif model_name == "convnext_tiny":
        checkpoint = checkpoint_dir_path / "convnext_tiny_best.pth"
        temperature_file = checkpoint_dir_path / "convnext_tiny_temperature.pth"
    else:
        print(f"[ERROR] Unknown model: {model_name}")
        return None, None

    try:
        model = _build_pretrained_model(model_name)
        if checkpoint.exists() and not _is_git_lfs_pointer(checkpoint):
            try:
                ckpt = torch.load(str(checkpoint), map_location=str(resolved_device), weights_only=False)
                if isinstance(ckpt, dict) and "model_state" in ckpt:
                    model.load_state_dict(ckpt["model_state"])
                elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                    model.load_state_dict(ckpt["state_dict"])
                else:
                    model.load_state_dict(ckpt)
                try:
                    rel_ckpt = checkpoint.relative_to(Path.cwd())
                except ValueError:
                    rel_ckpt = checkpoint
                print(f"[INFO] Loaded checkpoint weights from {rel_ckpt}")
            except Exception as exc:
                print(f"[WARN] Could not load checkpoint {checkpoint}: {exc}. Falling back to pretrained weights.")
        else:
            if checkpoint.exists() and _is_git_lfs_pointer(checkpoint):
                print(f"[WARN] {checkpoint} is a Git LFS pointer. Falling back to pretrained weights.")
            else:
                print(f"[WARN] Checkpoint not found: {checkpoint}. Falling back to pretrained weights.")

        model.to(resolved_device).eval()
        temperature = _load_temperature(str(temperature_file))
        return model, temperature

    except Exception as exc:
        print(f"[ERROR] Failed to load {model_name}: {exc}")
        return None, None


def preprocess_image(image_array: np.ndarray) -> torch.Tensor:
    """Convert numpy array (RGB, RGBA, Grayscale, Float, Int) to preprocessed tensor [1, 3, 224, 224]."""
    image_array = np.asarray(image_array)

    # Normalize float images in [0, 1] to uint8 [0, 255]
    if np.issubdtype(image_array.dtype, np.floating):
        if image_array.max() <= 1.0:
            image_array = (image_array * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image_array = image_array.clip(0, 255).astype(np.uint8)
    elif image_array.dtype != np.uint8:
        # Handle 16-bit or signed int arrays
        image_array = ((image_array - image_array.min()) / (image_array.ptp() + 1e-8) * 255.0).astype(np.uint8)

    # Convert shape dimensions to 3-channel RGB
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.ndim == 3 and image_array.shape[0] in (1, 3, 4) and image_array.shape[2] not in (1, 3, 4):
        # Convert channels-first (C, H, W) to channels-last (H, W, C)
        image_array = np.transpose(image_array, (1, 2, 0))

    if image_array.ndim == 3:
        if image_array.shape[2] == 1:
            image_array = np.concatenate([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:
            # Drop alpha channel
            image_array = image_array[:, :, :3]

    image = Image.fromarray(image_array, mode="RGB")
    return torch.unsqueeze(preprocess_transform(image), 0)


def predict_single(
    model: nn.Module,
    image_array: np.ndarray,
    temperature: Union[float, np.ndarray, torch.Tensor] = 1.0,
    device: Optional[str] = None,
) -> Dict:
    """Run inference on one image with scalar or vector temperature scaling."""
    resolved_device = _resolve_device(device)
    tensor = preprocess_image(image_array).to(resolved_device)

    with torch.no_grad():
        logits = model(tensor)

    raw_probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

    # Temperature scaling (scalar or class-wise vector)
    if isinstance(temperature, (np.ndarray, list)):
        temp_tensor = torch.tensor(temperature, device=resolved_device, dtype=torch.float32)
        cal_logits = logits / temp_tensor
    else:
        cal_logits = logits / float(temperature)

    cal_probs = torch.softmax(cal_logits, dim=1).cpu().numpy().flatten()

    pred_idx = int(cal_probs.argmax())
    prediction = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"
    confidence = float(cal_probs[pred_idx])
    raw_conf = float(raw_probs[pred_idx])

    if confidence > 0.99:
        print(f"  [WARN] Post-calibration confidence {confidence:.4f} is still > 0.99.")

    all_probs_dict = {
        CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}": float(cal_probs[i])
        for i in range(len(cal_probs))
    }

    return {
        "prediction": prediction,
        "confidence": confidence,
        "raw_confidence": raw_conf,
        "all_probabilities": all_probs_dict,
        "probabilities": list(all_probs_dict.values()),
        "temperature": temperature if isinstance(temperature, float) else np.array(temperature).tolist(),
        "class_index": pred_idx,
        "predicted_idx": pred_idx,
    }


def predict_batch(
    model: nn.Module,
    perturbations_dict: Dict[str, Dict],
    temperature: Union[float, np.ndarray, torch.Tensor] = 1.0,
    device: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Accelerated Tensor Batch Inference: Preprocesses and stacks all perturbation image variants
    into a single 4D tensor [N, 3, 224, 224] for parallel single-pass GPU/CPU forward execution.
    """
    resolved_device = _resolve_device(device)
    pert_keys = list(perturbations_dict.keys())
    if not pert_keys:
        return {}

    # Preprocess all perturbation images and concatenate into a single 4D tensor
    tensor_list = []
    for k in pert_keys:
        img = perturbations_dict[k]["image"]
        tensor_list.append(preprocess_image(img))

    batch_tensor = torch.cat(tensor_list, dim=0).to(resolved_device)

    with torch.no_grad():
        logits = model(batch_tensor)  # [N, num_classes]

    raw_probs = torch.softmax(logits, dim=1).cpu().numpy()  # [N, num_classes]

    # Temperature scaling (scalar or class-wise vector)
    if isinstance(temperature, (np.ndarray, list)):
        temp_tensor = torch.tensor(temperature, device=resolved_device, dtype=torch.float32)
        cal_logits = logits / temp_tensor
    else:
        cal_logits = logits / float(temperature)

    cal_probs = torch.softmax(cal_logits, dim=1).cpu().numpy()  # [N, num_classes]

    batch_results = {}
    for idx, pert_key in enumerate(pert_keys):
        pert_data = perturbations_dict[pert_key]
        c_probs = cal_probs[idx]
        r_probs = raw_probs[idx]

        pred_idx = int(c_probs.argmax())
        prediction = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"
        confidence = float(c_probs[pred_idx])
        raw_conf = float(r_probs[pred_idx])

        all_probs_dict = {
            CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}": float(c_probs[i])
            for i in range(len(c_probs))
        }

        batch_results[pert_key] = {
            "prediction": prediction,
            "confidence": confidence,
            "raw_confidence": raw_conf,
            "all_probabilities": all_probs_dict,
            "probabilities": list(all_probs_dict.values()),
            "temperature": temperature if isinstance(temperature, float) else np.array(temperature).tolist(),
            "class_index": pred_idx,
            "predicted_idx": pred_idx,
            "metadata": {
                "type": pert_data.get("type", pert_key),
                "parameter": pert_data.get("parameter"),
                "id": pert_data.get("id", pert_key),
            },
        }

    return batch_results


def run_batch_inference(image_path: str, checkpoint_dir: Optional[str] = None, device: Optional[str] = None) -> Dict:
    """Run accelerated batch inference on one image across all perturbations and all models."""
    resolved_device = _resolve_device(device)
    checkpoint_dir_path = _resolve_path(checkpoint_dir, CHECKPOINT_DIR)

    if not os.path.exists(_resolve_path(image_path)):
        raise ValueError(f"Image not found: {image_path}")

    _load_image_array(image_path)

    from perturbations.perturbation_engine import generate_perturbations

    perturbations = generate_perturbations(image_path)
    results = {}

    model_list = ["efficientnet_b0", "resnet50", "swin_t", "convnext_tiny"]
    for model_name in model_list:
        print(f"\nLoading {model_name}...")
        model, temperature = load_model(model_name, checkpoint_dir=checkpoint_dir_path, device=str(resolved_device))

        if model is None:
            results[model_name] = {"error": f"Could not load {model_name}"}
            continue

        temp_val = temperature if temperature is not None else 1.0
        # Accelerated Tensor Batch Inference
        model_results = predict_batch(model, perturbations, temperature=temp_val, device=str(resolved_device))
        results[model_name] = model_results

    return results


def print_inference_summary(results: Dict) -> None:
    """Print a formatted summary of batch inference results."""
    print("\n" + "=" * 65)
    print("BATCH INFERENCE SUMMARY")
    print("=" * 65)

    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}")
        if "error" in model_results:
            print(f"  ERROR: {model_results['error']}")
            continue

        confs = []
        raw_confs = []
        preds = []
        for pert_id, pred in model_results.items():
            if "error" in pred:
                continue
            confs.append(pred.get("confidence", 0.0))
            raw_confs.append(pred.get("raw_confidence", 0.0))
            preds.append(pred.get("prediction", "unknown"))

        if confs:
            from collections import Counter
            counts = Counter(preds)
            counts_str = ", ".join([f"{k}: {v}/{len(preds)}" for k, v in counts.items()])
            print(f"  Calibrated Confidence: {sum(confs) / len(confs):.4f}")
            print(f"  Raw Confidence:        {sum(raw_confs) / len(raw_confs):.4f}")
            print(f"  Prediction Breakdown:  {counts_str}")
        else:
            print("  No valid predictions.")

    print("\n" + "=" * 65)
