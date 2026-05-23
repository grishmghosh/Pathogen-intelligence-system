"""
Grad-CAM generation for the Pathogen Intelligence System (Step 7).

This module provides a lightweight, deterministic Grad-CAM implementation for
CNN backbones. It supports common convolutional architectures such as
EfficientNet and ResNet by auto-selecting the last convolutional layer when a
specific target layer is not provided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF


def _safe_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    elif isinstance(value, Image.Image):
        array = np.asarray(value)
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value)
    else:
        return None
    array = np.asarray(array, dtype=float)
    if array.size == 0:
        return None
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array


def _to_tensor_image(image: Any) -> Tuple[Optional[torch.Tensor], Optional[Image.Image], Dict[str, Any]]:
    """Convert an input image into a tensor plus a displayable PIL image."""
    metadata: Dict[str, Any] = {"input_kind": type(image).__name__}

    if isinstance(image, torch.Tensor):
        tensor = image.detach().clone().float()
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 4:
            return None, None, {"error": f"Expected tensor with 3 or 4 dims, got shape {tuple(tensor.shape)}"}
        display_tensor = tensor[0].detach().cpu()
        if display_tensor.dim() != 3:
            return None, None, {"error": "Tensor image must have 3 channels."}
        display = display_tensor
        display = display - display.min()
        if float(display.max()) > 0:
            display = display / display.max()
        display_image = TF.to_pil_image(display.clamp(0.0, 1.0))
        metadata["input_tensor_shape"] = tuple(tensor.shape)
        return tensor, display_image, metadata

    if isinstance(image, Image.Image):
        display_image = image.convert("RGB")
        tensor = TF.to_tensor(display_image).unsqueeze(0).float()
        metadata["input_tensor_shape"] = tuple(tensor.shape)
        return tensor, display_image, metadata

    array = _safe_array(image)
    if array is None:
        return None, None, {"error": "Unsupported image format."}

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3:
        return None, None, {"error": f"Expected image with 3 dimensions, got shape {array.shape}"}

    if array.shape[-1] not in (1, 3):
        return None, None, {"error": f"Expected channel dimension of 1 or 3, got {array.shape[-1]}"}
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)

    array = array.astype(float)
    if array.max() > 1.0:
        array = array / 255.0
    array = np.clip(array, 0.0, 1.0)
    display_image = Image.fromarray((array * 255.0).astype(np.uint8))
    tensor = TF.to_tensor(display_image).unsqueeze(0).float()
    metadata["input_tensor_shape"] = tuple(tensor.shape)
    return tensor, display_image, metadata


def _find_last_conv_layer(model: nn.Module) -> Tuple[Optional[str], Optional[nn.Module]]:
    last_name = None
    last_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
            last_layer = module
    return last_name, last_layer


def _resolve_target_layer(
    model: nn.Module,
    target_layer: Optional[Union[str, nn.Module]] = None,
) -> Tuple[Optional[str], Optional[nn.Module], Optional[str]]:
    if isinstance(target_layer, nn.Module):
        return getattr(target_layer, "_name", None), target_layer, None

    if isinstance(target_layer, str):
        for name, module in model.named_modules():
            if name == target_layer:
                if isinstance(module, nn.Conv2d):
                    return name, module, None
                return None, None, f"Target layer '{target_layer}' is not a Conv2d layer."
        return None, None, f"Target layer '{target_layer}' was not found."

    name, layer = _find_last_conv_layer(model)
    if layer is None:
        return None, None, "No convolutional layer found."
    return name, layer, None


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=float)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = np.maximum(heatmap, 0.0)
    if heatmap.size == 0:
        return heatmap
    maximum = float(heatmap.max())
    minimum = float(heatmap.min())
    if maximum <= minimum:
        return np.zeros_like(heatmap, dtype=float)
    return (heatmap - minimum) / (maximum - minimum)


def create_attention_overlay(
    image: Any,
    heatmap: Any,
    alpha: float = 0.45,
    cmap: str = "jet",
) -> Optional[Image.Image]:
    """Create a color overlay from an image and a normalized heatmap."""
    _, display_image, error = _to_tensor_image(image)
    if display_image is None:
        return None

    heatmap_array = _safe_array(heatmap)
    if heatmap_array is None:
        return None
    if heatmap_array.ndim == 3 and heatmap_array.shape[0] == 1:
        heatmap_array = heatmap_array[0]
    if heatmap_array.ndim != 2:
        return None

    heatmap_array = _normalize_heatmap(heatmap_array)
    heatmap_image = Image.fromarray(np.uint8(255 * heatmap_array))
    heatmap_image = heatmap_image.resize(display_image.size, resample=Image.BILINEAR)
    heatmap_resized = np.asarray(heatmap_image, dtype=float) / 255.0

    colormap = plt.get_cmap(cmap)
    colored = colormap(heatmap_resized)[..., :3]
    base = np.asarray(display_image.convert("RGB"), dtype=float) / 255.0
    blended = np.clip((1.0 - alpha) * base + alpha * colored, 0.0, 1.0)
    return Image.fromarray(np.uint8(blended * 255.0))


def generate_gradcam(
    model: nn.Module,
    image: Any,
    target_class: Optional[int] = None,
    target_layer: Optional[Union[str, nn.Module]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Generate a Grad-CAM heatmap for a CNN model.

    Returns a dictionary with raw heatmaps, normalized maps, overlay image,
    and metadata. Errors are returned in-band via the ``status`` field so the
    caller can handle unsupported backbones gracefully.
    """
    if model is None:
        return {"status": "error", "error": "Model is required."}

    tensor, display_image, image_meta = _to_tensor_image(image)
    if tensor is None or display_image is None:
        return {"status": "error", **image_meta}

    target_layer_name, layer, layer_error = _resolve_target_layer(model, target_layer=target_layer)
    if layer is None:
        return {"status": "unsupported_backbone", "error": layer_error or "No valid target layer found."}

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model = model.to(device)
    model.eval()
    tensor = tensor.to(device)
    tensor.requires_grad_(True)

    activations = []
    gradients = []

    def _forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def _backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = layer.register_forward_hook(_forward_hook)
    backward_handle = layer.register_full_backward_hook(_backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        output = model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.dim() == 1:
            output = output.unsqueeze(0)
        if output.dim() != 2:
            return {"status": "error", "error": f"Expected model output shape [N, C], got {tuple(output.shape)}"}

        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())
        if target_class < 0 or target_class >= output.shape[1]:
            return {"status": "error", "error": f"target_class {target_class} is out of range for {output.shape[1]} classes."}

        score = output[:, target_class].sum()
        score.backward(retain_graph=False)

        if not activations or not gradients:
            return {"status": "error", "error": "Failed to capture activations or gradients."}

        activation = activations[-1]
        gradient = gradients[-1]
        if activation.dim() != 4 or gradient.dim() != 4:
            return {"status": "error", "error": "Captured tensors are not 4D convolutional maps."}

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activation, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        raw_heatmap = cam.squeeze().detach().cpu().numpy()
        normalized_heatmap = _normalize_heatmap(raw_heatmap)
        overlay = create_attention_overlay(display_image, normalized_heatmap)

        return {
            "status": "ok",
            "target_class_index": int(target_class),
            "target_class_score": float(score.detach().cpu().item()),
            "target_layer_name": target_layer_name,
            "raw_heatmap": raw_heatmap,
            "normalized_heatmap": normalized_heatmap,
            "overlay_image": overlay,
            "input_image": display_image,
            "input_metadata": image_meta,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc), "target_layer_name": target_layer_name}
    finally:
        forward_handle.remove()
        backward_handle.remove()


def save_heatmap_png(
    heatmap: Any,
    output_path: Union[str, Path],
    cmap: str = "jet",
    title: Optional[str] = None,
) -> Path:
    """Save a raw heatmap as a PNG image."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    heatmap_array = _safe_array(heatmap)
    if heatmap_array is None:
        raise ValueError("Invalid heatmap input.")
    if heatmap_array.ndim == 3 and heatmap_array.shape[0] == 1:
        heatmap_array = heatmap_array[0]
    if heatmap_array.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape {heatmap_array.shape}")

    heatmap_array = _normalize_heatmap(heatmap_array)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(heatmap_array, cmap=cmap)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return output


def save_overlay_png(
    overlay_image: Image.Image,
    output_path: Union[str, Path],
) -> Path:
    """Save an attention overlay image as a PNG."""
    if overlay_image is None:
        raise ValueError("Overlay image is required.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay_image.save(output)
    return output
