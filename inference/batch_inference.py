"""
Batch Inference Module for Pathogen Intelligence System.

This module acts as the bridge between perturbation generation and robustness analysis.
It loads trained CNN models, processes perturbed images, and collects structured predictions.

Purpose:
    - Load trained EfficientNet-B0 and ResNet-50 models
    - Process perturbation variants from perturbation_engine
    - Run inference on original and perturbed images
    - Collect structured outputs with predictions, confidence, and metadata
    - Enable downstream robustness analysis

Architecture Flow:
    Perturbation Engine → Batch Inference → Robustness Analyzer
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.efficientnet_setup import build_efficientnet_b0
from models.resnet_setup import build_resnet50
from perturbations.perturbation_engine import generate_perturbations
from loaders.data_loader import IMAGENET_MEAN, IMAGENET_STD


# Class mapping for pathogen classification
CLASS_NAMES = ["e_coli", "k_pneumoniae", "p_aeruginosa", "s_aureus"]

# Default checkpoint paths
DEFAULT_EFFICIENTNET_CHECKPOINT = "checkpoints/efficientnet_b0_best.pth"
DEFAULT_RESNET_CHECKPOINT = "checkpoints/resnet50_best.pth"


def load_model(model_name, checkpoint_path, device):
    """
    Load a trained CNN model from checkpoint.
    
    Args:
        model_name: "efficientnet_b0" or "resnet50"
        checkpoint_path: Path to model checkpoint (.pth file)
        device: torch.device for model placement
        
    Returns:
        Loaded model in evaluation mode
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If model_name is invalid
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"[INFO] Loading {model_name} from {checkpoint_path}")
    
    # Build model architecture
    if model_name == "efficientnet_b0":
        model = build_efficientnet_b0(num_classes=len(CLASS_NAMES))
    elif model_name == "resnet50":
        model = build_resnet50(num_classes=len(CLASS_NAMES))
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'efficientnet_b0' or 'resnet50'")
    
    # Load trained weights
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[INFO] Successfully loaded weights for {model_name}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")
    
    # Set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_np, target_size=(224, 224)):
    """
    Preprocess numpy image for PyTorch inference.
    
    Applies:
        - Resize to target size
        - Convert to PIL Image
        - Convert to tensor
        - Normalize with ImageNet stats
        - Add batch dimension
    
    Args:
        image_np: Numpy array (H, W, 3) in RGB format, uint8
        target_size: Target image size (height, width)
        
    Returns:
        Preprocessed tensor (1, 3, H, W) ready for inference
    """
    # Convert numpy to PIL Image
    if image_np.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image_np.dtype}")
    
    pil_image = Image.fromarray(image_np)
    
    # Define preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Apply preprocessing
    tensor = preprocess(pil_image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor


def predict_single(model, image_tensor, device):
    """
    Run inference on a single preprocessed image.
    
    Args:
        model: Trained PyTorch model in eval mode
        image_tensor: Preprocessed tensor (1, 3, H, W)
        device: torch.device for computation
        
    Returns:
        Dictionary containing:
            - prediction: Predicted class name
            - confidence: Confidence score (0-1)
            - probabilities: Softmax probabilities for all classes
            - predicted_idx: Predicted class index
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Forward pass
        logits = model(image_tensor)
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=1)
        
        # Get predicted class
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        # Convert to Python types
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()
        probabilities = probabilities.squeeze().cpu().numpy().tolist()
    
    prediction = CLASS_NAMES[predicted_idx]
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "predicted_idx": predicted_idx
    }


def run_batch_inference(image_path, models_config=None, device=None):
    """
    Run batch inference on original image and all perturbation variants.
    
    This is the main entry point for the inference module. It:
        1. Generates perturbations from input image
        2. Loads specified models
        3. Runs inference on all variants
        4. Collects structured results with metadata
    
    Args:
        image_path: Path to input image
        models_config: Dictionary mapping model names to checkpoint paths
                      Example: {"efficientnet_b0": "path/to/checkpoint.pth"}
                      If None, uses default checkpoints
        device: torch.device for computation. If None, auto-detects GPU/CPU
        
    Returns:
        Dictionary with structure:
        {
            "model_name": {
                "perturbation_name": {
                    "prediction": str,
                    "confidence": float,
                    "probabilities": list,
                    "predicted_idx": int,
                    "metadata": {
                        "type": str,
                        "parameter": float,
                        "id": str
                    }
                }
            }
        }
        
    Raises:
        FileNotFoundError: If image or checkpoint not found
        ValueError: If image is invalid
    """
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Setup models configuration
    if models_config is None:
        models_config = {
            "efficientnet_b0": DEFAULT_EFFICIENTNET_CHECKPOINT,
            "resnet50": DEFAULT_RESNET_CHECKPOINT
        }
    
    # Generate perturbations
    print(f"\n[INFO] Generating perturbations for: {image_path}")
    try:
        perturbations = generate_perturbations(image_path)
        print(f"[INFO] Generated {len(perturbations)} perturbation variants")
    except Exception as e:
        raise ValueError(f"Failed to generate perturbations: {e}")
    
    # Initialize results structure
    results = {}
    
    # Process each model
    for model_name, checkpoint_path in models_config.items():
        print(f"\n{'='*80}")
        print(f"[INFO] Processing model: {model_name}")
        print(f"{'='*80}")
        
        try:
            # Load model
            model = load_model(model_name, checkpoint_path, device)
            
            # Initialize model results
            model_results = {}
            
            # Process each perturbation
            for perturbation_name, perturbation_data in perturbations.items():
                print(f"\n[INFO] Processing perturbation: {perturbation_name}")
                
                try:
                    # Extract image and metadata
                    image_np = perturbation_data["image"]
                    metadata = {
                        "type": perturbation_data["type"],
                        "parameter": perturbation_data["parameter"],
                        "id": perturbation_data["id"]
                    }
                    
                    # Preprocess image
                    image_tensor = preprocess_image(image_np)
                    
                    # Run inference
                    prediction_result = predict_single(model, image_tensor, device)
                    
                    # Add metadata to result
                    prediction_result["metadata"] = metadata
                    
                    # Store result
                    model_results[perturbation_name] = prediction_result
                    
                    # Debug logging
                    print(f"  → Prediction: {prediction_result['prediction']}")
                    print(f"  → Confidence: {prediction_result['confidence']:.4f}")
                    print(f"  → Metadata: {metadata['id']}")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process {perturbation_name}: {e}")
                    model_results[perturbation_name] = {
                        "error": str(e),
                        "metadata": metadata if 'metadata' in locals() else None
                    }
            
            # Store model results
            results[model_name] = model_results
            
            print(f"\n[INFO] Completed inference for {model_name}")
            print(f"[INFO] Processed {len(model_results)} perturbations")
            
        except Exception as e:
            print(f"[ERROR] Failed to process model {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    print(f"\n{'='*80}")
    print(f"[INFO] Batch inference completed")
    print(f"[INFO] Models processed: {list(results.keys())}")
    print(f"{'='*80}\n")
    
    return results


def print_inference_summary(results):
    """
    Print a human-readable summary of inference results.
    
    Args:
        results: Output from run_batch_inference()
    """
    print("\n" + "="*80)
    print("INFERENCE SUMMARY")
    print("="*80)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        print("-" * 80)
        
        if "error" in model_results:
            print(f"  ERROR: {model_results['error']}")
            continue
        
        for perturbation_name, pred_data in model_results.items():
            if "error" in pred_data:
                print(f"  {perturbation_name:20s} → ERROR: {pred_data['error']}")
            else:
                pred = pred_data['prediction']
                conf = pred_data['confidence']
                print(f"  {perturbation_name:20s} → {pred:15s} (confidence: {conf:.4f})")
    
    print("\n" + "="*80)


def main():
    """
    Test batch inference on a sample image.
    Demonstrates the complete inference pipeline.
    """
    # Get image path from user
    image_path = input("Enter path to pathogen image: ").strip()
    
    if not image_path:
        print("No path provided. Exiting.")
        return
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Run batch inference
    print("\n" + "="*80)
    print("STARTING BATCH INFERENCE")
    print("="*80)
    
    try:
        results = run_batch_inference(image_path)
        
        # Print summary
        print_inference_summary(results)
        
        # Example: Access specific result
        if "efficientnet_b0" in results and "original" in results["efficientnet_b0"]:
            original_pred = results["efficientnet_b0"]["original"]
            print(f"\nExample access:")
            print(f"EfficientNet prediction on original image: {original_pred['prediction']}")
            print(f"Confidence: {original_pred['confidence']:.4f}")
        
    except Exception as e:
        print(f"\n[ERROR] Batch inference failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
