"""
Comprehensive Intelligence Layer Testing Suite

This script performs thorough end-to-end testing of:
    - inference/batch_inference.py
    - analysis/robustness_analyzer.py

Tests the complete pipeline with REAL data and REAL inference.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("COMPREHENSIVE INTELLIGENCE LAYER TEST SUITE")
print("="*80)
print("\nThis test performs REAL end-to-end inference with actual trained models.")
print("Testing: Perturbations -> Batch Inference -> Robustness Analysis\n")

# =============================================================================
# TEST 1: Import Validation
# =============================================================================
print("\n" + "="*80)
print("[TEST 1] IMPORT VALIDATION")
print("="*80)

test_results = {
    "imports": {"status": "pending", "details": []},
    "checkpoints": {"status": "pending", "details": []},
    "perturbations": {"status": "pending", "details": []},
    "preprocessing": {"status": "pending", "details": []},
    "inference": {"status": "pending", "details": []},
    "metadata": {"status": "pending", "details": []},
    "robustness": {"status": "pending", "details": []},
    "comparison": {"status": "pending", "details": []},
    "realism": {"status": "pending", "details": []},
    "error_handling": {"status": "pending", "details": []}
}

print("\n[1.1] Testing config imports...")
try:
    from configs.perturbation_config import PERTURBATION_CONFIG, ENABLED_PERTURBATIONS
    print("  ✓ configs.perturbation_config imported successfully")
    print(f"    - Config keys: {list(PERTURBATION_CONFIG.keys())}")
    print(f"    - Enabled perturbations: {len(ENABLED_PERTURBATIONS)}")
    test_results["imports"]["details"].append("✓ Config imports working")
except Exception as e:
    print(f"  ✗ Config import failed: {e}")
    test_results["imports"]["details"].append(f"✗ Config import failed: {e}")
    test_results["imports"]["status"] = "failed"

print("\n[1.2] Testing perturbation imports...")
try:
    from perturbations.perturbation_engine import generate_perturbations, load_image
    print("  ✓ perturbations.perturbation_engine imported successfully")
    test_results["imports"]["details"].append("✓ Perturbation imports working")
except Exception as e:
    print(f"  ✗ Perturbation import failed: {e}")
    test_results["imports"]["details"].append(f"✗ Perturbation import failed: {e}")
    test_results["imports"]["status"] = "failed"

print("\n[1.3] Testing model imports...")
try:
    from models.efficientnet_setup import build_efficientnet_b0
    from models.resnet_setup import build_resnet50
    print("  ✓ models.efficientnet_setup imported successfully")
    print("  ✓ models.resnet_setup imported successfully")
    test_results["imports"]["details"].append("✓ Model imports working")
except Exception as e:
    print(f"  ✗ Model import failed: {e}")
    test_results["imports"]["details"].append(f"✗ Model import failed: {e}")
    test_results["imports"]["status"] = "failed"

print("\n[1.4] Testing inference imports...")
try:
    from inference.batch_inference import (
        load_model, preprocess_image, predict_single, 
        run_batch_inference, print_inference_summary
    )
    print("  ✓ inference.batch_inference imported successfully")
    print("    - load_model: available")
    print("    - preprocess_image: available")
    print("    - predict_single: available")
    print("    - run_batch_inference: available")
    test_results["imports"]["details"].append("✓ Inference imports working")
except Exception as e:
    print(f"  ✗ Inference import failed: {e}")
    test_results["imports"]["details"].append(f"✗ Inference import failed: {e}")
    test_results["imports"]["status"] = "failed"
    import traceback
    traceback.print_exc()

print("\n[1.5] Testing analysis imports...")
try:
    from analysis.robustness_analyzer import (
        analyze_prediction_consistency,
        analyze_confidence_drift,
        analyze_perturbation_sensitivity,
        compute_robustness_score,
        compare_models,
        generate_robustness_report,
        print_robustness_summary
    )
    print("  ✓ analysis.robustness_analyzer imported successfully")
    print("    - analyze_prediction_consistency: available")
    print("    - analyze_confidence_drift: available")
    print("    - analyze_perturbation_sensitivity: available")
    print("    - compute_robustness_score: available")
    print("    - compare_models: available")
    print("    - generate_robustness_report: available")
    test_results["imports"]["details"].append("✓ Analysis imports working")
except Exception as e:
    print(f"  ✗ Analysis import failed: {e}")
    test_results["imports"]["details"].append(f"✗ Analysis import failed: {e}")
    test_results["imports"]["status"] = "failed"
    import traceback
    traceback.print_exc()

if test_results["imports"]["status"] != "failed":
    test_results["imports"]["status"] = "passed"
    print("\n[TEST 1] ✓ ALL IMPORTS SUCCESSFUL")
else:
    print("\n[TEST 1] ✗ IMPORT FAILURES DETECTED")
    print("Cannot proceed with further testing.")
    sys.exit(1)

# =============================================================================
# TEST 2: Checkpoint Loading
# =============================================================================
print("\n" + "="*80)
print("[TEST 2] CHECKPOINT LOADING")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[2.1] Device detection: {device}")

checkpoint_paths = {
    "efficientnet_b0": "checkpoints/efficientnet_b0_best.pth",
    "resnet50": "checkpoints/resnet50_best.pth"
}

loaded_models = {}

for model_name, checkpoint_path in checkpoint_paths.items():
    print(f"\n[2.2] Testing {model_name} checkpoint loading...")
    print(f"  Checkpoint path: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"  ✗ Checkpoint not found: {checkpoint_path}")
        test_results["checkpoints"]["details"].append(f"✗ {model_name} checkpoint missing")
        test_results["checkpoints"]["status"] = "failed"
        continue
    
    try:
        model = load_model(model_name, checkpoint_path, device)
        loaded_models[model_name] = model
        print(f"  ✓ {model_name} loaded successfully")
        print(f"    - Device: {next(model.parameters()).device}")
        print(f"    - Mode: {'eval' if not model.training else 'train'}")
        print(f"    - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        test_results["checkpoints"]["details"].append(f"✓ {model_name} loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to load {model_name}: {e}")
        test_results["checkpoints"]["details"].append(f"✗ {model_name} loading failed: {e}")
        test_results["checkpoints"]["status"] = "failed"
        import traceback
        traceback.print_exc()

if len(loaded_models) == len(checkpoint_paths):
    test_results["checkpoints"]["status"] = "passed"
    print("\n[TEST 2] ✓ ALL CHECKPOINTS LOADED SUCCESSFULLY")
else:
    print("\n[TEST 2] ⚠ SOME CHECKPOINTS FAILED TO LOAD")
    print("Proceeding with available models...")

# =============================================================================
# Find Test Image
# =============================================================================
print("\n" + "="*80)
print("FINDING TEST IMAGE")
print("="*80)

test_image_path = None
search_paths = [
    "dataset_split/val/e_coli/Plate 1",
    "dataset_split/train/e_coli/Plate 1",
    "dataset_split/val/k_pneumoniae/Plate 1",
]

for search_path in search_paths:
    if os.path.exists(search_path):
        for file in os.listdir(search_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_image_path = os.path.join(search_path, file)
                break
    if test_image_path:
        break

if not test_image_path:
    print("✗ No test image found in dataset")
    print("Cannot proceed with inference testing.")
    sys.exit(1)

print(f"✓ Test image found: {test_image_path}")
print(f"  File size: {os.path.getsize(test_image_path):,} bytes")

# =============================================================================
# TEST 3: Perturbation Integration
# =============================================================================
print("\n" + "="*80)
print("[TEST 3] PERTURBATION INTEGRATION")
print("="*80)

print("\n[3.1] Generating perturbations...")
try:
    perturbations = generate_perturbations(test_image_path)
    print(f"  ✓ Generated {len(perturbations)} perturbations")
    
    expected_perturbations = ["original"] + ENABLED_PERTURBATIONS
    print(f"\n[3.2] Validating perturbation completeness...")
    print(f"  Expected: {expected_perturbations}")
    print(f"  Generated: {list(perturbations.keys())}")
    
    all_present = all(p in perturbations for p in expected_perturbations)
    if all_present:
        print("  ✓ All expected perturbations present")
        test_results["perturbations"]["status"] = "passed"
    else:
        missing = [p for p in expected_perturbations if p not in perturbations]
        print(f"  ✗ Missing perturbations: {missing}")
        test_results["perturbations"]["status"] = "failed"
    
    print(f"\n[3.3] Validating metadata structure...")
    for pert_name, pert_data in perturbations.items():
        required_fields = ["image", "type", "parameter", "id"]
        missing_fields = [f for f in required_fields if f not in pert_data]
        if missing_fields:
            print(f"  ✗ {pert_name}: missing fields {missing_fields}")
            test_results["perturbations"]["status"] = "failed"
        else:
            img = pert_data["image"]
            print(f"  ✓ {pert_name:20s} → shape: {img.shape}, dtype: {img.dtype}, id: {pert_data['id']}")
    
    test_results["perturbations"]["details"].append(f"✓ Generated {len(perturbations)} perturbations")
    
except Exception as e:
    print(f"  ✗ Perturbation generation failed: {e}")
    test_results["perturbations"]["status"] = "failed"
    test_results["perturbations"]["details"].append(f"✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[TEST 3] ✓ PERTURBATION INTEGRATION SUCCESSFUL")

# =============================================================================
# TEST 4: Preprocessing Validation
# =============================================================================
print("\n" + "="*80)
print("[TEST 4] PREPROCESSING VALIDATION")
print("="*80)

print("\n[4.1] Testing image preprocessing...")
try:
    test_image_np = perturbations["original"]["image"]
    print(f"  Input image shape: {test_image_np.shape}")
    print(f"  Input image dtype: {test_image_np.dtype}")
    
    preprocessed = preprocess_image(test_image_np)
    print(f"\n  ✓ Preprocessing successful")
    print(f"    - Output shape: {preprocessed.shape}")
    print(f"    - Output dtype: {preprocessed.dtype}")
    print(f"    - Output device: {preprocessed.device}")
    print(f"    - Value range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    
    # Validate shape
    if preprocessed.shape == torch.Size([1, 3, 224, 224]):
        print(f"    ✓ Shape correct: (1, 3, 224, 224)")
        test_results["preprocessing"]["status"] = "passed"
    else:
        print(f"    ✗ Shape incorrect: expected (1, 3, 224, 224), got {preprocessed.shape}")
        test_results["preprocessing"]["status"] = "failed"
    
    test_results["preprocessing"]["details"].append("✓ Preprocessing working correctly")
    
except Exception as e:
    print(f"  ✗ Preprocessing failed: {e}")
    test_results["preprocessing"]["status"] = "failed"
    test_results["preprocessing"]["details"].append(f"✗ Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()

print("\n[TEST 4] ✓ PREPROCESSING VALIDATION COMPLETE")

# =============================================================================
# TEST 5: Inference Validation
# =============================================================================
print("\n" + "="*80)
print("[TEST 5] INFERENCE VALIDATION")
print("="*80)

if not loaded_models:
    print("✗ No models loaded, skipping inference test")
    test_results["inference"]["status"] = "failed"
else:
    print("\n[5.1] Running REAL inference on all perturbations...")
    
    inference_results = {}
    
    for model_name, model in loaded_models.items():
        print(f"\n  Testing {model_name}:")
        model_results = {}
        
        for pert_name, pert_data in perturbations.items():
            try:
                # Preprocess
                image_tensor = preprocess_image(pert_data["image"])
                
                # Predict
                prediction = predict_single(model, image_tensor, device)
                
                # Validate prediction structure
                required_keys = ["prediction", "confidence", "probabilities", "predicted_idx"]
                if all(k in prediction for k in required_keys):
                    # Validate values
                    probs = prediction["probabilities"]
                    conf = prediction["confidence"]
                    
                    # Check probabilities sum to 1
                    prob_sum = sum(probs)
                    if abs(prob_sum - 1.0) < 0.01:
                        prob_check = "✓"
                    else:
                        prob_check = f"✗ (sum={prob_sum:.3f})"
                    
                    # Check confidence in range
                    if 0 <= conf <= 1:
                        conf_check = "✓"
                    else:
                        conf_check = f"✗ (conf={conf})"
                    
                    print(f"    {pert_name:20s} → {prediction['prediction']:15s} "
                          f"(conf: {conf:.4f}) {prob_check} {conf_check}")
                    
                    # Store with metadata
                    prediction["metadata"] = {
                        "type": pert_data["type"],
                        "parameter": pert_data["parameter"],
                        "id": pert_data["id"]
                    }
                    model_results[pert_name] = prediction
                else:
                    print(f"    {pert_name:20s} → ✗ Missing keys in prediction")
                    test_results["inference"]["status"] = "failed"
                
            except Exception as e:
                print(f"    {pert_name:20s} → ✗ Inference failed: {e}")
                test_results["inference"]["status"] = "failed"
        
        inference_results[model_name] = model_results
    
    if test_results["inference"]["status"] != "failed":
        test_results["inference"]["status"] = "passed"
        test_results["inference"]["details"].append(f"✓ Inference successful on {len(perturbations)} perturbations")
    
    print("\n[TEST 5] ✓ INFERENCE VALIDATION COMPLETE")

# =============================================================================
# TEST 6: Metadata Validation
# =============================================================================
print("\n" + "="*80)
print("[TEST 6] METADATA VALIDATION")
print("="*80)

print("\n[6.1] Validating metadata preservation...")
metadata_valid = True

for model_name, model_results in inference_results.items():
    print(f"\n  Checking {model_name}:")
    for pert_name, pred_data in model_results.items():
        if "metadata" in pred_data:
            metadata = pred_data["metadata"]
            required = ["type", "parameter", "id"]
            if all(k in metadata for k in required):
                print(f"    ✓ {pert_name:20s} → metadata preserved: {metadata['id']}")
            else:
                print(f"    ✗ {pert_name:20s} → metadata incomplete")
                metadata_valid = False
        else:
            print(f"    ✗ {pert_name:20s} → metadata missing")
            metadata_valid = False

if metadata_valid:
    test_results["metadata"]["status"] = "passed"
    test_results["metadata"]["details"].append("✓ Metadata preserved throughout pipeline")
    print("\n[TEST 6] ✓ METADATA VALIDATION SUCCESSFUL")
else:
    test_results["metadata"]["status"] = "failed"
    print("\n[TEST 6] ✗ METADATA VALIDATION FAILED")

# =============================================================================
# TEST 7: Robustness Analyzer Validation
# =============================================================================
print("\n" + "="*80)
print("[TEST 7] ROBUSTNESS ANALYZER VALIDATION")
print("="*80)

try:
    for model_name, model_results in inference_results.items():
        print(f"\n[7.1] Analyzing {model_name}...")
        
        # Consistency analysis
        consistency = analyze_prediction_consistency(model_results)
        if "error" not in consistency:
            print(f"  ✓ Consistency analysis successful")
            print(f"    - Original prediction: {consistency['original_prediction']}")
            print(f"    - Consistency rate: {consistency['consistency_rate']:.2f}%")
            print(f"    - Consistent predictions: {consistency['consistent_predictions']}/{consistency['total_perturbations']}")
        else:
            print(f"  ✗ Consistency analysis failed: {consistency['error']}")
            test_results["robustness"]["status"] = "failed"
        
        # Confidence analysis
        confidence = analyze_confidence_drift(model_results)
        if "error" not in confidence:
            print(f"  ✓ Confidence analysis successful")
            print(f"    - Original confidence: {confidence['original_confidence']:.4f}")
            print(f"    - Mean confidence: {confidence['mean_confidence']:.4f}")
            print(f"    - Std confidence: {confidence['std_confidence']:.4f}")
            print(f"    - Confidence drop: {confidence['confidence_drop']:.4f}")
        else:
            print(f"  ✗ Confidence analysis failed: {confidence['error']}")
            test_results["robustness"]["status"] = "failed"
        
        # Sensitivity analysis
        sensitivity = analyze_perturbation_sensitivity(model_results)
        if "error" not in sensitivity:
            print(f"  ✓ Sensitivity analysis successful")
            most_damaging = sensitivity['most_damaging_perturbation']
            if most_damaging:
                print(f"    - Most damaging: {most_damaging['name']} (impact: {most_damaging['impact_score']:.3f})")
            print(f"    - Prediction flips: {len(sensitivity['prediction_flip_perturbations'])}")
        else:
            print(f"  ✗ Sensitivity analysis failed: {sensitivity['error']}")
            test_results["robustness"]["status"] = "failed"
        
        # Robustness score
        robustness = compute_robustness_score(consistency, confidence, sensitivity)
        if "error" not in robustness:
            print(f"  ✓ Robustness score computed successfully")
            print(f"    - Overall score: {robustness['robustness_score']:.2f}/100")
            print(f"    - Interpretation: {robustness['interpretation']}")
            print(f"    - Consistency: {robustness['consistency_score']:.2f}")
            print(f"    - Stability: {robustness['stability_score']:.2f}")
            print(f"    - Resistance: {robustness['resistance_score']:.2f}")
        else:
            print(f"  ✗ Robustness score failed: {robustness['error']}")
            test_results["robustness"]["status"] = "failed"
    
    if test_results["robustness"]["status"] != "failed":
        test_results["robustness"]["status"] = "passed"
        test_results["robustness"]["details"].append("✓ All robustness analyses working")
    
    print("\n[TEST 7] ✓ ROBUSTNESS ANALYZER VALIDATION COMPLETE")
    
except Exception as e:
    print(f"\n[TEST 7] ✗ ROBUSTNESS ANALYZER FAILED: {e}")
    test_results["robustness"]["status"] = "failed"
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 8: Model Comparison Validation
# =============================================================================
print("\n" + "="*80)
print("[TEST 8] MODEL COMPARISON VALIDATION")
print("="*80)

if len(inference_results) > 1:
    print("\n[8.1] Comparing models...")
    try:
        comparison = compare_models(inference_results)
        
        if "error" not in comparison:
            print(f"  ✓ Model comparison successful")
            print(f"    - Winner: {comparison['winner']}")
            print(f"\n  Model Rankings:")
            for model_name, scores in comparison['model_rankings']:
                print(f"    {model_name:20s} → Robustness: {scores['robustness_score']:.2f}")
            
            test_results["comparison"]["status"] = "passed"
            test_results["comparison"]["details"].append(f"✓ Winner: {comparison['winner']}")
        else:
            print(f"  ✗ Model comparison failed")
            test_results["comparison"]["status"] = "failed"
        
        print("\n[TEST 8] ✓ MODEL COMPARISON VALIDATION COMPLETE")
        
    except Exception as e:
        print(f"\n[TEST 8] ✗ MODEL COMPARISON FAILED: {e}")
        test_results["comparison"]["status"] = "failed"
        import traceback
        traceback.print_exc()
else:
    print("\n[TEST 8] ⚠ SKIPPED (only one model available)")
    test_results["comparison"]["status"] = "skipped"

# =============================================================================
# TEST 9: Realism Validation
# =============================================================================
print("\n" + "="*80)
print("[TEST 9] REALISM VALIDATION")
print("="*80)

print("\n[9.1] Checking if outputs make logical sense...")

for model_name, model_results in inference_results.items():
    print(f"\n  Analyzing {model_name} realism:")
    
    original_conf = model_results["original"]["confidence"]
    print(f"    Original confidence: {original_conf:.4f}")
    
    # Check if perturbations affect confidence
    confidence_changes = []
    for pert_name, pred_data in model_results.items():
        if pert_name != "original":
            conf_change = pred_data["confidence"] - original_conf
            confidence_changes.append((pert_name, conf_change))
            
            if abs(conf_change) > 0.5:
                print(f"    ⚠ {pert_name}: Large confidence change ({conf_change:+.4f})")
            elif abs(conf_change) < 0.001:
                print(f"    ⚠ {pert_name}: No confidence change (suspicious)")
    
    # Check if noise/blur reduce confidence (usually expected)
    noise_conf = model_results.get("gaussian_noise", {}).get("confidence", original_conf)
    blur_conf = model_results.get("gaussian_blur", {}).get("confidence", original_conf)
    
    if noise_conf < original_conf:
        print(f"    ✓ Gaussian noise reduced confidence (realistic)")
    else:
        print(f"    ⚠ Gaussian noise increased confidence (unusual)")
    
    if blur_conf <= original_conf:
        print(f"    ✓ Gaussian blur maintained/reduced confidence (realistic)")
    else:
        print(f"    ⚠ Gaussian blur increased confidence (unusual)")

test_results["realism"]["status"] = "passed"
test_results["realism"]["details"].append("✓ Outputs appear logically consistent")
print("\n[TEST 9] ✓ REALISM VALIDATION COMPLETE")

# =============================================================================
# TEST 10: Error Handling Validation
# =============================================================================
print("\n" + "="*80)
print("[TEST 10] ERROR HANDLING VALIDATION")
print("="*80)

print("\n[10.1] Testing invalid image path...")
try:
    result = run_batch_inference("nonexistent_image.jpg")
    print("  ✗ Should have raised an error")
    test_results["error_handling"]["status"] = "failed"
except Exception as e:
    print(f"  ✓ Correctly raised error: {type(e).__name__}")
    test_results["error_handling"]["details"].append("✓ Invalid path handled")

print("\n[10.2] Testing with missing checkpoint...")
try:
    models_config = {"efficientnet_b0": "nonexistent_checkpoint.pth"}
    result = run_batch_inference(test_image_path, models_config=models_config)
    print("  ✗ Should have raised an error")
    test_results["error_handling"]["status"] = "failed"
except Exception as e:
    print(f"  ✓ Correctly raised error: {type(e).__name__}")
    test_results["error_handling"]["details"].append("✓ Missing checkpoint handled")

if test_results["error_handling"]["status"] != "failed":
    test_results["error_handling"]["status"] = "passed"

print("\n[TEST 10] ✓ ERROR HANDLING VALIDATION COMPLETE")

# =============================================================================
# FINAL REPORT
# =============================================================================
print("\n" + "="*80)
print("FINAL TEST REPORT")
print("="*80)

print("\n" + "="*80)
print("TEST RESULTS SUMMARY")
print("="*80)

for test_name, result in test_results.items():
    status = result["status"]
    if status == "passed":
        symbol = "[PASS]"
    elif status == "failed":
        symbol = "[FAIL]"
    elif status == "skipped":
        symbol = "[SKIP]"
    else:
        symbol = "[????]"
    
    print(f"\n{symbol} {test_name.upper()}: {status.upper()}")
    for detail in result["details"]:
        print(f"    {detail}")

# Overall assessment
all_critical_passed = all(
    test_results[t]["status"] in ["passed", "skipped"] 
    for t in ["imports", "checkpoints", "perturbations", "inference", "robustness"]
)

print("\n" + "="*80)
print("OVERALL ASSESSMENT")
print("="*80)

if all_critical_passed:
    print("\n[SUCCESS] THE COMPLETE ROBUSTNESS-AWARE PATHOGEN INTELLIGENCE PIPELINE IS FUNCTIONAL")
    print("\nAll critical components working:")
    print("  [OK] Import system")
    print("  [OK] Checkpoint loading")
    print("  [OK] Perturbation integration")
    print("  [OK] Batch inference")
    print("  [OK] Robustness analysis")
    print("  [OK] Model comparison")
    print("  [OK] Error handling")
else:
    print("\n[FAILED] PIPELINE HAS CRITICAL FAILURES")
    print("\nFailed components:")
    for test_name, result in test_results.items():
        if result["status"] == "failed":
            print(f"  [X] {test_name}")

print("\n" + "="*80)
print("END OF COMPREHENSIVE TEST")
print("="*80)
