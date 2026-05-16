"""
Complete Intelligence Pipeline Test

This script tests the full pathogen intelligence system:
    1. Perturbation Generation
    2. Batch Inference
    3. Robustness Analysis

Tests the complete flow:
    Image → Perturbations → CNN Inference → Robustness Intelligence
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.batch_inference import run_batch_inference, print_inference_summary
from analysis.robustness_analyzer import generate_robustness_report, print_robustness_summary


def test_intelligence_pipeline(image_path):
    """
    Test the complete intelligence pipeline.
    
    Args:
        image_path: Path to test image
    """
    print("="*80)
    print("PATHOGEN INTELLIGENCE SYSTEM - COMPLETE PIPELINE TEST")
    print("="*80)
    
    # Step 1: Batch Inference
    print("\n[STEP 1] Running batch inference...")
    print("-" * 80)
    
    try:
        inference_results = run_batch_inference(image_path)
        print_inference_summary(inference_results)
    except Exception as e:
        print(f"\n[ERROR] Batch inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Robustness Analysis
    print("\n[STEP 2] Running robustness analysis...")
    print("-" * 80)
    
    try:
        robustness_report = generate_robustness_report(inference_results)
        print_robustness_summary(robustness_report)
    except Exception as e:
        print(f"\n[ERROR] Robustness analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Summary
    print("\n" + "="*80)
    print("PIPELINE TEST COMPLETED SUCCESSFULLY")
    print("="*80)
    
    # Extract key insights
    print("\nKEY INSIGHTS:")
    print("-" * 80)
    
    for model_name, model_report in robustness_report.items():
        if model_name == "model_comparison":
            continue
        
        if "error" not in model_report:
            robustness = model_report.get("robustness_score", {})
            consistency = model_report.get("consistency_analysis", {})
            
            score = robustness.get("robustness_score", 0)
            interpretation = robustness.get("interpretation", "Unknown")
            original_pred = consistency.get("original_prediction", "Unknown")
            consistency_rate = consistency.get("consistency_rate", 0)
            
            print(f"\n{model_name}:")
            print(f"  Prediction: {original_pred}")
            print(f"  Robustness: {score:.2f}/100 ({interpretation})")
            print(f"  Consistency: {consistency_rate:.2f}%")
    
    if "model_comparison" in robustness_report:
        winner = robustness_report["model_comparison"].get("winner", "N/A")
        print(f"\nMost Robust Model: {winner}")
    
    print("\n" + "="*80)


def main():
    """Main entry point for pipeline test."""
    # Get image path
    image_path = input("Enter path to pathogen image: ").strip()
    
    if not image_path:
        print("No path provided. Exiting.")
        return
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Run complete pipeline test
    test_intelligence_pipeline(image_path)


if __name__ == "__main__":
    main()
