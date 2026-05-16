"""
Robustness Analyzer for Pathogen Intelligence System.

This module provides the intelligence layer for analyzing model robustness.
It evaluates prediction stability, confidence reliability, and perturbation sensitivity.

Purpose:
    - Analyze prediction consistency across perturbations
    - Measure confidence drift and stability
    - Identify perturbation vulnerabilities
    - Compare model robustness (EfficientNet vs ResNet)
    - Generate structured robustness reports

Architecture Flow:
    Batch Inference Results → Robustness Analyzer → Intelligence Reports

Philosophy:
    The goal is not just classification accuracy, but robustness-aware intelligence:
    - How stable are predictions under perturbations?
    - How reliable is model confidence?
    - Which perturbations expose model weaknesses?
"""

import os
import sys
import numpy as np
from collections import Counter, defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def analyze_prediction_consistency(model_results):
    """
    Analyze whether predictions remain consistent across perturbations.
    
    Prediction consistency is a key robustness metric. A robust model should
    maintain the same prediction even when the input is slightly perturbed.
    
    Args:
        model_results: Dictionary of perturbation results for a single model
                      Format: {perturbation_name: {prediction, confidence, ...}}
    
    Returns:
        Dictionary containing:
            - original_prediction: Prediction on original image
            - consistent_predictions: Number of perturbations with same prediction
            - total_perturbations: Total number of perturbations
            - consistency_rate: Percentage of consistent predictions (0-100)
            - prediction_distribution: Count of each predicted class
            - inconsistent_perturbations: List of perturbations with different predictions
    """
    # Extract predictions
    predictions = {}
    for pert_name, pert_data in model_results.items():
        if "error" not in pert_data and "prediction" in pert_data:
            predictions[pert_name] = pert_data["prediction"]
    
    if not predictions:
        return {"error": "No valid predictions found"}
    
    # Get original prediction as reference
    original_prediction = predictions.get("original", None)
    if original_prediction is None:
        return {"error": "Original prediction not found"}
    
    # Count consistent predictions
    consistent_count = sum(1 for pred in predictions.values() if pred == original_prediction)
    total_count = len(predictions)
    consistency_rate = (consistent_count / total_count) * 100
    
    # Analyze prediction distribution
    prediction_distribution = Counter(predictions.values())
    
    # Identify inconsistent perturbations
    inconsistent_perturbations = [
        name for name, pred in predictions.items() 
        if pred != original_prediction
    ]
    
    return {
        "original_prediction": original_prediction,
        "consistent_predictions": consistent_count,
        "total_perturbations": total_count,
        "consistency_rate": consistency_rate,
        "prediction_distribution": dict(prediction_distribution),
        "inconsistent_perturbations": inconsistent_perturbations
    }


def analyze_confidence_drift(model_results):
    """
    Analyze how confidence scores change across perturbations.
    
    Confidence drift measures how much the model's certainty changes when
    inputs are perturbed. Large drift indicates instability.
    
    Args:
        model_results: Dictionary of perturbation results for a single model
    
    Returns:
        Dictionary containing:
            - original_confidence: Confidence on original image
            - mean_confidence: Average confidence across all perturbations
            - std_confidence: Standard deviation of confidence
            - min_confidence: Minimum confidence observed
            - max_confidence: Maximum confidence observed
            - confidence_drop: Drop from original to minimum
            - confidence_variance: Variance in confidence scores
            - per_perturbation_drift: Confidence change for each perturbation
    """
    # Extract confidences
    confidences = {}
    for pert_name, pert_data in model_results.items():
        if "error" not in pert_data and "confidence" in pert_data:
            confidences[pert_name] = pert_data["confidence"]
    
    if not confidences:
        return {"error": "No valid confidence scores found"}
    
    # Get original confidence
    original_confidence = confidences.get("original", None)
    if original_confidence is None:
        return {"error": "Original confidence not found"}
    
    # Compute statistics
    confidence_values = list(confidences.values())
    mean_confidence = np.mean(confidence_values)
    std_confidence = np.std(confidence_values)
    min_confidence = np.min(confidence_values)
    max_confidence = np.max(confidence_values)
    confidence_variance = np.var(confidence_values)
    
    # Compute confidence drop from original
    confidence_drop = original_confidence - min_confidence
    
    # Per-perturbation drift (relative to original)
    per_perturbation_drift = {
        name: conf - original_confidence 
        for name, conf in confidences.items()
        if name != "original"
    }
    
    return {
        "original_confidence": original_confidence,
        "mean_confidence": mean_confidence,
        "std_confidence": std_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
        "confidence_drop": confidence_drop,
        "confidence_variance": confidence_variance,
        "per_perturbation_drift": per_perturbation_drift
    }


def analyze_perturbation_sensitivity(model_results):
    """
    Identify which perturbations most affect model predictions.
    
    Perturbation sensitivity reveals model vulnerabilities. Some perturbations
    may cause larger confidence drops or prediction changes than others.
    
    Args:
        model_results: Dictionary of perturbation results for a single model
    
    Returns:
        Dictionary containing:
            - most_damaging_perturbation: Perturbation causing largest confidence drop
            - least_damaging_perturbation: Perturbation causing smallest confidence drop
            - perturbation_impact_ranking: Sorted list of perturbations by impact
            - prediction_flip_perturbations: Perturbations causing prediction changes
    """
    # Extract data
    original_prediction = None
    original_confidence = None
    perturbation_impacts = {}
    
    for pert_name, pert_data in model_results.items():
        if "error" in pert_data:
            continue
        
        if pert_name == "original":
            original_prediction = pert_data.get("prediction")
            original_confidence = pert_data.get("confidence")
        else:
            # Compute impact score (confidence drop + prediction change penalty)
            confidence = pert_data.get("confidence", 0)
            prediction = pert_data.get("prediction")
            
            if original_confidence is not None:
                confidence_drop = original_confidence - confidence
                prediction_changed = (prediction != original_prediction)
                
                # Impact score: confidence drop + penalty for prediction flip
                impact_score = confidence_drop + (0.5 if prediction_changed else 0)
                
                perturbation_impacts[pert_name] = {
                    "impact_score": impact_score,
                    "confidence_drop": confidence_drop,
                    "prediction_changed": prediction_changed,
                    "new_prediction": prediction if prediction_changed else None
                }
    
    if not perturbation_impacts:
        return {"error": "Insufficient data for sensitivity analysis"}
    
    # Rank perturbations by impact
    ranked_perturbations = sorted(
        perturbation_impacts.items(),
        key=lambda x: x[1]["impact_score"],
        reverse=True
    )
    
    most_damaging = ranked_perturbations[0] if ranked_perturbations else None
    least_damaging = ranked_perturbations[-1] if ranked_perturbations else None
    
    # Identify prediction flips
    prediction_flip_perturbations = [
        name for name, data in perturbation_impacts.items()
        if data["prediction_changed"]
    ]
    
    return {
        "most_damaging_perturbation": {
            "name": most_damaging[0],
            "impact_score": most_damaging[1]["impact_score"],
            "confidence_drop": most_damaging[1]["confidence_drop"]
        } if most_damaging else None,
        "least_damaging_perturbation": {
            "name": least_damaging[0],
            "impact_score": least_damaging[1]["impact_score"],
            "confidence_drop": least_damaging[1]["confidence_drop"]
        } if least_damaging else None,
        "perturbation_impact_ranking": [
            {"name": name, **data} for name, data in ranked_perturbations
        ],
        "prediction_flip_perturbations": prediction_flip_perturbations
    }


def compute_robustness_score(consistency_analysis, confidence_analysis, sensitivity_analysis):
    """
    Compute an overall robustness score for a model.
    
    The robustness score combines multiple metrics into a single interpretable value:
        - Prediction consistency (40% weight)
        - Confidence stability (40% weight)
        - Perturbation resistance (20% weight)
    
    Score interpretation:
        90-100: Excellent robustness
        80-90:  Good robustness
        70-80:  Moderate robustness
        60-70:  Fair robustness
        <60:    Poor robustness
    
    Args:
        consistency_analysis: Output from analyze_prediction_consistency()
        confidence_analysis: Output from analyze_confidence_drift()
        sensitivity_analysis: Output from analyze_perturbation_sensitivity()
    
    Returns:
        Dictionary containing:
            - robustness_score: Overall score (0-100)
            - consistency_score: Prediction consistency component
            - stability_score: Confidence stability component
            - resistance_score: Perturbation resistance component
            - interpretation: Human-readable interpretation
    """
    # Check for errors
    if "error" in consistency_analysis or "error" in confidence_analysis:
        return {"error": "Cannot compute robustness score due to missing data"}
    
    # Component 1: Prediction consistency (0-100)
    consistency_score = consistency_analysis["consistency_rate"]
    
    # Component 2: Confidence stability (0-100)
    # Penalize high variance and large drops
    original_conf = confidence_analysis["original_confidence"]
    confidence_drop = confidence_analysis["confidence_drop"]
    std_confidence = confidence_analysis["std_confidence"]
    
    # Normalize: less drop and less variance = higher score
    drop_penalty = min(confidence_drop * 100, 50)  # Max 50 point penalty
    variance_penalty = min(std_confidence * 100, 50)  # Max 50 point penalty
    stability_score = max(0, 100 - drop_penalty - variance_penalty)
    
    # Component 3: Perturbation resistance (0-100)
    # Fewer prediction flips = higher resistance
    if "error" not in sensitivity_analysis:
        total_perturbations = len(sensitivity_analysis["perturbation_impact_ranking"])
        flip_count = len(sensitivity_analysis["prediction_flip_perturbations"])
        resistance_score = ((total_perturbations - flip_count) / total_perturbations) * 100
    else:
        resistance_score = 50  # Default if analysis unavailable
    
    # Weighted combination
    robustness_score = (
        0.40 * consistency_score +
        0.40 * stability_score +
        0.20 * resistance_score
    )
    
    # Interpretation
    if robustness_score >= 90:
        interpretation = "Excellent robustness"
    elif robustness_score >= 80:
        interpretation = "Good robustness"
    elif robustness_score >= 70:
        interpretation = "Moderate robustness"
    elif robustness_score >= 60:
        interpretation = "Fair robustness"
    else:
        interpretation = "Poor robustness"
    
    return {
        "robustness_score": robustness_score,
        "consistency_score": consistency_score,
        "stability_score": stability_score,
        "resistance_score": resistance_score,
        "interpretation": interpretation
    }


def compare_models(inference_results):
    """
    Compare robustness between multiple models (e.g., EfficientNet vs ResNet).
    
    Args:
        inference_results: Complete output from batch_inference.run_batch_inference()
    
    Returns:
        Dictionary containing:
            - model_rankings: Models ranked by robustness score
            - comparative_analysis: Side-by-side comparison
            - winner: Most robust model
    """
    model_scores = {}
    
    for model_name, model_results in inference_results.items():
        if "error" in model_results:
            continue
        
        # Run full analysis
        consistency = analyze_prediction_consistency(model_results)
        confidence = analyze_confidence_drift(model_results)
        sensitivity = analyze_perturbation_sensitivity(model_results)
        robustness = compute_robustness_score(consistency, confidence, sensitivity)
        
        model_scores[model_name] = {
            "robustness_score": robustness.get("robustness_score", 0),
            "consistency_rate": consistency.get("consistency_rate", 0),
            "confidence_drop": confidence.get("confidence_drop", 0),
            "prediction_flips": len(sensitivity.get("prediction_flip_perturbations", []))
        }
    
    # Rank models
    ranked_models = sorted(
        model_scores.items(),
        key=lambda x: x[1]["robustness_score"],
        reverse=True
    )
    
    winner = ranked_models[0][0] if ranked_models else None
    
    return {
        "model_rankings": ranked_models,
        "comparative_analysis": model_scores,
        "winner": winner
    }


def generate_robustness_report(inference_results):
    """
    Generate a comprehensive robustness analysis report.
    
    This is the main entry point for robustness analysis. It performs
    complete analysis on all models and generates structured reports.
    
    Args:
        inference_results: Output from batch_inference.run_batch_inference()
    
    Returns:
        Dictionary containing complete robustness analysis for all models
    """
    print("\n" + "="*80)
    print("ROBUSTNESS ANALYSIS")
    print("="*80)
    
    report = {}
    
    for model_name, model_results in inference_results.items():
        print(f"\n[INFO] Analyzing {model_name}...")
        
        if "error" in model_results:
            print(f"[ERROR] Skipping {model_name}: {model_results['error']}")
            report[model_name] = {"error": model_results["error"]}
            continue
        
        # Run all analyses
        consistency = analyze_prediction_consistency(model_results)
        confidence = analyze_confidence_drift(model_results)
        sensitivity = analyze_perturbation_sensitivity(model_results)
        robustness = compute_robustness_score(consistency, confidence, sensitivity)
        
        # Compile model report
        report[model_name] = {
            "consistency_analysis": consistency,
            "confidence_analysis": confidence,
            "sensitivity_analysis": sensitivity,
            "robustness_score": robustness
        }
        
        print(f"[INFO] Robustness score: {robustness.get('robustness_score', 0):.2f}/100")
    
    # Add model comparison
    if len(report) > 1:
        print(f"\n[INFO] Comparing models...")
        comparison = compare_models(inference_results)
        report["model_comparison"] = comparison
        print(f"[INFO] Most robust model: {comparison.get('winner', 'N/A')}")
    
    print("\n" + "="*80)
    print("[INFO] Robustness analysis completed")
    print("="*80 + "\n")
    
    return report


def print_robustness_summary(report):
    """
    Print a human-readable summary of robustness analysis.
    
    Args:
        report: Output from generate_robustness_report()
    """
    print("\n" + "="*80)
    print("ROBUSTNESS SUMMARY")
    print("="*80)
    
    for model_name, model_report in report.items():
        if model_name == "model_comparison":
            continue
        
        print(f"\n{model_name.upper()}:")
        print("-" * 80)
        
        if "error" in model_report:
            print(f"  ERROR: {model_report['error']}")
            continue
        
        # Robustness score
        robustness = model_report.get("robustness_score", {})
        score = robustness.get("robustness_score", 0)
        interpretation = robustness.get("interpretation", "Unknown")
        print(f"  Overall Robustness Score: {score:.2f}/100 ({interpretation})")
        
        # Consistency
        consistency = model_report.get("consistency_analysis", {})
        consistency_rate = consistency.get("consistency_rate", 0)
        print(f"  Prediction Consistency:   {consistency_rate:.2f}%")
        
        # Confidence
        confidence = model_report.get("confidence_analysis", {})
        original_conf = confidence.get("original_confidence", 0)
        conf_drop = confidence.get("confidence_drop", 0)
        print(f"  Original Confidence:      {original_conf:.4f}")
        print(f"  Max Confidence Drop:      {conf_drop:.4f}")
        
        # Sensitivity
        sensitivity = model_report.get("sensitivity_analysis", {})
        most_damaging = sensitivity.get("most_damaging_perturbation", {})
        if most_damaging:
            print(f"  Most Damaging:            {most_damaging.get('name', 'N/A')}")
    
    # Model comparison
    if "model_comparison" in report:
        comparison = report["model_comparison"]
        print(f"\n{'='*80}")
        print("MODEL COMPARISON:")
        print("-" * 80)
        print(f"  Winner: {comparison.get('winner', 'N/A')}")
        
        for model_name, score in comparison.get("model_rankings", []):
            print(f"  {model_name:20s} → Robustness: {score['robustness_score']:.2f}")
    
    print("\n" + "="*80)


def main():
    """
    Test robustness analyzer with sample inference results.
    Demonstrates the complete analysis pipeline.
    """
    print("Robustness Analyzer - Test Mode")
    print("="*80)
    print("\nThis module requires inference results from batch_inference.py")
    print("Run batch_inference.py first to generate predictions.\n")
    
    # Example: Load inference results and analyze
    # In practice, you would:
    # 1. Run batch_inference.run_batch_inference(image_path)
    # 2. Pass results to generate_robustness_report()
    
    print("Example usage:")
    print("-" * 80)
    print("from inference.batch_inference import run_batch_inference")
    print("from analysis.robustness_analyzer import generate_robustness_report")
    print("")
    print("# Run inference")
    print("results = run_batch_inference('path/to/image.jpg')")
    print("")
    print("# Analyze robustness")
    print("report = generate_robustness_report(results)")
    print("print_robustness_summary(report)")
    print("-" * 80)


if __name__ == "__main__":
    main()
