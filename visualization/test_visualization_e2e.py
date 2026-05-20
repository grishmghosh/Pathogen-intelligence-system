"""
End-to-end validation test for the visualization subsystem.

Constructs synthetic inference_results and robustness_report data matching
the exact schemas produced by batch_inference.py and robustness_analyzer.py,
then runs every visualization function and verifies that PNG / CSV / JSON
files are created successfully.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- build synthetic inference results matching batch_inference.run_batch_inference() schema ----

SYNTHETIC_INFERENCE_RESULTS = {
    "efficientnet_b0": {
        "original": {
            "prediction": "s_aureus",
            "confidence": 0.92,
            "probabilities": [0.02, 0.03, 0.03, 0.92],
            "predicted_idx": 3,
            "metadata": {"type": "none", "parameter": None, "id": "original"},
        },
        "bright": {
            "prediction": "s_aureus",
            "confidence": 0.88,
            "probabilities": [0.03, 0.04, 0.05, 0.88],
            "predicted_idx": 3,
            "metadata": {"type": "brightness", "parameter": 1.2, "id": "brightness_increase_1.2"},
        },
        "dark": {
            "prediction": "s_aureus",
            "confidence": 0.85,
            "probabilities": [0.04, 0.05, 0.06, 0.85],
            "predicted_idx": 3,
            "metadata": {"type": "brightness", "parameter": 0.8, "id": "brightness_decrease_0.8"},
        },
        "high_contrast": {
            "prediction": "s_aureus",
            "confidence": 0.90,
            "probabilities": [0.02, 0.03, 0.05, 0.90],
            "predicted_idx": 3,
            "metadata": {"type": "contrast", "parameter": 1.15, "id": "contrast_increase_1.15"},
        },
        "low_contrast": {
            "prediction": "e_coli",
            "confidence": 0.55,
            "probabilities": [0.55, 0.15, 0.15, 0.15],
            "predicted_idx": 0,
            "metadata": {"type": "contrast", "parameter": 0.85, "id": "contrast_decrease_0.85"},
        },
        "gaussian_noise": {
            "prediction": "s_aureus",
            "confidence": 0.78,
            "probabilities": [0.05, 0.07, 0.10, 0.78],
            "predicted_idx": 3,
            "metadata": {"type": "noise", "parameter": 8, "id": "gaussian_noise_sigma_8"},
        },
        "gaussian_blur": {
            "prediction": "p_aeruginosa",
            "confidence": 0.45,
            "probabilities": [0.15, 0.15, 0.45, 0.25],
            "predicted_idx": 2,
            "metadata": {"type": "blur", "parameter": 5, "id": "gaussian_blur_kernel_5"},
        },
    },
    "resnet50": {
        "original": {
            "prediction": "s_aureus",
            "confidence": 0.89,
            "probabilities": [0.03, 0.04, 0.04, 0.89],
            "predicted_idx": 3,
            "metadata": {"type": "none", "parameter": None, "id": "original"},
        },
        "bright": {
            "prediction": "s_aureus",
            "confidence": 0.86,
            "probabilities": [0.04, 0.04, 0.06, 0.86],
            "predicted_idx": 3,
            "metadata": {"type": "brightness", "parameter": 1.2, "id": "brightness_increase_1.2"},
        },
        "dark": {
            "prediction": "s_aureus",
            "confidence": 0.80,
            "probabilities": [0.05, 0.06, 0.09, 0.80],
            "predicted_idx": 3,
            "metadata": {"type": "brightness", "parameter": 0.8, "id": "brightness_decrease_0.8"},
        },
        "high_contrast": {
            "prediction": "s_aureus",
            "confidence": 0.87,
            "probabilities": [0.03, 0.04, 0.06, 0.87],
            "predicted_idx": 3,
            "metadata": {"type": "contrast", "parameter": 1.15, "id": "contrast_increase_1.15"},
        },
        "low_contrast": {
            "prediction": "s_aureus",
            "confidence": 0.60,
            "probabilities": [0.10, 0.15, 0.15, 0.60],
            "predicted_idx": 3,
            "metadata": {"type": "contrast", "parameter": 0.85, "id": "contrast_decrease_0.85"},
        },
        "gaussian_noise": {
            "prediction": "k_pneumoniae",
            "confidence": 0.52,
            "probabilities": [0.18, 0.52, 0.15, 0.15],
            "predicted_idx": 1,
            "metadata": {"type": "noise", "parameter": 8, "id": "gaussian_noise_sigma_8"},
        },
        "gaussian_blur": {
            "prediction": "s_aureus",
            "confidence": 0.55,
            "probabilities": [0.10, 0.15, 0.20, 0.55],
            "predicted_idx": 3,
            "metadata": {"type": "blur", "parameter": 5, "id": "gaussian_blur_kernel_5"},
        },
    },
}


def build_synthetic_robustness_report():
    """Generate a robustness report using the actual analyzer on synthetic data."""
    from analysis.robustness_analyzer import generate_robustness_report
    return generate_robustness_report(SYNTHETIC_INFERENCE_RESULTS)


def test_all_visualizations():
    """Run every visualization function and check output files."""
    print("=" * 80)
    print("VISUALIZATION SUBSYSTEM — END-TO-END VALIDATION")
    print("=" * 80)

    # Build robustness report from synthetic data
    print("\n[1/10] Building synthetic robustness report...")
    robustness_report = build_synthetic_robustness_report()
    print("       Robustness report built successfully.")

    results = {}

    # --- Robustness Plots ---
    from visualization.robustness_plots import (
        plot_accuracy_vs_severity,
        plot_confidence_vs_severity,
        plot_model_comparison,
        plot_per_class_robustness_degradation,
    )

    print("\n[2/10] plot_accuracy_vs_severity...")
    path = plot_accuracy_vs_severity(inference_results=SYNTHETIC_INFERENCE_RESULTS)
    results["accuracy_vs_severity"] = path
    print(f"       -> {path}")

    print("\n[3/10] plot_confidence_vs_severity...")
    path = plot_confidence_vs_severity(inference_results=SYNTHETIC_INFERENCE_RESULTS)
    results["confidence_vs_severity"] = path
    print(f"       -> {path}")

    print("\n[4/10] plot_model_comparison...")
    path = plot_model_comparison(robustness_report=robustness_report)
    results["model_comparison"] = path
    print(f"       -> {path}")

    print("\n[5/10] plot_per_class_robustness_degradation...")
    path = plot_per_class_robustness_degradation(inference_results=SYNTHETIC_INFERENCE_RESULTS)
    results["per_class_degradation"] = path
    print(f"       -> {path}")

    # --- Calibration Plots ---
    from visualization.calibration_plots import (
        plot_reliability_diagram,
        plot_confidence_histogram,
        plot_calibration_curve,
        plot_expected_vs_actual_accuracy,
    )

    print("\n[6/10] plot_reliability_diagram...")
    path = plot_reliability_diagram(inference_results=SYNTHETIC_INFERENCE_RESULTS)
    results["reliability_diagram"] = path
    print(f"       -> {path}")

    print("\n[7/10] plot_confidence_histogram...")
    path = plot_confidence_histogram(inference_results=SYNTHETIC_INFERENCE_RESULTS)
    results["confidence_histogram"] = path
    print(f"       -> {path}")

    print("\n[8/10] plot_calibration_curve...")
    path = plot_calibration_curve(inference_results=SYNTHETIC_INFERENCE_RESULTS)
    results["calibration_curve"] = path
    print(f"       -> {path}")

    print("\n[9/10] plot_expected_vs_actual_accuracy...")
    path = plot_expected_vs_actual_accuracy(inference_results=SYNTHETIC_INFERENCE_RESULTS)
    results["expected_vs_actual"] = path
    print(f"       -> {path}")

    # --- Heatmaps ---
    from visualization.heatmaps import (
        plot_prediction_flip_heatmap,
        plot_severity_vs_instability_heatmap,
        plot_model_instability_comparison,
    )

    print("\n[10/10] Heatmaps (3 plots)...")
    path1 = plot_prediction_flip_heatmap(inference_results=SYNTHETIC_INFERENCE_RESULTS)
    results["prediction_flip_heatmap"] = path1
    print(f"        flip heatmap -> {path1}")

    path2 = plot_severity_vs_instability_heatmap(inference_results=SYNTHETIC_INFERENCE_RESULTS)
    results["severity_instability"] = path2
    print(f"        severity instability -> {path2}")

    path3 = plot_model_instability_comparison(inference_results=SYNTHETIC_INFERENCE_RESULTS)
    results["instability_comparison"] = path3
    print(f"        instability comparison -> {path3}")

    # --- Experiment Summary ---
    from visualization.experiment_summary import (
        generate_experiment_summary,
        export_summary_csv,
        export_summary_json,
    )

    print("\n[SUMMARY] Generating experiment summary...")
    summary = generate_experiment_summary(
        inference_results=SYNTHETIC_INFERENCE_RESULTS,
        robustness_report=robustness_report,
    )

    csv_path = export_summary_csv(summary)
    results["summary_csv"] = csv_path
    print(f"  CSV -> {csv_path}")

    json_path = export_summary_json(summary)
    results["summary_json"] = json_path
    print(f"  JSON -> {json_path}")

    # --- Verification ---
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)

    all_ok = True
    for name, path in results.items():
        if path is not None and os.path.isfile(path):
            size = os.path.getsize(path)
            print(f"  [OK] {name:30s} -- {size:>8,} bytes -- {path}")
        else:
            print(f"  [FAIL] {name:30s} -- MISSING or FAILED")
            all_ok = False

    print("\n" + "=" * 80)
    if all_ok:
        print("ALL 13 OUTPUTS GENERATED SUCCESSFULLY.")
    else:
        print("SOME OUTPUTS FAILED — see above.")
    print("=" * 80)

    return all_ok


if __name__ == "__main__":
    success = test_all_visualizations()
    sys.exit(0 if success else 1)
