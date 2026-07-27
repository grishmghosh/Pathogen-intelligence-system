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

import numpy as np


def build_synthetic_inference_results():
    """Build a multi-sample synthetic inference dictionary representing realistic predictions across all 4 models."""
    np.random.seed(42)
    classes = ["e_coli", "k_pneumoniae", "p_aeruginosa", "s_aureus"]
    perts = ["original", "bright", "dark", "high_contrast", "low_contrast", "gaussian_noise", "gaussian_blur"]
    models = ["efficientnet_b0", "resnet50", "swin_t", "convnext_tiny"]

    results = {m: {} for m in models}

    for model in models:
        for pert in perts:
            samples = []
            sev_factor = 0.0 if pert == "original" else (0.04 if pert in ["bright", "dark", "high_contrast"] else 0.12)

            for i in range(40):  # 40 samples per perturbation = 280 samples per model
                true_cls = classes[i % 4]

                if model == "resnet50":
                    conf = float(np.random.beta(8, 1.2) - sev_factor * 0.25)
                elif model == "swin_t":
                    conf = float(np.random.beta(7, 1.5) - sev_factor * 0.25)
                elif model == "convnext_tiny":
                    conf = float(np.random.beta(6, 1.8) - sev_factor * 0.30)
                else:  # efficientnet_b0
                    conf = float(np.random.beta(5, 2.0) - sev_factor * 0.35)

                conf = float(np.clip(conf, 0.25, 0.99))
                is_correct = bool(np.random.rand() < conf)
                pred_cls = true_cls if is_correct else classes[(classes.index(true_cls) + 1) % 4]
                pred_idx = classes.index(pred_cls)

                probs = [0.05, 0.05, 0.05, 0.05]
                probs[pred_idx] = conf
                rem = (1.0 - conf) / 3.0
                for k in range(4):
                    if k != pred_idx:
                        probs[k] = rem

                samples.append({
                    "prediction": pred_cls,
                    "confidence": conf,
                    "probabilities": [float(p) for p in probs],
                    "predicted_idx": pred_idx,
                    "true_label": true_cls,
                    "correct": int(is_correct),
                    "metadata": {"type": pert, "parameter": 1.0, "id": pert},
                })

            results[model][pert] = samples
    return results


SYNTHETIC_INFERENCE_RESULTS = build_synthetic_inference_results()


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
