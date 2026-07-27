"""
experiments/run_statistical_benchmark.py
=========================================
Multi-Seed Statistical Significance Benchmark Runner
Executes evaluation across multiple random seeds (default: 42, 100, 2024, 777, 999),
computes Mean ± SD, paired t-tests, Wilcoxon tests, Cohen's d effect sizes,
and exports publication-grade Markdown tables.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from analysis.robustness_analyzer import generate_robustness_report
from analysis.statistical_testing import (
    compute_descriptive_stats,
    run_statistical_tests,
    generate_publication_table,
)
from inference.batch_inference import load_model, predict_batch
from perturbations.perturbation_engine import generate_perturbations


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-seed statistical significance benchmark")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--checkpoint-dir", type=str, default=str(ROOT_DIR / "checkpoints"), help="Path to checkpoints")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2024, 777, 999], help="List of random seeds")
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "results" / "experiments"), help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Execution device")
    return parser.parse_args()


def run_benchmark():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = args.image
    if not image_path or not os.path.exists(image_path):
        # Create a synthetic demo image if none provided
        from test_intelligence_pipeline import create_demo_image
        image_path = create_demo_image(output_dir / "synthetic_statistical_demo.png")
        print(f"[INFO] Using synthetic image for statistical benchmark: {image_path}")

    print("=" * 80)
    print("MULTI-SEED STATISTICAL SIGNIFICANCE BENCHMARK")
    print(f"Seeds ({len(args.seeds)}): {args.seeds}")
    print("=" * 80)

    models_to_evaluate = ["efficientnet_b0", "resnet50", "swin_t", "convnext_tiny"]
    seed_scores = {model_name: [] for model_name in models_to_evaluate}

    device = "cuda" if args.device == "auto" and os.environ.get("CUDA_VISIBLE_DEVICES") else ("cpu" if args.device == "auto" else args.device)

    # Evaluate across all seeds
    for seed_idx, seed in enumerate(args.seeds, 1):
        print(f"\n[Seed {seed_idx}/{len(args.seeds)}] Evaluating with seed={seed}...")
        perturbations = generate_perturbations(image_path, seed=seed)

        for model_name in models_to_evaluate:
            model, temperature = load_model(model_name, checkpoint_dir=args.checkpoint_dir, device=device)
            if model is None:
                continue

            temp_val = temperature if temperature is not None else 1.0
            batch_results = predict_batch(model, perturbations, temperature=temp_val, device=device)
            inference_results = {model_name: batch_results}
            report = generate_robustness_report(inference_results)

            score = report[model_name]["robustness_score"]["robustness_score"]
            seed_scores[model_name].append(score)
            print(f"  - {model_name:18s} (Seed {seed}): Robustness Score = {score:.2f}/100")

    # Compute Statistical Analysis
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS & SIGNIFICANCE TESTING")
    print("=" * 80)

    model_analysis = {}
    for model_name, scores in seed_scores.items():
        desc = compute_descriptive_stats(scores)
        model_analysis[model_name] = {"scores": scores, "descriptive": desc}

    # Find winning architecture (highest mean score)
    sorted_models = sorted(model_analysis.items(), key=lambda x: x[1]["descriptive"]["mean"], reverse=True)
    best_model_name = sorted_models[0][0]

    # Compute significance tests against best model
    for model_name, data in model_analysis.items():
        if model_name != best_model_name:
            stat_tests = run_statistical_tests(model_analysis[best_model_name]["scores"], data["scores"])
            data["comparison_vs_best"] = stat_tests
        else:
            data["comparison_vs_best"] = {"is_baseline": True}

    # Format Markdown Publication Table
    pub_table = generate_publication_table(model_analysis)
    print("\n### Statistical Significance Table (Mean ± SD, 95% CI, p-values)\n")
    print(pub_table)

    # Save structured results
    report_output = {
        "seeds": args.seeds,
        "best_model": best_model_name,
        "model_analysis": model_analysis,
        "markdown_table": pub_table,
    }

    report_path = output_dir / "statistical_significance_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report_output, handle, indent=2, ensure_ascii=False)

    table_path = output_dir / "statistical_significance_table.md"
    with open(table_path, "w", encoding="utf-8") as handle:
        handle.write("# Statistical Significance Benchmark Table\n\n" + pub_table + "\n")

    print(f"\nSaved statistical report to: {report_path}")
    print(f"Saved publication table to:  {table_path}")
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
