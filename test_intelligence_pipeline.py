"""
Complete Intelligence Pipeline Test

This script evaluates the full pathogen intelligence system end to end:
1. Perturbation generation
2. Batch inference
3. Robustness analysis

It supports both real-image execution and a built-in demo mode for environments
without a local dataset, making it suitable for reproducible technical demos.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.robustness_analyzer import generate_robustness_report, print_robustness_summary
from inference.batch_inference import print_inference_summary, run_batch_inference


def create_demo_image(output_path: Optional[Path | str] = None) -> str:
    """Create a synthetic image useful for demo runs without the original dataset."""
    rng = np.random.default_rng(42)
    height, width = 224, 224
    base = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)

    y, x = np.indices((height, width))
    pattern = ((np.sin(x / 16.0) + np.cos(y / 18.0)) / 2.0 + 0.5) * 255.0
    pattern = pattern.astype(np.uint8)

    image = np.clip(base * 0.6 + pattern[..., None] * 0.4, 0, 255).astype(np.uint8)

    if output_path is None:
        output_path = str(Path("results") / "demo" / "synthetic_pathogen_demo.png")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output)
    return str(output)


def test_intelligence_pipeline(image_path: str, checkpoint_dir: Optional[str] = None, device: Optional[str] = None, output_dir: Optional[str] = None):
    """Run the complete intelligence pipeline for a single input image."""
    output_dir_path = Path(output_dir or str(Path("results") / "reports"))
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PATHOGEN INTELLIGENCE SYSTEM - COMPLETE PIPELINE TEST")
    print("=" * 80)

    print("\n[STEP 1] Running batch inference...")
    print("-" * 80)

    try:
        inference_results = run_batch_inference(image_path, checkpoint_dir=checkpoint_dir, device=device)
        print_inference_summary(inference_results)
    except Exception as exc:
        print(f"\n[ERROR] Batch inference failed: {exc}")
        import traceback

        traceback.print_exc()
        return None

    print("\n[STEP 2] Running robustness analysis...")
    print("-" * 80)

    try:
        robustness_report = generate_robustness_report(inference_results)
        print_robustness_summary(robustness_report)

        from analysis.robustness_analyzer import print_sensitivity_summary
        components_dict = {
            m: data["robustness_score"]["components"]
            for m, data in robustness_report.items()
            if "robustness_score" in data and "components" in data["robustness_score"]
        }
        if components_dict:
            print_sensitivity_summary(components_dict)
    except Exception as exc:
        print(f"\n[ERROR] Robustness analysis failed: {exc}")
        import traceback

        traceback.print_exc()
        return None

    print("\n" + "=" * 80)
    print("PIPELINE TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)

    summary = {
        "image_path": str(Path(image_path).resolve()),
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()) if checkpoint_dir else None,
        "device": device,
        "robustness_report": robustness_report,
    }

    summary_path = output_dir_path / "pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\nSaved structured summary to: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pathogen intelligence pipeline")
    parser.add_argument("--image", type=str, default=None, help="Path to an input image. If omitted, a synthetic demo image is created.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory containing model checkpoints")
    parser.add_argument("--output-dir", type=str, default=str(Path("results") / "reports"), help="Directory for generated reports")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Execution device")
    parser.add_argument("--demo", action="store_true", help="Force creation of a synthetic demo image")
    return parser.parse_args()


def main():
    """Main entry point for the pipeline test."""
    args = parse_args()

    if args.image:
        image_path = args.image
    elif args.demo:
        image_path = create_demo_image(Path(args.output_dir) / "synthetic_pathogen_demo.png")
    else:
        image_path = create_demo_image(Path(args.output_dir) / "synthetic_pathogen_demo.png")
        print("No image provided. Created a synthetic demo image for reproducible execution.")

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    test_intelligence_pipeline(
        image_path=image_path,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
