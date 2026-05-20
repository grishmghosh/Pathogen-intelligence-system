"""
Synthetic validation test for the Disagreement Analysis subsystem.

Verifies:
    1.  Import integrity
    2.  Pairwise agreement matrix computation
    3.  Disagreement detection
    4.  Disagreement statistics
    5.  CSV export
    6.  JSON export
    7.  Multi-model (3+) compatibility
    8.  Edge cases: single model, empty inputs, missing columns
    9.  Column-alias normalisation
    10. Save-path correctness
"""

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
PASS = 0
FAIL = 0


def _report(label, ok, detail=""):
    global PASS, FAIL
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")


# ---------------------------------------------------------------------------
# 1. Import integrity
# ---------------------------------------------------------------------------
def test_imports():
    print("\n=== 1. Import Integrity ===")
    try:
        from analysis.disagreement import (
            compute_agreement_matrix,
            compute_disagreement_statistics,
            load_predictions,
            detect_disagreements,
            export_agreement_matrix_csv,
            export_agreement_matrix_json,
            export_disagreements_csv,
            export_disagreements_json,
            export_statistics_json,
        )
        _report("All public imports resolve", True)
    except Exception as e:
        _report("All public imports resolve", False, str(e))
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 2. load_predictions – list of dicts
# ---------------------------------------------------------------------------
def _make_sample_predictions():
    """4 samples, 2 models, with 1 disagreement on sample_003."""
    return [
        {"sample_id": "s001", "model_name": "efficientnet_b0", "predicted_class": "s_aureus",       "confidence": 0.95},
        {"sample_id": "s001", "model_name": "resnet50",        "predicted_class": "s_aureus",       "confidence": 0.91},
        {"sample_id": "s002", "model_name": "efficientnet_b0", "predicted_class": "e_coli",         "confidence": 0.88},
        {"sample_id": "s002", "model_name": "resnet50",        "predicted_class": "e_coli",         "confidence": 0.84},
        {"sample_id": "s003", "model_name": "efficientnet_b0", "predicted_class": "p_aeruginosa",   "confidence": 0.72},
        {"sample_id": "s003", "model_name": "resnet50",        "predicted_class": "k_pneumoniae",   "confidence": 0.68},
        {"sample_id": "s004", "model_name": "efficientnet_b0", "predicted_class": "s_aureus",       "confidence": 0.99},
        {"sample_id": "s004", "model_name": "resnet50",        "predicted_class": "s_aureus",       "confidence": 0.97},
    ]


def test_load_predictions_from_list():
    print("\n=== 2. load_predictions (list[dict]) ===")
    from analysis.disagreement.disagreement_utils import load_predictions

    preds = _make_sample_predictions()
    df = load_predictions(preds)
    _report("Returns DataFrame", isinstance(df, pd.DataFrame))
    _report("Row count matches", len(df) == len(preds), f"{len(df)} rows")
    _report("Has sample_id column", "sample_id" in df.columns)
    _report("Has confidence column", "confidence" in df.columns)


# ---------------------------------------------------------------------------
# 3. load_predictions – CSV round-trip
# ---------------------------------------------------------------------------
def test_load_predictions_from_csv(tmp_dir):
    print("\n=== 3. load_predictions (CSV) ===")
    from analysis.disagreement.disagreement_utils import load_predictions

    csv_path = tmp_dir / "preds.csv"
    pd.DataFrame(_make_sample_predictions()).to_csv(csv_path, index=False)
    df = load_predictions(csv_path)
    _report("CSV load succeeds", isinstance(df, pd.DataFrame))
    _report("Row count matches", len(df) == 8, f"{len(df)} rows")


# ---------------------------------------------------------------------------
# 4. Column alias normalisation
# ---------------------------------------------------------------------------
def test_column_aliases():
    print("\n=== 4. Column Alias Normalisation ===")
    from analysis.disagreement.disagreement_utils import load_predictions

    data = [
        {"id": "x1", "model": "m1", "prediction": "cls_a", "score": 0.9},
        {"id": "x1", "model": "m2", "prediction": "cls_b", "score": 0.8},
    ]
    df = load_predictions(data)
    _report("Alias 'id' -> 'sample_id'", "sample_id" in df.columns)
    _report("Alias 'model' -> 'model_name'", "model_name" in df.columns)
    _report("Alias 'prediction' -> 'predicted_class'", "predicted_class" in df.columns)
    _report("Alias 'score' -> 'confidence'", "confidence" in df.columns)


# ---------------------------------------------------------------------------
# 5. Pairwise agreement matrix
# ---------------------------------------------------------------------------
def test_agreement_matrix():
    print("\n=== 5. Pairwise Agreement Matrix ===")
    from analysis.disagreement.disagreement_utils import load_predictions
    from analysis.disagreement.agreement_metrics import compute_agreement_matrix

    df = load_predictions(_make_sample_predictions())
    pct, counts = compute_agreement_matrix(df)

    _report("Matrix is square", pct.shape[0] == pct.shape[1])
    _report("Diagonal is 100%", all(pct.loc[m, m] == 100.0 for m in pct.index))
    _report("Symmetric", pct.loc["efficientnet_b0", "resnet50"] == pct.loc["resnet50", "efficientnet_b0"])

    expected_agree = 75.0  # 3 out of 4 samples agree
    actual = pct.loc["efficientnet_b0", "resnet50"]
    _report(
        f"Agreement = {expected_agree}%",
        abs(actual - expected_agree) < 0.01,
        f"actual={actual:.2f}%",
    )


# ---------------------------------------------------------------------------
# 6. Disagreement detection
# ---------------------------------------------------------------------------
def test_disagreement_detection():
    print("\n=== 6. Disagreement Detection ===")
    from analysis.disagreement.disagreement_utils import load_predictions, detect_disagreements

    df = load_predictions(_make_sample_predictions())
    dis = detect_disagreements(df)

    disagreement_ids = dis["sample_id"].unique()
    _report("Returns DataFrame", isinstance(dis, pd.DataFrame))
    _report("Only s003 is disagreement", set(disagreement_ids) == {"s003"}, f"ids={list(disagreement_ids)}")
    _report("2 rows for disagreement sample", len(dis) == 2, f"{len(dis)} rows")


# ---------------------------------------------------------------------------
# 7. Disagreement statistics
# ---------------------------------------------------------------------------
def test_disagreement_statistics():
    print("\n=== 7. Disagreement Statistics ===")
    from analysis.disagreement.disagreement_utils import load_predictions, detect_disagreements
    from analysis.disagreement.agreement_metrics import compute_disagreement_statistics

    df = load_predictions(_make_sample_predictions())
    dis = detect_disagreements(df)
    stats = compute_disagreement_statistics(df, dis)

    _report("total_samples == 4", stats["total_samples"] == 4)
    _report("agreement_count == 3", stats["agreement_count"] == 3)
    _report("disagreement_count == 1", stats["disagreement_count"] == 1)
    _report("agreement_rate == 0.75", abs(stats["agreement_rate"] - 0.75) < 1e-6)
    _report("disagreement_rate == 0.25", abs(stats["disagreement_rate"] - 0.25) < 1e-6)
    _report("total_models == 2", stats["total_models"] == 2)


# ---------------------------------------------------------------------------
# 8. CSV export
# ---------------------------------------------------------------------------
def test_csv_export(tmp_dir):
    print("\n=== 8. CSV Export ===")
    from analysis.disagreement.disagreement_utils import load_predictions, detect_disagreements
    from analysis.disagreement.agreement_metrics import compute_agreement_matrix
    from analysis.disagreement.disagreement_export import (
        export_agreement_matrix_csv,
        export_disagreements_csv,
    )

    df = load_predictions(_make_sample_predictions())
    pct, _ = compute_agreement_matrix(df)
    dis = detect_disagreements(df)

    p1 = export_agreement_matrix_csv(pct, output_dir=tmp_dir)
    p2 = export_disagreements_csv(dis, output_dir=tmp_dir)

    _report("Agreement CSV exists", p1.exists(), str(p1))
    _report("Disagreements CSV exists", p2.exists(), str(p2))

    # Verify CSV is readable
    reloaded = pd.read_csv(p2)
    _report("Disagreements CSV re-readable", len(reloaded) == 2)


# ---------------------------------------------------------------------------
# 9. JSON export
# ---------------------------------------------------------------------------
def test_json_export(tmp_dir):
    print("\n=== 9. JSON Export ===")
    from analysis.disagreement.disagreement_utils import load_predictions, detect_disagreements
    from analysis.disagreement.agreement_metrics import (
        compute_agreement_matrix,
        compute_disagreement_statistics,
    )
    from analysis.disagreement.disagreement_export import (
        export_agreement_matrix_json,
        export_disagreements_json,
        export_statistics_json,
    )

    df = load_predictions(_make_sample_predictions())
    pct, _ = compute_agreement_matrix(df)
    dis = detect_disagreements(df)
    stats = compute_disagreement_statistics(df, dis)

    p1 = export_agreement_matrix_json(pct, output_dir=tmp_dir)
    p2 = export_disagreements_json(dis, output_dir=tmp_dir)
    p3 = export_statistics_json(stats, output_dir=tmp_dir)

    _report("Agreement JSON exists", p1.exists())
    _report("Disagreements JSON exists", p2.exists())
    _report("Statistics JSON exists", p3.exists())

    # Verify JSON parseable
    with open(p3, "r") as f:
        loaded_stats = json.load(f)
    _report("Statistics JSON parseable", loaded_stats["total_samples"] == 4)


# ---------------------------------------------------------------------------
# 10. Multi-model (3+) compatibility
# ---------------------------------------------------------------------------
def test_multi_model():
    print("\n=== 10. Multi-Model (3+) Compatibility ===")
    from analysis.disagreement.disagreement_utils import load_predictions, detect_disagreements
    from analysis.disagreement.agreement_metrics import compute_agreement_matrix, compute_disagreement_statistics

    preds = [
        {"sample_id": "s1", "model_name": "m1", "predicted_class": "A", "confidence": 0.9},
        {"sample_id": "s1", "model_name": "m2", "predicted_class": "A", "confidence": 0.8},
        {"sample_id": "s1", "model_name": "m3", "predicted_class": "B", "confidence": 0.7},
        {"sample_id": "s2", "model_name": "m1", "predicted_class": "C", "confidence": 0.6},
        {"sample_id": "s2", "model_name": "m2", "predicted_class": "C", "confidence": 0.5},
        {"sample_id": "s2", "model_name": "m3", "predicted_class": "C", "confidence": 0.4},
    ]
    df = load_predictions(preds)
    pct, _ = compute_agreement_matrix(df)
    dis = detect_disagreements(df)
    stats = compute_disagreement_statistics(df, dis)

    _report("Matrix shape 3×3", pct.shape == (3, 3))
    _report("m1-m2 agree 100%", abs(pct.loc["m1", "m2"] - 100.0) < 0.01)
    _report("m1-m3 agree 50%", abs(pct.loc["m1", "m3"] - 50.0) < 0.01)
    _report("1 disagreement sample", stats["disagreement_count"] == 1)
    _report("1 agreement sample", stats["agreement_count"] == 1)


# ---------------------------------------------------------------------------
# 11. Edge cases
# ---------------------------------------------------------------------------
def test_edge_cases():
    print("\n=== 11. Edge Cases ===")
    from analysis.disagreement.disagreement_utils import load_predictions, detect_disagreements

    # Empty list
    try:
        load_predictions([])
        _report("Empty list raises ValueError", False)
    except ValueError:
        _report("Empty list raises ValueError", True)

    # Missing required column
    try:
        load_predictions([{"sample_id": "s1", "predicted_class": "A"}])
        _report("Missing model_name raises ValueError", False)
    except ValueError:
        _report("Missing model_name raises ValueError", True)

    # Single model -> detect_disagreements raises
    try:
        df = load_predictions([
            {"sample_id": "s1", "model_name": "m1", "predicted_class": "A"},
        ])
        detect_disagreements(df)
        _report("Single model raises ValueError", False)
    except ValueError:
        _report("Single model raises ValueError", True)

    # Missing confidence column (should gracefully add NaN)
    df = load_predictions([
        {"sample_id": "s1", "model_name": "m1", "predicted_class": "A"},
        {"sample_id": "s1", "model_name": "m2", "predicted_class": "B"},
    ])
    _report("Missing confidence -> NaN fill", "confidence" in df.columns and df["confidence"].isna().all())

    # Nonexistent CSV path
    try:
        load_predictions("/nonexistent/path/to/file.csv")
        _report("Bad CSV path raises FileNotFoundError", False)
    except FileNotFoundError:
        _report("Bad CSV path raises FileNotFoundError", True)

    # Full agreement (no disagreements)
    df = load_predictions([
        {"sample_id": "s1", "model_name": "m1", "predicted_class": "A"},
        {"sample_id": "s1", "model_name": "m2", "predicted_class": "A"},
    ])
    dis = detect_disagreements(df)
    _report("Full agreement -> empty disagreements", len(dis) == 0)


# ---------------------------------------------------------------------------
# 12. Default save paths
# ---------------------------------------------------------------------------
def test_default_save_paths():
    print("\n=== 12. Default Save Paths ===")
    from analysis.disagreement.disagreement_export import _DEFAULT_OUTPUT_DIR
    expected = Path("results") / "disagreement"
    _report("Default output dir correct", _DEFAULT_OUTPUT_DIR == expected, str(_DEFAULT_OUTPUT_DIR))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  DISAGREEMENT ANALYSIS — SYNTHETIC VALIDATION SUITE")
    print("=" * 70)

    tmp_dir = Path("results") / "disagreement" / "_test_scratch"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        test_imports()
        test_load_predictions_from_list()
        test_load_predictions_from_csv(tmp_dir)
        test_column_aliases()
        test_agreement_matrix()
        test_disagreement_detection()
        test_disagreement_statistics()
        test_csv_export(tmp_dir)
        test_json_export(tmp_dir)
        test_multi_model()
        test_edge_cases()
        test_default_save_paths()
    finally:
        # Clean up scratch files
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f"  RESULTS:  {PASS} passed,  {FAIL} failed")
    print("=" * 70)

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
