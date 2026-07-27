"""
Synthetic validation test for Step 2: Confidence-Aware Disagreement Analysis.

Verifies:
    1.  Import integrity (all Step 2 APIs)
    2.  Step 1 backward compatibility
    3.  Confidence gap computation
    4.  Pairwise confidence analysis
    5.  Confidence spread summary
    6.  Severity classification - pairwise
    7.  Severity classification - sample roll-up
    8.  Severity summary statistics
    9.  Disagreement scoring - score ranges
   10.  Score summary
   11.  CSV exports (confidence + severity)
   12.  JSON exports (confidence + severity)
   13.  Multi-model (3+) compatibility
   14.  Edge cases (missing confidence, single model, empty, full agreement)
   15.  Severity threshold correctness
   16.  Architecture: no redundancy between Step 1 and Step 2
"""

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

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
# Test data factories
# ---------------------------------------------------------------------------

def _two_model_predictions():
    """4 samples, 2 models: s003 disagrees, rest agree."""
    return [
        {"sample_id": "s001", "model_name": "efficientnet_b0", "predicted_class": "s_aureus",     "confidence": 0.95},
        {"sample_id": "s001", "model_name": "resnet50",        "predicted_class": "s_aureus",     "confidence": 0.91},
        {"sample_id": "s002", "model_name": "efficientnet_b0", "predicted_class": "e_coli",       "confidence": 0.88},
        {"sample_id": "s002", "model_name": "resnet50",        "predicted_class": "e_coli",       "confidence": 0.84},
        {"sample_id": "s003", "model_name": "efficientnet_b0", "predicted_class": "p_aeruginosa", "confidence": 0.92},
        {"sample_id": "s003", "model_name": "resnet50",        "predicted_class": "k_pneumoniae", "confidence": 0.88},
        {"sample_id": "s004", "model_name": "efficientnet_b0", "predicted_class": "s_aureus",     "confidence": 0.99},
        {"sample_id": "s004", "model_name": "resnet50",        "predicted_class": "s_aureus",     "confidence": 0.97},
    ]


def _severity_test_predictions():
    """Samples designed to hit every severity tier."""
    return [
        # low: same class, small gap
        {"sample_id": "low1",  "model_name": "m1", "predicted_class": "A", "confidence": 0.90},
        {"sample_id": "low1",  "model_name": "m2", "predicted_class": "A", "confidence": 0.88},
        # moderate: different classes, one model uncertain
        {"sample_id": "mod1",  "model_name": "m1", "predicted_class": "A", "confidence": 0.40},
        {"sample_id": "mod1",  "model_name": "m2", "predicted_class": "B", "confidence": 0.35},
        # high: different classes, both above uncertain but not both high
        {"sample_id": "high1", "model_name": "m1", "predicted_class": "A", "confidence": 0.65},
        {"sample_id": "high1", "model_name": "m2", "predicted_class": "B", "confidence": 0.70},
        # critical: different classes, both highly confident
        {"sample_id": "crit1", "model_name": "m1", "predicted_class": "A", "confidence": 0.95},
        {"sample_id": "crit1", "model_name": "m2", "predicted_class": "B", "confidence": 0.92},
    ]


def _three_model_predictions():
    """3 models, 3 samples with mixed agreement."""
    return [
        {"sample_id": "s1", "model_name": "m1", "predicted_class": "A", "confidence": 0.90},
        {"sample_id": "s1", "model_name": "m2", "predicted_class": "A", "confidence": 0.85},
        {"sample_id": "s1", "model_name": "m3", "predicted_class": "B", "confidence": 0.80},
        {"sample_id": "s2", "model_name": "m1", "predicted_class": "C", "confidence": 0.60},
        {"sample_id": "s2", "model_name": "m2", "predicted_class": "C", "confidence": 0.55},
        {"sample_id": "s2", "model_name": "m3", "predicted_class": "C", "confidence": 0.50},
        {"sample_id": "s3", "model_name": "m1", "predicted_class": "A", "confidence": 0.95},
        {"sample_id": "s3", "model_name": "m2", "predicted_class": "B", "confidence": 0.93},
        {"sample_id": "s3", "model_name": "m3", "predicted_class": "C", "confidence": 0.91},
    ]


# ===========================================================================
# Tests
# ===========================================================================

def test_01_imports():
    print("\n=== 1. Import Integrity (Step 2 APIs) ===")
    try:
        from analysis.disagreement import (
            # Step 2 - confidence
            compute_confidence_gaps,
            compute_pairwise_confidence_analysis,
            compute_confidence_spread_summary,
            # Step 2 - severity
            classify_pairwise_severity,
            classify_sample_severity,
            compute_severity_summary,
            # Step 2 - scoring
            compute_disagreement_scores,
            compute_score_summary,
            # Step 2 - exports
            export_confidence_gaps_csv,
            export_confidence_gaps_json,
            export_pairwise_analysis_csv,
            export_pairwise_analysis_json,
            export_confidence_summary_json,
            export_severity_csv,
            export_severity_json,
            export_severity_summary_json,
            export_score_summary_json,
        )
        _report("All Step 2 imports resolve", True)
    except Exception as e:
        _report("All Step 2 imports resolve", False, str(e))
        traceback.print_exc()


def test_02_step1_backward_compat():
    print("\n=== 2. Step 1 Backward Compatibility ===")
    try:
        from analysis.disagreement import (
            load_predictions, detect_disagreements,
            compute_agreement_matrix, compute_disagreement_statistics,
            export_agreement_matrix_csv, export_statistics_json,
        )
        df = load_predictions(_two_model_predictions())
        pct, _ = compute_agreement_matrix(df)
        dis = detect_disagreements(df)
        stats = compute_disagreement_statistics(df, dis)
        _report("Step 1 pipeline still works", stats["total_samples"] == 4)
    except Exception as e:
        _report("Step 1 pipeline still works", False, str(e))


def test_03_confidence_gaps():
    print("\n=== 3. Confidence Gap Computation ===")
    from analysis.disagreement.disagreement_utils import load_predictions
    from analysis.disagreement.confidence_disagreement import compute_confidence_gaps

    df = load_predictions(_two_model_predictions())
    gaps = compute_confidence_gaps(df)

    _report("Returns DataFrame", isinstance(gaps, pd.DataFrame))
    _report("4 samples", len(gaps) == 4, f"got {len(gaps)}")
    _report("Has confidence_gap column", "confidence_gap" in gaps.columns)
    _report("Has classes_agree column", "classes_agree" in gaps.columns)

    s003 = gaps[gaps["sample_id"] == "s003"].iloc[0]
    _report("s003 classes_agree=False", s003["classes_agree"] == False)
    _report("s003 gap = 0.04", abs(s003["confidence_gap"] - 0.04) < 1e-6, f"got {s003['confidence_gap']:.6f}")

    s001 = gaps[gaps["sample_id"] == "s001"].iloc[0]
    _report("s001 classes_agree=True", s001["classes_agree"] == True)


def test_04_pairwise_confidence():
    print("\n=== 4. Pairwise Confidence Analysis ===")
    from analysis.disagreement.disagreement_utils import load_predictions
    from analysis.disagreement.confidence_disagreement import compute_pairwise_confidence_analysis

    df = load_predictions(_two_model_predictions())
    pw = compute_pairwise_confidence_analysis(df)

    _report("Returns DataFrame", isinstance(pw, pd.DataFrame))
    _report("4 rows (4 samples x 1 pair)", len(pw) == 4, f"got {len(pw)}")
    _report("Has confidence_diff", "confidence_diff" in pw.columns)
    _report("Has classes_agree", "classes_agree" in pw.columns)

    s003_row = pw[pw["sample_id"] == "s003"].iloc[0]
    _report("s003 classes disagree", s003_row["classes_agree"] == False)


def test_05_confidence_spread_summary():
    print("\n=== 5. Confidence Spread Summary ===")
    from analysis.disagreement.disagreement_utils import load_predictions
    from analysis.disagreement.confidence_disagreement import (
        compute_pairwise_confidence_analysis, compute_confidence_spread_summary,
    )

    df = load_predictions(_two_model_predictions())
    pw = compute_pairwise_confidence_analysis(df)
    summary = compute_confidence_spread_summary(pw)

    _report("Is dict", isinstance(summary, dict))
    _report("Has mean_confidence_diff", "mean_confidence_diff" in summary)
    _report("Has total_pairs", summary["total_pairs"] == 4)
    _report("Has disagree_pairs", summary["disagree_pairs"] == 1)
    _report("Has agree_pairs", summary["agree_pairs"] == 3)


def test_06_severity_classification():
    print("\n=== 6. Severity Classification (Pairwise) ===")
    from analysis.disagreement.disagreement_utils import load_predictions
    from analysis.disagreement.confidence_disagreement import compute_pairwise_confidence_analysis
    from analysis.disagreement.disagreement_severity import classify_pairwise_severity

    df = load_predictions(_severity_test_predictions())
    pw = compute_pairwise_confidence_analysis(df)
    sev = classify_pairwise_severity(pw)

    _report("Has severity column", "severity" in sev.columns)

    low_row = sev[sev["sample_id"] == "low1"].iloc[0]
    mod_row = sev[sev["sample_id"] == "mod1"].iloc[0]
    high_row = sev[sev["sample_id"] == "high1"].iloc[0]
    crit_row = sev[sev["sample_id"] == "crit1"].iloc[0]

    _report("low1 -> 'low'", low_row["severity"] == "low", f"got '{low_row['severity']}'")
    _report("mod1 -> 'moderate'", mod_row["severity"] == "moderate", f"got '{mod_row['severity']}'")
    _report("high1 -> 'high'", high_row["severity"] == "high", f"got '{high_row['severity']}'")
    _report("crit1 -> 'critical'", crit_row["severity"] == "critical", f"got '{crit_row['severity']}'")


def test_07_sample_severity_rollup():
    print("\n=== 7. Per-Sample Severity Roll-up ===")
    from analysis.disagreement.disagreement_utils import load_predictions
    from analysis.disagreement.confidence_disagreement import compute_pairwise_confidence_analysis
    from analysis.disagreement.disagreement_severity import classify_pairwise_severity, classify_sample_severity

    df = load_predictions(_severity_test_predictions())
    pw = compute_pairwise_confidence_analysis(df)
    sev = classify_pairwise_severity(pw)
    sample_sev = classify_sample_severity(sev)

    _report("Returns DataFrame", isinstance(sample_sev, pd.DataFrame))
    _report("4 sample rows", len(sample_sev) == 4, f"got {len(sample_sev)}")
    _report("Has n_critical column", "n_critical" in sample_sev.columns)

    crit_row = sample_sev[sample_sev["sample_id"] == "crit1"].iloc[0]
    _report("crit1 worst = 'critical'", crit_row["severity"] == "critical")


def test_08_severity_summary():
    print("\n=== 8. Severity Summary Statistics ===")
    from analysis.disagreement.disagreement_utils import load_predictions
    from analysis.disagreement.confidence_disagreement import compute_pairwise_confidence_analysis
    from analysis.disagreement.disagreement_severity import classify_pairwise_severity, compute_severity_summary

    df = load_predictions(_severity_test_predictions())
    pw = compute_pairwise_confidence_analysis(df)
    sev = classify_pairwise_severity(pw)
    summary = compute_severity_summary(sev)

    _report("Is dict", isinstance(summary, dict))
    _report("total_pairs == 4", summary["total_pairs"] == 4)
    _report("severity_counts has all keys", all(k in summary["severity_counts"] for k in ("low", "moderate", "high", "critical")))
    _report("critical_sample_ids includes crit1", "crit1" in summary["critical_sample_ids"])
    _report("high_sample_ids includes high1", "high1" in summary["high_sample_ids"])


def test_09_disagreement_scoring():
    print("\n=== 9. Disagreement Scoring ===")
    from analysis.disagreement.disagreement_utils import load_predictions
    from analysis.disagreement.confidence_disagreement import compute_pairwise_confidence_analysis
    from analysis.disagreement.disagreement_scoring import compute_disagreement_scores

    df = load_predictions(_severity_test_predictions())
    pw = compute_pairwise_confidence_analysis(df)
    scored = compute_disagreement_scores(pw)

    _report("Has disagreement_score column", "disagreement_score" in scored.columns)

    scores = scored["disagreement_score"]
    _report("All scores >= 0", (scores >= 0).all())
    _report("All scores <= 1", (scores <= 1).all())

    low_score = scored[scored["sample_id"] == "low1"]["disagreement_score"].iloc[0]
    crit_score = scored[scored["sample_id"] == "crit1"]["disagreement_score"].iloc[0]
    _report("low1 score < crit1 score", low_score < crit_score, f"low={low_score:.4f}, crit={crit_score:.4f}")

    agree_scores = scored[scored["classes_agree"]]["disagreement_score"]
    disagree_scores = scored[~scored["classes_agree"]]["disagreement_score"]
    _report(
        "Agreement scores < disagreement scores (means)",
        agree_scores.mean() < disagree_scores.mean(),
        f"agree_mean={agree_scores.mean():.4f}, disagree_mean={disagree_scores.mean():.4f}",
    )


def test_10_score_summary():
    print("\n=== 10. Score Summary ===")
    from analysis.disagreement.disagreement_utils import load_predictions
    from analysis.disagreement.confidence_disagreement import compute_pairwise_confidence_analysis
    from analysis.disagreement.disagreement_scoring import compute_disagreement_scores, compute_score_summary

    df = load_predictions(_severity_test_predictions())
    pw = compute_pairwise_confidence_analysis(df)
    scored = compute_disagreement_scores(pw)
    summary = compute_score_summary(scored)

    _report("Is dict", isinstance(summary, dict))
    _report("Has mean_score", "mean_score" in summary)
    _report("total_scored == 4", summary["total_scored"] == 4)
    _report("min_score <= max_score", summary["min_score"] <= summary["max_score"])


def test_11_csv_exports(tmp_dir):
    print("\n=== 11. CSV Exports ===")
    from analysis.disagreement.disagreement_utils import load_predictions
    from analysis.disagreement.confidence_disagreement import (
        compute_confidence_gaps, compute_pairwise_confidence_analysis,
    )
    from analysis.disagreement.disagreement_severity import classify_pairwise_severity
    from analysis.disagreement.disagreement_scoring import (
        compute_disagreement_scores,
        export_confidence_gaps_csv, export_pairwise_analysis_csv,
        export_severity_csv,
    )

    df = load_predictions(_two_model_predictions())
    gaps = compute_confidence_gaps(df)
    pw = compute_pairwise_confidence_analysis(df)
    sev = classify_pairwise_severity(pw)
    scored = compute_disagreement_scores(sev)

    p1 = export_confidence_gaps_csv(gaps, output_dir=tmp_dir)
    p2 = export_pairwise_analysis_csv(scored, output_dir=tmp_dir)
    p3 = export_severity_csv(scored, output_dir=tmp_dir)

    _report("Confidence gaps CSV exists", p1.exists())
    _report("Pairwise analysis CSV exists", p2.exists())
    _report("Severity CSV exists", p3.exists())

    reload = pd.read_csv(p3)
    _report("Severity CSV re-readable", "severity" in reload.columns)


def test_12_json_exports(tmp_dir):
    print("\n=== 12. JSON Exports ===")
    from analysis.disagreement.disagreement_utils import load_predictions
    from analysis.disagreement.confidence_disagreement import (
        compute_confidence_gaps, compute_pairwise_confidence_analysis,
        compute_confidence_spread_summary,
    )
    from analysis.disagreement.disagreement_severity import (
        classify_pairwise_severity, compute_severity_summary,
    )
    from analysis.disagreement.disagreement_scoring import (
        compute_disagreement_scores, compute_score_summary,
        export_confidence_gaps_json, export_pairwise_analysis_json,
        export_confidence_summary_json, export_severity_json,
        export_severity_summary_json, export_score_summary_json,
    )

    df = load_predictions(_two_model_predictions())
    gaps = compute_confidence_gaps(df)
    pw = compute_pairwise_confidence_analysis(df)
    spread = compute_confidence_spread_summary(pw)
    sev = classify_pairwise_severity(pw)
    sev_summ = compute_severity_summary(sev)
    scored = compute_disagreement_scores(sev)
    sc_summ = compute_score_summary(scored)

    p1 = export_confidence_gaps_json(gaps, output_dir=tmp_dir)
    p2 = export_pairwise_analysis_json(scored, output_dir=tmp_dir)
    p3 = export_confidence_summary_json(spread, output_dir=tmp_dir)
    p4 = export_severity_json(scored, output_dir=tmp_dir)
    p5 = export_severity_summary_json(sev_summ, output_dir=tmp_dir)
    p6 = export_score_summary_json(sc_summ, output_dir=tmp_dir)

    for path, name in [(p1, "gaps"), (p2, "pairwise"), (p3, "spread"),
                        (p4, "severity"), (p5, "sev_summary"), (p6, "score_summary")]:
        _report(f"{name} JSON exists", path.exists())

    # Verify parseability
    with open(p5, "r") as f:
        data = json.load(f)
    _report("Severity summary JSON parseable", "total_pairs" in data)

    with open(p6, "r") as f:
        data = json.load(f)
    _report("Score summary JSON parseable", "mean_score" in data)


def test_13_multi_model():
    print("\n=== 13. Multi-Model (3+) Compatibility ===")
    from analysis.disagreement.disagreement_utils import load_predictions
    from analysis.disagreement.confidence_disagreement import (
        compute_confidence_gaps, compute_pairwise_confidence_analysis,
    )
    from analysis.disagreement.disagreement_severity import classify_pairwise_severity
    from analysis.disagreement.disagreement_scoring import compute_disagreement_scores

    df = load_predictions(_three_model_predictions())
    gaps = compute_confidence_gaps(df)
    pw = compute_pairwise_confidence_analysis(df)
    sev = classify_pairwise_severity(pw)
    scored = compute_disagreement_scores(sev)

    _report("Gaps: 3 samples", len(gaps) == 3, f"got {len(gaps)}")
    # 3 models -> 3 pairs (C(3,2)=3) x 3 samples = 9 rows
    _report("Pairwise: 9 rows", len(pw) == 9, f"got {len(pw)}")
    _report("Severity column present", "severity" in sev.columns)
    _report("Score column present", "disagreement_score" in scored.columns)

    # s2 all agree -> all pairs should be 'low'
    s2_sevs = sev[sev["sample_id"] == "s2"]["severity"].unique()
    _report("s2 (all agree) -> all 'low'", list(s2_sevs) == ["low"], f"got {list(s2_sevs)}")

    # s3 all different classes, all high confidence -> should have 'critical' pairs
    s3_sevs = set(sev[sev["sample_id"] == "s3"]["severity"].unique())
    _report("s3 (all disagree, high conf) has 'critical'", "critical" in s3_sevs, f"got {s3_sevs}")


def test_14_edge_cases():
    print("\n=== 14. Edge Cases ===")
    from analysis.disagreement.disagreement_utils import load_predictions
    from analysis.disagreement.confidence_disagreement import (
        compute_confidence_gaps, compute_pairwise_confidence_analysis,
    )
    from analysis.disagreement.disagreement_severity import classify_pairwise_severity
    from analysis.disagreement.disagreement_scoring import compute_disagreement_scores

    # Empty DataFrame
    try:
        compute_confidence_gaps(pd.DataFrame())
        _report("Empty DF raises ValueError", False)
    except ValueError:
        _report("Empty DF raises ValueError", True)

    # Missing confidence entirely
    try:
        df_no_conf = pd.DataFrame([
            {"sample_id": "s1", "model_name": "m1", "predicted_class": "A"},
            {"sample_id": "s1", "model_name": "m2", "predicted_class": "B"},
        ])
        df_no_conf["confidence"] = np.nan
        compute_confidence_gaps(df_no_conf)
        _report("All-NaN confidence raises ValueError", False)
    except ValueError:
        _report("All-NaN confidence raises ValueError", True)

    # Single model
    try:
        df_single = load_predictions([
            {"sample_id": "s1", "model_name": "m1", "predicted_class": "A", "confidence": 0.9},
        ])
        compute_pairwise_confidence_analysis(df_single)
        _report("Single model raises ValueError", False)
    except ValueError:
        _report("Single model raises ValueError", True)

    # Full agreement -> severity should all be 'low'
    agree_data = [
        {"sample_id": "s1", "model_name": "m1", "predicted_class": "A", "confidence": 0.9},
        {"sample_id": "s1", "model_name": "m2", "predicted_class": "A", "confidence": 0.8},
    ]
    df_agree = load_predictions(agree_data)
    pw = compute_pairwise_confidence_analysis(df_agree)
    sev = classify_pairwise_severity(pw)
    _report("Full agreement -> severity 'low'", sev["severity"].iloc[0] == "low")

    # Empty pairwise -> scoring handles gracefully
    empty_pw = pd.DataFrame(columns=["classes_agree", "confidence_a", "confidence_b", "confidence_diff"])
    scored_empty = compute_disagreement_scores(empty_pw)
    _report("Empty pairwise -> empty scored", len(scored_empty) == 0)


def test_15_severity_thresholds():
    print("\n=== 15. Severity Threshold Correctness ===")
    from analysis.disagreement.disagreement_severity import (
        _classify_single, UNCERTAIN_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD,
    )

    # Same class -> always low
    _report("same class -> low", _classify_single(True, 0.99, 0.01, 0.98) == "low")

    # Different class, both None conf -> moderate
    _report("diff class, no conf -> moderate", _classify_single(False, None, None, None) == "moderate")

    # Different class, one uncertain
    _report("diff class, one uncertain -> moderate",
            _classify_single(False, 0.30, 0.70, 0.40) == "moderate")

    # Different class, both above uncertain, not both high
    _report("diff class, both moderate conf -> high",
            _classify_single(False, 0.60, 0.70, 0.10) == "high")

    # Different class, both high
    _report("diff class, both high -> critical",
            _classify_single(False, 0.85, 0.90, 0.05) == "critical")

    # Boundary: exactly at UNCERTAIN
    _report("boundary: exactly at uncertain -> high",
            _classify_single(False, UNCERTAIN_THRESHOLD, UNCERTAIN_THRESHOLD, 0.0) == "high")

    # Boundary: exactly at HIGH_CONFIDENCE
    _report("boundary: exactly at high_conf -> critical",
            _classify_single(False, HIGH_CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD, 0.0) == "critical")


def test_16_no_architecture_redundancy():
    print("\n=== 16. Architecture Redundancy Check ===")
    from analysis.disagreement import disagreement_scoring as scoring_mod
    import inspect

    src = inspect.getsource(scoring_mod)
    # Scoring module should reuse Step 1's _make_serialisable, not redefine it
    _report("No _make_serialisable redefinition in scoring",
            "def _make_serialisable" not in src)

    # Step 2 modules should not re-import load_predictions or detect_disagreements
    from analysis.disagreement import confidence_disagreement as conf_mod
    conf_src = inspect.getsource(conf_mod)
    _report("confidence_disagreement does not import load_predictions",
            "from analysis.disagreement.disagreement_utils import load_predictions" not in conf_src)


# ===========================================================================
# Runner
# ===========================================================================

def main():
    print("=" * 70)
    print("  STEP 2 VALIDATION: CONFIDENCE-AWARE DISAGREEMENT ANALYSIS")
    print("=" * 70)

    tmp_dir = Path("results") / "disagreement" / "_test_step2_scratch"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        test_01_imports()
        test_02_step1_backward_compat()
        test_03_confidence_gaps()
        test_04_pairwise_confidence()
        test_05_confidence_spread_summary()
        test_06_severity_classification()
        test_07_sample_severity_rollup()
        test_08_severity_summary()
        test_09_disagreement_scoring()
        test_10_score_summary()
        test_11_csv_exports(tmp_dir)
        test_12_json_exports(tmp_dir)
        test_13_multi_model()
        test_14_edge_cases()
        test_15_severity_thresholds()
        test_16_no_architecture_redundancy()
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f"  RESULTS:  {PASS} passed,  {FAIL} failed")
    print("=" * 70)

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
