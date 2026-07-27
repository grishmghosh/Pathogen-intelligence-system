"""
Synthetic validation test for Step 3: Perturbation-Aware Disagreement Analysis.

Verifies:
    1.  Import integrity (all Step 3 APIs)
    2.  Step 1 + Step 2 backward compatibility
    3.  load_perturbation_predictions (column normalisation)
    4.  Perturbation-induced disagreement detection
    5.  Perturbation sensitivity ranking
    6.  Severity-level disagreement rates
    7.  Consensus stability tracking
    8.  Model instability scoring
    9.  Sample instability scoring
   10.  Instability summary
   11.  Escalation trend generation
   12.  Model comparison trend
   13.  Perturbation ranking trend
   14.  CSV exports
   15.  JSON exports
   16.  Multi-model (3+) compatibility
   17.  Edge cases (empty, single model, no clean, no perturbed)
   18.  Instability score range validation
   19.  Architecture redundancy check
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

def _perturbation_dataset():
    """
    2 models, 3 samples, each with clean + 3 perturbation types x 3 severities.
    Designed so:
      - s001: models agree on clean and all perturbations (stable)
      - s002: models agree on clean, disagree under gaussian_noise/severe (induced)
      - s003: models disagree on clean (not induced)
    """
    data = []
    models = ["efficientnet_b0", "resnet50"]
    pert_types = ["blur", "gaussian_noise", "brightness"]
    severities = ["mild", "moderate", "severe"]

    # --- s001: always agrees ---
    for m in models:
        data.append({"sample_id": "s001", "model_name": m, "predicted_class": "s_aureus",
                      "confidence": 0.95, "perturbation_type": "clean", "severity_level": "clean"})
    for pt in pert_types:
        for sv in severities:
            for m in models:
                data.append({"sample_id": "s001", "model_name": m, "predicted_class": "s_aureus",
                              "confidence": 0.90, "perturbation_type": pt, "severity_level": sv})

    # --- s002: agrees on clean, disagrees on gaussian_noise/severe ---
    for m in models:
        data.append({"sample_id": "s002", "model_name": m, "predicted_class": "e_coli",
                      "confidence": 0.92, "perturbation_type": "clean", "severity_level": "clean"})
    for pt in pert_types:
        for sv in severities:
            if pt == "gaussian_noise" and sv == "severe":
                data.append({"sample_id": "s002", "model_name": "efficientnet_b0",
                              "predicted_class": "e_coli", "confidence": 0.70,
                              "perturbation_type": pt, "severity_level": sv})
                data.append({"sample_id": "s002", "model_name": "resnet50",
                              "predicted_class": "k_pneumoniae", "confidence": 0.65,
                              "perturbation_type": pt, "severity_level": sv})
            elif pt == "gaussian_noise" and sv == "moderate":
                # Also disagree at moderate
                data.append({"sample_id": "s002", "model_name": "efficientnet_b0",
                              "predicted_class": "e_coli", "confidence": 0.80,
                              "perturbation_type": pt, "severity_level": sv})
                data.append({"sample_id": "s002", "model_name": "resnet50",
                              "predicted_class": "p_aeruginosa", "confidence": 0.55,
                              "perturbation_type": pt, "severity_level": sv})
            else:
                for m in models:
                    data.append({"sample_id": "s002", "model_name": m,
                                  "predicted_class": "e_coli", "confidence": 0.88,
                                  "perturbation_type": pt, "severity_level": sv})

    # --- s003: disagrees even on clean ---
    data.append({"sample_id": "s003", "model_name": "efficientnet_b0",
                  "predicted_class": "p_aeruginosa", "confidence": 0.85,
                  "perturbation_type": "clean", "severity_level": "clean"})
    data.append({"sample_id": "s003", "model_name": "resnet50",
                  "predicted_class": "s_aureus", "confidence": 0.82,
                  "perturbation_type": "clean", "severity_level": "clean"})
    for pt in pert_types:
        for sv in severities:
            data.append({"sample_id": "s003", "model_name": "efficientnet_b0",
                          "predicted_class": "p_aeruginosa", "confidence": 0.80,
                          "perturbation_type": pt, "severity_level": sv})
            data.append({"sample_id": "s003", "model_name": "resnet50",
                          "predicted_class": "s_aureus", "confidence": 0.78,
                          "perturbation_type": pt, "severity_level": sv})
    return data


def _three_model_perturbation_dataset():
    """3 models, 2 samples, clean + 2 perturbation severities."""
    data = []
    models = ["m1", "m2", "m3"]

    # s1: all agree on clean, m3 flips at severe
    for m in models:
        data.append({"sample_id": "s1", "model_name": m, "predicted_class": "A",
                      "confidence": 0.90, "perturbation_type": "clean", "severity_level": "clean"})
    for m in models:
        data.append({"sample_id": "s1", "model_name": m, "predicted_class": "A",
                      "confidence": 0.85, "perturbation_type": "noise", "severity_level": "mild"})
    for m in ["m1", "m2"]:
        data.append({"sample_id": "s1", "model_name": m, "predicted_class": "A",
                      "confidence": 0.80, "perturbation_type": "noise", "severity_level": "severe"})
    data.append({"sample_id": "s1", "model_name": "m3", "predicted_class": "B",
                  "confidence": 0.75, "perturbation_type": "noise", "severity_level": "severe"})

    # s2: all agree everywhere
    for m in models:
        for sv in ["clean", "mild", "severe"]:
            pt = "clean" if sv == "clean" else "noise"
            data.append({"sample_id": "s2", "model_name": m, "predicted_class": "C",
                          "confidence": 0.90, "perturbation_type": pt, "severity_level": sv})
    return data


# ===========================================================================
# Tests
# ===========================================================================

def test_01_imports():
    print("\n=== 1. Import Integrity (Step 3 APIs) ===")
    try:
        from analysis.disagreement import (
            load_perturbation_predictions,
            detect_perturbation_induced_disagreements,
            compute_perturbation_sensitivity,
            track_consensus_stability,
            compute_severity_disagreement_rates,
            compute_model_instability,
            compute_sample_instability,
            compute_instability_summary,
            generate_escalation_trend,
            generate_model_comparison_trend,
            generate_perturbation_ranking_trend,
            generate_full_trend_report,
            export_induced_disagreements_csv,
            export_induced_disagreements_json,
            export_perturbation_sensitivity_csv,
            export_perturbation_sensitivity_json,
            export_severity_rates_csv,
            export_consensus_stability_csv,
            export_consensus_stability_json,
            export_model_instability_csv,
            export_model_instability_json,
            export_sample_instability_csv,
            export_sample_instability_json,
            export_instability_summary_json,
            export_trend_report_json,
        )
        _report("All Step 3 imports resolve", True)
    except Exception as e:
        _report("All Step 3 imports resolve", False, str(e))
        traceback.print_exc()


def test_02_backward_compat():
    print("\n=== 2. Step 1 + Step 2 Backward Compatibility ===")
    try:
        from analysis.disagreement import (
            load_predictions, detect_disagreements, compute_agreement_matrix,
            compute_pairwise_confidence_analysis, classify_pairwise_severity,
            compute_disagreement_scores,
        )
        preds = [
            {"sample_id": "x1", "model_name": "m1", "predicted_class": "A", "confidence": 0.9},
            {"sample_id": "x1", "model_name": "m2", "predicted_class": "B", "confidence": 0.8},
        ]
        df = load_predictions(preds)
        pct, _ = compute_agreement_matrix(df)
        pw = compute_pairwise_confidence_analysis(df)
        sev = classify_pairwise_severity(pw)
        scored = compute_disagreement_scores(sev)
        _report("Step 1+2 pipeline still works", len(scored) == 1)
    except Exception as e:
        _report("Step 1+2 pipeline still works", False, str(e))


def test_03_load_perturbation_predictions():
    print("\n=== 3. load_perturbation_predictions ===")
    from analysis.disagreement.perturbation_disagreement import load_perturbation_predictions

    df = load_perturbation_predictions(_perturbation_dataset())
    _report("Returns DataFrame", isinstance(df, pd.DataFrame))
    _report("Has perturbation_type column", "perturbation_type" in df.columns)
    _report("Has severity_level column", "severity_level" in df.columns)

    # Column alias normalisation
    alias_data = [
        {"sample_id": "x", "model_name": "m1", "predicted_class": "A",
         "confidence": 0.9, "perturbation": "blur", "severity": "mild"},
        {"sample_id": "x", "model_name": "m2", "predicted_class": "A",
         "confidence": 0.8, "perturbation": "blur", "severity": "mild"},
    ]
    df2 = load_perturbation_predictions(alias_data)
    _report("Alias 'perturbation' -> 'perturbation_type'", "perturbation_type" in df2.columns)
    _report("Alias 'severity' -> 'severity_level'", "severity_level" in df2.columns)

    # Missing perturbation fields -> graceful fill
    df3 = load_perturbation_predictions([
        {"sample_id": "x", "model_name": "m1", "predicted_class": "A", "confidence": 0.9},
        {"sample_id": "x", "model_name": "m2", "predicted_class": "A", "confidence": 0.8},
    ])
    _report("Missing pert fields -> 'unknown'", (df3["perturbation_type"] == "unknown").all())


def test_04_induced_disagreements():
    print("\n=== 4. Perturbation-Induced Disagreement Detection ===")
    from analysis.disagreement.perturbation_disagreement import (
        load_perturbation_predictions, detect_perturbation_induced_disagreements,
    )

    df = load_perturbation_predictions(_perturbation_dataset())
    induced = detect_perturbation_induced_disagreements(df)

    _report("Returns DataFrame", isinstance(induced, pd.DataFrame))
    _report("Has perturbation_induced column", "perturbation_induced" in induced.columns)

    induced_samples = induced["sample_id"].unique()
    _report("s002 is induced", "s002" in induced_samples)
    _report("s001 is NOT induced (always agrees)", "s001" not in induced_samples)
    _report("s003 is NOT induced (disagrees on clean)", "s003" not in induced_samples)

    # Only gaussian_noise at moderate+severe should appear
    induced_pts = induced["perturbation_type"].unique()
    _report("Only gaussian_noise triggers induced", list(induced_pts) == ["gaussian_noise"],
            f"got {list(induced_pts)}")


def test_05_perturbation_sensitivity():
    print("\n=== 5. Perturbation Sensitivity Ranking ===")
    from analysis.disagreement.perturbation_disagreement import (
        load_perturbation_predictions, compute_perturbation_sensitivity,
    )

    df = load_perturbation_predictions(_perturbation_dataset())
    sens = compute_perturbation_sensitivity(df)

    _report("Returns DataFrame", isinstance(sens, pd.DataFrame))
    _report("Has perturbation_type column", "perturbation_type" in sens.columns)
    _report("Has disagreement_rate column", "disagreement_rate" in sens.columns)

    # gaussian_noise should have highest disagreement rate
    top = sens.iloc[0]
    _report("gaussian_noise is most disruptive", top["perturbation_type"] == "gaussian_noise",
            f"got '{top['perturbation_type']}'")


def test_06_severity_disagreement_rates():
    print("\n=== 6. Severity-Level Disagreement Rates ===")
    from analysis.disagreement.perturbation_disagreement import (
        load_perturbation_predictions, compute_severity_disagreement_rates,
    )

    df = load_perturbation_predictions(_perturbation_dataset())
    rates = compute_severity_disagreement_rates(df)

    _report("Returns DataFrame", isinstance(rates, pd.DataFrame))
    _report("Has severity_rank column", "severity_rank" in rates.columns)

    # Sorted by severity_rank
    ranks = rates["severity_rank"].tolist()
    _report("Sorted by severity_rank", ranks == sorted(ranks))

    # Severe should have >= moderate disagreement rate for this dataset
    sev_rates = dict(zip(rates["severity_level"], rates["disagreement_rate"]))
    if "severe" in sev_rates and "mild" in sev_rates:
        _report("severe_rate >= mild_rate", sev_rates["severe"] >= sev_rates["mild"],
                f"severe={sev_rates['severe']}, mild={sev_rates['mild']}")
    else:
        _report("severe_rate >= mild_rate", False, "missing severity levels")


def test_07_consensus_stability():
    print("\n=== 7. Consensus Stability Tracking ===")
    from analysis.disagreement.perturbation_disagreement import (
        load_perturbation_predictions, track_consensus_stability,
    )

    df = load_perturbation_predictions(_perturbation_dataset())
    stability = track_consensus_stability(df)

    _report("Returns DataFrame", isinstance(stability, pd.DataFrame))
    _report("Has stability_breakpoint", "stability_breakpoint" in stability.columns)
    _report("Has collapse_severity", "collapse_severity" in stability.columns)

    # s001 should never break
    s001 = stability[stability["sample_id"] == "s001"]
    s001_breaks = s001[s001["stability_breakpoint"] == True]
    _report("s001 never breaks", len(s001_breaks) == 0)

    # s002 should break at moderate (gaussian_noise/moderate disagrees)
    s002 = stability[stability["sample_id"] == "s002"]
    s002_collapse = s002["collapse_severity"].dropna().unique()
    _report("s002 collapses at moderate", "moderate" in s002_collapse if len(s002_collapse) > 0 else False,
            f"got {list(s002_collapse)}")


def test_08_model_instability():
    print("\n=== 8. Model Instability Scoring ===")
    from analysis.disagreement.perturbation_disagreement import load_perturbation_predictions
    from analysis.disagreement.instability_analysis import compute_model_instability

    df = load_perturbation_predictions(_perturbation_dataset())
    mi = compute_model_instability(df)

    _report("Returns DataFrame", isinstance(mi, pd.DataFrame))
    _report("Has instability_score", "instability_score" in mi.columns)
    _report("Has flip_rate", "flip_rate" in mi.columns)
    _report("2 model rows", len(mi) == 2, f"got {len(mi)}")

    scores = mi["instability_score"]
    _report("All scores >= 0", (scores >= 0).all())
    _report("All scores <= 1", (scores <= 1).all())

    # resnet50 flips on s002 (gaussian_noise moderate+severe), efficientnet does not flip at all
    resnet_row = mi[mi["model_name"] == "resnet50"]
    eff_row = mi[mi["model_name"] == "efficientnet_b0"]
    if not resnet_row.empty and not eff_row.empty:
        _report("resnet50 more unstable than efficientnet",
                resnet_row.iloc[0]["instability_score"] > eff_row.iloc[0]["instability_score"])


def test_09_sample_instability():
    print("\n=== 9. Sample Instability Scoring ===")
    from analysis.disagreement.perturbation_disagreement import load_perturbation_predictions
    from analysis.disagreement.instability_analysis import compute_sample_instability

    df = load_perturbation_predictions(_perturbation_dataset())
    si = compute_sample_instability(df)

    _report("Returns DataFrame", isinstance(si, pd.DataFrame))
    _report("Has instability_score", "instability_score" in si.columns)

    scores = si["instability_score"]
    _report("All scores >= 0", (scores >= 0).all())
    _report("All scores <= 1", (scores <= 1).all())

    # s001 should be most stable (score = 0)
    s001 = si[si["sample_id"] == "s001"]
    if not s001.empty:
        _report("s001 instability = 0", s001.iloc[0]["instability_score"] == 0.0)

    # s002 or s003 should have nonzero instability
    s002 = si[si["sample_id"] == "s002"]
    s003 = si[si["sample_id"] == "s003"]
    has_unstable = False
    if not s002.empty:
        has_unstable = has_unstable or s002.iloc[0]["instability_score"] > 0
    if not s003.empty:
        has_unstable = has_unstable or s003.iloc[0]["instability_score"] > 0
    _report("At least one sample has nonzero instability", has_unstable)


def test_10_instability_summary():
    print("\n=== 10. Instability Summary ===")
    from analysis.disagreement.perturbation_disagreement import (
        load_perturbation_predictions, compute_severity_disagreement_rates,
    )
    from analysis.disagreement.instability_analysis import (
        compute_model_instability, compute_sample_instability, compute_instability_summary,
    )

    df = load_perturbation_predictions(_perturbation_dataset())
    mi = compute_model_instability(df)
    si = compute_sample_instability(df)
    sr = compute_severity_disagreement_rates(df)
    summary = compute_instability_summary(mi, si, sr)

    _report("Is dict", isinstance(summary, dict))
    _report("Has overall_instability", "overall_instability" in summary)
    _report("Has most_unstable_model", "most_unstable_model" in summary)
    _report("Has severity_escalation", "severity_escalation" in summary)
    _report("overall_instability in [0,1]", 0 <= summary["overall_instability"] <= 1)


def test_11_escalation_trend():
    print("\n=== 11. Escalation Trend Generation ===")
    from analysis.disagreement.perturbation_disagreement import (
        load_perturbation_predictions, compute_severity_disagreement_rates,
    )
    from analysis.disagreement.disagreement_trends import generate_escalation_trend

    df = load_perturbation_predictions(_perturbation_dataset())
    rates = compute_severity_disagreement_rates(df)
    trend = generate_escalation_trend(rates)

    _report("Is dict", isinstance(trend, dict))
    _report("Has trend_direction", "trend_direction" in trend)
    _report("Has severity_steps", len(trend["severity_steps"]) > 0)
    _report("Has max_rate", "max_rate" in trend)
    _report("Trend is increasing or stable", trend["trend_direction"] in ("increasing", "stable"),
            f"got '{trend['trend_direction']}'")


def test_12_model_comparison_trend():
    print("\n=== 12. Model Comparison Trend ===")
    from analysis.disagreement.perturbation_disagreement import load_perturbation_predictions
    from analysis.disagreement.instability_analysis import compute_model_instability
    from analysis.disagreement.disagreement_trends import generate_model_comparison_trend

    df = load_perturbation_predictions(_perturbation_dataset())
    mi = compute_model_instability(df)
    trend = generate_model_comparison_trend(mi)

    _report("Is dict", isinstance(trend, dict))
    _report("Has model_ranking", len(trend["model_ranking"]) == 2)
    _report("Has most_stable", trend["most_stable"] is not None)
    _report("Has summary", len(trend["summary"]) > 0)


def test_13_perturbation_ranking_trend():
    print("\n=== 13. Perturbation Ranking Trend ===")
    from analysis.disagreement.perturbation_disagreement import (
        load_perturbation_predictions, compute_perturbation_sensitivity,
    )
    from analysis.disagreement.disagreement_trends import generate_perturbation_ranking_trend

    df = load_perturbation_predictions(_perturbation_dataset())
    sens = compute_perturbation_sensitivity(df)
    trend = generate_perturbation_ranking_trend(sens)

    _report("Is dict", isinstance(trend, dict))
    _report("Has ranking", len(trend["ranking"]) > 0)
    _report("most_disruptive is gaussian_noise", trend["most_disruptive"] == "gaussian_noise",
            f"got '{trend['most_disruptive']}'")
    _report("Has summary", "gaussian_noise" in trend["summary"].lower())


def test_14_csv_exports(tmp_dir):
    print("\n=== 14. CSV Exports ===")
    from analysis.disagreement.perturbation_disagreement import (
        load_perturbation_predictions, detect_perturbation_induced_disagreements,
        compute_perturbation_sensitivity, compute_severity_disagreement_rates,
        track_consensus_stability,
    )
    from analysis.disagreement.instability_analysis import (
        compute_model_instability, compute_sample_instability,
    )
    from analysis.disagreement.disagreement_trends import (
        export_induced_disagreements_csv, export_perturbation_sensitivity_csv,
        export_severity_rates_csv, export_consensus_stability_csv,
        export_model_instability_csv, export_sample_instability_csv,
    )

    df = load_perturbation_predictions(_perturbation_dataset())
    induced = detect_perturbation_induced_disagreements(df)
    sens = compute_perturbation_sensitivity(df)
    rates = compute_severity_disagreement_rates(df)
    stab = track_consensus_stability(df)
    mi = compute_model_instability(df)
    si = compute_sample_instability(df)

    paths = [
        ("induced CSV", export_induced_disagreements_csv(induced, output_dir=tmp_dir)),
        ("sensitivity CSV", export_perturbation_sensitivity_csv(sens, output_dir=tmp_dir)),
        ("rates CSV", export_severity_rates_csv(rates, output_dir=tmp_dir)),
        ("stability CSV", export_consensus_stability_csv(stab, output_dir=tmp_dir)),
        ("model instab CSV", export_model_instability_csv(mi, output_dir=tmp_dir)),
        ("sample instab CSV", export_sample_instability_csv(si, output_dir=tmp_dir)),
    ]
    for name, p in paths:
        _report(f"{name} exists", p.exists())

    # Re-read one
    reload = pd.read_csv(paths[1][1])
    _report("Sensitivity CSV re-readable", "perturbation_type" in reload.columns)


def test_15_json_exports(tmp_dir):
    print("\n=== 15. JSON Exports ===")
    from analysis.disagreement.perturbation_disagreement import (
        load_perturbation_predictions, detect_perturbation_induced_disagreements,
        compute_perturbation_sensitivity, compute_severity_disagreement_rates,
        track_consensus_stability,
    )
    from analysis.disagreement.instability_analysis import (
        compute_model_instability, compute_sample_instability, compute_instability_summary,
    )
    from analysis.disagreement.disagreement_trends import (
        export_induced_disagreements_json, export_perturbation_sensitivity_json,
        export_consensus_stability_json, export_model_instability_json,
        export_sample_instability_json, export_instability_summary_json,
        generate_full_trend_report, export_trend_report_json,
    )

    df = load_perturbation_predictions(_perturbation_dataset())
    induced = detect_perturbation_induced_disagreements(df)
    sens = compute_perturbation_sensitivity(df)
    rates = compute_severity_disagreement_rates(df)
    stab = track_consensus_stability(df)
    mi = compute_model_instability(df)
    si = compute_sample_instability(df)
    inst_sum = compute_instability_summary(mi, si, rates)
    trends = generate_full_trend_report(rates, mi, sens)

    paths = [
        ("induced JSON", export_induced_disagreements_json(induced, output_dir=tmp_dir)),
        ("sensitivity JSON", export_perturbation_sensitivity_json(sens, output_dir=tmp_dir)),
        ("stability JSON", export_consensus_stability_json(stab, output_dir=tmp_dir)),
        ("model instab JSON", export_model_instability_json(mi, output_dir=tmp_dir)),
        ("sample instab JSON", export_sample_instability_json(si, output_dir=tmp_dir)),
        ("instab summary JSON", export_instability_summary_json(inst_sum, output_dir=tmp_dir)),
        ("trend report JSON", export_trend_report_json(trends, output_dir=tmp_dir)),
    ]
    for name, p in paths:
        _report(f"{name} exists", p.exists())

    # Verify parseability
    with open(paths[-1][1], "r") as f:
        data = json.load(f)
    _report("Trend report JSON parseable", "escalation_trend" in data)


def test_16_multi_model():
    print("\n=== 16. Multi-Model (3+) Compatibility ===")
    from analysis.disagreement.perturbation_disagreement import (
        load_perturbation_predictions, detect_perturbation_induced_disagreements,
        compute_perturbation_sensitivity, track_consensus_stability,
    )
    from analysis.disagreement.instability_analysis import compute_model_instability

    df = load_perturbation_predictions(_three_model_perturbation_dataset())
    induced = detect_perturbation_induced_disagreements(df)
    sens = compute_perturbation_sensitivity(df)
    stab = track_consensus_stability(df)
    mi = compute_model_instability(df)

    _report("Induced detection works with 3 models", isinstance(induced, pd.DataFrame))
    _report("s1 has induced disagreement", "s1" in induced["sample_id"].unique() if not induced.empty else False)
    _report("Model instability has 3 rows", len(mi) == 3, f"got {len(mi)}")

    # m3 should be most unstable
    top_model = mi.iloc[0]["model_name"]
    _report("m3 is most unstable", top_model == "m3", f"got '{top_model}'")


def test_17_edge_cases():
    print("\n=== 17. Edge Cases ===")
    from analysis.disagreement.perturbation_disagreement import (
        load_perturbation_predictions, detect_perturbation_induced_disagreements,
        compute_perturbation_sensitivity,
    )
    from analysis.disagreement.instability_analysis import compute_model_instability

    # Empty DataFrame
    try:
        detect_perturbation_induced_disagreements(pd.DataFrame())
        _report("Empty DF raises ValueError", False)
    except ValueError:
        _report("Empty DF raises ValueError", True)

    # Single model
    try:
        df = load_perturbation_predictions([
            {"sample_id": "s1", "model_name": "m1", "predicted_class": "A",
             "confidence": 0.9, "perturbation_type": "clean", "severity_level": "clean"},
        ])
        detect_perturbation_induced_disagreements(df)
        _report("Single model raises ValueError", False)
    except ValueError:
        _report("Single model raises ValueError", True)

    # No clean predictions
    df_no_clean = load_perturbation_predictions([
        {"sample_id": "s1", "model_name": "m1", "predicted_class": "A",
         "confidence": 0.9, "perturbation_type": "blur", "severity_level": "mild"},
        {"sample_id": "s1", "model_name": "m2", "predicted_class": "B",
         "confidence": 0.8, "perturbation_type": "blur", "severity_level": "mild"},
    ])
    induced = detect_perturbation_induced_disagreements(df_no_clean)
    _report("No clean -> empty induced", len(induced) == 0)

    # No perturbed predictions
    df_no_pert = load_perturbation_predictions([
        {"sample_id": "s1", "model_name": "m1", "predicted_class": "A",
         "confidence": 0.9, "perturbation_type": "clean", "severity_level": "clean"},
        {"sample_id": "s1", "model_name": "m2", "predicted_class": "A",
         "confidence": 0.8, "perturbation_type": "clean", "severity_level": "clean"},
    ])
    induced2 = detect_perturbation_induced_disagreements(df_no_pert)
    _report("No perturbed -> empty induced", len(induced2) == 0)

    # All agree -> instability should be 0
    df_agree = load_perturbation_predictions([
        {"sample_id": "s1", "model_name": "m1", "predicted_class": "A",
         "confidence": 0.9, "perturbation_type": "clean", "severity_level": "clean"},
        {"sample_id": "s1", "model_name": "m2", "predicted_class": "A",
         "confidence": 0.8, "perturbation_type": "clean", "severity_level": "clean"},
        {"sample_id": "s1", "model_name": "m1", "predicted_class": "A",
         "confidence": 0.85, "perturbation_type": "blur", "severity_level": "mild"},
        {"sample_id": "s1", "model_name": "m2", "predicted_class": "A",
         "confidence": 0.8, "perturbation_type": "blur", "severity_level": "mild"},
    ])
    mi = compute_model_instability(df_agree)
    _report("Full agreement -> all instability=0",
            (mi["instability_score"] == 0).all() if not mi.empty else True)


def test_18_score_ranges():
    print("\n=== 18. Instability Score Range Validation ===")
    from analysis.disagreement.perturbation_disagreement import load_perturbation_predictions
    from analysis.disagreement.instability_analysis import (
        compute_model_instability, compute_sample_instability,
    )

    df = load_perturbation_predictions(_perturbation_dataset())
    mi = compute_model_instability(df)
    si = compute_sample_instability(df)

    for name, frame in [("model", mi), ("sample", si)]:
        if frame.empty:
            _report(f"{name} instability scores in [0,1]", True, "empty")
            continue
        scores = frame["instability_score"]
        _report(f"{name} instability min >= 0", scores.min() >= 0, f"min={scores.min():.6f}")
        _report(f"{name} instability max <= 1", scores.max() <= 1, f"max={scores.max():.6f}")


def test_19_architecture_check():
    print("\n=== 19. Architecture Redundancy Check ===")
    import inspect
    from analysis.disagreement import disagreement_trends as trends_mod
    from analysis.disagreement import instability_analysis as instab_mod

    # Trends module should NOT redefine _make_serialisable
    trends_src = inspect.getsource(trends_mod)
    _report("No _make_serialisable redefinition in trends",
            "def _make_serialisable" not in trends_src)

    # Instability module should NOT re-import load_predictions
    instab_src = inspect.getsource(instab_mod)
    _report("instability_analysis does not import load_predictions",
            "from analysis.disagreement.disagreement_utils import load_predictions" not in instab_src)

    # perturbation_disagreement should reuse load_predictions from Step 1
    from analysis.disagreement import perturbation_disagreement as pert_mod
    pert_src = inspect.getsource(pert_mod)
    _report("perturbation module reuses Step 1 load_predictions",
            "from analysis.disagreement.disagreement_utils import load_predictions" in pert_src)


# ===========================================================================
# Runner
# ===========================================================================

def main():
    print("=" * 70)
    print("  STEP 3 VALIDATION: PERTURBATION-AWARE DISAGREEMENT ANALYSIS")
    print("=" * 70)

    tmp_dir = Path("results") / "disagreement" / "_test_step3_scratch"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        test_01_imports()
        test_02_backward_compat()
        test_03_load_perturbation_predictions()
        test_04_induced_disagreements()
        test_05_perturbation_sensitivity()
        test_06_severity_disagreement_rates()
        test_07_consensus_stability()
        test_08_model_instability()
        test_09_sample_instability()
        test_10_instability_summary()
        test_11_escalation_trend()
        test_12_model_comparison_trend()
        test_13_perturbation_ranking_trend()
        test_14_csv_exports(tmp_dir)
        test_15_json_exports(tmp_dir)
        test_16_multi_model()
        test_17_edge_cases()
        test_18_score_ranges()
        test_19_architecture_check()
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
