"""
Comprehensive validation tests for Step 4: Consensus Reliability Analysis.

Validates:
    1. Consensus reliability scoring (score range, component weights)
    2. False consensus detection (fragile, false, unstable)
    3. Trust classification and labelling
    4. Consensus breakdown analysis
    5. Consensus consistency metrics
    6. Model trust contribution
    7. CSV and JSON export
    8. Compatibility with Steps 1-3
    9. Error handling for missing/malformed data
    10. Synthetic perturbation stability tests
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("step4_validation")

# ===========================================================================
# Synthetic test data generation
# ===========================================================================

def create_synthetic_predictions():
    """Create synthetic predictions across clean and perturbed conditions."""
    records = []
    models = ["resnet50", "efficientnet_b0", "densenet121"]
    samples = [f"sample_{i:03d}" for i in range(1, 11)]

    # --- Clean predictions ---
    for sid in samples:
        for model in models:
            if sid in ("sample_001", "sample_002", "sample_003", "sample_004",
                       "sample_005"):
                # All agree on clean
                pred = "s_aureus"
                conf = 0.85 + np.random.RandomState(hash(sid + model) % 2**31).uniform(-0.05, 0.05)
            elif sid in ("sample_006", "sample_007"):
                # All agree but low confidence
                pred = "e_coli"
                conf = 0.35 + np.random.RandomState(hash(sid + model) % 2**31).uniform(-0.05, 0.05)
            elif sid in ("sample_008", "sample_009"):
                # All agree but with large confidence gap
                pred = "k_pneumoniae"
                if model == "resnet50":
                    conf = 0.95
                elif model == "efficientnet_b0":
                    conf = 0.45
                else:
                    conf = 0.70
            else:
                # Disagree on clean
                if model == "resnet50":
                    pred = "s_aureus"
                    conf = 0.90
                else:
                    pred = "e_coli"
                    conf = 0.85

            records.append({
                "sample_id": sid,
                "model_name": model,
                "predicted_class": pred,
                "confidence": round(conf, 4),
                "perturbation_type": "clean",
                "severity_level": "clean",
            })

    # --- Perturbed predictions ---
    perturbation_types = ["gaussian_noise", "contrast"]
    severity_levels = ["mild", "moderate", "severe"]

    for sid in samples[:8]:  # Only first 8 samples get perturbations
        for pert in perturbation_types:
            for sev in severity_levels:
                for model in models:
                    rng = np.random.RandomState(
                        hash(sid + model + pert + sev) % 2**31
                    )

                    if sid in ("sample_001", "sample_002"):
                        # Stable consensus - survives all perturbations
                        pred = "s_aureus"
                        conf = 0.80 + rng.uniform(-0.10, 0.05)
                    elif sid in ("sample_003", "sample_004"):
                        # Fragile - collapses at mild
                        if sev == "mild" and model == "densenet121":
                            pred = "e_coli"
                            conf = 0.55
                        elif sev in ("moderate", "severe") and model in ("densenet121", "efficientnet_b0"):
                            pred = "e_coli"
                            conf = 0.60 + rng.uniform(-0.05, 0.05)
                        else:
                            pred = "s_aureus"
                            conf = 0.75 + rng.uniform(-0.10, 0.05)
                    elif sid == "sample_005":
                        # Oscillating - agree/disagree/agree pattern
                        if sev == "mild":
                            pred = "s_aureus"  # still agrees
                            conf = 0.70
                        elif sev == "moderate":
                            if model == "resnet50":
                                pred = "e_coli"  # disagrees
                            else:
                                pred = "s_aureus"
                            conf = 0.60
                        else:  # severe
                            pred = "s_aureus"  # agrees again
                            conf = 0.50
                    elif sid in ("sample_006", "sample_007"):
                        # Low confidence consensus collapse
                        if sev in ("moderate", "severe") and model == "resnet50":
                            pred = "k_pneumoniae"
                            conf = 0.40
                        else:
                            pred = "e_coli"
                            conf = 0.30 + rng.uniform(-0.05, 0.05)
                    elif sid == "sample_008":
                        # High gap consensus, collapses at severe
                        if sev == "severe" and model == "efficientnet_b0":
                            pred = "s_aureus"
                            conf = 0.50
                        else:
                            pred = "k_pneumoniae"
                            conf = 0.70 if model == "resnet50" else 0.55
                    else:
                        pred = "e_coli"
                        conf = 0.50

                    records.append({
                        "sample_id": sid,
                        "model_name": model,
                        "predicted_class": pred,
                        "confidence": round(max(0, min(1, conf)), 4),
                        "perturbation_type": pert,
                        "severity_level": sev,
                    })

    return pd.DataFrame(records)


# ===========================================================================
# Validation functions
# ===========================================================================

def validate_imports():
    """V1: Verify all Step 4 imports resolve correctly."""
    logger.info("=" * 60)
    logger.info("V1: Validating imports...")

    errors = []

    try:
        from analysis.disagreement.consensus_reliability import (
            compute_consensus_reliability,
            compute_reliability_summary,
            compute_consensus_breakdown,
            compute_consensus_consistency_metrics,
            compute_model_trust_contribution,
            compute_model_trust_summary,
        )
    except ImportError as e:
        errors.append(f"consensus_reliability import failed: {e}")

    try:
        from analysis.disagreement.false_consensus_detection import (
            detect_fragile_consensus,
            detect_false_consensus,
            detect_unstable_agreement,
            compute_false_consensus_summary,
        )
    except ImportError as e:
        errors.append(f"false_consensus_detection import failed: {e}")

    try:
        from analysis.disagreement.trust_analysis import (
            classify_trust_level,
            assign_trust_labels,
            compute_trust_summary,
            export_consensus_reliability_csv,
            export_consensus_reliability_json,
            export_reliability_summary_json,
            export_consensus_breakdown_csv,
            export_consensus_breakdown_json,
            export_consistency_metrics_json,
            export_model_trust_contribution_csv,
            export_model_trust_contribution_json,
            export_model_trust_summary_json,
            export_false_consensus_csv,
            export_false_consensus_json,
            export_fragile_consensus_csv,
            export_fragile_consensus_json,
            export_unstable_agreement_csv,
            export_unstable_agreement_json,
            export_false_consensus_summary_json,
            export_trust_labels_csv,
            export_trust_labels_json,
            export_trust_summary_json,
        )
    except ImportError as e:
        errors.append(f"trust_analysis import failed: {e}")

    # Test package-level imports
    try:
        from analysis.disagreement import (
            compute_consensus_reliability,
            detect_false_consensus,
            assign_trust_labels,
        )
    except ImportError as e:
        errors.append(f"Package-level import failed: {e}")

    if errors:
        for e in errors:
            logger.error("  FAIL: %s", e)
        return False

    logger.info("  PASS: All Step 4 imports resolve correctly.")
    return True


def validate_consensus_reliability_scoring(predictions, confidence_gaps,
                                           sample_severity, consensus_stability,
                                           sample_instability):
    """V2: Validate consensus reliability score range and computation."""
    logger.info("=" * 60)
    logger.info("V2: Validating consensus reliability scoring...")

    from analysis.disagreement.consensus_reliability import (
        compute_consensus_reliability,
        compute_reliability_summary,
    )

    errors = []

    reliability = compute_consensus_reliability(
        predictions=predictions,
        confidence_gaps=confidence_gaps,
        sample_severity=sample_severity,
        consensus_stability=consensus_stability,
        sample_instability=sample_instability,
    )

    # Check not empty
    if reliability.empty:
        errors.append("Reliability DataFrame is empty.")
        return False, None

    # Check required columns
    expected_cols = [
        "sample_id", "agreement_strength", "confidence_consistency",
        "perturbation_stability", "severity_reliability",
        "instability_reliability", "consensus_reliability_score",
    ]
    missing = set(expected_cols) - set(reliability.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")

    # Check score range [0, 1]
    score_col = "consensus_reliability_score"
    if score_col in reliability.columns:
        min_s = reliability[score_col].min()
        max_s = reliability[score_col].max()
        if min_s < 0.0 or max_s > 1.0:
            errors.append(f"Score out of range: min={min_s}, max={max_s}")

    # Check component ranges
    for col in expected_cols[1:]:
        if col in reliability.columns:
            vals = reliability[col]
            if vals.min() < 0.0 or vals.max() > 1.0:
                errors.append(f"Component {col} out of [0,1]: min={vals.min()}, max={vals.max()}")

    # Validate summary
    summary = compute_reliability_summary(reliability)
    if not isinstance(summary, dict):
        errors.append("Summary is not a dict.")
    else:
        for key in ["total_samples", "mean_reliability", "component_means", "weakest_component"]:
            if key not in summary:
                errors.append(f"Summary missing key: {key}")

    if errors:
        for e in errors:
            logger.error("  FAIL: %s", e)
        return False, reliability

    logger.info("  PASS: Reliability scoring validated. %d samples scored.", len(reliability))
    logger.info("         Score range: [%.4f, %.4f], mean=%.4f",
                reliability[score_col].min(),
                reliability[score_col].max(),
                reliability[score_col].mean())
    return True, reliability


def validate_false_consensus_detection(predictions, confidence_gaps,
                                       consensus_stability, sample_instability):
    """V3: Validate false consensus detection logic."""
    logger.info("=" * 60)
    logger.info("V3: Validating false consensus detection...")

    from analysis.disagreement.false_consensus_detection import (
        detect_fragile_consensus,
        detect_false_consensus,
        detect_unstable_agreement,
        compute_false_consensus_summary,
    )

    errors = []

    # --- Fragile consensus ---
    fragile = detect_fragile_consensus(predictions, consensus_stability)
    logger.info("  Fragile consensus: %d samples detected.", len(fragile))

    if not fragile.empty:
        if "flag" not in fragile.columns:
            errors.append("Fragile DF missing 'flag' column.")
        else:
            if not (fragile["flag"] == "fragile_consensus").all():
                errors.append("Not all fragile flags are 'fragile_consensus'.")

    # --- False consensus ---
    false_cons = detect_false_consensus(
        predictions, confidence_gaps, consensus_stability, sample_instability
    )
    logger.info("  False consensus: %d samples detected.", len(false_cons))

    if not false_cons.empty:
        required = {"sample_id", "flags", "flag_count", "false_consensus_severity"}
        missing = required - set(false_cons.columns)
        if missing:
            errors.append(f"False consensus DF missing columns: {missing}")

        # Check severity values
        if "false_consensus_severity" in false_cons.columns:
            valid_sevs = {"mild", "moderate", "severe"}
            actual = set(false_cons["false_consensus_severity"].unique())
            invalid = actual - valid_sevs
            if invalid:
                errors.append(f"Invalid false consensus severity: {invalid}")

    # --- Unstable agreement ---
    unstable = detect_unstable_agreement(consensus_stability)
    logger.info("  Unstable agreement: %d samples detected.", len(unstable))

    if not unstable.empty:
        if "oscillation_rate" in unstable.columns:
            if unstable["oscillation_rate"].min() <= 0:
                errors.append("Unstable samples should have positive oscillation rate.")

    # --- Summary ---
    summary = compute_false_consensus_summary(false_cons, fragile, unstable)
    if not isinstance(summary, dict):
        errors.append("False consensus summary is not a dict.")
    else:
        for key in ["total_false_consensus", "total_fragile_consensus",
                     "total_unique_flagged"]:
            if key not in summary:
                errors.append(f"Summary missing key: {key}")

    if errors:
        for e in errors:
            logger.error("  FAIL: %s", e)
        return False, fragile, false_cons, unstable

    logger.info("  PASS: False consensus detection validated.")
    return True, fragile, false_cons, unstable


def validate_trust_classification(reliability, false_consensus):
    """V4: Validate trust classification logic."""
    logger.info("=" * 60)
    logger.info("V4: Validating trust classification...")

    from analysis.disagreement.trust_analysis import (
        classify_trust_level,
        assign_trust_labels,
        compute_trust_summary,
    )

    errors = []

    # --- classify_trust_level ---
    test_cases = [
        (0.0, "critical"),
        (0.20, "critical"),
        (0.25, "low"),
        (0.44, "low"),
        (0.45, "moderate"),
        (0.69, "moderate"),
        (0.70, "high"),
        (0.84, "high"),
        (0.85, "very_high"),
        (1.0, "very_high"),
    ]
    for score, expected in test_cases:
        actual = classify_trust_level(score)
        if actual != expected:
            errors.append(f"classify_trust_level({score}): expected={expected}, got={actual}")

    # --- assign_trust_labels ---
    trust_labels = assign_trust_labels(reliability, false_consensus)

    if trust_labels.empty:
        errors.append("Trust labels DataFrame is empty.")
    else:
        required = {
            "sample_id", "consensus_reliability_score",
            "base_trust_level", "adjusted_trust_level",
            "false_consensus_flag",
        }
        missing = required - set(trust_labels.columns)
        if missing:
            errors.append(f"Trust labels missing columns: {missing}")

        # Valid trust levels
        valid_levels = {"very_high", "high", "moderate", "low", "critical"}
        if "adjusted_trust_level" in trust_labels.columns:
            actual_levels = set(trust_labels["adjusted_trust_level"].unique())
            invalid = actual_levels - valid_levels
            if invalid:
                errors.append(f"Invalid trust levels: {invalid}")

    # --- Trust summary ---
    summary = compute_trust_summary(trust_labels)
    if not isinstance(summary, dict):
        errors.append("Trust summary is not a dict.")
    else:
        for key in ["total_samples", "trust_distribution", "trust_rates"]:
            if key not in summary:
                errors.append(f"Summary missing key: {key}")

    if errors:
        for e in errors:
            logger.error("  FAIL: %s", e)
        return False, trust_labels

    logger.info("  PASS: Trust classification validated.")
    dist = trust_labels["adjusted_trust_level"].value_counts().to_dict()
    logger.info("         Distribution: %s", dist)
    return True, trust_labels


def validate_consensus_breakdown(consensus_stability, sample_instability):
    """V5: Validate consensus breakdown analysis."""
    logger.info("=" * 60)
    logger.info("V5: Validating consensus breakdown analysis...")

    from analysis.disagreement.consensus_reliability import (
        compute_consensus_breakdown,
    )

    errors = []

    breakdown = compute_consensus_breakdown(consensus_stability, sample_instability)
    logger.info("  Breakdown records: %d", len(breakdown))

    if not breakdown.empty:
        required = {
            "sample_id", "collapse_severity", "stability_duration",
            "escalation_speed",
        }
        missing = required - set(breakdown.columns)
        if missing:
            errors.append(f"Breakdown missing columns: {missing}")

        if "escalation_speed" in breakdown.columns:
            if breakdown["escalation_speed"].min() < 0:
                errors.append("Escalation speed should not be negative.")

    if errors:
        for e in errors:
            logger.error("  FAIL: %s", e)
        return False, breakdown

    logger.info("  PASS: Consensus breakdown validated.")
    return True, breakdown


def validate_consistency_metrics(predictions, confidence_gaps,
                                 consensus_stability, model_instability):
    """V6: Validate consensus consistency metrics."""
    logger.info("=" * 60)
    logger.info("V6: Validating consistency metrics...")

    from analysis.disagreement.consensus_reliability import (
        compute_consensus_consistency_metrics,
    )

    errors = []

    metrics = compute_consensus_consistency_metrics(
        predictions, confidence_gaps, consensus_stability, model_instability
    )

    if not isinstance(metrics, dict):
        errors.append("Metrics is not a dict.")
    else:
        expected_keys = [
            "agreement_persistence", "confidence_stability",
            "perturbation_survival_rate", "cross_model_consistency",
            "overall_consistency",
        ]
        for key in expected_keys:
            if key not in metrics:
                errors.append(f"Missing metric: {key}")
            elif not (0.0 <= metrics[key] <= 1.0):
                errors.append(f"Metric {key}={metrics[key]} out of [0,1].")

    if errors:
        for e in errors:
            logger.error("  FAIL: %s", e)
        return False, metrics

    logger.info("  PASS: Consistency metrics validated.")
    for k, v in metrics.items():
        logger.info("         %s: %.4f", k, v)
    return True, metrics


def validate_model_trust_contribution(model_instability):
    """V7: Validate model trust contribution analysis."""
    logger.info("=" * 60)
    logger.info("V7: Validating model trust contribution...")

    from analysis.disagreement.consensus_reliability import (
        compute_model_trust_contribution,
        compute_model_trust_summary,
    )

    errors = []

    trust_contrib = compute_model_trust_contribution(model_instability)

    if trust_contrib.empty:
        errors.append("Trust contribution DataFrame is empty.")
    else:
        required = {
            "model_name", "instability_score", "trust_contribution",
            "trust_rank", "stability_label",
        }
        missing = required - set(trust_contrib.columns)
        if missing:
            errors.append(f"Missing columns: {missing}")

        valid_labels = {"stabiliser", "neutral", "destabiliser"}
        if "stability_label" in trust_contrib.columns:
            actual = set(trust_contrib["stability_label"].unique())
            invalid = actual - valid_labels
            if invalid:
                errors.append(f"Invalid stability labels: {invalid}")

    # Summary
    summary = compute_model_trust_summary(trust_contrib)
    if not isinstance(summary, dict):
        errors.append("Trust summary is not a dict.")

    if errors:
        for e in errors:
            logger.error("  FAIL: %s", e)
        return False, trust_contrib

    logger.info("  PASS: Model trust contribution validated.")
    for _, row in trust_contrib.iterrows():
        logger.info("         %s: contribution=%.4f (%s)",
                    row["model_name"], row["trust_contribution"],
                    row["stability_label"])
    return True, trust_contrib


def validate_exports(reliability, breakdown, metrics, trust_contrib,
                     false_consensus, fragile, unstable, trust_labels,
                     reliability_summary, fc_summary, trust_summary,
                     model_trust_summary):
    """V8: Validate CSV and JSON exports."""
    logger.info("=" * 60)
    logger.info("V8: Validating CSV and JSON exports...")

    from analysis.disagreement.trust_analysis import (
        export_consensus_reliability_csv,
        export_consensus_reliability_json,
        export_reliability_summary_json,
        export_consensus_breakdown_csv,
        export_consensus_breakdown_json,
        export_consistency_metrics_json,
        export_model_trust_contribution_csv,
        export_model_trust_contribution_json,
        export_model_trust_summary_json,
        export_false_consensus_csv,
        export_false_consensus_json,
        export_fragile_consensus_csv,
        export_fragile_consensus_json,
        export_unstable_agreement_csv,
        export_unstable_agreement_json,
        export_false_consensus_summary_json,
        export_trust_labels_csv,
        export_trust_labels_json,
        export_trust_summary_json,
    )

    errors = []
    export_dir = Path("results") / "disagreement"

    def _check(export_func, *args, **kwargs):
        try:
            fp = export_func(*args, **kwargs)
            if not fp.exists():
                errors.append(f"Export not created: {fp}")
            elif fp.stat().st_size == 0:
                errors.append(f"Export is empty: {fp}")
            else:
                # Validate JSON files
                if fp.suffix == ".json":
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not data:
                        errors.append(f"JSON is empty: {fp}")
                # Validate CSV files
                elif fp.suffix == ".csv":
                    df = pd.read_csv(fp)
                    if df.empty and len(args[0]) > 0 if isinstance(args[0], pd.DataFrame) else False:
                        errors.append(f"CSV is empty: {fp}")
                logger.info("    OK: %s", fp)
        except Exception as e:
            errors.append(f"Export failed ({export_func.__name__}): {e}")

    # Consensus reliability exports
    _check(export_consensus_reliability_csv, reliability)
    _check(export_consensus_reliability_json, reliability)
    _check(export_reliability_summary_json, reliability_summary)
    _check(export_consensus_breakdown_csv, breakdown)
    _check(export_consensus_breakdown_json, breakdown)
    _check(export_consistency_metrics_json, metrics)

    # Model trust exports
    _check(export_model_trust_contribution_csv, trust_contrib)
    _check(export_model_trust_contribution_json, trust_contrib)
    _check(export_model_trust_summary_json, model_trust_summary)

    # False consensus exports
    _check(export_false_consensus_csv, false_consensus)
    _check(export_false_consensus_json, false_consensus)
    _check(export_fragile_consensus_csv, fragile)
    _check(export_fragile_consensus_json, fragile)
    _check(export_unstable_agreement_csv, unstable)
    _check(export_unstable_agreement_json, unstable)
    _check(export_false_consensus_summary_json, fc_summary)

    # Trust analysis exports
    _check(export_trust_labels_csv, trust_labels)
    _check(export_trust_labels_json, trust_labels)
    _check(export_trust_summary_json, trust_summary)

    if errors:
        for e in errors:
            logger.error("  FAIL: %s", e)
        return False

    logger.info("  PASS: All exports validated successfully.")
    return True


def validate_error_handling():
    """V9: Validate graceful error handling."""
    logger.info("=" * 60)
    logger.info("V9: Validating error handling...")

    from analysis.disagreement.consensus_reliability import (
        compute_consensus_reliability,
        compute_consensus_breakdown,
        compute_consensus_consistency_metrics,
        compute_model_trust_contribution,
    )
    from analysis.disagreement.false_consensus_detection import (
        detect_fragile_consensus,
        detect_unstable_agreement,
    )
    from analysis.disagreement.trust_analysis import (
        assign_trust_labels,
    )

    errors = []

    # Empty DataFrame
    empty_df = pd.DataFrame()

    # Test empty predictions
    try:
        compute_consensus_reliability(empty_df)
        errors.append("Should raise ValueError for empty predictions.")
    except ValueError:
        logger.info("  OK: Empty predictions raises ValueError.")

    # Test empty consensus stability
    result = compute_consensus_breakdown(empty_df)
    if not result.empty or len(result) != 0:
        errors.append("Empty stability should return empty breakdown.")
    else:
        logger.info("  OK: Empty stability returns empty breakdown.")

    # Test empty model instability
    result = compute_model_trust_contribution(empty_df)
    if not result.empty or len(result) != 0:
        errors.append("Empty instability should return empty trust contribution.")
    else:
        logger.info("  OK: Empty instability returns empty trust contribution.")

    # Test missing columns
    bad_stability = pd.DataFrame({"sample_id": ["s1"], "bad_col": [1]})
    result = compute_consensus_breakdown(bad_stability)
    if len(result) != 0:
        errors.append("Missing columns should return empty result.")
    else:
        logger.info("  OK: Missing columns handled gracefully.")

    # Test empty trust labels
    empty_reliability = pd.DataFrame(columns=[
        "sample_id", "consensus_reliability_score"
    ])
    result = assign_trust_labels(empty_reliability)
    if not isinstance(result, pd.DataFrame):
        errors.append("Empty reliability should return DataFrame.")
    else:
        logger.info("  OK: Empty reliability returns empty trust labels.")

    # Test with None optional params
    min_pred = pd.DataFrame({
        "sample_id": ["s1", "s1"],
        "model_name": ["m1", "m2"],
        "predicted_class": ["a", "a"],
        "confidence": [0.9, 0.8],
    })
    result = compute_consensus_reliability(min_pred)
    if result.empty:
        errors.append("Minimal predictions should still produce results.")
    else:
        logger.info("  OK: Minimal predictions (no optional data) works.")

    # Test consistency metrics with all None
    metrics = compute_consensus_consistency_metrics(min_pred)
    if not isinstance(metrics, dict):
        errors.append("Consistency metrics should return dict even with minimal data.")
    else:
        logger.info("  OK: Consistency metrics works with minimal data.")

    # Test unstable detection with empty data
    result = detect_unstable_agreement(empty_df)
    if not isinstance(result, pd.DataFrame):
        errors.append("Empty data should return DataFrame for unstable detection.")
    else:
        logger.info("  OK: Empty unstable detection returns empty DataFrame.")

    if errors:
        for e in errors:
            logger.error("  FAIL: %s", e)
        return False

    logger.info("  PASS: Error handling validated.")
    return True


def validate_step_compatibility(predictions):
    """V10: Validate compatibility with Steps 1-3."""
    logger.info("=" * 60)
    logger.info("V10: Validating Step 1-3 compatibility...")

    errors = []

    # Step 1
    from analysis.disagreement.disagreement_utils import load_predictions, detect_disagreements
    from analysis.disagreement.agreement_metrics import compute_agreement_matrix, compute_disagreement_statistics

    # Step 2
    from analysis.disagreement.confidence_disagreement import compute_confidence_gaps
    from analysis.disagreement.disagreement_severity import classify_pairwise_severity, classify_sample_severity
    from analysis.disagreement.confidence_disagreement import compute_pairwise_confidence_analysis

    # Step 3
    from analysis.disagreement.perturbation_disagreement import (
        load_perturbation_predictions,
        track_consensus_stability,
        compute_perturbation_sensitivity,
    )
    from analysis.disagreement.instability_analysis import (
        compute_model_instability,
        compute_sample_instability,
    )

    # Step 4
    from analysis.disagreement.consensus_reliability import compute_consensus_reliability
    from analysis.disagreement.false_consensus_detection import detect_false_consensus
    from analysis.disagreement.trust_analysis import assign_trust_labels

    try:
        # Step 1 pipeline
        preds = load_perturbation_predictions(predictions)
        disagrees = detect_disagreements(preds)
        _, _ = compute_agreement_matrix(preds)
        _ = compute_disagreement_statistics(preds, disagrees)

        # Step 2 pipeline
        conf_gaps = compute_confidence_gaps(preds)
        pairwise = compute_pairwise_confidence_analysis(preds)
        pairwise_sev = classify_pairwise_severity(pairwise)
        sample_sev = classify_sample_severity(pairwise_sev)

        # Step 3 pipeline
        stability = track_consensus_stability(preds)
        model_inst = compute_model_instability(preds)
        sample_inst = compute_sample_instability(preds)

        # Step 4 pipeline (using Step 1-3 outputs)
        reliability = compute_consensus_reliability(
            preds, conf_gaps, sample_sev, stability, sample_inst
        )
        false_cons = detect_false_consensus(
            preds, conf_gaps, stability, sample_inst
        )
        trust = assign_trust_labels(reliability, false_cons)

        # Verify chain produces valid output
        if reliability.empty:
            errors.append("Full pipeline produced empty reliability.")
        if trust.empty:
            errors.append("Full pipeline produced empty trust labels.")

        logger.info("  Full Step 1→2→3→4 pipeline executed successfully.")
        logger.info("    Reliability: %d samples", len(reliability))
        logger.info("    False consensus: %d detections", len(false_cons))
        logger.info("    Trust labels: %d samples", len(trust))

    except Exception as e:
        errors.append(f"Pipeline compatibility error: {e}")
        import traceback
        traceback.print_exc()

    if errors:
        for e in errors:
            logger.error("  FAIL: %s", e)
        return False

    logger.info("  PASS: Full pipeline compatibility confirmed.")
    return True


def validate_architectural_redundancy():
    """V11: Detect architectural redundancy."""
    logger.info("=" * 60)
    logger.info("V11: Checking for architectural redundancy...")

    errors = []

    # Check that Step 4 modules don't duplicate Step 1-3 functions
    import analysis.disagreement.consensus_reliability as cr
    import analysis.disagreement.false_consensus_detection as fcd
    import analysis.disagreement.trust_analysis as ta

    step4_funcs = set()
    for mod in [cr, fcd, ta]:
        for name in dir(mod):
            if not name.startswith("_") and callable(getattr(mod, name)):
                step4_funcs.add(name)

    # Known Step 1-3 public function names
    step123_funcs = {
        "compute_agreement_matrix", "compute_disagreement_statistics",
        "load_predictions", "detect_disagreements",
        "compute_confidence_gaps", "compute_pairwise_confidence_analysis",
        "classify_pairwise_severity", "classify_sample_severity",
        "compute_severity_summary", "compute_disagreement_scores",
        "load_perturbation_predictions",
        "detect_perturbation_induced_disagreements",
        "compute_perturbation_sensitivity", "track_consensus_stability",
        "compute_model_instability", "compute_sample_instability",
        "compute_instability_summary",
    }

    overlap = step4_funcs & step123_funcs
    if overlap:
        errors.append(f"Function name overlap with Steps 1-3: {overlap}")

    if errors:
        for e in errors:
            logger.error("  FAIL: %s", e)
        return False

    logger.info("  PASS: No architectural redundancy detected.")
    logger.info("         Step 4 defines %d unique public functions.", len(step4_funcs))
    return True


# ===========================================================================
# Main validation runner
# ===========================================================================

def main():
    logger.info("=" * 60)
    logger.info("STEP 4 VALIDATION: Consensus Reliability Analysis")
    logger.info("=" * 60)

    # Generate synthetic data
    predictions = create_synthetic_predictions()
    logger.info("Synthetic data: %d rows, %d samples, %d models",
                len(predictions),
                predictions["sample_id"].nunique(),
                predictions["model_name"].nunique())

    # Pre-compute Step 2-3 outputs
    from analysis.disagreement.confidence_disagreement import compute_confidence_gaps
    from analysis.disagreement.disagreement_severity import classify_pairwise_severity, classify_sample_severity
    from analysis.disagreement.confidence_disagreement import compute_pairwise_confidence_analysis
    from analysis.disagreement.perturbation_disagreement import (
        load_perturbation_predictions,
        track_consensus_stability,
    )
    from analysis.disagreement.instability_analysis import (
        compute_model_instability,
        compute_sample_instability,
    )

    preds = load_perturbation_predictions(predictions)
    confidence_gaps = compute_confidence_gaps(preds)
    pairwise = compute_pairwise_confidence_analysis(preds)
    pairwise_sev = classify_pairwise_severity(pairwise)
    sample_severity = classify_sample_severity(pairwise_sev)
    consensus_stability = track_consensus_stability(preds)
    model_instability = compute_model_instability(preds)
    sample_instability = compute_sample_instability(preds)

    results = {}

    # V1: Imports
    results["imports"] = validate_imports()

    # V2: Consensus reliability scoring
    ok, reliability = validate_consensus_reliability_scoring(
        preds, confidence_gaps, sample_severity, consensus_stability, sample_instability
    )
    results["reliability_scoring"] = ok

    # V3: False consensus detection
    ok, fragile, false_cons, unstable = validate_false_consensus_detection(
        preds, confidence_gaps, consensus_stability, sample_instability
    )
    results["false_consensus"] = ok

    # V4: Trust classification
    ok, trust_labels = validate_trust_classification(reliability, false_cons)
    results["trust_classification"] = ok

    # V5: Consensus breakdown
    ok, breakdown = validate_consensus_breakdown(consensus_stability, sample_instability)
    results["breakdown"] = ok

    # V6: Consistency metrics
    ok, metrics = validate_consistency_metrics(
        preds, confidence_gaps, consensus_stability, model_instability
    )
    results["consistency_metrics"] = ok

    # V7: Model trust contribution
    ok, trust_contrib = validate_model_trust_contribution(model_instability)
    results["model_trust"] = ok

    # Compute summaries for exports
    from analysis.disagreement.consensus_reliability import (
        compute_reliability_summary,
        compute_model_trust_summary,
    )
    from analysis.disagreement.false_consensus_detection import compute_false_consensus_summary
    from analysis.disagreement.trust_analysis import compute_trust_summary

    reliability_summary = compute_reliability_summary(reliability)
    fc_summary = compute_false_consensus_summary(false_cons, fragile, unstable)
    trust_summary = compute_trust_summary(trust_labels)
    model_trust_summary = compute_model_trust_summary(trust_contrib)

    # V8: Exports
    results["exports"] = validate_exports(
        reliability, breakdown, metrics, trust_contrib,
        false_cons, fragile, unstable, trust_labels,
        reliability_summary, fc_summary, trust_summary, model_trust_summary
    )

    # V9: Error handling
    results["error_handling"] = validate_error_handling()

    # V10: Step compatibility
    results["step_compatibility"] = validate_step_compatibility(predictions)

    # V11: Architectural redundancy
    results["architecture"] = validate_architectural_redundancy()

    # ==== Final report ====
    logger.info("=" * 60)
    logger.info("VALIDATION REPORT")
    logger.info("=" * 60)

    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        logger.info("  [%s] %s", status, name)

    logger.info("-" * 60)
    if all_pass:
        logger.info("ALL VALIDATIONS PASSED ✓")
    else:
        logger.info("SOME VALIDATIONS FAILED ✗")
    logger.info("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
