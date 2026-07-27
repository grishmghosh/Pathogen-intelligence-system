"""
Disagreement Analysis Subsystem for Pathogen Intelligence System.

This package provides pairwise model agreement analysis and basic
disagreement detection. It computes agreement matrices, identifies
prediction disagreements across models, and exports structured results.

Architecture Flow:
    Inference Predictions -> Agreement Metrics -> Disagreement Export

Modules (Step 1):
    agreement_metrics   - Pairwise agreement matrix and disagreement statistics
    disagreement_utils  - Input parsing, validation, and disagreement detection
    disagreement_export - CSV and JSON export of analysis results

Modules (Step 2):
    confidence_disagreement - Confidence gap analysis and enriched detection
    disagreement_severity   - Rule-based severity classification
    disagreement_scoring    - Normalised disagreement scoring and Step 2 exports

Modules (Step 3):
    perturbation_disagreement - Perturbation-aware disagreement detection
    instability_analysis      - Model and sample instability scoring
    disagreement_trends       - Trend analysis and Step 3 exports

Modules (Step 4):
    consensus_reliability      - Consensus reliability scoring and breakdown
    false_consensus_detection  - Fragile/false/unstable consensus detection
    trust_analysis             - Trust labelling and Step 4 exports

Modules (Step 5):
    risk_estimation            - Deterministic reliability risk scoring and model profiling
    risk_classification        - Risk level labels and distribution summaries
    risk_summary               - Risk summaries, fragility analysis, trend analysis, and exports
"""

# ---- Step 1 ----
from analysis.disagreement.agreement_metrics import (
    compute_agreement_matrix,
    compute_disagreement_statistics,
)
from analysis.disagreement.disagreement_utils import (
    load_predictions,
    detect_disagreements,
)
from analysis.disagreement.disagreement_export import (
    export_agreement_matrix_csv,
    export_agreement_matrix_json,
    export_disagreements_csv,
    export_disagreements_json,
    export_statistics_json,
)

# ---- Step 2 ----
from analysis.disagreement.confidence_disagreement import (
    compute_confidence_gaps,
    compute_pairwise_confidence_analysis,
    compute_confidence_spread_summary,
)
from analysis.disagreement.disagreement_severity import (
    classify_pairwise_severity,
    classify_sample_severity,
    compute_severity_summary,
)
from analysis.disagreement.disagreement_scoring import (
    compute_disagreement_scores,
    compute_score_summary,
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

# ---- Step 3 ----
from analysis.disagreement.perturbation_disagreement import (
    load_perturbation_predictions,
    detect_perturbation_induced_disagreements,
    compute_perturbation_sensitivity,
    track_consensus_stability,
    compute_severity_disagreement_rates,
)
from analysis.disagreement.instability_analysis import (
    compute_model_instability,
    compute_sample_instability,
    compute_instability_summary,
)
from analysis.disagreement.disagreement_trends import (
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

# ---- Step 4 ----
from analysis.disagreement.consensus_reliability import (
    compute_consensus_reliability,
    compute_reliability_summary,
    compute_consensus_breakdown,
    compute_consensus_consistency_metrics,
    compute_model_trust_contribution,
    compute_model_trust_summary,
)
from analysis.disagreement.false_consensus_detection import (
    detect_fragile_consensus,
    detect_false_consensus,
    detect_unstable_agreement,
    compute_false_consensus_summary,
)
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

# ---- Step 5 ----
from analysis.disagreement.risk_estimation import (
    compute_reliability_risk_scores,
    compute_model_reliability_risk_profiles,
    compute_reliability_risk_summary,
)
from analysis.disagreement.risk_classification import (
    classify_reliability_risk_level,
    assign_reliability_risk_labels,
    compute_risk_classification_summary,
)
from analysis.disagreement.risk_summary import (
    compute_risk_summary,
    compute_risk_contributor_summary,
    compute_perturbation_fragility_analysis,
    compute_model_risk_summary,
    generate_risk_trend_summary,
    export_risk_analysis_csv,
    export_risk_analysis_json,
    export_model_risk_profile_csv,
    export_model_risk_profile_json,
    export_fragility_analysis_csv,
    export_fragility_analysis_json,
    export_risk_trends_csv,
    export_risk_trends_json,
)

__all__ = [
    # Step 1
    "compute_agreement_matrix",
    "compute_disagreement_statistics",
    "load_predictions",
    "detect_disagreements",
    "export_agreement_matrix_csv",
    "export_agreement_matrix_json",
    "export_disagreements_csv",
    "export_disagreements_json",
    "export_statistics_json",
    # Step 2 - Confidence
    "compute_confidence_gaps",
    "compute_pairwise_confidence_analysis",
    "compute_confidence_spread_summary",
    # Step 2 - Severity
    "classify_pairwise_severity",
    "classify_sample_severity",
    "compute_severity_summary",
    # Step 2 - Scoring
    "compute_disagreement_scores",
    "compute_score_summary",
    # Step 2 - Exports
    "export_confidence_gaps_csv",
    "export_confidence_gaps_json",
    "export_pairwise_analysis_csv",
    "export_pairwise_analysis_json",
    "export_confidence_summary_json",
    "export_severity_csv",
    "export_severity_json",
    "export_severity_summary_json",
    "export_score_summary_json",
    # Step 3 - Perturbation
    "load_perturbation_predictions",
    "detect_perturbation_induced_disagreements",
    "compute_perturbation_sensitivity",
    "track_consensus_stability",
    "compute_severity_disagreement_rates",
    # Step 3 - Instability
    "compute_model_instability",
    "compute_sample_instability",
    "compute_instability_summary",
    # Step 3 - Trends
    "generate_escalation_trend",
    "generate_model_comparison_trend",
    "generate_perturbation_ranking_trend",
    "generate_full_trend_report",
    # Step 3 - Exports
    "export_induced_disagreements_csv",
    "export_induced_disagreements_json",
    "export_perturbation_sensitivity_csv",
    "export_perturbation_sensitivity_json",
    "export_severity_rates_csv",
    "export_consensus_stability_csv",
    "export_consensus_stability_json",
    "export_model_instability_csv",
    "export_model_instability_json",
    "export_sample_instability_csv",
    "export_sample_instability_json",
    "export_instability_summary_json",
    "export_trend_report_json",
    # Step 4 - Consensus Reliability
    "compute_consensus_reliability",
    "compute_reliability_summary",
    "compute_consensus_breakdown",
    "compute_consensus_consistency_metrics",
    "compute_model_trust_contribution",
    "compute_model_trust_summary",
    # Step 4 - False Consensus Detection
    "detect_fragile_consensus",
    "detect_false_consensus",
    "detect_unstable_agreement",
    "compute_false_consensus_summary",
    # Step 4 - Trust Analysis
    "classify_trust_level",
    "assign_trust_labels",
    "compute_trust_summary",
    # Step 4 - Exports
    "export_consensus_reliability_csv",
    "export_consensus_reliability_json",
    "export_reliability_summary_json",
    "export_consensus_breakdown_csv",
    "export_consensus_breakdown_json",
    "export_consistency_metrics_json",
    "export_model_trust_contribution_csv",
    "export_model_trust_contribution_json",
    "export_model_trust_summary_json",
    "export_false_consensus_csv",
    "export_false_consensus_json",
    "export_fragile_consensus_csv",
    "export_fragile_consensus_json",
    "export_unstable_agreement_csv",
    "export_unstable_agreement_json",
    "export_false_consensus_summary_json",
    "export_trust_labels_csv",
    "export_trust_labels_json",
    "export_trust_summary_json",
    # Step 5 - Risk estimation
    "compute_reliability_risk_scores",
    "compute_model_reliability_risk_profiles",
    "compute_reliability_risk_summary",
    # Step 5 - Risk classification
    "classify_reliability_risk_level",
    "assign_reliability_risk_labels",
    "compute_risk_classification_summary",
    # Step 5 - Risk summary and exports
    "compute_risk_summary",
    "compute_risk_contributor_summary",
    "compute_perturbation_fragility_analysis",
    "compute_model_risk_summary",
    "generate_risk_trend_summary",
    "export_risk_analysis_csv",
    "export_risk_analysis_json",
    "export_model_risk_profile_csv",
    "export_model_risk_profile_json",
    "export_fragility_analysis_csv",
    "export_fragility_analysis_json",
    "export_risk_trends_csv",
    "export_risk_trends_json",
]


