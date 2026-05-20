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
]
