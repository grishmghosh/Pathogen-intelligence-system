"""
Uncertainty Quantification Foundation Layer for Pathogen Intelligence System (Step 6).

This package provides deterministic uncertainty metrics, entropy analysis,
confidence dispersion analysis, and summary/export helpers. It is designed to
sit on top of Steps 1–5 without rewriting or altering earlier layers.
"""

from analysis.uncertainty.uncertainty_metrics import (
    normalize_probability_vector,
    compute_prediction_entropy,
    compute_entropy_metrics,
    compute_entropy_summary,
)
from analysis.uncertainty.entropy_analysis import (
    compute_entropy_analysis,
    aggregate_sample_uncertainty,
    compute_uncertainty_severity_curve,
    rank_perturbation_uncertainty,
    detect_confidence_collapse,
)
from analysis.uncertainty.confidence_dispersion import (
    compute_confidence_dispersion,
    aggregate_confidence_dispersion,
    compute_dispersion_summary,
)
from analysis.uncertainty.uncertainty_summary import (
    classify_uncertainty_level,
    assign_uncertainty_labels,
    compute_uncertainty_summary,
    detect_uncertainty_conflicts,
    compute_disagreement_uncertainty_relationship,
    compute_model_uncertainty_profiles,
    compute_model_uncertainty_summary,
    generate_uncertainty_trend_summary,
    compute_confidence_collapse_summary,
    export_entropy_analysis_csv,
    export_entropy_analysis_json,
    export_confidence_dispersion_csv,
    export_confidence_dispersion_json,
    export_confidence_collapse_csv,
    export_confidence_collapse_json,
    export_uncertainty_trends_csv,
    export_uncertainty_trends_json,
    export_uncertainty_summary_json,
    export_relationship_summary_json,
    export_uncertainty_conflicts_csv,
    export_uncertainty_conflicts_json,
    export_model_uncertainty_profiles_csv,
    export_model_uncertainty_profiles_json,
    export_model_uncertainty_summary_json,
)

__all__ = [
    "normalize_probability_vector",
    "compute_prediction_entropy",
    "compute_entropy_metrics",
    "compute_entropy_summary",
    "compute_entropy_analysis",
    "aggregate_sample_uncertainty",
    "compute_uncertainty_severity_curve",
    "rank_perturbation_uncertainty",
    "detect_confidence_collapse",
    "compute_confidence_dispersion",
    "aggregate_confidence_dispersion",
    "compute_dispersion_summary",
    "classify_uncertainty_level",
    "assign_uncertainty_labels",
    "compute_uncertainty_summary",
    "detect_uncertainty_conflicts",
    "compute_disagreement_uncertainty_relationship",
    "compute_model_uncertainty_profiles",
    "compute_model_uncertainty_summary",
    "generate_uncertainty_trend_summary",
    "compute_confidence_collapse_summary",
    "export_entropy_analysis_csv",
    "export_entropy_analysis_json",
    "export_confidence_dispersion_csv",
    "export_confidence_dispersion_json",
    "export_confidence_collapse_csv",
    "export_confidence_collapse_json",
    "export_uncertainty_trends_csv",
    "export_uncertainty_trends_json",
    "export_uncertainty_summary_json",
    "export_relationship_summary_json",
    "export_uncertainty_conflicts_csv",
    "export_uncertainty_conflicts_json",
    "export_model_uncertainty_profiles_csv",
    "export_model_uncertainty_profiles_json",
    "export_model_uncertainty_summary_json",
]
