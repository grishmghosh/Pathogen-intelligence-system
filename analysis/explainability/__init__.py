"""
Explainability & Attention Analysis Layer for Pathogen Intelligence System (Step 7).
"""

from analysis.explainability.gradcam_generator import (
    generate_gradcam,
    create_attention_overlay,
    save_heatmap_png,
    save_overlay_png,
)
from analysis.explainability.attention_analysis import (
    normalize_attention_map,
    compute_attention_features,
    compute_attention_analysis,
    aggregate_attention_summary,
)
from analysis.explainability.attention_drift import (
    compute_attention_drift_metrics,
    compute_attention_drift_analysis,
    compute_attention_stability_curve,
    compute_attention_collapse_report,
)
from analysis.explainability.attention_comparison import (
    compare_attention_maps,
    compute_cross_model_attention_comparison,
    compute_attention_divergence_ranking,
    compute_attention_consistency_summary,
)
from analysis.explainability.explainability_summary import (
    compute_explainability_summary,
    compute_explainability_drift_summary,
    compute_explainability_relationship_summary,
    export_attention_heatmap_png,
    export_attention_overlay_png,
    export_attention_drift_png,
    export_attention_comparison_png,
    export_attention_analysis_csv,
    export_attention_analysis_json,
    export_attention_drift_csv,
    export_attention_drift_json,
    export_attention_comparison_csv,
    export_attention_comparison_json,
    export_explainability_summary_csv,
    export_explainability_summary_json,
    export_drift_summary_json,
    export_consistency_summary_json,
    export_relationship_summary_json,
)

__all__ = [
    "generate_gradcam",
    "create_attention_overlay",
    "save_heatmap_png",
    "save_overlay_png",
    "normalize_attention_map",
    "compute_attention_features",
    "compute_attention_analysis",
    "aggregate_attention_summary",
    "compute_attention_drift_metrics",
    "compute_attention_drift_analysis",
    "compute_attention_stability_curve",
    "compute_attention_collapse_report",
    "compare_attention_maps",
    "compute_cross_model_attention_comparison",
    "compute_attention_divergence_ranking",
    "compute_attention_consistency_summary",
    "compute_explainability_summary",
    "compute_explainability_drift_summary",
    "compute_explainability_relationship_summary",
    "export_attention_heatmap_png",
    "export_attention_overlay_png",
    "export_attention_drift_png",
    "export_attention_comparison_png",
    "export_attention_analysis_csv",
    "export_attention_analysis_json",
    "export_attention_drift_csv",
    "export_attention_drift_json",
    "export_attention_comparison_csv",
    "export_attention_comparison_json",
    "export_explainability_summary_csv",
    "export_explainability_summary_json",
    "export_drift_summary_json",
    "export_consistency_summary_json",
    "export_relationship_summary_json",
]
