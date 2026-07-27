"""
Visualization & Experimentation Layer for Pathogen Intelligence System.

This package provides modular scientific visualization infrastructure
for generating plots and experiment summaries from evaluation outputs.

Modules:
    - visualization_utils: Shared helpers, theme setup, data loaders
    - robustness_plots: Accuracy/confidence vs severity, model comparison
    - calibration_plots: Reliability diagrams, confidence histograms
    - heatmaps: Prediction flip heatmaps, instability comparisons
    - experiment_summary: Aggregated statistics, CSV/JSON reports
"""

from visualization.visualization_utils import setup_plot_theme, ensure_output_dirs
from visualization.robustness_plots import (
    plot_accuracy_vs_severity,
    plot_confidence_vs_severity,
    plot_model_comparison,
    plot_per_class_robustness_degradation,
)
from visualization.calibration_plots import (
    plot_reliability_diagram,
    plot_confidence_histogram,
    plot_calibration_curve,
    plot_expected_vs_actual_accuracy,
)
from visualization.heatmaps import (
    plot_prediction_flip_heatmap,
    plot_severity_vs_instability_heatmap,
    plot_model_instability_comparison,
)
from visualization.experiment_summary import (
    generate_experiment_summary,
    export_summary_csv,
    export_summary_json,
)

__all__ = [
    "setup_plot_theme",
    "ensure_output_dirs",
    "plot_accuracy_vs_severity",
    "plot_confidence_vs_severity",
    "plot_model_comparison",
    "plot_per_class_robustness_degradation",
    "plot_reliability_diagram",
    "plot_confidence_histogram",
    "plot_calibration_curve",
    "plot_expected_vs_actual_accuracy",
    "plot_prediction_flip_heatmap",
    "plot_severity_vs_instability_heatmap",
    "plot_model_instability_comparison",
    "generate_experiment_summary",
    "export_summary_csv",
    "export_summary_json",
]
