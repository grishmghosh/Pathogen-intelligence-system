"""
Publication and research reporting layer.
"""

from reporting.experiment_provenance import (
    build_provenance_record,
    build_traceability_summary,
    export_provenance_manifest_json,
    export_traceability_summary_json,
)
from reporting.figure_packager import (
    package_figures,
    build_figure_manifest,
    build_figure_bundle,
    export_figure_manifest_json,
    export_figure_manifest_csv,
)
from reporting.narrative_summary import (
    build_narrative_summaries,
    export_narrative_json,
    export_narrative_text,
)
from reporting.publication_tables import (
    build_publication_table,
    build_publication_tables,
    build_model_ranking_table,
    build_perturbation_sensitivity_table,
    build_instability_table,
    build_uncertainty_table,
    build_attention_table,
    build_consensus_table,
    build_risk_table,
    export_publication_tables,
)
from reporting.reproducibility_report import (
    build_reproducibility_report,
    build_seed_tracking_summary,
    build_configuration_snapshot,
    build_dataset_metadata_summary,
    build_model_version_summary,
    export_reproducibility_report_json,
    export_configuration_snapshot_json,
)
from reporting.reporting_summary import (
    build_reporting_summary,
    build_reporting_summary_frame,
    export_reporting_summary_json,
    export_reporting_summary_csv,
)
from reporting.report_generator import ReportGenerator
