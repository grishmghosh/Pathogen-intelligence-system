"""
Integrated publication and research report generator.
"""

from pathlib import Path
import json
import shutil

import pandas as pd

from reporting.experiment_provenance import build_provenance_record, build_traceability_summary, export_provenance_manifest_json, export_traceability_summary_json
from reporting.figure_packager import package_figures, build_figure_bundle, export_figure_manifest_csv, export_figure_manifest_json
from reporting.narrative_summary import build_narrative_summaries, export_narrative_json, export_narrative_text
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
from reporting.reporting_summary import build_reporting_summary, export_reporting_summary_csv, export_reporting_summary_json


DEFAULT_REPORT_ROOT = Path("results") / "reports"


def _read_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_series_to_frame(data):
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame([data])


class ReportGenerator:
    def __init__(self, report_id=None, report_root=None, experiment_paths=None, benchmark_paths=None, figure_sources=None):
        self.report_id = report_id or "report"
        self.report_root = Path(report_root) if report_root is not None else DEFAULT_REPORT_ROOT / f"report_{self.report_id}"
        self.experiment_paths = [Path(path) for path in (experiment_paths or [])]
        self.benchmark_paths = [Path(path) for path in (benchmark_paths or [])]
        self.figure_sources = [Path(path) for path in (figure_sources or [Path("results") / "plots", Path("results") / "benchmarks"])]

    def _ensure_directories(self):
        directories = {
            "tables": self.report_root / "tables",
            "figures": self.report_root / "figures",
            "narratives": self.report_root / "narratives",
            "manifests": self.report_root / "manifests",
            "provenance": self.report_root / "provenance",
            "summaries": self.report_root / "summaries",
        }
        for path in directories.values():
            path.mkdir(parents=True, exist_ok=True)
        return directories

    def _load_experiment_summary(self):
        for path in self.experiment_paths:
            if path.is_file():
                payload = _read_json(path)
                if payload is not None:
                    return payload, path
            if path.is_dir():
                candidate = path / "summaries" / "experiment_summary.json"
                if candidate.exists():
                    payload = _read_json(candidate)
                    if payload is not None:
                        return payload, candidate
        default_candidates = sorted(Path("results").glob("experiments/experiment_*/summaries/experiment_summary.json"))
        for candidate in default_candidates:
            payload = _read_json(candidate)
            if payload is not None:
                return payload, candidate
        legacy_candidates = [Path("results") / "plots" / "summaries" / "experiment_summary.json"]
        for candidate in legacy_candidates:
            payload = _read_json(candidate)
            if payload is not None:
                return payload, candidate
        return None, None

    def _load_benchmark_summary(self):
        for path in self.benchmark_paths:
            if path.is_file():
                payload = _read_json(path)
                if payload is not None:
                    return payload, path
            if path.is_dir():
                candidate = path / "statistical_summaries" / "benchmark_summary.json"
                if candidate.exists():
                    payload = _read_json(candidate)
                    if payload is not None:
                        return payload, candidate
        default_candidates = sorted(Path("results").glob("benchmarks/statistical_summaries/benchmark_summary.json"))
        for candidate in default_candidates:
            payload = _read_json(candidate)
            if payload is not None:
                return payload, candidate
        return None, None

    def _summary_to_tables(self, summary):
        summary = summary if isinstance(summary, dict) else {}
        model_rows = []
        for model_name, model_stats in (summary.get("per_model_stats", {}) or {}).items():
            if isinstance(model_stats, dict):
                row = {"model_name": model_name}
                row.update(model_stats)
                model_rows.append(row)
        model_table = pd.DataFrame(model_rows)

        perturbation_rows = []
        for perturbation_name, perturbation_stats in (summary.get("degradation_summary", {}) or {}).items():
            if isinstance(perturbation_stats, dict):
                row = {"perturbation_type": perturbation_name}
                row.update(perturbation_stats)
                row["degradation_impact"] = perturbation_stats.get("mean_drop")
                perturbation_rows.append(row)
        perturbation_table = pd.DataFrame(perturbation_rows)

        severity_rows = []
        if isinstance(summary.get("calibration_summary", {}), dict):
            for model_name, calibration_stats in summary.get("calibration_summary", {}).items():
                if isinstance(calibration_stats, dict):
                    severity_rows.append({
                        "model_name": model_name,
                        "mean_uncertainty_score": calibration_stats.get("ece"),
                        "mean_consensus_reliability_score": None,
                        "severity_level": "aggregated",
                    })
        severity_table = pd.DataFrame(severity_rows)

        correlation_table = pd.DataFrame()
        benchmark_like = {
            "model_summary": {
                "best_robustness_model": (summary.get("best_robustness", {}) or {}).get("model"),
                "most_stable_model": (summary.get("best_robustness", {}) or {}).get("model"),
                "lowest_risk_model": None,
                "best_attention_stability_model": None,
                "strongest_perturbation_resilience": (summary.get("best_robustness", {}) or {}).get("model"),
            },
            "perturbation_summary": {
                "most_disruptive_perturbation": None,
            },
            "severity_summary": {
                "trend_direction": "stable",
            },
            "correlation_summary": {},
            "summary_line": "Experiment summary loaded from visualization outputs.",
        }
        if not perturbation_table.empty and "mean_drop" in perturbation_table.columns:
            ordered = perturbation_table.sort_values("mean_drop", ascending=False)
            benchmark_like["perturbation_summary"]["most_disruptive_perturbation"] = str(ordered.iloc[0]["perturbation_type"])
        return model_table, perturbation_table, severity_table, correlation_table, benchmark_like

    def _load_benchmark_table(self, benchmark_root, relative_path, json_name=None):
        benchmark_root = Path(benchmark_root) if benchmark_root is not None else None
        candidates = []
        if benchmark_root is not None:
            candidates.append(benchmark_root / relative_path)
            if json_name is not None:
                candidates.append(benchmark_root / json_name)
        for candidate in candidates:
            if candidate is None or not candidate.exists():
                continue
            if candidate.suffix.lower() == ".csv":
                frame = _read_csv(candidate)
                if not frame.empty:
                    return frame, candidate
            else:
                payload = _read_json(candidate)
                if payload is not None:
                    return _safe_series_to_frame(payload), candidate
        return pd.DataFrame(), None

    def _load_config_from_summary(self, experiment_summary):
        if not isinstance(experiment_summary, dict):
            return {}
        manifest = experiment_summary.get("manifest") or experiment_summary.get("experiment_manifest") or {}
        config = manifest.get("config") if isinstance(manifest, dict) else {}
        if isinstance(config, dict):
            return config
        return experiment_summary.get("config", {}) if isinstance(experiment_summary.get("config"), dict) else {}

    def build(self):
        directories = self._ensure_directories()
        experiment_summary, experiment_summary_path = self._load_experiment_summary()
        benchmark_summary, benchmark_summary_path = self._load_benchmark_summary()
        if not isinstance(benchmark_summary, dict):
            benchmark_summary = {}
        if not isinstance(experiment_summary, dict):
            experiment_summary = {}
        if not benchmark_summary and experiment_summary:
            _, _, _, _, benchmark_summary = self._summary_to_tables(experiment_summary)

        config = self._load_config_from_summary(experiment_summary)
        manifest = experiment_summary.get("manifest", {}) if isinstance(experiment_summary, dict) else {}
        reproducibility = experiment_summary.get("reproducibility", {}) if isinstance(experiment_summary, dict) else {}
        provenance = build_provenance_record(
            experiment_id=(manifest.get("experiment_id") if isinstance(manifest, dict) else None) or (benchmark_summary.get("benchmark", {}).get("experiment_id") if isinstance(benchmark_summary, dict) else None),
            config=config,
            manifest=manifest,
            reproducibility=reproducibility,
            benchmark_summary=benchmark_summary,
        )
        traceability = build_traceability_summary(provenance)

        model_table = pd.DataFrame()
        perturbation_table = pd.DataFrame()
        severity_table = pd.DataFrame()
        correlation_table = pd.DataFrame()
        model_result = benchmark_summary.get("model_result", {}) if isinstance(benchmark_summary, dict) else {}
        perturbation_result = benchmark_summary.get("perturbation_result", {}) if isinstance(benchmark_summary, dict) else {}
        reliability_result = benchmark_summary.get("reliability_result", {}) if isinstance(benchmark_summary, dict) else {}
        if isinstance(model_result, dict):
            model_table = _safe_series_to_frame(model_result.get("rankings"))
        if isinstance(perturbation_result, dict):
            perturbation_table = _safe_series_to_frame(perturbation_result.get("rankings"))
            severity_table = _safe_series_to_frame(perturbation_result.get("severity_frame"))
        if isinstance(reliability_result, dict):
            correlation_table = _safe_series_to_frame(reliability_result.get("correlation_frame"))

        if model_table.empty:
            model_table = _safe_series_to_frame(benchmark_summary.get("summary", {}).get("model_summary", {}))
        if perturbation_table.empty:
            perturbation_table = _safe_series_to_frame(benchmark_summary.get("summary", {}).get("perturbation_summary", {}))
        if severity_table.empty:
            severity_table = _safe_series_to_frame(benchmark_summary.get("summary", {}).get("severity_summary", {}))

        benchmark_root = None
        if benchmark_summary_path is not None:
            benchmark_root = benchmark_summary_path.parent.parent if benchmark_summary_path.parent.name == "statistical_summaries" else benchmark_summary_path.parent
        if model_table.empty and benchmark_root is not None:
            model_table, _ = self._load_benchmark_table(benchmark_root, Path("model_rankings") / "model_benchmark_rankings.csv")
        if perturbation_table.empty and benchmark_root is not None:
            perturbation_table, _ = self._load_benchmark_table(benchmark_root, Path("perturbation_rankings") / "perturbation_benchmark_rankings.csv")
        if severity_table.empty and benchmark_root is not None:
            severity_table, _ = self._load_benchmark_table(benchmark_root, Path("perturbation_rankings") / "severity_benchmark.csv")
        if correlation_table.empty and benchmark_root is not None:
            correlation_table, _ = self._load_benchmark_table(benchmark_root, Path("correlation_analysis") / "correlation_matrix.csv")

        if model_table.empty and experiment_summary:
            model_table, perturbation_from_summary, severity_from_summary, correlation_from_summary, benchmark_from_experiment = self._summary_to_tables(experiment_summary)
            if perturbation_table.empty:
                perturbation_table = perturbation_from_summary
            if severity_table.empty:
                severity_table = severity_from_summary
            if correlation_table.empty:
                correlation_table = correlation_from_summary
            if not benchmark_summary:
                benchmark_summary = benchmark_from_experiment

        tables = build_publication_tables({
            "model_rankings": model_table if not model_table.empty else build_model_ranking_table(benchmark_summary.get("summary", {}).get("model_summary", {}) if isinstance(benchmark_summary, dict) else {}),
            "perturbation_sensitivity": perturbation_table if not perturbation_table.empty else build_perturbation_sensitivity_table(benchmark_summary.get("summary", {}).get("perturbation_summary", {}) if isinstance(benchmark_summary, dict) else {}),
            "instability_rankings": build_instability_table(benchmark_summary.get("summary", {}).get("statistics_summary", {}) if isinstance(benchmark_summary, dict) else {}),
            "uncertainty_rankings": build_uncertainty_table(benchmark_summary.get("summary", {}).get("statistics_summary", {}) if isinstance(benchmark_summary, dict) else {}),
            "attention_stability": build_attention_table(benchmark_summary.get("summary", {}).get("statistics_summary", {}) if isinstance(benchmark_summary, dict) else {}),
            "consensus_reliability": build_consensus_table(benchmark_summary.get("summary", {}).get("statistics_summary", {}) if isinstance(benchmark_summary, dict) else {}),
            "risk_summary": build_risk_table(benchmark_summary.get("summary", {}).get("statistics_summary", {}) if isinstance(benchmark_summary, dict) else {}),
        })

        if isinstance(benchmark_summary, dict):
            if not model_table.empty:
                tables["model_rankings"] = build_publication_table(model_table, name="model_rankings")
            if not perturbation_table.empty:
                tables["perturbation_rankings"] = build_publication_table(perturbation_table, name="perturbation_rankings")
            if not severity_table.empty:
                tables["severity_analysis"] = build_publication_table(severity_table, name="severity_analysis")
            if not correlation_table.empty:
                tables["correlation_analysis"] = build_publication_table(correlation_table, name="correlation_analysis")

        exported_tables = export_publication_tables(tables, directories["tables"])

        package_result = package_figures(self.figure_sources, directories["figures"])
        figure_manifest_csv = export_figure_manifest_csv(package_result, directories["manifests"] / "figure_manifest.csv")
        figure_manifest_json = export_figure_manifest_json(package_result, directories["manifests"] / "figure_manifest.json")
        figure_bundle = build_figure_bundle(package_result)

        narrative = build_narrative_summaries(
            model_table=tables.get("model_rankings", {}).get("dataframe") if isinstance(tables.get("model_rankings"), dict) else None,
            perturbation_table=tables.get("perturbation_rankings", {}).get("dataframe") if isinstance(tables.get("perturbation_rankings"), dict) else None,
            severity_table=severity_table,
            correlation_table=correlation_table,
            benchmark_summary=benchmark_summary if isinstance(benchmark_summary, dict) else {},
            provenance=provenance,
            reproducibility=reproducibility,
        )

        reproducibility_report = build_reproducibility_report(
            config=config,
            manifest=manifest,
            provenance=provenance,
            benchmark_summary=benchmark_summary,
            report_id=self.report_id,
        )
        config_snapshot = build_configuration_snapshot(config)
        seed_summary = build_seed_tracking_summary(reproducibility_report)
        dataset_summary = build_dataset_metadata_summary(config)
        model_summary = build_model_version_summary(config)

        report_summary = build_reporting_summary(
            report_id=self.report_id,
            provenance=provenance,
            reproducibility=reproducibility_report,
            tables=tables,
            figures=figure_bundle,
            narratives=narrative,
            benchmark_summary=benchmark_summary if isinstance(benchmark_summary, dict) else {},
        )

        provenance_path = export_provenance_manifest_json(provenance, directories["provenance"] / "provenance_manifest.json")
        traceability_path = export_traceability_summary_json(traceability, directories["provenance"] / "traceability_summary.json")
        reproducibility_path = export_reproducibility_report_json(reproducibility_report, directories["manifests"] / "reproducibility_report.json")
        config_snapshot_path = export_configuration_snapshot_json(config_snapshot, directories["manifests"] / "configuration_snapshot.json")
        export_narrative_json(narrative, directories["narratives"] / "narrative_summary.json")
        export_narrative_text(narrative, directories["narratives"] / "narrative_summary.txt")
        export_reporting_summary_json(report_summary, directories["summaries"] / "report_summary.json")
        export_reporting_summary_csv(report_summary, directories["summaries"] / "report_summary.csv")

        report_manifest = {
            "report_id": self.report_id,
            "report_root": str(self.report_root),
            "created_at": manifest.get("created_at") if isinstance(manifest, dict) else None,
            "experiment_summary_source": str(experiment_summary_path) if experiment_summary_path is not None else None,
            "benchmark_summary_source": str(benchmark_summary_path) if benchmark_summary_path is not None else None,
            "artifacts": {
                "tables": exported_tables,
                "figures": str(package_result.get("destination_dir")) if isinstance(package_result, dict) else None,
                "narratives": str(directories["narratives"]),
                "manifests": str(directories["manifests"]),
                "provenance": str(directories["provenance"]),
                "summaries": str(directories["summaries"]),
            },
            "summary_line": report_summary.get("summary_line"),
            "traceability": traceability,
            "seed_tracking": seed_summary,
            "dataset_summary": dataset_summary,
            "model_summary": model_summary,
            "figure_manifest_csv": str(figure_manifest_csv),
            "figure_manifest_json": str(figure_manifest_json),
            "reproducibility_report": str(reproducibility_path),
            "configuration_snapshot": str(config_snapshot_path),
            "provenance_manifest": str(provenance_path),
            "traceability_summary": str(traceability_path),
        }

        with (directories["manifests"] / "report_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(report_manifest, handle, indent=2, ensure_ascii=False)

        return {
            "report_id": self.report_id,
            "report_root": self.report_root,
            "report_manifest": report_manifest,
            "tables": tables,
            "figures": figure_bundle,
            "narrative": narrative,
            "provenance": provenance,
            "reproducibility": reproducibility_report,
            "report_summary": report_summary,
        }
