"""
Step 9 benchmark runner for large-scale experiment evaluation.
"""

from pathlib import Path
import json

import pandas as pd

from experiments.benchmark_summary import (
    build_benchmark_summary,
    build_benchmark_summary_frame,
    export_benchmark_summary_csv,
    export_benchmark_summary_json,
)
from experiments.experiment_statistics import export_csv, export_json, to_json_ready
from experiments.model_benchmarking import benchmark_models, export_model_benchmark_outputs
from experiments.perturbation_benchmarking import benchmark_perturbations, export_perturbation_benchmark_outputs
from experiments.reliability_benchmarking import benchmark_reliability_relationships, export_reliability_benchmark_outputs


DEFAULT_BENCHMARK_ROOT = Path("results") / "benchmarks"


class BenchmarkRunner:
    def __init__(self, records=None, benchmark_root=None, experiment_paths=None):
        self.records = records
        self.benchmark_root = Path(benchmark_root) if benchmark_root is not None else DEFAULT_BENCHMARK_ROOT
        self.experiment_paths = [Path(path) for path in (experiment_paths or [])]

    def discover_experiment_summaries(self, root=None):
        root_path = Path(root) if root is not None else Path("results") / "experiments"
        if not root_path.exists():
            return []
        return sorted(root_path.glob("experiment_*/summaries/experiment_summary.json"))

    def _load_json(self, path):
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    def _records_from_summary(self, summary):
        if summary is None:
            return []
        if isinstance(summary, list):
            return summary
        if isinstance(summary, pd.DataFrame):
            return summary.to_dict(orient="records")
        if not isinstance(summary, dict):
            return []
        if "benchmark_records" in summary and isinstance(summary["benchmark_records"], list):
            return summary["benchmark_records"]
        if "records" in summary and isinstance(summary["records"], list):
            return summary["records"]
        if "model_rankings" in summary or "perturbation_rankings" in summary:
            records = []
            model_summary = summary.get("model_rankings")
            if isinstance(model_summary, list):
                for row in model_summary:
                    if isinstance(row, dict):
                        row = dict(row)
                        row["benchmark_scope"] = "model"
                        records.append(row)
            perturbation_summary = summary.get("perturbation_rankings")
            if isinstance(perturbation_summary, list):
                for row in perturbation_summary:
                    if isinstance(row, dict):
                        row = dict(row)
                        row["benchmark_scope"] = "perturbation"
                        records.append(row)
            return records
        if "stage_summaries" in summary and isinstance(summary["stage_summaries"], dict):
            records = []
            for stage_name, stage_payload in summary["stage_summaries"].items():
                if isinstance(stage_payload, dict):
                    record = {"benchmark_scope": stage_name}
                    record.update(stage_payload)
                    records.append(record)
            return records
        return []

    def load_records(self):
        if self.records is not None:
            if isinstance(self.records, pd.DataFrame):
                return self.records.copy()
            if isinstance(self.records, list):
                return pd.DataFrame(self.records)
            if isinstance(self.records, dict):
                return pd.DataFrame([self.records])
            return pd.DataFrame()
        records = []
        paths = self.experiment_paths or self.discover_experiment_summaries()
        for path in paths:
            summary = self._load_json(path)
            records.extend(self._records_from_summary(summary))
        return pd.DataFrame(records)

    def _ensure_directories(self):
        directories = {
            "model_rankings": self.benchmark_root / "model_rankings",
            "perturbation_rankings": self.benchmark_root / "perturbation_rankings",
            "severity_analysis": self.benchmark_root / "severity_analysis",
            "correlation_analysis": self.benchmark_root / "correlation_analysis",
            "leaderboards": self.benchmark_root / "leaderboards",
            "statistical_summaries": self.benchmark_root / "statistical_summaries",
        }
        for path in directories.values():
            path.mkdir(parents=True, exist_ok=True)
        return directories

    def run(self):
        records_frame = self.load_records()
        directories = self._ensure_directories()

        model_result = benchmark_models(records_frame)
        perturbation_result = benchmark_perturbations(records_frame)
        reliability_result = benchmark_reliability_relationships(records_frame)

        benchmark_summary = build_benchmark_summary(
            model_result=model_result,
            perturbation_result=perturbation_result,
            reliability_result=reliability_result,
            statistics_result={
                "model_statistics": model_result.get("statistics", {}),
                "perturbation_statistics": perturbation_result.get("statistics", pd.DataFrame()).to_dict(orient="records") if isinstance(perturbation_result.get("statistics"), pd.DataFrame) else perturbation_result.get("statistics", {}),
            },
        )

        export_model_benchmark_outputs(model_result, directories["model_rankings"])
        export_perturbation_benchmark_outputs(perturbation_result, directories["perturbation_rankings"])
        export_reliability_benchmark_outputs(reliability_result, directories["correlation_analysis"])

        combined_leaderboards = pd.DataFrame([
            {"leaderboard": "best_robustness_model", "value": benchmark_summary["leaderboards"].get("best_robustness_model")},
            {"leaderboard": "most_stable_model", "value": benchmark_summary["leaderboards"].get("most_stable_model")},
            {"leaderboard": "lowest_risk_model", "value": benchmark_summary["leaderboards"].get("lowest_risk_model")},
            {"leaderboard": "best_attention_stability_model", "value": benchmark_summary["leaderboards"].get("best_attention_stability_model")},
            {"leaderboard": "strongest_perturbation_resilience", "value": benchmark_summary["leaderboards"].get("strongest_perturbation_resilience")},
            {"leaderboard": "most_disruptive_perturbation", "value": benchmark_summary["leaderboards"].get("most_disruptive_perturbation")},
        ])

        export_csv(directories["leaderboards"] / "benchmark_leaderboards.csv", combined_leaderboards)
        export_json(directories["leaderboards"] / "benchmark_leaderboards.json", combined_leaderboards.to_dict(orient="records"))

        benchmark_statistics_frame = build_benchmark_summary_frame(benchmark_summary)
        export_csv(directories["statistical_summaries"] / "benchmark_summary.csv", benchmark_statistics_frame)
        export_json(directories["statistical_summaries"] / "benchmark_summary.json", benchmark_summary)
        export_benchmark_summary_csv(benchmark_summary, directories["statistical_summaries"] / "benchmark_summary_flat.csv")
        export_benchmark_summary_json(benchmark_summary, directories["statistical_summaries"] / "benchmark_summary_full.json")

        return {
            "benchmark_root": self.benchmark_root,
            "records": records_frame,
            "model_result": model_result,
            "perturbation_result": perturbation_result,
            "reliability_result": reliability_result,
            "summary": benchmark_summary,
            "leaderboards": combined_leaderboards,
        }
