"""
Integrated experimental evaluation pipeline for Step 8.

The runner orchestrates the existing subsystems without rewriting them. It is
failure-tolerant, supports selective stage execution, and emits reproducibility
manifests plus consolidated experiment summaries.
"""

from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from inference.batch_inference import run_batch_inference
from perturbations.perturbation_engine import generate_perturbations
from analysis.robustness_analyzer import generate_robustness_report
from analysis.disagreement import (
    load_predictions,
    load_perturbation_predictions,
    detect_disagreements,
    compute_agreement_matrix,
    compute_disagreement_statistics,
    compute_confidence_gaps,
    compute_pairwise_confidence_analysis,
    compute_confidence_spread_summary,
    classify_pairwise_severity,
    classify_sample_severity,
    compute_severity_summary,
    compute_disagreement_scores,
    compute_score_summary,
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
    compute_consensus_reliability,
    compute_reliability_summary,
    compute_consensus_breakdown,
    compute_consensus_consistency_metrics,
    compute_model_trust_contribution,
    compute_model_trust_summary,
    detect_fragile_consensus,
    detect_false_consensus,
    detect_unstable_agreement,
    compute_false_consensus_summary,
    classify_trust_level,
    assign_trust_labels,
    compute_trust_summary,
    compute_reliability_risk_scores,
    compute_model_reliability_risk_profiles,
    compute_reliability_risk_summary,
    classify_reliability_risk_level,
    assign_reliability_risk_labels,
    compute_risk_classification_summary,
    compute_risk_summary,
    compute_risk_contributor_summary,
    compute_perturbation_fragility_analysis,
    compute_model_risk_summary,
    generate_risk_trend_summary,
)
from analysis.uncertainty import (
    normalize_probability_vector,
    compute_prediction_entropy,
    compute_entropy_metrics,
    compute_entropy_summary,
    compute_entropy_analysis,
    aggregate_sample_uncertainty,
    compute_uncertainty_severity_curve,
    rank_perturbation_uncertainty,
    detect_confidence_collapse,
    compute_confidence_dispersion,
    aggregate_confidence_dispersion,
    compute_dispersion_summary,
    classify_uncertainty_level,
    assign_uncertainty_labels,
    compute_uncertainty_summary,
    detect_uncertainty_conflicts,
    compute_disagreement_uncertainty_relationship,
    compute_model_uncertainty_profiles,
    compute_model_uncertainty_summary,
    generate_uncertainty_trend_summary,
    compute_confidence_collapse_summary,
)
from analysis.explainability import (
    generate_gradcam,
    compute_attention_analysis,
    aggregate_attention_summary,
    compute_attention_drift_analysis,
    compute_attention_stability_curve,
    compute_attention_collapse_report,
    compute_cross_model_attention_comparison,
    compute_attention_divergence_ranking,
    compute_attention_consistency_summary,
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

from pipeline.experiment_config import load_experiment_config, validate_experiment_config, save_experiment_config_json
from pipeline.experiment_registry import create_experiment_manifest, register_experiment, DEFAULT_REGISTRY_PATH
from pipeline.reproducibility import build_reproducibility_manifest, build_experiment_snapshot, export_reproducibility_manifest_json, export_experiment_snapshot_json
from pipeline.experiment_summary import build_consolidated_summary, build_summary_frame, export_consolidated_summary_csv, export_consolidated_summary_json
from pipeline.pipeline_utils import (
    build_experiment_root,
    ensure_experiment_directories,
    ensure_directory,
    current_timestamp,
    generate_experiment_id,
    collect_execution_report,
    stage_record,
    safe_json_dump,
    coerce_dataframe,
    dataframe_metrics,
    experiment_snapshot,
    traceback_string,
)


DEFAULT_STAGE_ORDER = [
    "inference",
    "perturbation_generation",
    "robustness",
    "calibration",
    "disagreement",
    "consensus",
    "risk",
    "uncertainty",
    "explainability",
]


class ExperimentRunner:
    def __init__(self, config=None, stage_handlers=None, output_root=None, registry_path=None):
        validation = validate_experiment_config(config)
        self.config = validation["config"]
        self.config_validation = validation
        self.stage_handlers = stage_handlers or {}
        self.experiment_id = self.config.get("experiment_id") or generate_experiment_id()
        self.output_root = build_experiment_root(self.experiment_id, root_dir=output_root)
        self.directories = ensure_experiment_directories(self.output_root)
        self.registry_path = Path(registry_path) if registry_path is not None else DEFAULT_REGISTRY_PATH
        self.stage_results = {}
        self.stage_outputs = {}
        self.samples = self._build_samples()
        self.manifest = None
        self.registry_record = None
        self.reproducibility_manifest = None
        self.summary = None

    def _build_samples(self):
        samples = []
        dataset = self.config.get("dataset", {}) if isinstance(self.config, dict) else {}
        if dataset.get("samples"):
            for sample in dataset.get("samples", []):
                if isinstance(sample, dict):
                    entry = dict(sample)
                else:
                    entry = {"image_path": sample}
                entry.setdefault("sample_id", entry.get("id") or entry.get("image_id") or f"sample_{len(samples) + 1}")
                if entry.get("image_path") is not None:
                    entry["image_path"] = str(entry["image_path"])
                samples.append(entry)
        elif dataset.get("sample_paths"):
            for index, image_path in enumerate(dataset.get("sample_paths", [])):
                samples.append({"sample_id": f"sample_{index + 1}", "image_path": str(image_path)})
        return samples

    def run(self):
        self.manifest = create_experiment_manifest(self.config, self.experiment_id, self.output_root, validation=self.config_validation)
        save_experiment_config_json(self.config, self.directories["configs"] / "experiment_config.json")
        safe_json_dump(self.directories["manifests"] / "experiment_manifest.json", self.manifest)
        self.registry_record = register_experiment(self.manifest, self.registry_path)

        for stage_name in self.config.get("enabled_stages", DEFAULT_STAGE_ORDER):
            self._run_stage(stage_name)

        self.stage_results["execution_report"] = collect_execution_report(self.stage_results)
        self.reproducibility_manifest = build_reproducibility_manifest(
            self.config,
            self.manifest,
            self.stage_results,
            seed=self.config.get("seed"),
            model_versions=self.config.get("models", {}),
            dataset_metadata=self.config.get("dataset", {}),
            perturbation_metadata=self.config.get("perturbations", {}),
        )
        self.stage_results["reproducibility_manifest"] = self.reproducibility_manifest
        self.stage_results["registry_record"] = self.registry_record
        self.stage_results["experiment_manifest"] = self.manifest
        self.stage_results["summary"] = build_consolidated_summary(
            self.stage_results,
            manifest=self.manifest,
            registry_record=self.registry_record,
            reproducibility_manifest=self.reproducibility_manifest,
            config=self.config,
        )
        self.stage_results["summary_frame"] = build_summary_frame(self.stage_results["summary"])

        self._export_run_artifacts()
        return {
            "experiment_id": self.experiment_id,
            "output_root": str(self.output_root),
            "manifest": self.manifest,
            "registry_record": self.registry_record,
            "reproducibility_manifest": self.reproducibility_manifest,
            "execution_report": self.stage_results["execution_report"],
            "summary": self.stage_results["summary"],
            "stage_results": self.stage_results,
        }

    def _stage_enabled(self, stage_name):
        stage_flags = self.config.get("stage_flags", {})
        if stage_flags:
            return bool(stage_flags.get(stage_name, False))
        return stage_name in (self.config.get("enabled_stages") or DEFAULT_STAGE_ORDER)

    def _run_stage(self, stage_name):
        if not self._stage_enabled(stage_name):
            started = time.time()
            finished = time.time()
            self.stage_results[stage_name] = stage_record(stage_name, "skipped", started, finished, skipped_reason="disabled in config")
            return self.stage_results[stage_name]

        handler = self.stage_handlers.get(stage_name)
        started = time.time()
        try:
            if handler is not None:
                output = handler(self)
            else:
                output = getattr(self, f"_stage_{stage_name}")()
            finished = time.time()
            record = stage_record(stage_name, "completed", started, finished, output=_serialise_stage_output(output))
            self.stage_results[stage_name] = record
            self.stage_outputs[stage_name] = output
            return record
        except Exception:
            finished = time.time()
            record = stage_record(stage_name, "failed", started, finished, error=traceback_string())
            self.stage_results[stage_name] = record
            self.stage_outputs[stage_name] = {"error": record["error"], "status": "failed"}
            return record

    def _models_config(self):
        models = self.config.get("models", {})
        if isinstance(models, dict) and models:
            return models
        return None

    def _stage_inference(self):
        if not self.samples:
            return {"status": "skipped", "reason": "No samples available for inference"}
        model_config = self._models_config()
        per_sample = []
        for sample in self.samples:
            image_path = sample.get("image_path")
            if image_path is None:
                continue
            inference_result = run_batch_inference(image_path, models_config=model_config)
            per_sample.append({
                "sample_id": sample.get("sample_id"),
                "image_path": image_path,
                "true_label": sample.get("true_label"),
                "inference_result": inference_result,
            })
        frame = build_inference_frame(per_sample)
        return {
            "status": "completed",
            "summary": {
                "sample_count": int(len(per_sample)),
                "model_count": int(len(model_config or {})),
                "inference_row_count": int(frame.shape[0]),
            },
            "data": frame,
            "per_sample": per_sample,
        }

    def _stage_perturbation_generation(self):
        if not self.samples:
            return {"status": "skipped", "reason": "No samples available for perturbation generation"}
        records = []
        for sample in self.samples:
            image_path = sample.get("image_path")
            if image_path is None:
                continue
            perturbations = generate_perturbations(image_path)
            for name, payload in (perturbations or {}).items():
                records.append({
                    "sample_id": sample.get("sample_id"),
                    "image_path": image_path,
                    "perturbation_name": name,
                    "perturbation_type": payload.get("type") if isinstance(payload, dict) else None,
                    "severity_level": payload.get("severity_level") if isinstance(payload, dict) else None,
                    "parameter": payload.get("parameter") if isinstance(payload, dict) else None,
                    "metadata": payload.get("metadata") if isinstance(payload, dict) else None,
                })
        frame = pd.DataFrame(records)
        return {
            "status": "completed",
            "summary": {
                "sample_count": int(len(self.samples)),
                "perturbation_count": int(frame.shape[0]),
            },
            "data": frame,
        }

    def _stage_robustness(self):
        inference_payload = self.stage_outputs.get("inference", {})
        per_sample = inference_payload.get("per_sample", []) if isinstance(inference_payload, dict) else []
        reports = []
        for entry in per_sample:
            inference_result = entry.get("inference_result")
            if inference_result is None:
                continue
            report = generate_robustness_report(inference_result)
            reports.append({"sample_id": entry.get("sample_id"), "report": report})
        frame = pd.DataFrame(reports)
        return {
            "status": "completed",
            "summary": {
                "sample_count": int(len(reports)),
                "report_count": int(len(reports)),
            },
            "reports": reports,
            "data": frame,
        }

    def _stage_calibration(self):
        frame = build_calibration_frame(self.stage_outputs.get("inference", {}))
        summary = build_calibration_summary(frame)
        return {"status": "completed", "summary": summary, "data": frame}

    def _stage_disagreement(self):
        frame = build_flat_prediction_frame(self.stage_outputs.get("inference", {}))
        if frame.empty:
            return {"status": "skipped", "reason": "No flattened prediction rows available"}
        loaded = load_perturbation_predictions(frame)
        agreement_matrix = compute_agreement_matrix(loaded)
        disagreements = detect_disagreements(loaded)
        disagreement_stats = compute_disagreement_statistics(loaded, disagreements)
        confidence_gaps = compute_confidence_gaps(loaded)
        pairwise_analysis = compute_pairwise_confidence_analysis(loaded)
        confidence_summary = compute_confidence_spread_summary(pairwise_analysis)
        severity_summary = compute_severity_summary(classify_pairwise_severity(pairwise_analysis))
        disagreement_scores = compute_disagreement_scores(pairwise_analysis)
        score_summary = compute_score_summary(disagreement_scores)
        perturbation_disagreements = detect_perturbation_induced_disagreements(loaded)
        perturbation_sensitivity = compute_perturbation_sensitivity(loaded)
        consensus_stability = track_consensus_stability(loaded)
        severity_rates = compute_severity_disagreement_rates(loaded)
        instability_model = compute_model_instability(loaded)
        instability_sample = compute_sample_instability(loaded)
        instability_summary = compute_instability_summary(
            instability_model,
            instability_sample,
            severity_rates,
        )
        trend_report = generate_full_trend_report(
            severity_rates=severity_rates,
            model_instability=instability_model,
            sensitivity=perturbation_sensitivity,
        )
        return {
            "status": "completed",
            "summary": {
                "loaded_rows": int(frame.shape[0]),
                "agreement_rows": int(agreement_matrix.shape[0]) if hasattr(agreement_matrix, "shape") else 0,
                "disagreement_rows": int(disagreements.shape[0]) if hasattr(disagreements, "shape") else 0,
                "perturbation_disagreements": int(perturbation_disagreements.shape[0]) if hasattr(perturbation_disagreements, "shape") else 0,
            },
            "data": loaded,
            "agreement_matrix": agreement_matrix,
            "disagreement_statistics": disagreement_stats,
            "disagreements": disagreements,
            "confidence_gaps": confidence_gaps,
            "pairwise_analysis": pairwise_analysis,
            "confidence_summary": confidence_summary,
            "severity_summary": severity_summary,
            "disagreement_scores": disagreement_scores,
            "score_summary": score_summary,
            "perturbation_disagreements": perturbation_disagreements,
            "perturbation_sensitivity": perturbation_sensitivity,
            "consensus_stability": consensus_stability,
            "severity_rates": severity_rates,
            "instability_model": instability_model,
            "instability_sample": instability_sample,
            "instability_summary": instability_summary,
            "trend_report": trend_report,
        }

    def _stage_consensus(self):
        disagreement_stage = self.stage_outputs.get("disagreement", {})
        consensus_stability = disagreement_stage.get("consensus_stability")
        predictions = self.stage_outputs.get("disagreement", {}).get("data")
        if predictions is None or not isinstance(predictions, pd.DataFrame) or predictions.empty:
            predictions = build_flat_prediction_frame(self.stage_outputs.get("inference", {}))
        if predictions is None or predictions.empty:
            return {"status": "skipped", "reason": "No prediction frame available for consensus analysis"}
        reliability = compute_consensus_reliability(predictions)
        reliability_summary = compute_reliability_summary(reliability)
        consensus_breakdown = compute_consensus_breakdown(
            consensus_stability,
            sample_instability=disagreement_stage.get("instability_sample"),
        )
        consistency_metrics = compute_consensus_consistency_metrics(
            predictions,
            confidence_gaps=disagreement_stage.get("confidence_gaps"),
            consensus_stability=consensus_stability,
            model_instability=disagreement_stage.get("instability_model"),
        )
        model_trust_contribution = compute_model_trust_contribution(
            disagreement_stage.get("instability_model"),
            predictions=predictions,
        )
        model_trust_summary = compute_model_trust_summary(model_trust_contribution)
        false_consensus = detect_false_consensus(
            predictions,
            confidence_gaps=disagreement_stage.get("confidence_gaps"),
            consensus_stability=disagreement_stage.get("consensus_stability"),
            sample_instability=disagreement_stage.get("instability_sample"),
        )
        fragile_consensus = detect_fragile_consensus(
            predictions,
            consensus_stability=disagreement_stage.get("consensus_stability"),
        )
        unstable_agreement = detect_unstable_agreement(consensus_stability)
        trust_labels = assign_trust_labels(reliability, false_consensus=false_consensus)
        trust_summary = compute_trust_summary(trust_labels)
        false_consensus_summary = compute_false_consensus_summary(
            false_consensus,
            fragile_consensus,
            unstable_agreement,
        )
        return {
            "status": "completed",
            "summary": {
                "reliability_rows": int(reliability.shape[0]),
                "trust_rows": int(trust_labels.shape[0]),
            },
            "reliability": reliability,
            "reliability_summary": reliability_summary,
            "consensus_breakdown": consensus_breakdown,
            "consistency_metrics": consistency_metrics,
            "model_trust_contribution": model_trust_contribution,
            "model_trust_summary": model_trust_summary,
            "false_consensus": false_consensus,
            "fragile_consensus": fragile_consensus,
            "unstable_agreement": unstable_agreement,
            "false_consensus_summary": false_consensus_summary,
            "trust_labels": trust_labels,
            "trust_summary": trust_summary,
        }

    def _stage_risk(self):
        disagreement_stage = self.stage_outputs.get("disagreement", {})
        consensus_stage = self.stage_outputs.get("consensus", {})
        predictions = disagreement_stage.get("data")
        if predictions is None or not isinstance(predictions, pd.DataFrame) or predictions.empty:
            predictions = build_flat_prediction_frame(self.stage_outputs.get("inference", {}))
        if predictions is None or predictions.empty:
            return {"status": "skipped", "reason": "No prediction frame available for risk analysis"}
        pairwise_disagreement = disagreement_stage.get("pairwise_analysis")
        confidence_gaps = disagreement_stage.get("confidence_gaps")
        consensus_reliability = consensus_stage.get("reliability")
        sample_instability = disagreement_stage.get("instability_sample")
        consensus_breakdown = consensus_stage.get("consensus_breakdown")
        consensus_stability = disagreement_stage.get("consensus_stability")
        false_consensus = consensus_stage.get("false_consensus")
        fragile_consensus = consensus_stage.get("fragile_consensus")
        risk_scores = compute_reliability_risk_scores(
            pairwise_disagreement=pairwise_disagreement,
            confidence_gaps=confidence_gaps,
            consensus_reliability=consensus_reliability,
            sample_instability=sample_instability,
            fragile_consensus=fragile_consensus,
            false_consensus=false_consensus,
            consensus_breakdown=consensus_breakdown,
            consensus_stability=consensus_stability,
        )
        model_risk_profiles = compute_model_reliability_risk_profiles(
            pairwise_disagreement=pairwise_disagreement,
            model_instability=disagreement_stage.get("instability_model"),
            model_trust_contribution=consensus_stage.get("model_trust_contribution"),
        )
        risk_summary = compute_risk_summary(risk_scores)
        contributor_summary = compute_risk_contributor_summary(risk_scores)
        fragility_analysis = compute_perturbation_fragility_analysis(
            risk_scores=risk_scores,
            perturbation_sensitivity=disagreement_stage.get("perturbation_sensitivity"),
            severity_rates=disagreement_stage.get("severity_rates"),
            consensus_stability=consensus_stability,
        )
        model_risk_summary = compute_model_risk_summary(model_risk_profiles)
        trend_summary = generate_risk_trend_summary(risk_scores, consensus_stability=consensus_stability)
        classified = assign_reliability_risk_labels(risk_scores)
        risk_classification_summary = compute_risk_classification_summary(classified)
        return {
            "status": "completed",
            "summary": risk_summary,
            "risk_scores": risk_scores,
            "model_risk_profiles": model_risk_profiles,
            "contributor_summary": contributor_summary,
            "fragility_analysis": fragility_analysis,
            "model_risk_summary": model_risk_summary,
            "trend_summary": trend_summary,
            "risk_classification_summary": risk_classification_summary,
            "classified_risk": classified,
        }

    def _stage_uncertainty(self):
        frame = build_flat_prediction_frame(self.stage_outputs.get("inference", {}))
        if frame.empty:
            return {"status": "skipped", "reason": "No prediction frame available for uncertainty analysis"}
        entropy_input = ensure_probability_columns(frame)
        entropy_metrics = compute_entropy_metrics(entropy_input)
        entropy_summary = compute_entropy_summary(entropy_input)
        entropy_analysis = compute_entropy_analysis(entropy_input)
        sample_uncertainty = aggregate_sample_uncertainty(entropy_analysis)
        severity_curve = compute_uncertainty_severity_curve(entropy_analysis)
        perturbation_ranking = rank_perturbation_uncertainty(entropy_analysis)
        collapse_events = detect_confidence_collapse(entropy_analysis)
        dispersion = compute_confidence_dispersion(entropy_input)
        dispersion_summary = compute_dispersion_summary(dispersion)
        dispersion_aggregate = aggregate_confidence_dispersion(dispersion)
        labels = assign_uncertainty_labels(entropy_analysis)
        uncertainty_summary = compute_uncertainty_summary(entropy_analysis)
        conflicts = detect_uncertainty_conflicts(entropy_analysis, dispersion)
        disagreement_relationship = compute_disagreement_uncertainty_relationship(
            entropy_analysis,
            disagreement_scores=self.stage_outputs.get("disagreement", {}).get("data"),
            sample_instability=self.stage_outputs.get("disagreement", {}).get("instability_sample"),
        )
        model_profiles = compute_model_uncertainty_profiles(entropy_analysis)
        model_summary = compute_model_uncertainty_summary(model_profiles)
        trend_summary = generate_uncertainty_trend_summary(entropy_analysis, dispersion_analysis=dispersion)
        collapse_summary = compute_confidence_collapse_summary(collapse_events)
        return {
            "status": "completed",
            "summary": entropy_summary,
            "entropy_metrics": entropy_metrics,
            "entropy_analysis": entropy_analysis,
            "sample_uncertainty": sample_uncertainty,
            "severity_curve": severity_curve,
            "perturbation_ranking": perturbation_ranking,
            "collapse_events": collapse_events,
            "dispersion": dispersion,
            "dispersion_summary": dispersion_summary,
            "dispersion_aggregate": dispersion_aggregate,
            "labels": labels,
            "uncertainty_summary": uncertainty_summary,
            "conflicts": conflicts,
            "disagreement_relationship": disagreement_relationship,
            "model_profiles": model_profiles,
            "model_summary": model_summary,
            "trend_summary": trend_summary,
            "collapse_summary": collapse_summary,
        }

    def _stage_explainability(self):
        explainability_inputs = self.config.get("metadata", {}).get("explainability_inputs", [])
        if not explainability_inputs:
            return {"status": "skipped", "reason": "No explainability inputs were provided"}
        records = []
        for index, item in enumerate(explainability_inputs):
            model = item.get("model")
            image = item.get("image")
            if model is None or image is None:
                continue
            gradcam = generate_gradcam(model, image, target_class=item.get("target_class"), target_layer=item.get("target_layer"))
            records.append({
                "sample_id": item.get("sample_id", f"explainability_{index + 1}"),
                "model_name": item.get("model_name", f"model_{index + 1}"),
                "severity_level": item.get("severity_level", "clean"),
                "perturbation_type": item.get("perturbation_type", "clean"),
                "gradcam": gradcam,
            })
        if not records:
            return {"status": "skipped", "reason": "No valid explainability records were generated"}
        attention_frame = build_attention_frame(records)
        drift_frame = compute_attention_drift_analysis(attention_frame)
        comparison_frame = compute_cross_model_attention_comparison(attention_frame)
        summary = compute_explainability_summary(
            attention_frame,
            drift_analysis=drift_frame,
            comparison_analysis=comparison_frame,
            uncertainty_data=self.stage_outputs.get("uncertainty", {}).get("entropy_analysis"),
            disagreement_data=self.stage_outputs.get("disagreement", {}).get("data"),
            instability_data=self.stage_outputs.get("disagreement", {}).get("instability_sample"),
        )
        drift_summary = compute_explainability_drift_summary(drift_frame)
        relationship_summary = compute_explainability_relationship_summary(
            drift_frame,
            uncertainty_data=self.stage_outputs.get("uncertainty", {}).get("entropy_analysis"),
            disagreement_data=self.stage_outputs.get("disagreement", {}).get("data"),
            instability_data=self.stage_outputs.get("disagreement", {}).get("instability_sample"),
        )
        consistency_summary = compute_attention_consistency_summary(comparison_frame)
        return {
            "status": "completed",
            "summary": summary,
            "attention_analysis": attention_frame,
            "drift_analysis": drift_frame,
            "comparison_analysis": comparison_frame,
            "drift_summary": drift_summary,
            "relationship_summary": relationship_summary,
            "consistency_summary": consistency_summary,
        }

    def _export_run_artifacts(self):
        config_path = self.directories["configs"] / "experiment_config.json"
        manifest_path = self.directories["manifests"] / "experiment_manifest.json"
        reproducibility_path = self.directories["manifests"] / "reproducibility_manifest.json"
        snapshot_path = self.directories["manifests"] / "experiment_snapshot.json"
        summary_json_path = self.directories["summaries"] / "experiment_summary.json"
        summary_csv_path = self.directories["summaries"] / "experiment_summary.csv"
        safe_json_dump(manifest_path, self.manifest)
        save_reproducibility = export_reproducibility_manifest_json(self.reproducibility_manifest, reproducibility_path)
        snapshot = build_experiment_snapshot(self.stage_results)
        export_experiment_snapshot_json(snapshot, snapshot_path)
        export_consolidated_summary_json(self.stage_results["summary"], summary_json_path)
        export_consolidated_summary_csv(self.stage_results["summary"], summary_csv_path)
        safe_json_dump(self.directories["logs"] / "execution_report.json", self.stage_results["execution_report"])
        safe_json_dump(self.directories["logs"] / "stage_snapshot.json", snapshot)
        safe_json_dump(self.directories["logs"] / "registry_record.json", self.registry_record)
        return {
            "config_path": str(config_path),
            "manifest_path": str(manifest_path),
            "reproducibility_path": str(save_reproducibility),
            "snapshot_path": str(snapshot_path),
            "summary_json_path": str(summary_json_path),
            "summary_csv_path": str(summary_csv_path),
        }


def _serialise_stage_output(output):
    if isinstance(output, pd.DataFrame):
        return dataframe_metrics(output, "stage_output")
    if isinstance(output, dict):
        serialised = {}
        for key, value in output.items():
            if isinstance(value, pd.DataFrame):
                serialised[key] = dataframe_metrics(value, key)
            else:
                serialised[key] = value if not isinstance(value, (pd.Series, np.ndarray)) else str(type(value).__name__)
        return serialised
    return output


def build_inference_frame(per_sample_outputs):
    rows = []
    for sample in per_sample_outputs or []:
        sample_id = sample.get("sample_id")
        inference_result = sample.get("inference_result") or {}
        if not isinstance(inference_result, dict):
            continue
        for model_name, model_result in inference_result.items():
            if not isinstance(model_result, dict) or "error" in model_result:
                continue
            for perturbation_name, prediction in model_result.items():
                if not isinstance(prediction, dict) or "error" in prediction:
                    continue
                metadata = prediction.get("metadata") or {}
                rows.append({
                    "sample_id": str(sample_id),
                    "model_name": str(model_name),
                    "perturbation": str(perturbation_name),
                    "perturbation_type": str(metadata.get("type", perturbation_name)),
                    "severity_level": str(metadata.get("severity_level", metadata.get("severity", perturbation_name))),
                    "predicted_class": prediction.get("prediction"),
                    "confidence": prediction.get("confidence"),
                    "probabilities": prediction.get("probabilities"),
                    "predicted_idx": prediction.get("predicted_idx"),
                    "image_path": sample.get("image_path"),
                    "true_label": sample.get("true_label"),
                })
    return pd.DataFrame(rows)


def build_flat_prediction_frame(inference_stage):
    if not isinstance(inference_stage, dict):
        return pd.DataFrame()
    per_sample = inference_stage.get("per_sample") or []
    return build_inference_frame(per_sample)


def ensure_probability_columns(frame):
    frame = coerce_dataframe(frame)
    if frame.empty:
        return frame
    if "probabilities" not in frame.columns:
        frame["probabilities"] = [[] for _ in range(len(frame))]
    def _row_probs(row):
        probabilities = row.get("probabilities")
        if isinstance(probabilities, list) and probabilities:
            array = np.asarray(probabilities, dtype=float)
            total = float(array.sum())
            if total > 0:
                return (array / total).tolist()
        confidence = row.get("confidence")
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = float(np.clip(confidence, 0.0, 1.0))
        remaining = max(0.0, 1.0 - confidence)
        if "predicted_idx" in row and pd.notna(row.get("predicted_idx")):
            predicted_idx = int(row.get("predicted_idx"))
        else:
            predicted_idx = 0
        size = 4
        values = np.full(size, remaining / max(size - 1, 1), dtype=float)
        values[predicted_idx % size] = confidence
        return values.tolist()
    frame = frame.copy()
    frame["probabilities"] = frame.apply(_row_probs, axis=1)
    return frame


def build_calibration_frame(inference_stage):
    frame = build_flat_prediction_frame(inference_stage)
    if frame.empty:
        return pd.DataFrame()
    frame = ensure_probability_columns(frame)
    records = []
    for model_name, group in frame.groupby("model_name"):
        confidences = pd.to_numeric(group["confidence"], errors="coerce").fillna(0.0)
        if "true_label" in group.columns and group["true_label"].notna().any():
            correctness = (group["true_label"].astype(str) == group["predicted_class"].astype(str)).astype(int)
            mode = "true_label"
        else:
            original = group[group["perturbation"].astype(str).str.lower() == "original"]
            if original.empty:
                original_prediction = group.iloc[0]["predicted_class"]
            else:
                original_prediction = original.iloc[0]["predicted_class"]
            correctness = (group["predicted_class"].astype(str) == str(original_prediction)).astype(int)
            mode = "proxy"
        bins = np.linspace(0, 1, 11)
        bin_ids = np.digitize(confidences, bins, right=True)
        ece = 0.0
        total = float(len(group))
        calibration_rows = []
        for bin_index in range(1, len(bins)):
            mask = bin_ids == bin_index
            if not mask.any():
                continue
            bin_conf = float(confidences[mask].mean())
            bin_acc = float(correctness[mask].mean())
            weight = float(mask.sum()) / total if total > 0 else 0.0
            ece += weight * abs(bin_acc - bin_conf)
            calibration_rows.append({
                "model_name": model_name,
                "bin": int(bin_index),
                "bin_confidence": round(bin_conf, 6),
                "bin_accuracy": round(bin_acc, 6),
                "bin_count": int(mask.sum()),
            })
        records.append({
            "model_name": model_name,
            "sample_count": int(len(group)),
            "mean_confidence": round(float(confidences.mean()), 6),
            "confidence_std": round(float(confidences.std(ddof=0)), 6) if len(confidences) > 1 else 0.0,
            "prediction_consistency": round(float(correctness.mean()), 6),
            "expected_calibration_error": round(float(ece), 6),
            "calibration_mode": mode,
            "calibration_bins": calibration_rows,
        })
    return pd.DataFrame(records)


def build_calibration_summary(frame):
    if frame is None or frame.empty:
        return {"status": "empty", "total_models": 0, "mean_expected_calibration_error": 0.0}
    ece = pd.to_numeric(frame["expected_calibration_error"], errors="coerce").dropna()
    consistency = pd.to_numeric(frame["prediction_consistency"], errors="coerce").dropna()
    return {
        "status": "completed",
        "total_models": int(len(frame)),
        "mean_expected_calibration_error": round(float(ece.mean()), 6) if not ece.empty else 0.0,
        "max_expected_calibration_error": round(float(ece.max()), 6) if not ece.empty else 0.0,
        "mean_prediction_consistency": round(float(consistency.mean()), 6) if not consistency.empty else 0.0,
        "calibration_mode_distribution": frame["calibration_mode"].astype(str).value_counts().to_dict() if "calibration_mode" in frame.columns else {},
    }


def build_attention_frame(records):
    rows = []
    for record in records or []:
        gradcam = record.get("gradcam") or {}
        if not isinstance(gradcam, dict):
            continue
        if gradcam.get("status") != "ok":
            continue
        heatmap = gradcam.get("normalized_heatmap")
        if heatmap is None:
            heatmap = gradcam.get("raw_heatmap")
        rows.append({
            "sample_id": record.get("sample_id"),
            "model_name": record.get("model_name"),
            "severity_level": record.get("severity_level"),
            "perturbation_type": record.get("perturbation_type"),
            "attention_map": heatmap,
        })
    return pd.DataFrame(rows)
