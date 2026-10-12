"""Domain adapters: owned artifacts preserve every transformation's lineage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.artifacts import ArtifactStore
from app.core.context import SkillContext
from app.core.errors import SkillError
from app.schemas.response import Explanation, ModelInfo, Prediction

if TYPE_CHECKING:
    from app.models.base import BasePredictor


class AnalysisService:
    def __init__(
        self,
        store: ArtifactStore,
        predictor: BasePredictor | None = None,
        *,
        retrieval=None,
        reports=None,
    ):
        self.store = store
        self.predictor = predictor
        self.retrieval = retrieval
        self.reports = reports

    def create_upload(self, content: bytes, context: SkillContext) -> str:
        return self.store.put(content, kind="upload", context=context)

    def inspect_csv(self, upload_ref: str, context: SkillContext) -> dict:
        from app.pipeline.preprocess import inspect_data, parse_csv

        content = self.store.get(upload_ref, kind="upload", context=context)
        frame = parse_csv(content)
        quality = inspect_data(frame)
        ref = self.store.put(
            {"frame": frame, "upload_ref": upload_ref, "quality_report": quality},
            kind="dataset",
            context=context,
            source_ref=upload_ref,
        )
        return {
            "dataset_ref": ref,
            "columns": list(frame.columns),
            "row_count": len(frame),
            "quality_report": quality,
        }

    def process_signal(self, dataset_ref: str, context: SkillContext) -> dict:
        from app.pipeline.preprocess import preprocess

        dataset = self.store.get(dataset_ref, kind="dataset", context=context)
        windows, report = preprocess(dataset["frame"])
        ref = self.store.put(
            {"windows": windows, "dataset_ref": dataset_ref, "cleaning_report": report},
            kind="window_set",
            context=context,
            source_ref=dataset_ref,
        )
        return {
            "window_set_ref": ref,
            "window_count": len(windows),
            "cleaning_report": report,
            "warnings": report.get("warnings", []),
        }

    def extract(self, window_set_ref: str, context: SkillContext) -> dict:
        from app.pipeline.feature_engineering import FEATURE_SCHEMA_VERSION, extract_features

        window_set = self.store.get(window_set_ref, kind="window_set", context=context)
        frame = extract_features(window_set["windows"])
        ref = self.store.put(
            {"frame": frame, "window_set_ref": window_set_ref},
            kind="feature_set",
            context=context,
            source_ref=window_set_ref,
            schema_version=FEATURE_SCHEMA_VERSION,
        )
        return {
            "feature_set_ref": ref,
            "feature_names": list(frame.columns),
            "window_count": len(frame),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        }

    def predict(self, feature_set_ref: str, context: SkillContext) -> dict:
        feature_set = self.store.get(feature_set_ref, kind="feature_set", context=context)
        predictor = self._predictor()
        raw = predictor.predict(feature_set["frame"])
        prediction = self._prediction(raw)
        ref = self.store.put(
            {"result": prediction.model_dump(mode="json"), "feature_set_ref": feature_set_ref},
            kind="prediction",
            context=context,
            source_ref=feature_set_ref,
        )
        return {"prediction_ref": ref, **prediction.model_dump(mode="json")}

    def explain(self, prediction_ref: str, context: SkillContext, top_k: int = 5) -> dict:
        stored = self.store.get(prediction_ref, kind="prediction", context=context)
        features = self.store.get(stored["feature_set_ref"], kind="feature_set", context=context)
        raw = self._predictor().explain(features["frame"], top_k=top_k)
        explanation = self._explanation(raw, stored["result"]["highest_risk_window"])
        ref = self.store.put(
            {"prediction_ref": prediction_ref, "explanation": explanation.model_dump(mode="json")},
            kind="analysis",
            context=context,
            source_ref=prediction_ref,
        )
        return {"analysis_ref": ref, **explanation.model_dump(mode="json")}

    def visualize(self, analysis_ref: str, plot_types: list[str], context: SkillContext) -> dict:
        from app.visualization.plots import render_plots

        analysis = self.store.get(analysis_ref, kind="analysis", context=context)
        prediction = self.store.get(analysis["prediction_ref"], kind="prediction", context=context)
        features = self.store.get(
            prediction["feature_set_ref"], kind="feature_set", context=context
        )
        window_set = self.store.get(features["window_set_ref"], kind="window_set", context=context)
        plots = render_plots(
            window_set["windows"], prediction["result"], analysis["explanation"], plot_types
        )
        artifacts = []
        for plot in plots:
            ref = self.store.put(plot, kind="image", context=context, source_ref=analysis_ref)
            artifacts.append(
                {
                    "artifact_ref": ref,
                    "kind": plot["kind"],
                    "mime_type": plot["mime_type"],
                    "url": f"/artifacts/{ref}",
                }
            )
        return {"artifacts": artifacts}

    def _predictor(self) -> BasePredictor:
        if self.predictor is None:
            raise SkillError("model_unavailable", "The configured model is not available.")
        return self.predictor

    def _prediction(self, raw: dict) -> Prediction:
        model = dict(raw["model"])
        name, version = model.pop("name"), model.pop("version")
        bundle_metadata = self._predictor().metadata
        # The model card is public; training subjects and reference arrays stay in the bundle.
        metadata = {
            key: bundle_metadata[key]
            for key in (
                "input_schema_version",
                "feature_schema_version",
                "input_profile",
                "score_semantics",
                "aggregation",
                "threshold",
                "threshold_source",
                "risk_thresholds",
                "processing_config",
                "limitations",
            )
            if key in bundle_metadata
        }
        scope = model.pop(
            "task_scope", bundle_metadata.get("task_scope", "synthetic_physiology_v1")
        )
        metadata.update(model)
        return Prediction.model_validate(
            {
                **{
                    key: raw[key]
                    for key in (
                        "prediction",
                        "risk_score",
                        "important_features",
                        "aggregation",
                        "window_count",
                        "anomalous_ratio",
                        "highest_risk_window",
                        "window_predictions",
                    )
                },
                "risk_level": raw.get(
                    "risk_level",
                    "high"
                    if raw["risk_score"] >= 0.8
                    else "medium"
                    if raw["risk_score"] >= 0.5
                    else "low",
                ),
                "model": ModelInfo(name=name, version=version, task_scope=scope, metadata=metadata),
            }
        )

    @staticmethod
    def _explanation(raw: dict, highest_risk_window: int) -> Explanation:
        return Explanation.model_validate(
            {
                **raw,
                "highest_risk_window": raw.get("highest_risk_window", highest_risk_window),
            }
        )
