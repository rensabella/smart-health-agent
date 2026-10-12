"""Deterministic execution and publication guards, not a medical safety classifier."""

from __future__ import annotations

import math
from copy import deepcopy

from app.core.errors import SkillError
from app.harness.config import AGENT_ROLES, HarnessConfig, load_config, with_limits
from app.schemas.response import AnalyzeResponse, Explanation, Prediction

_LINEAGE = {
    "csv_analysis": ("dataset_ref", "dataset", "upload_ref"),
    "signal_processing": ("window_set_ref", "window_set", "dataset_ref"),
    "feature_extraction": ("feature_set_ref", "feature_set", "window_set_ref"),
    "anomaly_prediction": ("prediction_ref", "prediction", "feature_set_ref"),
    "feature_explanation": ("analysis_ref", "analysis", "prediction_ref"),
}
UNCERTAINTY = "模型仅用合成数据训练，未经临床验证；分数和特征贡献不能证明病因或患病概率。"
HIGH_SCORE_RECOMMENDATION = (
    "请先核对输入协议、数据完整性和采集质量，再复测分析；合成模型高分不能作为诊疗依据。"
)


def analysis_summary(predicted: Prediction) -> str:
    return (
        f"分析了 {predicted.window_count} 个 30 秒窗口；最高合成异常模型分数为 "
        f"{predicted.risk_score:.3f}，对应窗口 {predicted.highest_risk_window}（从 0 编号）。"
        "该分数不表示临床患病概率或医学诊断；局部贡献描述模型行为，不证明异常原因。"
    )


def score_definition(metadata: dict) -> dict:
    return {
        "kind": "score_definition",
        "summary": "risk_score 是当前模型对合成异常模式的分数。每 30 秒窗口产生一个分数，会话取最高窗口分数；它不是临床患病概率。",
        "source": {
            "type": "model_card",
            "name": metadata["name"],
            "version": metadata["version"],
            "score_semantics": metadata["score_semantics"],
            "aggregation": metadata["aggregation"],
        },
    }


def _require(condition, code="harness_invalid_output"):
    if not condition:
        raise SkillError(code, "Output does not satisfy the active Harness policy.")


class HarnessValidator:
    def __init__(self, config: HarnessConfig | None = None, *, limits=None):
        self._config = with_limits(config if config is not None else load_config(), limits)

    @property
    def config(self):
        return self._config

    @property
    def limits(self):
        return self.config.agent.limits

    @property
    def version(self):
        return self.config.agent.version

    @property
    def fingerprint(self):
        return self.config.fingerprint

    @property
    def enabled_skills(self):
        return frozenset(s.name for s in self.config.skills.skills if s.enabled)

    def allowed_skills(self, role, *, simple=False):
        if simple or role not in AGENT_ROLES:
            return frozenset()
        policy = getattr(self.config.agent.roles, role)
        if policy is None:
            return frozenset()
        return frozenset(policy.allowed_skills) & self.enabled_skills

    def role_tool_limit(self, role):
        field = {
            "data_analysis": "max_data_tool_calls",
            "rag": "max_rag_tool_calls",
            "report_writer": "max_writer_tool_calls",
            "safety_verification": "max_safety_tool_calls",
        }.get(role)
        return getattr(self.limits, field) if field else 0

    def delegation_allowed(self, role, target, *, depth):
        return depth == 0 and target in self.allowed_delegates(role)

    def allowed_delegates(self, role):
        if role != "planner":
            return frozenset()
        return frozenset(self.config.agent.roles.planner.allowed_delegates)

    def validate_catalogue(self, descriptions):
        catalogue = {item["name"]: item for item in descriptions}
        declared = {item.name: item for item in self.config.skills.skills}
        if set(catalogue) != set(declared) or any(
            catalogue[name]["version"] != policy.version for name, policy in declared.items()
        ):
            raise ValueError("Harness Skill catalogue/version does not match installed Skills")

    def skill_timeout(self, name, version, implementation_timeout):
        policy = next((p for p in self.config.skills.skills if p.name == name), None)
        if policy is None or not policy.enabled:
            raise SkillError("skill_disabled", "This Skill is disabled by operator policy.")
        if version != policy.version:
            raise SkillError("skill_version_mismatch", "Skill version differs from active policy.")
        return min(policy.timeout_seconds, implementation_timeout)

    def check_call(self, role, name, *, simple, used, role_used, context):
        context.check_active()
        if name not in self.allowed_skills(role, simple=simple):
            raise SkillError("permission_denied", "This role is not allowed to call this Skill.")
        if used >= self.limits.max_tool_calls or role_used >= self.role_tool_limit(role):
            raise SkillError("budget_exceeded", "The configured tool call budget is exhausted.")

    def check_prediction(self, predicted: Prediction, metadata: dict):
        """Schema checks alone cannot prove cross-field consistency or model scope."""
        windows = predicted.window_predictions
        _require(len(windows) == predicted.window_count)
        _require([w.window_index for w in windows] == list(range(len(windows))))
        highest = max(windows, key=lambda w: w.risk_score)
        _require(predicted.highest_risk_window == highest.window_index)
        _require(math.isclose(predicted.risk_score, highest.risk_score, abs_tol=1e-12))
        _require(metadata.get("task_scope") == "synthetic_anomaly_only")
        _require(metadata.get("score_semantics") == "synthetic_anomaly_model_score")
        _require(metadata.get("input_profile") == "synthetic_physiology_v1")
        _require(metadata.get("aggregation") == predicted.aggregation == "max")
        _require(predicted.model.task_scope == metadata["task_scope"])
        _require(
            (predicted.model.name, predicted.model.version)
            == (metadata["name"], metadata["version"])
        )
        for key in (
            "score_semantics",
            "input_profile",
            "aggregation",
            "threshold",
            "risk_thresholds",
            "processing_config",
            "feature_schema_version",
            "limitations",
        ):
            _require(key in metadata and predicted.model.metadata.get(key) == metadata[key])
        threshold = metadata["threshold"]
        labels = ["anomaly" if w.risk_score >= threshold else "normal" for w in windows]
        _require(labels == [w.prediction for w in windows])
        _require(
            predicted.prediction == ("anomaly" if predicted.risk_score >= threshold else "normal")
        )
        _require(
            math.isclose(
                predicted.anomalous_ratio, labels.count("anomaly") / len(labels), abs_tol=1e-12
            )
        )
        levels = metadata["risk_thresholds"]
        level = (
            "high"
            if predicted.risk_score >= levels["high"]
            else "medium"
            if predicted.risk_score >= levels["medium"]
            else "low"
        )
        _require(predicted.risk_level == level)
        _require(bool(metadata["limitations"]))

    def check_skill_output(self, name, payload, data, *, context, service):
        """Check declared artifact lineage before observations or API responses receive facts."""
        try:
            if name in _LINEAGE:
                ref_field, kind, parent_field = _LINEAGE[name]
                stored = service.store.get(data[ref_field], kind=kind, context=context)
                service.store.check_source(
                    data[ref_field], source_ref=payload[parent_field], kind=kind, context=context
                )
                _require(stored[parent_field] == payload[parent_field])
                if name == "csv_analysis":
                    _require(data["row_count"] == len(stored["frame"]) > 0)
                    _require(data["columns"] == list(stored["frame"].columns))
                    _require(data["quality_report"] == stored["quality_report"])
                elif name == "signal_processing":
                    _require(data["window_count"] == len(stored["windows"]) > 0)
                    _require(data["cleaning_report"] == stored["cleaning_report"])
                    _require(data["warnings"] == stored["cleaning_report"].get("warnings", []))
                elif name == "feature_extraction":
                    from app.pipeline.feature_engineering import FEATURE_SCHEMA_VERSION

                    _require(data["window_count"] == len(stored["frame"]) > 0)
                    _require(data["feature_names"] == list(stored["frame"].columns))
                    _require(data["feature_schema_version"] == FEATURE_SCHEMA_VERSION)
                elif name == "anomaly_prediction":
                    prediction = Prediction.model_validate(
                        {k: data[k] for k in Prediction.model_fields}
                    )
                    self.check_prediction(prediction, service.predictor.metadata)
                    feature_set = service.store.get(
                        payload["feature_set_ref"], kind="feature_set", context=context
                    )
                    _require(prediction.window_count == len(feature_set["frame"]))
                    _require(prediction.model_dump(mode="json") == stored["result"])
                elif name == "feature_explanation":
                    explanation = Explanation.model_validate(
                        {k: data[k] for k in Explanation.model_fields}
                    )
                    _require(explanation.model_dump(mode="json") == stored["explanation"])
                    _require(bool(explanation.local_evidence) and bool(explanation.limitations))
            elif name == "visualization":
                _require(
                    sorted(a["kind"] for a in data["artifacts"]) == sorted(payload["plot_types"])
                )
                for artifact in data["artifacts"]:
                    service.store.check_source(
                        artifact["artifact_ref"],
                        source_ref=payload["analysis_ref"],
                        kind="image",
                        context=context,
                    )
                    image = service.store.get(
                        artifact["artifact_ref"], kind="image", context=context
                    )
                    _require(artifact["url"] == f"/artifacts/{artifact['artifact_ref']}")
                    _require(
                        image["kind"] == artifact["kind"] and image["mime_type"] == "image/png"
                    )
            elif name == "rag_retrieval":
                from app.rag.schemas import RetrievalResult

                stored = service.store.get(data["evidence_ref"], kind="evidence", context=context)
                service.store.check_source(
                    data["evidence_ref"],
                    source_ref=payload["analysis_ref"],
                    kind="evidence",
                    context=context,
                )
                result = RetrievalResult.model_validate(stored["result"])
                _require(stored["analysis_ref"] == payload["analysis_ref"])
                _require(data["hit_count"] == len(result.hits) and data["status"] == result.status)
                _require(data["knowledge_version"] == result.knowledge_version)
                _require(data["embedding_id"] == result.embedding_id)
            elif name in {"report_generation", "safety_verification"}:
                service.reports.validate_skill_output(name, payload, data, context)
        except (ValueError, TypeError, KeyError, AttributeError):
            raise SkillError(
                "harness_invalid_output", "Skill facts failed Harness validation."
            ) from None

    def validate_publication(self, candidate, *, objective, context, service):
        """Admit fixed facts only. No free-form claim can obtain a medical safety approval."""
        try:
            context.check_active()
            if objective == "health_report" and self.config.safety.medical_report_enabled:
                _require(set(candidate) == {"kind", "analysis", "report", "review_mode"})
                _require(candidate["kind"] == "health_report")
                _require(candidate["review_mode"] in {"llm_assisted", "scripted_test"})
                checked = self.validate_publication(
                    {"kind": "data_analysis", "analysis": candidate["analysis"]},
                    objective="data_analysis",
                    context=context,
                    service=service,
                )
                supplied = candidate["report"]
                report = service.reports.publication(
                    supplied["draft_ref"], supplied["review_ref"], context
                )
                _require(report.safety.policy_hash == self.fingerprint, "report_policy_changed")
                _require(report.model_dump(mode="json") == supplied)
                _require(report.analysis_ref == candidate["analysis"]["analysis_ref"])
                result = deepcopy(candidate)
                result["publication"] = {
                    **checked["publication"],
                    "medical_review": candidate["review_mode"],
                    "draft_ref": report.draft_ref,
                    "review_ref": report.review_ref,
                }
                return result
            _require(objective != "health_report", "safety_unavailable")
            _require(candidate.get("kind") == objective, "harness_publication_denied")
            metadata = service.predictor.metadata
            high = False
            if objective == "score_definition":
                _require(metadata.get("task_scope") == "synthetic_anomaly_only")
                _require(metadata.get("score_semantics") == "synthetic_anomaly_model_score")
                _require(candidate == score_definition(metadata), "harness_publication_denied")
                evidence = [candidate["source"]]
            else:
                _require(set(candidate) <= {"kind", "analysis", "source_run_id", "publication"})
                analysis = AnalyzeResponse.model_validate(candidate["analysis"], strict=True)
                _require(
                    (analysis.session_id, analysis.input_version)
                    == (context.session_id, context.input_version),
                    "stale_input",
                )
                self.check_prediction(analysis.prediction, metadata)
                _require(analysis.risk_level == analysis.prediction.risk_level)
                _require(
                    analysis.summary == analysis_summary(analysis.prediction),
                    "harness_publication_denied",
                )
                # Current owned artifacts, not a prior safety flag or LLM-provided source string.
                stored = service.store.get(analysis.analysis_ref, kind="analysis", context=context)
                predicted = service.store.get(
                    stored["prediction_ref"], kind="prediction", context=context
                )
                _require(stored["explanation"] == analysis.explanation.model_dump(mode="json"))
                _require(predicted["result"] == analysis.prediction.model_dump(mode="json"))
                features = service.store.get(
                    predicted["feature_set_ref"], kind="feature_set", context=context
                )
                windows = service.store.get(
                    features["window_set_ref"], kind="window_set", context=context
                )
                _require(analysis.quality_report == windows["cleaning_report"])
                _require(analysis.warnings == windows["cleaning_report"].get("warnings", []))
                _require(
                    bool(analysis.explanation.local_evidence)
                    and bool(analysis.explanation.limitations)
                )
                _require(
                    analysis.explanation.highest_risk_window
                    == analysis.prediction.highest_risk_window
                )
                for plot in analysis.visualization:
                    service.store.check_source(
                        plot.artifact_ref,
                        source_ref=analysis.analysis_ref,
                        kind="image",
                        context=context,
                    )
                evidence = [
                    {
                        "type": "analysis_artifact",
                        "ref": analysis.analysis_ref,
                        "input_version": context.input_version,
                    },
                    {
                        "type": "model_card",
                        "name": metadata["name"],
                        "version": metadata["version"],
                    },
                ]
                high = analysis.risk_level == "high"
            result = deepcopy(candidate)
            # Always rebuilt under the current policy, including when displaying a prior result.
            result["publication"] = {
                "policy": self.config.safety.publication_policy,
                "harness_version": self.version,
                "harness_hash": self.fingerprint,
                "evidence": evidence,
                "uncertainty": UNCERTAINTY,
                "recommendations": [HIGH_SCORE_RECOMMENDATION] if high else [],
                "medical_review": "not_performed",
            }
            return result
        except (ValueError, TypeError, KeyError, AttributeError):
            raise SkillError(
                "harness_publication_denied", "Publication failed Harness validation."
            ) from None
