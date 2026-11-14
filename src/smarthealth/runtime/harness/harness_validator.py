"""Deterministic execution and publication guards, not a medical safety classifier."""

from __future__ import annotations

import math
from copy import deepcopy

from smarthealth.pipeline.profiles import DataProfile, get_profile
from smarthealth.runtime.errors import SkillError
from smarthealth.runtime.harness.config import AGENT_ROLES, HarnessConfig, load_config, with_limits
from smarthealth.schemas.response import AnalyzeResponse, Explanation, Prediction

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
# Evaluation-only ablation baselines. Production callers never pass this parameter;
# the default keeps every guard enabled. Each value disables exactly one mechanism:
#   guard_off  - review verdicts are recorded but do not block publication.
#   no_lineage - input-version lineage is not enforced (artifact-store binding,
#                prior-result redisplay, publication-boundary staleness).
#   no_budget  - Skill tool-call budgets are not enforced (harness ceilings, the
#                runtime CallBudget cap and the safety-call reservation).
#                Token admission, decision/LLM-request limits and timeouts stay on.
BASELINES = frozenset({"full_guard", "guard_off", "no_lineage", "no_budget"})


def _profile_or_none(name) -> DataProfile | None:
    try:
        return get_profile(name)
    except ValueError:
        return None


def uncertainty_text(profile: DataProfile | None = None) -> str:
    """Guard copy for one profile; the default keeps the synthetic constant."""
    if profile is None:
        from smarthealth.pipeline.profiles import SYNTHETIC_V1

        profile = SYNTHETIC_V1
    return profile.guard_text.uncertainty


def analysis_summary(predicted: Prediction, *, profile_name: str | None = None) -> str:
    profile = get_profile(profile_name) if profile_name is not None else _profile_or_none(
        predicted.model.metadata.get("input_profile")
    ) or _profile_or_none("synthetic_physiology_v1")
    return profile.guard_text.analysis_summary_template.format(
        window_count=predicted.window_count,
        window_seconds=profile.window_seconds,
        risk_score=predicted.risk_score,
        highest_risk_window=predicted.highest_risk_window,
    )


def score_definition(metadata: dict, *, profile_name: str | None = None) -> dict:
    profile = (
        get_profile(profile_name)
        if profile_name is not None
        else _profile_or_none(metadata.get("input_profile"))
        or _profile_or_none("synthetic_physiology_v1")
    )
    return {
        "kind": "score_definition",
        "summary": profile.guard_text.score_definition_template.format(
            window_seconds=profile.window_seconds
        ),
        "source": {
            "type": "model_card",
            "name": metadata["name"],
            "version": metadata["version"],
            "score_semantics": metadata["score_semantics"],
            "aggregation": metadata["aggregation"],
        },
    }


def _require(condition, code="harness_invalid_output", rule=None):
    if not condition:
        # ``rule`` is a stable, compile-time check identifier for sanitized feedback;
        # it never derives from data values.
        raise SkillError(
            code,
            "Output does not satisfy the active Harness policy.",
            details={"rule": rule} if rule else None,
        )


class HarnessValidator:
    def __init__(self, config: HarnessConfig | None = None, *, limits=None, baseline="full_guard"):
        if baseline not in BASELINES:
            raise ValueError("Unknown evaluation baseline")
        self._baseline = baseline
        self._config = with_limits(config if config is not None else load_config(), limits)

    @property
    def baseline(self):
        return self._baseline

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
        if self._baseline == "no_budget":
            return
        if used >= self.limits.max_tool_calls or role_used >= self.role_tool_limit(role):
            raise SkillError("budget_exceeded", "The configured tool call budget is exhausted.")

    def check_prediction(self, predicted: Prediction, metadata: dict):
        """Schema checks alone cannot prove cross-field consistency or model scope."""
        windows = predicted.window_predictions
        _require(len(windows) == predicted.window_count, rule="prediction.window_count")
        _require(
            [w.window_index for w in windows] == list(range(len(windows))),
            rule="prediction.window_indices",
        )
        highest = max(windows, key=lambda w: w.risk_score)
        _require(
            predicted.highest_risk_window == highest.window_index,
            rule="prediction.highest_risk_window",
        )
        _require(
            math.isclose(predicted.risk_score, highest.risk_score, abs_tol=1e-12),
            rule="prediction.risk_score",
        )
        # Model scope is profile-pinned; unknown input profiles fail closed here.
        profile = _profile_or_none(metadata.get("input_profile"))
        _require(profile is not None, rule="prediction.metadata_input_profile")
        _require(
            metadata.get("task_scope") == profile.task_scope,
            rule="prediction.metadata_task_scope",
        )
        _require(
            metadata.get("score_semantics") == profile.score_semantics,
            rule="prediction.metadata_score_semantics",
        )
        _require(
            metadata.get("input_profile") == profile.name,
            rule="prediction.metadata_input_profile",
        )
        _require(
            metadata.get("aggregation") == predicted.aggregation == "max",
            rule="prediction.metadata_aggregation",
        )
        _require(
            predicted.model.task_scope == metadata["task_scope"],
            rule="prediction.model_task_scope",
        )
        _require(
            (predicted.model.name, predicted.model.version)
            == (metadata["name"], metadata["version"]),
            rule="prediction.model_identity",
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
            _require(
                key in metadata and predicted.model.metadata.get(key) == metadata[key],
                rule=f"prediction.model_metadata.{key}",
            )
        threshold = metadata["threshold"]
        risk_label = profile.risk_class
        # Threshold-first rule; binary models reduce exactly to the historical labels.
        _require(
            all((w.risk_score >= threshold) == (w.prediction == risk_label) for w in windows),
            rule="prediction.window_labels",
        )
        _require(
            (predicted.risk_score >= threshold) == (predicted.prediction == risk_label),
            rule="prediction.label",
        )
        _require(
            math.isclose(
                predicted.anomalous_ratio,
                sum(1 for w in windows if w.prediction == risk_label) / len(windows),
                abs_tol=1e-12,
            ),
            rule="prediction.anomalous_ratio",
        )
        levels = metadata["risk_thresholds"]
        level = (
            "high"
            if predicted.risk_score >= levels["high"]
            else "medium"
            if predicted.risk_score >= levels["medium"]
            else "low"
        )
        _require(predicted.risk_level == level, rule="prediction.risk_level")
        _require(bool(metadata["limitations"]), rule="prediction.limitations")

    def check_skill_output(self, name, payload, data, *, context, service):
        """Check declared artifact lineage before observations or API responses receive facts."""
        try:
            if name in _LINEAGE:
                ref_field, kind, parent_field = _LINEAGE[name]
                stored = service.store.get(data[ref_field], kind=kind, context=context)
                service.store.check_source(
                    data[ref_field], source_ref=payload[parent_field], kind=kind, context=context
                )
                _require(stored[parent_field] == payload[parent_field], rule=f"{name}.lineage_parent")
                if name == "csv_analysis":
                    _require(
                        data["row_count"] == len(stored["frame"]) > 0,
                        rule="csv_analysis.row_count",
                    )
                    _require(
                        data["columns"] == list(stored["frame"].columns),
                        rule="csv_analysis.columns",
                    )
                    _require(
                        data["quality_report"] == stored["quality_report"],
                        rule="csv_analysis.quality_report",
                    )
                elif name == "signal_processing":
                    _require(
                        data["window_count"] == len(stored["windows"]) > 0,
                        rule="signal_processing.window_count",
                    )
                    _require(
                        data["cleaning_report"] == stored["cleaning_report"],
                        rule="signal_processing.cleaning_report",
                    )
                    _require(
                        data["warnings"] == stored["cleaning_report"].get("warnings", []),
                        rule="signal_processing.warnings",
                    )
                elif name == "feature_extraction":
                    window_set = service.store.get(
                        stored["window_set_ref"], kind="window_set", context=context
                    )
                    profile = _profile_or_none(
                        window_set["cleaning_report"].get("input_profile")
                    )
                    _require(profile is not None, rule="feature_extraction.input_profile")
                    _require(
                        data["window_count"] == len(stored["frame"]) > 0,
                        rule="feature_extraction.window_count",
                    )
                    _require(
                        data["feature_names"] == list(stored["frame"].columns),
                        rule="feature_extraction.feature_names",
                    )
                    _require(
                        data["feature_names"] == list(profile.feature_names),
                        rule="feature_extraction.feature_names_profile",
                    )
                    _require(
                        data["feature_schema_version"] == profile.feature_schema_version,
                        rule="feature_extraction.feature_schema_version",
                    )
                elif name == "anomaly_prediction":
                    prediction = Prediction.model_validate(
                        {k: data[k] for k in Prediction.model_fields}
                    )
                    self.check_prediction(prediction, service.predictor.metadata)
                    feature_set = service.store.get(
                        payload["feature_set_ref"], kind="feature_set", context=context
                    )
                    _require(
                        prediction.window_count == len(feature_set["frame"]),
                        rule="anomaly_prediction.window_count",
                    )
                    _require(
                        prediction.model_dump(mode="json") == stored["result"],
                        rule="anomaly_prediction.stored_result",
                    )
                elif name == "feature_explanation":
                    explanation = Explanation.model_validate(
                        {k: data[k] for k in Explanation.model_fields}
                    )
                    _require(
                        explanation.model_dump(mode="json") == stored["explanation"],
                        rule="feature_explanation.stored_explanation",
                    )
                    _require(
                        bool(explanation.local_evidence) and bool(explanation.limitations),
                        rule="feature_explanation.evidence_bounds",
                    )
            elif name == "visualization":
                _require(
                    sorted(a["kind"] for a in data["artifacts"]) == sorted(payload["plot_types"]),
                    rule="visualization.plot_types",
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
                    _require(
                        artifact["url"] == f"/artifacts/{artifact['artifact_ref']}",
                        rule="visualization.artifact_url",
                    )
                    _require(
                        image["kind"] == artifact["kind"] and image["mime_type"] == "image/png",
                        rule="visualization.artifact_kind",
                    )
            elif name == "rag_retrieval":
                from smarthealth.retrieval.schemas import RetrievalResult

                stored = service.store.get(data["evidence_ref"], kind="evidence", context=context)
                service.store.check_source(
                    data["evidence_ref"],
                    source_ref=payload["analysis_ref"],
                    kind="evidence",
                    context=context,
                )
                result = RetrievalResult.model_validate(stored["result"])
                _require(
                    stored["analysis_ref"] == payload["analysis_ref"],
                    rule="rag_retrieval.analysis_ref",
                )
                _require(
                    data["hit_count"] == len(result.hits) and data["status"] == result.status,
                    rule="rag_retrieval.hit_count_status",
                )
                _require(
                    data["knowledge_version"] == result.knowledge_version,
                    rule="rag_retrieval.knowledge_version",
                )
                _require(
                    data["embedding_id"] == result.embedding_id,
                    rule="rag_retrieval.embedding_id",
                )
            elif name in {"report_generation", "safety_verification"}:
                service.reports.validate_skill_output(name, payload, data, context)
        except (ValueError, TypeError, KeyError, AttributeError):
            raise SkillError(
                "harness_invalid_output",
                "Skill facts failed Harness validation.",
                details={"rule": "skill_output.malformed_facts"},
            ) from None

    def validate_publication(self, candidate, *, objective, context, service):
        """Admit fixed facts only. No free-form claim can obtain a medical safety approval."""
        try:
            context.check_active()
            if objective == "health_report" and self.config.safety.medical_report_enabled:
                _require(
                    set(candidate) == {"kind", "analysis", "report", "review_mode"},
                    rule="publication.candidate_keys",
                )
                _require(candidate["kind"] == "health_report", rule="publication.candidate_kind")
                _require(
                    candidate["review_mode"] in {"llm_assisted", "scripted_test"},
                    rule="publication.review_mode",
                )
                checked = self.validate_publication(
                    {"kind": "data_analysis", "analysis": candidate["analysis"]},
                    objective="data_analysis",
                    context=context,
                    service=service,
                )
                supplied = candidate["report"]
                report = service.reports.publication(
                    supplied["draft_ref"],
                    supplied["review_ref"],
                    context,
                    enforce_review=self._baseline != "guard_off",
                )
                _require(report.safety.policy_hash == self.fingerprint, "report_policy_changed")
                _require(
                    report.model_dump(mode="json") == supplied,
                    rule="publication.report_reconstruction",
                )
                _require(
                    report.analysis_ref == candidate["analysis"]["analysis_ref"],
                    rule="publication.analysis_ref",
                )
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
            profile = _profile_or_none(metadata.get("input_profile"))
            high = False
            if objective == "score_definition":
                _require(profile is not None, rule="publication.metadata_input_profile")
                _require(
                    metadata.get("task_scope") == profile.task_scope,
                    rule="publication.metadata_task_scope",
                )
                _require(
                    metadata.get("score_semantics") == profile.score_semantics,
                    rule="publication.metadata_score_semantics",
                )
                _require(
                    candidate == score_definition(metadata, profile_name=profile.name),
                    "harness_publication_denied",
                )
                evidence = [candidate["source"]]
            else:
                _require(
                    set(candidate) <= {"kind", "analysis", "source_run_id", "publication"},
                    rule="publication.candidate_keys",
                )
                analysis = AnalyzeResponse.model_validate(candidate["analysis"], strict=True)
                if self._baseline != "no_lineage":
                    _require(
                        (analysis.session_id, analysis.input_version)
                        == (context.session_id, context.input_version),
                        "stale_input",
                    )
                self.check_prediction(analysis.prediction, metadata)
                _require(
                    analysis.risk_level == analysis.prediction.risk_level,
                    rule="publication.risk_level",
                )
                _require(
                    analysis.summary
                    == analysis_summary(analysis.prediction, profile_name=profile.name),
                    "harness_publication_denied",
                )
                # Current owned artifacts, not a prior safety flag or LLM-provided source string.
                stored = service.store.get(analysis.analysis_ref, kind="analysis", context=context)
                predicted = service.store.get(
                    stored["prediction_ref"], kind="prediction", context=context
                )
                _require(
                    stored["explanation"] == analysis.explanation.model_dump(mode="json"),
                    rule="publication.stored_explanation",
                )
                _require(
                    predicted["result"] == analysis.prediction.model_dump(mode="json"),
                    rule="publication.stored_prediction",
                )
                features = service.store.get(
                    predicted["feature_set_ref"], kind="feature_set", context=context
                )
                windows = service.store.get(
                    features["window_set_ref"], kind="window_set", context=context
                )
                _require(
                    analysis.quality_report == windows["cleaning_report"],
                    rule="publication.quality_report",
                )
                _require(
                    analysis.warnings == windows["cleaning_report"].get("warnings", []),
                    rule="publication.warnings",
                )
                _require(
                    bool(analysis.explanation.local_evidence)
                    and bool(analysis.explanation.limitations),
                    rule="publication.explanation_bounds",
                )
                _require(
                    analysis.explanation.highest_risk_window
                    == analysis.prediction.highest_risk_window,
                    rule="publication.highest_risk_window",
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
                "uncertainty": uncertainty_text(profile),
                "recommendations": (
                    [profile.guard_text.high_score_recommendation] if high else []
                ),
                "medical_review": "not_performed",
            }
            return result
        except (ValueError, TypeError, KeyError, AttributeError):
            raise SkillError(
                "harness_publication_denied",
                "Publication failed Harness validation.",
                details={"rule": "publication.malformed"},
            ) from None
