"""Deterministic intent routing; decisions describe requirements, never grant access.

Only complete, narrowly recognized requests qualify for the simple path. Unknown
language asks for clarification. Artifact availability is a server-verified fact,
not something inferred from the query or a caller's objective hint.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import Field, model_validator

from smarthealth.runtime.errors import SkillError
from smarthealth.skills.base_skill import SkillModel

Capability = Literal[
    "model_card",
    "verified_result_read",
    "csv_analysis",
    "signal_processing",
    "feature_extraction",
    "anomaly_prediction",
    "feature_explanation",
    "visualization",
    "rag_retrieval",
    "report_generation",
    "safety_verification",
]
RouteObjective = Literal["score_definition", "data_analysis", "health_report"]
ReasonCode = Literal[
    "score_definition",
    "existing_result",
    "data_analysis",
    "health_reasoning",
    "health_report",
    "missing_upload",
    "missing_prior_result",
    "clarify_task",
]


class RouteDecision(SkillModel):
    workflow_type: Literal["simple", "complex"]
    reason_code: ReasonCode
    objective: RouteObjective
    entry_agent: Literal["planner", "data_analysis"]
    required_capabilities: list[Capability] = Field(max_length=11)
    needs_input: Literal["missing_upload", "missing_prior_result", "clarify_task"] | None = None
    router_version: Literal["1.0"] = "1.0"

    @model_validator(mode="after")
    def consistent_requirements(self) -> RouteDecision:
        if len(self.required_capabilities) != len(set(self.required_capabilities)):
            raise ValueError("Route capabilities must be unique.")
        if self.entry_agent != ("data_analysis" if self.workflow_type == "simple" else "planner"):
            raise ValueError("The entry agent must match the workflow.")
        if self.objective == "health_report":
            required = {"rag_retrieval", "report_generation", "safety_verification"}
            if self.workflow_type != "complex" or not required.issubset(self.required_capabilities):
                raise ValueError("Health explanations require the complete verification path.")
        return self


_DATA_CAPABILITIES: tuple[Capability, ...] = (
    "csv_analysis",
    "signal_processing",
    "feature_extraction",
    "anomaly_prediction",
    "feature_explanation",
)
_REPORT_CAPABILITIES: tuple[Capability, ...] = (
    "rag_retrieval",
    "report_generation",
    "safety_verification",
)
_SCORE = r"(?:risk[ _-]*score|风险(?:评分|分数|得分))"
_DEFINITION = re.compile(
    rf"(?:"
    rf"(?:please )?(?:explain|define|describe) (?:the )?{_SCORE}"
    rf"|(?:please )?what (?:is|does) (?:the )?{_SCORE}(?: mean)?"
    rf"|(?:the )?{_SCORE} (?:definition|meaning)"
    rf"|(?:请)?(?:解释|说明)(?:一下)?\s*{_SCORE}\s*(?:的(?:含义|意思|定义))?"
    rf"|{_SCORE}\s*(?:是(?:什么|什么意思)|的(?:含义|意思|定义)(?:是什么)?)"
    rf"|(?:什么是|如何理解)\s*{_SCORE}"
    rf")"
)
_EXISTING_RESULT = re.compile(
    r"(?:"
    r"(?:please )?(?:show|display|view|open) (?:me )?(?:(?:the|my) )?"
    r"(?:last|previous|existing|prior|approved|verified) (?:analysis )?(?:results?|reports?)"
    r"(?: and (?:(?:the|its) )?(?:existing |previous |approved |verified )?(?:plots|charts))?"
    r"|(?:请)?(?:显示|展示|查看|打开)(?:我(?:的)?)?(?:上次|之前|已有|已审核|已验证|历史)"
    r"(?:的)?(?:分析)?(?:结果|报告)(?:(?:和|及)(?:已有|现有|之前)?(?:的)?(?:图表|可视化))?"
    r")"
)
_MEDICAL = re.compile(
    r"\b(?:diagnos\w*|diseases?|symptoms?|treatments?|medications?|prescrib\w*|"
    r"clinical|medical|therapy|therapies|illness|illnesses)\b"
    r"|\bhealth (?:advice|recommendations?|guidance|assessment)\b"
    r"|诊断|疾病|病症|症状|治疗|用药|处方|就医|病因|患病|医学|健康(?:建议|评估|指导)"
)
_REPORT = re.compile(r"\b(?:reports?|reporting)\b|报告")
_RISK_OR_ANOMALY = re.compile(
    rf"{_SCORE}|\brisks?\b|\banomal(?:y|ies|ous)\b|\babnormal\w*\b|风险|异常"
)
_REASONING = re.compile(
    r"\b(?:why|causes?|caused|reasons?|explain)\b|为什么|为何|原因|成因|解释|建议|怎么办"
)
_INDIVIDUAL = re.compile(
    r"\b(?:my|mine|patient(?:'s)?|user(?:'s)?|person(?:'s)?|his|her|their)\b"
    r"|这个(?:用户|人)|该(?:用户|人)|我的|患者|这次|本次"
)
_DATA_VERB = re.compile(
    r"\b(?:analys[ei]|analy[sz]e|analy[sz]ing|analysis|check|inspect|process|clean|"
    r"predict|upload|plot|visuali[sz]e)\b|分析|检查|检测|预测|清洗|处理|上传|绘制|画图|可视化"
)
_DATA_NOUN = re.compile(
    r"(?<![a-z0-9_])(?:csv|data|dataset|file|input|signals?|physiological|sensors?)(?![a-z0-9_])"
    r"|数据|文件|信号|生理|传感器"
)
_VISUALIZATION = re.compile(
    r"\b(?:plots?|charts?|graphs?|visuali[sz](?:e|ation|ations))\b|图表|绘图|画图|绘制|可视化"
)


def _normalize(query: str) -> str:
    normalized = unicodedata.normalize("NFKC", query).casefold()
    # Preserve internal punctuation: appended instructions must not become part
    # of an otherwise valid, complete simple request.
    return " ".join(normalized.split()).strip(" ?!.。!？")


class HealthRouter:
    """Rule-first router with fail-closed ambiguity and one-way upgrades."""

    def route(
        self,
        *,
        query: str,
        has_upload: bool = False,
        has_existing_result: bool = False,
        objective_hint: str | None = None,
        include_visualizations: bool = False,
    ) -> RouteDecision:
        text = _normalize(query)
        include_visualizations = include_visualizations or bool(_VISUALIZATION.search(text))
        # Hints may request stricter handling. They never authorize a shortcut.
        if objective_hint == "health_report":
            return self._health_route("health_report", include_visualizations)

        # Read-only display is narrowly anchored, and still needs a verified
        # source supplied by the runtime. It cannot generate a fresh report.
        if _EXISTING_RESULT.fullmatch(text):
            return RouteDecision(
                workflow_type="simple",
                reason_code="existing_result" if has_existing_result else "missing_prior_result",
                objective="data_analysis",
                entry_agent="data_analysis",
                required_capabilities=["verified_result_read", "visualization"]
                if include_visualizations
                else ["verified_result_read"],
                needs_input=None if has_existing_result else "missing_prior_result",
            )

        if _REPORT.search(text):
            return self._health_route("health_report", include_visualizations)
        if _MEDICAL.search(text):
            return self._health_route("health_reasoning", include_visualizations)

        if _DEFINITION.fullmatch(text):
            if include_visualizations:
                return self._analysis_route(has_upload, include_visualizations)
            return RouteDecision(
                workflow_type="simple",
                reason_code="score_definition",
                objective="score_definition",
                entry_agent="data_analysis",
                required_capabilities=["model_card"],
            )

        if _RISK_OR_ANOMALY.search(text) and (_REASONING.search(text) or _INDIVIDUAL.search(text)):
            return self._health_route("health_reasoning", include_visualizations)

        if _DATA_VERB.search(text) and _DATA_NOUN.search(text):
            return self._analysis_route(has_upload, include_visualizations)

        return RouteDecision(
            workflow_type="complex",
            reason_code="clarify_task",
            objective="data_analysis",
            entry_agent="planner",
            required_capabilities=[],
            needs_input="clarify_task",
        )

    def upgrade(self, current: RouteDecision, *, objective: str, has_upload: bool) -> RouteDecision:
        """Describe a stricter workflow; the runtime retains budgets and identity."""
        if current.workflow_type != "simple" or objective not in {"data_analysis", "health_report"}:
            raise SkillError("workflow_upgrade_denied", "Only a simple workflow may upgrade once.")
        include_visualizations = "visualization" in current.required_capabilities
        if objective == "health_report":
            return self._health_route("health_report", include_visualizations)
        return self._analysis_route(has_upload, include_visualizations)

    @staticmethod
    def _analysis_route(has_upload: bool, include_visualizations: bool) -> RouteDecision:
        capabilities = list(_DATA_CAPABILITIES)
        if include_visualizations:
            capabilities.append("visualization")
        return RouteDecision(
            workflow_type="complex",
            reason_code="data_analysis" if has_upload else "missing_upload",
            objective="data_analysis",
            entry_agent="planner",
            required_capabilities=capabilities,
            needs_input=None if has_upload else "missing_upload",
        )

    @staticmethod
    def _health_route(
        reason: Literal["health_reasoning", "health_report"], include_visualizations: bool
    ) -> RouteDecision:
        capabilities = list(_DATA_CAPABILITIES)
        if include_visualizations:
            capabilities.append("visualization")
        capabilities.extend(_REPORT_CAPABILITIES)
        return RouteDecision(
            workflow_type="complex",
            reason_code=reason,
            objective="health_report",
            entry_agent="planner",
            required_capabilities=capabilities,
        )
