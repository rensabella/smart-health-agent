"""Register only implemented and executable physiological data capabilities."""

from smarthealth.runtime.harness.config import IMPLEMENTED_SKILLS
from smarthealth.services.analysis_service import AnalysisService
from smarthealth.skills.anomaly_prediction_skill import AnomalyPredictionSkill
from smarthealth.skills.csv_analysis_skill import CSVAnalysisSkill
from smarthealth.skills.feature_explanation_skill import FeatureExplanationSkill
from smarthealth.skills.feature_extraction_skill import FeatureExtractionSkill
from smarthealth.skills.registry import SkillRegistry
from smarthealth.skills.signal_processing_skill import SignalProcessingSkill
from smarthealth.skills.visualization_skill import VisualizationSkill

DATA_SKILL_NAMES = IMPLEMENTED_SKILLS


def create_registry(
    service: AnalysisService, *, max_concurrency: int = 4, trace_sink=None, harness=None
) -> SkillRegistry:
    registry = SkillRegistry(max_concurrency=max_concurrency, trace_sink=trace_sink)
    for skill_type in (
        CSVAnalysisSkill,
        SignalProcessingSkill,
        FeatureExtractionSkill,
        AnomalyPredictionSkill,
        FeatureExplanationSkill,
        VisualizationSkill,
    ):
        registry.register(skill_type(service))
    if harness is not None and any(s.name == "rag_retrieval" for s in harness.config.skills.skills):
        from smarthealth.skills.rag_retrieval_skill import RAGRetrievalSkill
        from smarthealth.skills.report_skill import ReportSkill
        from smarthealth.skills.safety_verification_skill import SafetyVerificationSkill

        for skill_type in (RAGRetrievalSkill, ReportSkill, SafetyVerificationSkill):
            registry.register(skill_type(service))
    if harness is not None:
        registry.bind_harness(harness)
    return registry
