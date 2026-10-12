"""Register only implemented and executable physiological data capabilities."""

from app.harness.config import IMPLEMENTED_SKILLS
from app.services.analysis_service import AnalysisService
from skills.anomaly_prediction_skill import AnomalyPredictionSkill
from skills.csv_analysis_skill import CSVAnalysisSkill
from skills.feature_explanation_skill import FeatureExplanationSkill
from skills.feature_extraction_skill import FeatureExtractionSkill
from skills.registry import SkillRegistry
from skills.signal_processing_skill import SignalProcessingSkill
from skills.visualization_skill import VisualizationSkill

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
        from skills.rag_retrieval_skill import RAGRetrievalSkill
        from skills.report_skill import ReportSkill
        from skills.safety_verification_skill import SafetyVerificationSkill

        for skill_type in (RAGRetrievalSkill, ReportSkill, SafetyVerificationSkill):
            registry.register(skill_type(service))
    if harness is not None:
        registry.bind_harness(harness)
    return registry
