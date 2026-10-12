"""Offline demonstration policy. This fixture is not an LLM and never a production default."""

from smarthealth.agents.contracts import (
    CallSkillAction,
    DelegateAction,
    FinishAction,
    RequestInputAction,
)


def scripted_analysis(request):
    if request.objective == "score_definition":
        return FinishAction(result_kind="score_definition")
    if request.available_refs.get("published_analysis") and request.workflow_type == "simple":
        return FinishAction(
            result_kind="data_analysis", result_ref=request.available_refs["published_analysis"][-1]
        )
    if request.agent == "planner":
        completed = [
            o for o in request.observations if o.action == "delegate" and o.status == "success"
        ]
        if completed:
            return FinishAction(
                result_kind=request.objective, result_ref=completed[-1].data["analysis_ref"]
            )
        return DelegateAction(
            agent="data_analysis", task="Produce a validated physiological data analysis."
        )
    if not request.available_refs.get("upload"):
        return RequestInputAction(reason="missing_upload")
    errors = [o for o in request.observations if o.status == "error"]
    if errors and errors[-1].error_code in {"poor_data_quality", "invalid_csv", "invalid_columns"}:
        return RequestInputAction(reason="invalid_data")
    for kind, skill, field_name, produced in (
        ("upload", "csv_analysis", "upload_ref", "dataset"),
        ("dataset", "signal_processing", "dataset_ref", "window_set"),
        ("window_set", "feature_extraction", "window_set_ref", "feature_set"),
        ("feature_set", "anomaly_prediction", "feature_set_ref", "prediction"),
        ("prediction", "feature_explanation", "prediction_ref", "analysis"),
    ):
        if not request.available_refs.get(produced):
            return CallSkillAction(
                skill=skill, arguments={field_name: request.available_refs[kind][-1]}
            )
    analysis_ref = request.available_refs["analysis"][-1]
    if request.include_visualizations and not any(
        o.skill == "visualization" and o.status == "success" for o in request.observations
    ):
        return CallSkillAction(
            skill="visualization", arguments={"analysis_ref": analysis_ref, "plot_types": ["risk"]}
        )
    return FinishAction(result_kind="data_analysis", result_ref=analysis_ref)
