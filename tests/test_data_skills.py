from dataclasses import replace

import pytest

from app.core.artifacts import ArtifactStore
from app.core.context import SkillContext
from app.services.analysis_service import AnalysisService
from skills.bootstrap import DATA_SKILL_NAMES, create_registry


@pytest.mark.parametrize("backend", ["random_forest", "lightgbm"])
async def test_six_skills_real_models_and_private_artifacts(bundles, csv_bytes, backend):
    from app.models.model_loader import load_model

    store = ArtifactStore()
    service = AnalysisService(store, load_model(bundles[backend]))
    registry = create_registry(service)
    context = SkillContext(owner_id="owner", session_id="session", allowed_skills=DATA_SKILL_NAMES)
    upload_ref = service.create_upload(csv_bytes, context)
    try:
        next_input = {"upload_ref": upload_ref}
        outputs = {}
        for name, output_field in [
            ("csv_analysis", "dataset_ref"),
            ("signal_processing", "window_set_ref"),
            ("feature_extraction", "feature_set_ref"),
            ("anomaly_prediction", "prediction_ref"),
            ("feature_explanation", "analysis_ref"),
            ("visualization", None),
        ]:
            result = await registry.execute(name, next_input, context)
            assert result.error is None, result.model_dump()
            outputs[name] = result.data
            if output_field:
                next_input = {output_field: result.data[output_field]}
        assert context.budget.used == 6
        assert {s["name"] for s in registry.describe()} == DATA_SKILL_NAMES
        assert len(outputs["visualization"]["artifacts"]) == 3
        for artifact in outputs["visualization"]["artifacts"]:
            image = store.get(artifact["artifact_ref"], kind="image", context=context)
            assert image["content"].startswith(b"\x89PNG\r\n\x1a\n")
        denied = await registry.execute(
            "csv_analysis", {"upload_ref": upload_ref}, replace(context, owner_id="another")
        )
        assert denied.error.code == "permission_denied"
        assert (
            len(
                [
                    e
                    for e in registry.trace_sink.events_for(context)
                    if e["event"] == "skill.completed"
                ]
            )
            == 6
        )
    finally:
        await registry.aclose()
