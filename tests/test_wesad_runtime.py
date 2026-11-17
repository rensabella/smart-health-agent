"""WESAD runtime: gated full chain on a deploy bundle and cross-profile fail-closed."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from wesad_helpers import canonical_frame

from smarthealth.agents.backends import ScriptedDecisionBackend
from smarthealth.agents.contracts import CallSkillAction
from smarthealth.cli.train import train_wesad
from smarthealth.config.settings import Settings
from smarthealth.evaluation.live_llm.budget import EvaluationBudget
from smarthealth.evaluation.live_llm.config import EvalConfig
from smarthealth.evaluation.live_llm.runner import execute_task, synthetic_input
from smarthealth.evaluation.live_llm.tasks import load_tasks
from smarthealth.evaluation.live_llm.trace_collector import ArtifactCollector
from smarthealth.examples.demo_policy import scripted_analysis
from smarthealth.models.model_loader import load_model
from smarthealth.pipeline.inference import analyze_csv
from smarthealth.runtime.context import SkillContext
from smarthealth.runtime.errors import SkillError
from smarthealth.services.analysis_service import AnalysisService

SUBJECTS = ["S2", "S3", "S4", "S5"]


def _wesad_scripted(request):
    """The scripted chain plus the explicit WESAD data_profile declaration."""
    action = scripted_analysis(request)
    if isinstance(action, CallSkillAction) and action.skill == "csv_analysis":
        return CallSkillAction(
            skill="csv_analysis",
            arguments={**action.arguments, "data_profile": "wesad_physiology_v1"},
        )
    return action


@pytest.fixture(scope="module")
def wesad_deploy(tmp_path_factory):
    root = tmp_path_factory.mktemp("wesad_runtime")
    frames = [
        canonical_frame(subject, [(130, 0), (130, 1), (130, 2)], seed=index)
        for index, subject in enumerate(SUBJECTS)
    ]
    training_csv = root / "training.csv"
    pd.concat(frames, ignore_index=True).to_csv(training_csv, index=False)
    output = root / "bundles"
    train_wesad(training_csv, output, holdout_subject="S5", backends=("lightgbm",))
    return load_model(output / "lightgbm_v1.0")


@pytest.fixture(scope="module")
def wesad_content():
    frame = canonical_frame("S5", [(130, 0), (130, 1), (130, 2)], seed=99)
    return frame.drop(columns="label").to_csv(index=False).encode("utf-8")


async def test_wesad_full_guard_chain_publishes_a_validated_stress_analysis(
    wesad_deploy, wesad_content
):
    predictor = wesad_deploy
    suite, _ = load_tasks("wesad")
    task = next(t for t in suite.tasks if t.task_id == "csv_analysis")
    config = EvalConfig.model_validate(
        {"suite": "wesad", "model_dir": "models_saved/wesad_v1/lightgbm_v1.0"}
    )
    raw = analyze_csv(wesad_content, predictor)["prediction"]
    record = await execute_task(
        task,
        repetition=1,
        config=config,
        settings=Settings(),
        predictor=predictor,
        retrieval=None,
        content=wesad_content,
        budget=EvaluationBudget(config),
        reference_prediction={k: raw[k] for k in ("risk_score", "prediction", "window_count")},
        mode="infrastructure_test",
        make_backend=lambda: ScriptedDecisionBackend(_wesad_scripted),
    )
    assert record["suite"] == "wesad"
    assert record["snapshot"]["input_version"] == "wesad_v1"
    assert record["snapshot"]["status"] == "completed"
    assert record["snapshot"]["memory_status"] == "stored"
    assert record["grades"]["success"], record
    analysis = record["snapshot"]["final_result"]["analysis"]
    assert analysis["prediction"]["prediction"] in {"baseline", "stress", "amusement"}
    assert analysis["prediction"]["risk_score"] == pytest.approx(raw["risk_score"])
    assert analysis["prediction"]["window_count"] == raw["window_count"] == 12
    kinds = {a["kind"] for a in record["artifacts"]}
    assert {"upload", "dataset", "window_set", "feature_set", "prediction", "analysis"} <= kinds
    serialized = json.dumps(record)
    assert "synthetic_eval" not in serialized and '"frame"' not in serialized


async def test_cross_profile_uploads_fail_closed_with_recovery_hints(
    wesad_deploy, wesad_content, predictor
):
    with pytest.raises(SkillError) as exc_info:
        analyze_csv(wesad_content, predictor)
    assert "wesad_physiology_v1" in exc_info.value.message
    with pytest.raises(SkillError) as exc_info:
        analyze_csv(synthetic_input(10042), wesad_deploy)
    assert "synthetic_physiology_v1" in exc_info.value.message


async def test_service_pins_profiles_to_the_dataset_lineage(wesad_deploy, wesad_content):
    service = AnalysisService(ArtifactCollector(), wesad_deploy)
    context = SkillContext(
        owner_id="wesad_live_evaluation", session_id="runtime_check", input_version="wesad_v1"
    )
    upload = service.create_upload(wesad_content, context)
    # The data profile is never inferred from bytes: the default parse of a
    # WESAD CSV fails closed and names the matching registered profile.
    with pytest.raises(SkillError) as exc_info:
        service.inspect_csv(upload, context)
    assert "wesad_physiology_v1" in exc_info.value.message
    inspected = service.inspect_csv(upload, context, data_profile="wesad_physiology_v1")
    assert inspected["row_count"] == 390
    assert inspected["quality_report"]["input_profile"] == "wesad_physiology_v1"
    dataset_ref = inspected["dataset_ref"]
    with pytest.raises(SkillError) as exc_info:
        service.process_signal(dataset_ref, context, processing_profile="1hz_30s_v1")
    assert exc_info.value.code == "invalid_input"
    assert exc_info.value.details == {"expected_processing_profile": "1hz_60s_stride30_v1"}
    windows = service.process_signal(dataset_ref, context)
    assert windows["window_count"] == 12
    with pytest.raises(SkillError) as exc_info:
        service.extract(
            windows["window_set_ref"], context, feature_profile="physiology_statistics_v1"
        )
    assert exc_info.value.details == {"expected_feature_profile": "wesad_physiology_statistics_v1"}
    extracted = service.extract(windows["window_set_ref"], context)
    assert extracted["window_count"] == 12
    assert len(extracted["feature_names"]) == 59
    predicted = service.predict(extracted["feature_set_ref"], context)
    assert predicted["prediction"] in {"baseline", "stress", "amusement"}
    assert predicted["model"]["task_scope"] == "wesad_stress_only"
