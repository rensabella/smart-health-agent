"""Infrastructure tests only: scripted decisions/MockTransport are never live results."""

import copy
import json
from argparse import Namespace
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from rag_helpers import ScriptedEmbedding

from smarthealth.agents.backends import OpenAICompatibleBackend, ScriptedDecisionBackend
from smarthealth.config.settings import Settings
from smarthealth.evaluation.live_llm.budget import EvaluationBudget
from smarthealth.evaluation.live_llm.config import EvalConfig, load_config
from smarthealth.evaluation.live_llm.graders import grade_task
from smarthealth.evaluation.live_llm.metrics import aggregate
from smarthealth.evaluation.live_llm.report import render_markdown
from smarthealth.evaluation.live_llm.runner import execute_task, run, select_tasks, synthetic_input
from smarthealth.evaluation.live_llm.tasks import BenchmarkSuite, load_tasks
from smarthealth.examples.demo_policy import scripted_analysis
from smarthealth.examples.report_policy import scripted_report
from smarthealth.pipeline.inference import analyze_csv
from smarthealth.retrieval.retrieval import RetrievalService, build_index


def task_named(name):
    return next(t for t in load_tasks()[0].tasks if t.task_id == name)


def test_benchmark_validates_real_capabilities_and_rejects_contract_drift():
    suite, digest = load_tasks()
    assert len(suite.tasks) == 20 and len(digest) == 64
    assert {t.category for t in suite.tasks} == {
        "basic",
        "workflow",
        "rag",
        "failure",
        "safety",
        "unsupported",
    }
    for mutate in (
        lambda d: d["tasks"][0].update(required_skills=["imaginary_skill"]),
        lambda d: d["tasks"][1].update(task_id=d["tasks"][0]["task_id"]),
        lambda d: d["tasks"][0].update(fixture="patient.csv"),
        lambda d: d["tasks"][0].update(fault="tool_timeout"),
    ):
        data = suite.model_dump()
        mutate(data)
        with pytest.raises(ValidationError):
            BenchmarkSuite.model_validate(data)
    with pytest.raises(ValueError):
        select_tasks(suite, ["no_such_task"], 1)


@pytest.mark.parametrize(
    "values",
    [
        {"concurrency": 5},
        {"max_tasks": 21},
        {"baseline": "guards_off"},
        {"api_key": "not-accepted"},
        {"temperature": float("nan")},
        {"input_price_per_million": 1.0},
        {"max_estimated_cost": 1.0},
    ],
)
def test_config_rejects_unsafe_or_incomplete_options(values):
    with pytest.raises(ValidationError):
        EvalConfig.model_validate(values)


async def test_dry_run_ignores_credentials_and_cannot_construct_provider_or_model(
    tmp_path, monkeypatch
):
    import smarthealth.evaluation.live_llm.runner as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run must not access providers, models or environment secrets")

    monkeypatch.setattr(module, "new_backend", forbidden)
    monkeypatch.setattr(module, "load_model", forbidden)
    monkeypatch.setattr(module.Settings, "from_env", forbidden)
    output = tmp_path / "dry"
    args = Namespace(
        config=None,
        max_tasks=20,
        repetitions=2,
        concurrency=2,
        model=None,
        live=False,
        task=None,
        output=output,
    )
    assert await run(args) == 0
    report = json.loads((output / "summary.json").read_text())
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 40
    assert all(r["mode"] == "dry_run" and r["grades"]["success"] is None for r in rows)
    assert report["metrics"]["task_completion_rate"]["value"] is None
    assert report["budget"]["admitted_requests"] == 0
    assert "unexecuted" in (output / "report.md").read_text()
    with pytest.raises(FileExistsError):
        await run(args)  # Prior evidence must never be silently overwritten.


async def execute(name, predictor, *, retrieval=None, config=None, policy=scripted_analysis):
    content = synthetic_input(10042)
    raw = analyze_csv(content, predictor)["prediction"]
    config = config or EvalConfig()
    return await execute_task(
        task_named(name),
        repetition=1,
        config=config,
        settings=Settings(),
        predictor=predictor,
        retrieval=retrieval,
        content=content,
        budget=EvaluationBudget(config),
        reference_prediction={k: raw[k] for k in ("risk_score", "prediction", "window_count")},
        mode="infrastructure_test",
        make_backend=lambda: ScriptedDecisionBackend(policy),
    )


async def test_actual_pipeline_and_grader_detect_phantom_or_wrong_results(predictor):
    captured = []

    def policy(state):
        captured.append(state.model_dump())
        return scripted_analysis(state)

    record = await execute("csv_analysis", predictor, policy=policy)
    assert record["grades"]["success"], record
    assert record["snapshot"]["tool_calls"] == 5
    assert len(record["artifacts"]) == 6
    assert not record["grades"]["live_exercised"]
    assert all("expected" not in state and "success_criteria" not in state for state in captured)
    serialized = json.dumps(record)
    assert "heart_rate," not in serialized and '"frame"' not in serialized
    task = task_named("csv_analysis")
    forged = copy.deepcopy(record)
    forged["snapshot"]["final_result"]["analysis"]["prediction"]["risk_score"] = 0.12345
    assert not grade_task(task, forged)["checks"]["prediction_matches_reference"]
    forged = copy.deepcopy(record)
    forged["artifacts"] = []
    assert not grade_task(task, forged)["checks"]["artifact_provenance"]
    forged = copy.deepcopy(record)
    forged["events"] = [e for e in forged["events"] if e["event"] != "publication.checked"]
    assert not grade_task(task, forged)["checks"]["publication_verified"]


@pytest.mark.parametrize(
    "name",
    [
        "tool_timeout",
        "invalid_tool_output",
        "invalid_arguments",
        "stale_artifact",
        "duplicate_execution",
    ],
)
async def test_failure_cases_use_real_guards_and_observed_recovery(predictor, name):
    record = await execute(name, predictor)
    assert record["fault"]["triggered"]
    assert record["grades"]["fault"]["recovered"], record
    if name == "duplicate_execution":
        assert record["snapshot"]["tool_calls"] == 6
        assert record["grades"]["counts"]["physical_tool_calls"] == 6
        assert len(record["skill_calls"]) == 5


@pytest.mark.parametrize(
    "name",
    [
        "missing_upload",
        "ambiguous_task",
        "unsupported_capability",
        "invalid_csv",
        "execution_budget",
    ],
)
async def test_clarification_and_budget_controls_do_not_publish(predictor, name):
    record = await execute(name, predictor)
    assert record["grades"]["success"], record
    assert record["snapshot"]["final_result"] is None
    if name in {"missing_upload", "ambiguous_task", "unsupported_capability"}:
        assert record["provider_attempts"] == []
        assert record["grades"]["router_only"]


async def test_real_report_and_citation_grader_with_mock_embedding_only(predictor, tmp_path):
    index = tmp_path / "index"
    build_index(Path("knowledge/corpus.json"), index, ScriptedEmbedding())
    retrieval = RetrievalService(index, ScriptedEmbedding(), min_score=0.2)
    try:
        record = await execute(
            "grounded_signal_report", predictor, retrieval=retrieval, policy=scripted_report
        )
        assert record["grades"]["success"], record
        assert record["grades"]["counts"]["citations"] > 0
        task = task_named("grounded_signal_report")
        corrupted = copy.deepcopy(record)
        corrupted["snapshot"]["final_result"]["report"]["claims"][0]["sources"][0]["source"][
            "id"
        ] = "fabricated"
        assert not grade_task(task, corrupted)["checks"]["citation_identity"]
        corrupted = copy.deepcopy(record)
        corrupted["events"].insert(0, {"event": "report.chunk"})
        assert not grade_task(task, corrupted)["invariants"]["publication_precedes_chunks"]
    finally:
        retrieval.close()


async def test_missing_rag_is_skipped_without_provider_attempt(predictor):
    row = await execute("grounded_signal_report", predictor)
    assert row["status"] == "skipped" and row["provider_attempts"] == []
    assert aggregate([row])["task_completion_rate"]["denominator"] == 0


async def test_budget_rejects_before_transport_and_mock_cannot_be_labeled_live(predictor):
    config = EvalConfig(max_total_tokens=1)
    calls = []

    def backend():
        return OpenAICompatibleBackend(
            "https://provider.example/v1",
            "test-key",
            "test-model",
            transport=httpx.MockTransport(lambda request: calls.append(request)),
        )

    kwargs = dict(
        repetition=1,
        config=config,
        settings=Settings(),
        predictor=predictor,
        retrieval=None,
        content=synthetic_input(10042),
        budget=EvaluationBudget(config),
        make_backend=backend,
    )
    with pytest.raises(ValueError):
        await execute_task(task_named("score_definition"), mode="live", **kwargs)
    row = await execute_task(task_named("score_definition"), mode="infrastructure_test", **kwargs)
    assert calls == [] and row["status"] == "skipped"
    assert row["skip_reason"] == "evaluation_token_budget_exceeded"
    assert kwargs["budget"].snapshot()["admitted_requests"] == 0


def test_report_does_not_mutate_records_or_invent_empty_success():
    row = {
        "task_id": "case",
        "mode": "infrastructure_test",
        "status": "infrastructure_error",
        "error_code": "backend_unavailable",
        "grades": {
            "eligible": False,
            "success": None,
            "failures": [],
            "invariant_violations": ["test"],
            "fault": {"configured": False, "triggered": False},
        },
    }
    saved = copy.deepcopy(row)
    report = {"mode": "infrastructure_test", "records": [row], "config": {}}
    first = render_markdown(report)
    assert first == render_markdown(report) and row == saved
    assert "not live LLM evaluation" in first and "API cost is unavailable" in first
    assert aggregate([row])["task_completion_rate"] == {
        "numerator": 0,
        "denominator": 1,
        "value": 0.0,
    }
    assert aggregate([row])["failure_categories"]["backend_unavailable"] == 1


def test_config_example_is_valid():
    assert load_config(Path("configs/live_llm.example.yaml")).max_tasks == 1


async def test_published_png_and_execution_spans_survive_fixture_cleanup(predictor, tmp_path):
    import hashlib

    content = synthetic_input(10042)
    raw = analyze_csv(content, predictor)["prediction"]
    config = EvalConfig()
    path = tmp_path / "artifacts"
    record = await execute_task(
        task_named("risk_plot_parameters"),
        repetition=1,
        config=config,
        settings=Settings(),
        predictor=predictor,
        retrieval=None,
        content=content,
        budget=EvaluationBudget(config),
        reference_prediction={k: raw[k] for k in ("risk_score", "prediction", "window_count")},
        mode="infrastructure_test",
        make_backend=lambda: ScriptedDecisionBackend(scripted_analysis),
        artifact_directory=path,
    )
    assert record["grades"]["success"], record
    assert len(record["artifact_files"]) == 1
    artifact = record["artifact_files"][0]
    png = (path / artifact["file"]).read_bytes()
    assert png.startswith(b"\x89PNG") and hashlib.sha256(png).hexdigest() == artifact["sha256"]
    assert all(s["end_time"] is not None for s in record["execution_spans"])
    assert not any(f.suffix == ".csv" for f in path.iterdir())


async def test_cleanup_failure_is_reported_without_losing_artifact_or_call_trace(predictor):
    class CloseFailure(ScriptedDecisionBackend):
        async def aclose(self):
            await super().aclose()
            raise RuntimeError("SECRET must never be logged")

    config = EvalConfig()
    record = await execute_task(
        task_named("csv_analysis"),
        repetition=1,
        config=config,
        settings=Settings(),
        predictor=predictor,
        retrieval=None,
        content=synthetic_input(10042),
        budget=EvaluationBudget(config),
        mode="infrastructure_test",
        make_backend=lambda: CloseFailure(scripted_analysis),
    )
    assert record["status"] == "infrastructure_error"
    assert record["error_code"] == "evaluation_cleanup_failed"
    assert len(record["artifacts"]) == 6 and len(record["skill_calls"]) == 5
    assert "SECRET" not in json.dumps(record)
