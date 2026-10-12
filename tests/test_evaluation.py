import json
from pathlib import Path

import pytest
from rag_helpers import ScriptedEmbedding

from app.agents.backends import DecisionBackend, ScriptedDecisionBackend
from app.agents.contracts import DecisionReply, FinishAction
from app.core.errors import SkillError
from app.rag.retrieval import RetrievalService, build_index
from app.rag.schemas import KnowledgeChunk
from evaluation.cases import load_cases
from evaluation.metrics import aggregate, distribution, ratio, render_markdown
from evaluation.runner import (
    LiveBudget,
    MeteredBackend,
    backend_factory,
    evaluate_safety_case,
    evaluate_suite,
)
from examples.report_policy import scripted_report


@pytest.fixture
def eval_index(tmp_path):
    path = tmp_path / "index"
    build_index(Path("knowledge/corpus.json"), path, ScriptedEmbedding())
    service = RetrievalService(path, ScriptedEmbedding(), min_score=0.2)
    chunks = tuple(
        KnowledgeChunk.model_validate(c, strict=True)
        for c in json.loads((path / "chunks.json").read_text())
    )
    yield service, chunks
    service.close()


def test_missing_observations_are_not_perfect_scores():
    assert ratio(0, 0)["value"] is None
    assert distribution([None])["mean"] is None
    metrics = aggregate([], [], [], 1)
    assert metrics["safety"]["unsafe_response_rate"]["value"] is None
    assert metrics["rag"]["citation_precision"]["value"] is None
    assert metrics["system"]["first_token_latency"]["status"] == "not_measured"
    assert ratio(3, 4)["value"] == 0.75


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_nonfinite_or_negative_latency_is_rejected(value):
    with pytest.raises(ValueError):
        distribution([value])


async def test_evaluation_exposes_semantic_guard_gap_and_correct_denominators(
    predictor, csv_bytes, eval_index
):
    retrieval, chunks = eval_index
    suite = load_cases(Path("evaluation/test_cases.json"))
    report = await evaluate_suite(suite, predictor, retrieval, chunks, content=csv_bytes)
    metrics = report["metrics"]
    assert metrics["agent"]["scenario_pass_rate"]["value"] == 1.0
    assert metrics["agent"]["task_completion_rate"]["denominator"] == 8
    assert metrics["agent"]["tool_success_rate"]["denominator"] == 45
    assert metrics["agent"]["tool_success_rate"]["numerator"] == 44
    assert metrics["rag"]["retrieval_hit_rate"]["denominator"] == 8
    assert metrics["rag"]["citation_precision"]["value"] == 0.75
    assert metrics["safety"]["unsafe_response_rate"]["value"] == 0.25
    assert metrics["safety"]["decision_accuracy"]["value"] == 7 / 8
    assert report["external_llm_requests"] == 0
    markdown = render_markdown(report)
    assert "unsupported_sleep_equivalence" in markdown and "not measured" in markdown


async def test_expected_verdict_is_never_sent_to_safety_backend(predictor, csv_bytes, eval_index):
    _, chunks = eval_index
    suite = load_cases(Path("evaluation/test_cases.json"))
    case = suite.safety_cases[0]
    captured = []

    def policy(state):
        captured.append(state.model_dump(mode="json"))
        return scripted_report(state)

    result = await evaluate_safety_case(
        case, predictor, chunks, csv_bytes, lambda: ScriptedDecisionBackend(policy)
    )
    assert result["published"] and result["correct"]
    serialized = json.dumps(captured, ensure_ascii=False)
    assert case.rationale not in serialized and case.id not in serialized
    assert '"expected"' not in serialized and '"should_publish"' not in serialized


async def test_provider_failure_does_not_count_as_correct_safety_rejection(
    predictor, csv_bytes, eval_index
):
    _, chunks = eval_index
    case = load_cases(Path("evaluation/test_cases.json")).safety_cases[3]

    def unavailable(state):
        raise RuntimeError("Simulated provider outage")

    result = await evaluate_safety_case(
        case, predictor, chunks, csv_bytes, lambda: ScriptedDecisionBackend(unavailable)
    )
    assert result["operational_error"] and not result["correct"] and not result["published"]


async def test_changed_index_does_not_evaluate_against_stale_labels(
    predictor, csv_bytes, eval_index
):
    retrieval, chunks = eval_index
    suite = load_cases(Path("evaluation/test_cases.json"))
    with pytest.raises(ValueError, match="do not match"):
        await evaluate_suite(suite, predictor, retrieval, chunks[:-1], content=csv_bytes)


def test_live_mode_cannot_silently_use_disabled_backend(monkeypatch):
    monkeypatch.setenv("HEALTH_AGENT_BACKEND", "disabled")
    with pytest.raises(SkillError, match="Live evaluation"):
        backend_factory("live", LiveBudget())()


async def test_live_request_cap_is_shared_across_cases():
    class LiveFixture(DecisionBackend):
        mode = "live"

        async def decide(self, request):
            return DecisionReply(decision=FinishAction(result_kind="score_definition"))

        async def aclose(self):
            pass

    budget = LiveBudget(limit=1)
    first, second = MeteredBackend(LiveFixture(), budget), MeteredBackend(LiveFixture(), budget)
    await first.decide(None)
    with pytest.raises(SkillError) as error:
        await second.decide(None)
    assert error.value.code == "evaluation_live_budget_exceeded" and budget.used == 1
