"""Mutation tests for canonical grading; these are infrastructure records, not live scores."""

from copy import deepcopy

import pytest
from test_live_runner import execute, task_named
from test_report_runtime import retrieval as retrieval

from smarthealth.evaluation.live_llm.graders import GRADING_CONTRACT_VERSION, grade_task
from smarthealth.examples.report_policy import scripted_report


@pytest.mark.parametrize(
    "corruption,check",
    [
        ("input", "score_no_input_request"),
        ("upgrade", "score_no_workflow_upgrade"),
        ("skill", "score_no_skill_or_delegation"),
    ],
)
async def test_completed_score_cannot_hide_disallowed_behavior(predictor, corruption, check):
    row = await execute("score_definition", predictor)
    assert row["grades"]["success"]
    changed = deepcopy(row)
    if corruption == "input":
        changed["decision_actions"].append({"action": "request_input", "reason": "missing_upload"})
    elif corruption == "upgrade":
        changed["snapshot"]["workflow_upgrades"] = 1
    else:
        changed["events"].insert(
            -2, {"event": "skill.dispatched", "data": {"skill": "csv_analysis"}}
        )
    grade = grade_task(task_named("score_definition"), changed)
    assert grade["contract_version"] == GRADING_CONTRACT_VERSION
    assert not grade["success"] and not grade["checks"][check]
    assert row["grades"]["success"]  # Grading never rewrites an existing saved grade.


@pytest.mark.parametrize(
    "corruption,check",
    [
        ("review", "required_safety_review"),
        ("completion", "reviewer_returns_to_planner"),
        ("return", "reviewer_returns_to_planner"),
        ("wrong_parent", "reviewer_returns_to_planner"),
        ("duplicate", "one_review_per_draft_ref"),
        ("early_publication", "publication_current_review"),
        ("stale_draft", "publication_current_review"),
        ("hash", "publication_current_review"),
        ("artifacts", "required_artifacts"),
    ],
)
async def test_report_cannot_pass_with_missing_or_invalid_review_lifecycle(
    predictor, retrieval, corruption, check
):
    row = await execute(
        "grounded_signal_report", predictor, retrieval=retrieval, policy=scripted_report
    )
    assert row["grades"]["success"], row["grades"]
    changed = deepcopy(row)
    events = changed["events"]
    if corruption in {"review", "completion", "return"}:
        removed = {
            "review": "safety.reviewed",
            "completion": "agent.completed",
            "return": "agent.delegation_completed",
        }[corruption]
        changed["events"] = [
            e
            for e in events
            if not (
                e["event"] == removed
                and (
                    removed == "safety.reviewed"
                    or e.get("agent") == "safety_verification"
                    or e.get("data", {}).get("agent") == "safety_verification"
                )
            )
        ]
    elif corruption == "wrong_parent":
        next(
            e
            for e in events
            if e["event"] == "agent.delegation_completed"
            and e["data"].get("agent") == "safety_verification"
        )["step_id"] = "unrelated_delegation"
    elif corruption == "duplicate":
        dispatch = next(
            e
            for e in events
            if e["event"] == "skill.dispatched" and e["data"]["skill"] == "safety_verification"
        )
        events.insert(events.index(dispatch) + 1, deepcopy(dispatch))
    elif corruption == "early_publication":
        publication = next(e for e in events if e["event"] == "publication.checked")
        events.remove(publication)
        events.insert(0, publication)
    elif corruption == "stale_draft":
        draft = deepcopy(next(a for a in changed["artifacts"] if a["kind"] == "report_draft"))
        draft["ref"] = "art_" + "f" * 32
        changed["artifacts"].append(draft)
    elif corruption == "hash":
        next(a for a in changed["artifacts"] if a["kind"] == "report_review")["review_hash"] = (
            "f" * 64
        )
    else:
        changed["artifacts"] = []
    grade = grade_task(task_named("grounded_signal_report"), changed)
    assert not grade["success"] and not grade["checks"][check]
    assert row["grades"]["success"]


async def test_safe_blocking_is_not_completion_for_result_bearing_task(predictor):
    row = await execute("execution_budget", predictor)
    assert row["grades"]["success"]  # This task explicitly tests a negative control.
    task = task_named("csv_analysis")
    # Even broadening a fixture's allowed status cannot bypass required completion.
    task = task.model_copy(
        update={
            "expected": task.expected.model_copy(
                update={"statuses": ["blocked", "failed", "completed"]}
            )
        }
    )
    grade = grade_task(task, row)
    assert grade["checks"]["expected_status"] and not grade["checks"]["required_completion"]
    assert not grade["success"]
