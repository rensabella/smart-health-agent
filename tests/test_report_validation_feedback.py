"""Real report schema/service with synthetic facts; never a provider evaluation."""

import json
import logging
from copy import deepcopy
from dataclasses import replace

import pytest
from test_harness_validator import facts as facts
from test_report_runtime import close, execute, make
from test_report_runtime import retrieval as retrieval
from test_reports import report_facts as report_facts

from smarthealth.agents.contracts import CallSkillAction
from smarthealth.examples.report_policy import scripted_report
from smarthealth.reports.schemas import GenerationInput
from smarthealth.skills.registry import SkillRegistry
from smarthealth.skills.report_skill import ReportSkill


@pytest.mark.parametrize(
    "case,path,category,actual",
    [
        ("missing", "$.claims[0].claim_id", "missing", "missing"),
        ("type", "$.claims[0].text", "string_type", "boolean"),
        ("evidence", "$.claims[0].evidence_ids", "list_type", "string"),
        ("length", "$.claims[0].text", "string_too_long", "string"),
        ("qualitative", "$.claims[0].text", "value_error", "string"),
        ("revision", "$", "value_error", "object"),
        ("extra", "$.<unknown>", "extra_forbidden", "string"),
    ],
)
async def test_report_schema_diagnostics_are_safe_and_valid_correction_executes(
    report_facts, case, path, category, actual, caplog
):
    facts, reports, args = report_facts
    service = facts["service"]
    service.reports = reports
    context = replace(
        facts["context"], allowed_skills={"report_generation"}, agent_name="report_writer"
    )
    registry = SkillRegistry()
    registry.register(ReportSkill(service))
    caplog.set_level(logging.INFO, logger="smart_health")
    invalid = deepcopy(args)
    claim = invalid["claims"][0]
    marker = "PRIVATE_PHYSIOLOGICAL_VALUE_API_SECRET"
    if case == "missing":
        del claim["claim_id"]
    elif case == "type":
        claim["text"] = True
    elif case == "evidence":
        claim["evidence_ids"] = marker
    elif case == "length":
        claim["text"] = marker * 40
    elif case == "qualitative":
        claim["text"] = marker + " 83.1234567"
    elif case == "revision":
        invalid["previous_draft_ref"] = "art_" + "a" * 32
    elif case == "extra":
        invalid[marker] = marker
    before = service.store.item_count
    try:
        # Exported wire schema is the actual strict Skill input schema, including
        # constraints implemented by custom validators rather than JSON Schema.
        tool = registry.tool_definitions({"report_generation"})[0]["function"]
        assert tool["parameters"] == GenerationInput.model_json_schema()
        assert (
            "qualitative"
            in tool["parameters"]["$defs"]["ClaimDraft"]["properties"]["text"]["description"]
        )
        rejected = await registry.execute("report_generation", invalid, context)
        assert rejected.error.code == "invalid_input"
        error = rejected.error.validation_error
        issue = error.issues[0]
        assert (issue.field_path, issue.category, issue.actual_type) == (path, category, actual)
        assert issue.expected and error.retryable
        assert context.budget.used == 0 and service.store.item_count == before
        trace = registry.trace_sink.events_for(context)
        assert [e["event"] for e in trace] == ["skill.rejected"]
        assert trace[0]["validation_error"] == error.model_dump()
        serialized = json.dumps([rejected.model_dump(), trace]) + caplog.text
        for sensitive in (marker, "83.1234567", args["claims"][0]["text"], "https://"):
            assert sensitive not in serialized
        accepted = await registry.execute("report_generation", args, context)
        assert accepted.status == "success" and accepted.data["draft_version"] == 1
        assert context.budget.used == 1 and service.store.item_count == before + 1
    finally:
        await registry.aclose()


async def test_many_private_extra_keys_are_bounded_and_redacted(report_facts):
    facts, reports, args = report_facts
    facts["service"].reports = reports
    registry = SkillRegistry()
    registry.register(ReportSkill(facts["service"]))
    context = replace(facts["context"], allowed_skills={"report_generation"})
    args.update({f"PRIVATE_FIELD_{i}": "PRIVATE_VALUE" for i in range(20)})
    try:
        result = await registry.execute("report_generation", args, context)
        diagnostic = result.error.validation_error
        assert len(diagnostic.issues) == 4 and diagnostic.truncated
        assert all(i.field_path == "$.<unknown>" for i in diagnostic.issues)
        assert "PRIVATE" not in json.dumps(registry.trace_sink.events_for(context))
    finally:
        await registry.aclose()


@pytest.mark.parametrize("recover", [True, False])
async def test_writer_gets_bounded_feedback_then_completes_or_safely_blocks(
    predictor, csv_bytes, retrieval, recover
):
    attempts = 0
    feedback = []

    def policy(state):
        nonlocal attempts
        if state.agent == "report_writer":
            observed = [
                o.data["validation_error"]
                for o in state.observations
                if o.skill == "report_generation" and "validation_error" in o.data
            ]
            if observed:
                feedback.append(observed[-1])
        action = scripted_report(state)
        if isinstance(action, CallSkillAction) and action.skill == "report_generation":
            attempts += 1
            if attempts == 1 or not recover:
                action.arguments["claims"][0]["text"] = "PRIVATE 91.123456"
        return action

    runtime = make(predictor, retrieval, policy)
    try:
        result = await execute(runtime, csv_bytes)
        journal = runtime.get_events(result.run_id, owner_id="owner", session_id=result.session_id)
        assert attempts == 2 and feedback
        assert feedback[0]["remaining_corrections"] == 1 and feedback[0]["retryable"]
        assert feedback[0]["issues"][0]["field_path"] == "$.claims[0].text"
        assert "PRIVATE" not in json.dumps([e.model_dump() for e in journal])
        failures = [
            e
            for e in journal
            if e.event == "skill.failed" and e.data.get("skill") == "report_generation"
        ]
        if recover:
            assert result.status == "completed" and result.tool_calls == 8
            assert len(failures) == 1
            assert result.final_result["report"]["safety"]["safe"] is True
            assert sum(e.event == "safety.reviewed" for e in journal) == 1
        else:
            assert result.status == "blocked" and result.error_code == "skill_validation_exhausted"
            assert result.final_result is None and result.tool_calls == 6
            assert len(failures) == 2
            last = failures[-1].data["validation_error"]
            assert last["remaining_corrections"] == 0 and not last["retryable"]
            assert not any(
                e.event in {"safety.reviewed", "publication.checked", "report.chunk"}
                for e in journal
            )
            assert journal[-1].event == "run.blocked"
    finally:
        await close(runtime)


async def test_writer_correction_allowance_does_not_reset_on_redelegation(
    predictor, csv_bytes, retrieval
):
    writes = 0

    def policy(state):
        nonlocal writes
        action = scripted_report(state)
        if isinstance(action, CallSkillAction) and action.skill == "report_generation":
            writes += 1
            if writes in {1, 3}:
                action.arguments["claims"][0]["text"] = "Invalid numerical claim 42"
        if isinstance(action, CallSkillAction) and action.skill == "safety_verification":
            for review in action.arguments["reviews"]:
                review.update(supported=False, reason="insufficient_support")
        return action

    runtime = make(predictor, retrieval, policy)
    try:
        result = await execute(runtime, csv_bytes)
        assert writes == 3 and result.status == "blocked"
        assert result.error_code == "skill_validation_exhausted" and result.final_result is None
        assert runtime._runs[result.run_id].writer_schema_failures == 2
        assert result.tool_calls == 8  # Two rejected calls never enter the Skill body.
    finally:
        await close(runtime)
