"""Build the Planner/role decision view from authoritative, permission-scoped state.

The backend selects the next action; execution and budgets stay in the executor.
No static sequence or fabricated LLM decision is introduced here.
"""

from copy import deepcopy

from smarthealth.agents.contracts import DecisionRequest, TerminalResult
from smarthealth.runtime.action_policy import action_policy
from smarthealth.runtime.memory.schemas import MemoryContext


def build_decision_request(runtime, record, role, task):
    report_context = runtime._report_context(record, role)
    policy = action_policy(runtime, record, role, report_context)
    terminal = (
        TerminalResult(
            reason=policy.terminal_reason,
            result_kind=policy.finish_kind,
            result_ref=policy.terminal_ref,
        )
        if policy.terminal_reason
        else None
    )
    if terminal:
        # The semantic review is complete. Keep its status and exact references;
        # source text remains in the store and is checked again on publication.
        report_context = {
            k: v
            for k, v in report_context.items()
            if k in {"review_ref", "draft_ref", "review_safe", "review_current", "review"}
        }
    budget = record.budget_snapshot()
    return DecisionRequest(
        run_id=record.context.run_id,
        agent=role,
        task=task,
        objective=record.request.objective,
        include_visualizations=record.request.include_visualizations,
        available_refs=deepcopy(record.refs),
        observations=[
            o.model_copy(deep=True)
            for o in (list(record.observations)[-1:] if terminal else record.observations)
        ],
        tools=runtime.registry.tool_definitions(policy.skills),
        allowed_delegates=list(policy.delegates),
        remaining_tool_calls=0
        if runtime._is_simple(record) or terminal
        else min(
            runtime.limits.max_tool_calls - record.context.budget.used,
            runtime.harness.role_tool_limit(role) - record.role_tool_calls.get(role, 0),
        ),
        remaining_decisions=runtime._decision_cap(record) - record.decision_count,
        workflow_type=record.route.workflow_type if record.route else None,
        can_upgrade_workflow=bool(policy.upgrade_objectives),
        allowed_upgrade_objectives=list(policy.upgrade_objectives),
        allowed_finish_kinds=[policy.finish_kind],
        allowed_input_reasons=list(policy.input_reasons),
        terminal_result=terminal,
        remaining_tokens=budget.available_tokens if budget else None,
        memory=record.memory_context.model_copy(deep=True)
        if runtime.memory is not None
        and runtime.memory.is_current(
            owner_id=record.context.owner_id,
            session_id=record.context.session_id,
            run_id=record.context.run_id,
        )
        else MemoryContext(),
        report_context=report_context,
    )
