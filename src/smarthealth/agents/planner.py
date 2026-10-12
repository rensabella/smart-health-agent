"""Build the Planner/role decision view from authoritative, permission-scoped state.

The backend selects the next action; execution and budgets stay in the executor.
No static sequence or fabricated LLM decision is introduced here.
"""

from copy import deepcopy

from smarthealth.agents.contracts import DecisionRequest
from smarthealth.runtime.memory.schemas import MemoryContext


def build_decision_request(runtime, record, role, task):
    return DecisionRequest(
        run_id=record.context.run_id,
        agent=role,
        task=task,
        objective=record.request.objective,
        include_visualizations=record.request.include_visualizations,
        available_refs=deepcopy(record.refs),
        observations=[o.model_copy(deep=True) for o in record.observations],
        tools=runtime.registry.tool_definitions(
            runtime.harness.allowed_skills(role, simple=runtime._is_simple(record))
        ),
        allowed_delegates=sorted(runtime.harness.allowed_delegates(role))
        if not runtime._is_simple(record)
        else [],
        remaining_tool_calls=0
        if runtime._is_simple(record)
        else min(
            runtime.limits.max_tool_calls - record.context.budget.used,
            runtime.harness.role_tool_limit(role) - record.role_tool_calls.get(role, 0),
        ),
        remaining_decisions=runtime._decision_cap(record) - record.decision_count,
        workflow_type=record.route.workflow_type if record.route else None,
        can_upgrade_workflow=runtime._is_simple(record) and record.workflow_upgrades == 0,
        memory=record.memory_context.model_copy(deep=True)
        if runtime.memory is not None
        and runtime.memory.is_current(
            owner_id=record.context.owner_id,
            session_id=record.context.session_id,
            run_id=record.context.run_id,
        )
        else MemoryContext(),
        report_context=runtime._report_context(record, role),
    )
