"""State-derived action constraints shared by model views and runtime checks."""

from dataclasses import dataclass

from smarthealth.agents.contracts import (
    CallSkillAction,
    DelegateAction,
    FinishAction,
    RequestInputAction,
    UpgradeWorkflowAction,
)
from smarthealth.runtime.errors import SkillError
from smarthealth.runtime.harness.config import IMPLEMENTED_SKILLS, REPORT_SKILLS

GUARD_VERSION = "2.0"


@dataclass(frozen=True)
class ActionPolicy:
    skills: frozenset[str]
    delegates: tuple[str, ...]
    upgrade_objectives: tuple[str, ...]
    finish_kind: str
    input_reasons: tuple[str, ...] = ()
    terminal_ref: str | None = None
    terminal_reason: str | None = None

    def validate(self, action):
        if self.terminal_reason:
            if not isinstance(action, FinishAction):
                code = (
                    "review_already_exists"
                    if self.terminal_reason == "review_completed"
                    else "role_complete"
                )
                raise SkillError(code, "Return the current terminal result reference.")
            if action.result_ref != self.terminal_ref:
                raise SkillError("unobserved_reference", "Return the current terminal result.")
        if isinstance(action, FinishAction) and action.result_kind != self.finish_kind:
            raise SkillError("objective_mismatch", "Finish must match the current role objective.")
        if isinstance(action, UpgradeWorkflowAction):
            if action.objective not in self.upgrade_objectives:
                raise SkillError("workflow_upgrade_denied", "This objective cannot upgrade.")
        if isinstance(action, CallSkillAction) and action.skill not in self.skills:
            raise SkillError("permission_denied", "This Skill is not available in this state.")
        if isinstance(action, DelegateAction) and action.agent not in self.delegates:
            raise SkillError("delegation_denied", "This role is not available for delegation.")
        if isinstance(action, RequestInputAction) and action.reason not in self.input_reasons:
            raise SkillError("input_not_required", "No required input is missing for this action.")


def required_input_reasons(record) -> tuple[str, ...]:
    """Only server-observed requirements can authorize asking the user for input."""
    if record.request.objective == "score_definition" or record.refs.get("published_analysis"):
        return ()
    reasons = []
    if not record.request.upload_refs:
        reasons.append("missing_upload")
    # Argument/schema errors are the Agent's responsibility, not invalid user data.
    data_errors = {
        "invalid_csv",
        "invalid_columns",
        "invalid_data",
        "invalid_timestamp",
        "file_too_large",
        "multiple_sessions",
        "insufficient_data",
        "poor_data_quality",
    }
    for skill, produced in (("csv_analysis", "dataset"), ("signal_processing", "window_set")):
        latest = next((o for o in reversed(record.observations) if o.skill == skill), None)
        if not record.refs.get(produced) and latest and latest.error_code in data_errors:
            reasons.append("invalid_data")
            break
    retrieval = next((o for o in reversed(record.observations) if o.skill == "rag_retrieval"), None)
    if (record.route and record.route.needs_input == "clarify_task") or (
        retrieval and retrieval.data.get("status") == "insufficient_evidence"
    ):
        reasons.append("clarify_task")
    return tuple(reasons)


def action_policy(runtime, record, role, report_context) -> ActionPolicy:
    simple = runtime._is_simple(record)
    skills = runtime.harness.allowed_skills(role, simple=simple)
    delegates = tuple(sorted(runtime.harness.allowed_delegates(role))) if not simple else ()
    if record.request.objective == "score_definition":
        # Also enforce the objective when an embedding application omits routing.
        skills, delegates = frozenset(), ()
    finish_kind = {
        "planner": record.request.objective,
        "data_analysis": (
            "score_definition"
            if record.request.objective == "score_definition"
            else "data_analysis"
        ),
        "rag": "evidence",
        "report_writer": "draft",
        "safety_verification": "review",
    }[role]
    upgrades = ()
    if simple and record.workflow_upgrades == 0 and record.request.objective != "score_definition":
        # An available capability is not authorization to expand the user's task.
        # A pluggable router may describe a stricter requirement while initially
        # selecting the simple role; only those requirements can be upgraded.
        required = set(record.route.required_capabilities)
        if required & IMPLEMENTED_SKILLS:
            upgrades = ("data_analysis",)
        if REPORT_SKILLS <= required and runtime._reports_available():
            upgrades += ("health_report",)

    terminal_ref = terminal_reason = None
    # Both pass and fail complete the reviewer role. Only Planner may revise/publish.
    if report_context.get("review_current"):
        if role == "safety_verification":
            terminal_ref, terminal_reason = report_context["review_ref"], "review_completed"
        elif role == "planner" and report_context.get("review_safe"):
            terminal_ref, terminal_reason = report_context["review_ref"], "report_ready"
    if simple and record.route.reason_code == "existing_result":
        refs = record.refs.get("published_analysis", [])
        if refs:
            terminal_ref, terminal_reason = refs[-1], "existing_result_ready"
    if terminal_reason:
        skills, delegates, upgrades = frozenset(), (), ()
    return ActionPolicy(
        skills=frozenset(skills),
        delegates=delegates,
        upgrade_objectives=upgrades,
        finish_kind=finish_kind,
        input_reasons=() if terminal_reason else required_input_reasons(record),
        terminal_ref=terminal_ref,
        terminal_reason=terminal_reason,
    )
