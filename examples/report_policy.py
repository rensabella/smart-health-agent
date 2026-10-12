"""Deterministic multi-role fixture, never an LLM or production default."""

from app.agents.contracts import CallSkillAction, DelegateAction, FinishAction, RequestInputAction
from examples.demo_policy import scripted_analysis


def scripted_report(state):
    if state.objective != "health_report" or state.agent == "data_analysis":
        return scripted_analysis(state)
    refs, context = state.available_refs, state.report_context
    if state.agent == "planner":
        for key, agent in (
            ("analysis", "data_analysis"),
            ("evidence", "rag"),
            ("draft", "report_writer"),
        ):
            if not refs.get(key):
                return DelegateAction(
                    agent=agent,
                    task="Produce the next scoped result using current evidence and observations.",
                )
        if not context.get("review_current"):
            return DelegateAction(
                agent="safety_verification",
                task="Independently assess each claim against its cited evidence.",
            )
        if not context.get("review_safe"):
            return DelegateAction(
                agent="report_writer", task="Revise the current rejected draft using its review."
            )
        return FinishAction(result_kind="health_report", result_ref=refs["review"][-1])
    if state.agent == "rag":
        if refs.get("evidence"):
            if any(o.data.get("status") == "insufficient_evidence" for o in state.observations):
                return RequestInputAction(reason="clarify_task")
            return FinishAction(result_kind="evidence", result_ref=refs["evidence"][-1])
        return CallSkillAction(
            skill="rag_retrieval",
            arguments={
                "analysis_ref": refs["analysis"][-1],
                "query": "EDA 皮电 activity 活动与传感器测量背景",
                "top_k": 4,
            },
        )
    if state.agent == "report_writer":
        # A newly generated draft has no current review; return it to Planner.
        if refs.get("draft") and not context.get("review_current"):
            return FinishAction(result_kind="draft", result_ref=refs["draft"][-1])
        hits = context["retrieved_evidence"]["hits"]
        hit = next((h for h in hits if h["source"]["id"] == "empatica_eda_context"), hits[0])
        text = (
            "身体活动和兴奋等情境可能伴随皮电变化；皮电不能单独确定个人的压力状态或异常原因。"
            if hit["source"]["id"] == "empatica_eda_context"
            else "该检索资料提供一般信号与测量背景，不能单独证明当前个体异常的原因。"
        )
        args = {
            "analysis_ref": refs["analysis"][-1],
            "evidence_ref": refs["evidence"][-1],
            "claims": [
                {"claim_id": "signal_context", "text": text, "evidence_ids": [hit["evidence_id"]]}
            ],
        }
        if refs.get("draft"):
            args.update(
                previous_draft_ref=refs["draft"][-1], previous_review_ref=refs["review"][-1]
            )
        return CallSkillAction(skill="report_generation", arguments=args)
    if state.agent == "safety_verification":
        if context.get("review_current"):
            return FinishAction(result_kind="review", result_ref=refs["review"][-1])
        return CallSkillAction(
            skill="safety_verification",
            arguments={
                "draft_ref": refs["draft"][-1],
                "draft_hash": context["draft_hash"],
                "evidence_hash": context["evidence_hash"],
                "reviews": [
                    {
                        "claim_id": c["claim_id"],
                        "supported": True,
                        "within_scope": True,
                        "no_diagnosis": True,
                        "reason": "supported",
                    }
                    for c in context["claims"]
                ],
            },
        )
    raise ValueError("Unknown report fixture role")
