"""Ungated decision backend: the same wire protocol plus a final_answer tool.

The final answer tool is a plain tool entry, so the existing reply parser in
``agents.backends`` turns its calls into an ordinary ``CallSkillAction``; the
ungated loop itself recognizes the pseudo tool and terminates. Nothing in the
production parsing path changes. Only ``_prepare_request`` is overridden: the
inherited ``prepare`` builds the ``PreparedDecision`` for this instance and the
inherited transport, admission bounds and reply parsing are reused unchanged.
"""

from __future__ import annotations

from pydantic import Field

from smarthealth.agents.backends import (
    _CONTROL_NAMES,
    _TOOL_NAME,
    MAX_CONTEXT_BYTES,
    MAX_TOOLS,
    OpenAICompatibleBackend,
    _error,
    _serialize_payload,
    _validated_request,
)
from smarthealth.agents.contracts import DecisionRequest
from smarthealth.evaluation.baselines.prompts import build_ungated_messages
from smarthealth.skills.base_skill import SkillModel

FINAL_ANSWER_TOOL = "final_answer"


class FinalAnswerInput(SkillModel):
    text: str = Field(min_length=1, max_length=8000)


def final_answer_tool() -> dict:
    """One-field tool entry, shaped exactly like the runtime control tools."""
    schema = FinalAnswerInput.model_json_schema()
    schema.pop("title", None)
    for prop in schema["properties"].values():
        prop.pop("default", None)
    schema["required"] = list(schema["properties"])
    return {
        "type": "function",
        "function": {
            "name": FINAL_ANSWER_TOOL,
            "description": (
                "Deliver the final answer for the task. The text field must contain "
                "the complete user-facing answer."
            ),
            "parameters": schema,
            "strict": True,
        },
    }


class UngatedBackend(OpenAICompatibleBackend):
    """Chat Completions adapter whose tool list ends with final_answer.

    The gate-shaped control tools and the role prompt are never sent; the model
    selects among the task's Skill tools plus the final answer tool.
    """

    def __init__(self, *args, no_rag: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_rag = no_rag

    def _prepare_request(self, request: DecisionRequest) -> tuple[bytes, frozenset[str]]:
        request = _validated_request(request)
        tools = request.tools.copy()
        if len(tools) + 1 > MAX_TOOLS:
            raise _error("backend_invalid_request", "Too many decision tools were supplied.")
        names: set[str] = set()
        for tool in tools:
            function = tool.get("function")
            if tool.get("type") != "function" or not isinstance(function, dict):
                raise _error("backend_invalid_request", "The decision tool schema is invalid.")
            name = function.get("name")
            schema = function.get("parameters")
            if (
                not isinstance(name, str)
                or _TOOL_NAME.fullmatch(name) is None
                or name in names
                or name in _CONTROL_NAMES
                or name == FINAL_ANSWER_TOOL
                or not isinstance(schema, dict)
                or schema.get("type") != "object"
                or not isinstance(schema.get("properties"), dict)
            ):
                raise _error("backend_invalid_request", "The decision tool schema is invalid.")
            names.add(name)
        tools.append(final_answer_tool())
        names.add(FINAL_ANSWER_TOOL)
        try:
            payload = _serialize_payload(
                model=self._model,
                tools=tools,
                messages=build_ungated_messages(request, no_rag=self.no_rag),
                max_completion_tokens=self._max_completion_tokens,
                api_profile=self.api_profile,
                temperature=self._temperature,
                seed=self._seed,
            )
        except Exception:
            raise _error(
                "backend_invalid_request", "The decision state cannot be encoded."
            ) from None
        if len(payload) > MAX_CONTEXT_BYTES:
            raise _error(
                "backend_context_too_large", "The decision context exceeds its size limit."
            )
        return payload, frozenset(names)
