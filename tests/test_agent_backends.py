import json

import httpx
import pytest

from app.agents.backends import (
    OpenAICompatibleBackend,
    ScriptedDecisionBackend,
    backend_from_env,
)
from app.agents.contracts import DecisionRequest, FinishAction
from app.core.errors import SkillError


def state(agent="data_analysis"):
    return DecisionRequest(
        run_id="run1",
        agent=agent,
        task="Explain the score",
        objective="score_definition",
        include_visualizations=False,
        available_refs={},
        observations=[],
        tools=[],
        allowed_delegates=["data_analysis"] if agent == "planner" else [],
        remaining_tool_calls=3,
        remaining_decisions=4,
    )


def response(name="finish", arguments=None):
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(
                                    arguments
                                    or {"result_kind": "score_definition", "result_ref": None}
                                ),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 31, "completion_tokens": 9},
    }


async def test_live_adapter_tool_protocol_and_usage():
    captured = []

    def handle(request):
        captured.append(request)
        return httpx.Response(200, json=response())

    backend = OpenAICompatibleBackend(
        "https://llm.example/v1",
        "test-secret",
        "configured-model",
        transport=httpx.MockTransport(handle),
    )
    try:
        reply = await backend.decide(state("planner"))
        assert isinstance(reply.decision, FinishAction)
        assert (reply.input_tokens, reply.output_tokens) == (31, 9)
        request = captured[0]
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-secret"
        assert payload["tool_choice"] == "required" and not payload["parallel_tool_calls"]
        assert not payload["store"] and payload["max_completion_tokens"] == 2048
        assert {t["function"]["name"] for t in payload["tools"]} == {
            "finish",
            "delegate",
            "request_input",
        }
        assert "test-secret" not in request.content.decode()
    finally:
        await backend.aclose()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["choices"][0]["message"].update(content="SECRET free diagnosis"),
        lambda r: r["choices"][0]["message"].update(refusal="SECRET"),
        lambda r: r["choices"][0]["message"]["tool_calls"].extend(
            r["choices"][0]["message"]["tool_calls"][:]
        ),
        lambda r: r["choices"][0]["message"]["tool_calls"][0]["function"].update(
            arguments='{"result_kind":"score_definition","extra":"SECRET"}'
        ),
        lambda r: r["choices"][0]["message"]["tool_calls"][0]["function"].update(
            arguments='{"result_kind":"score_definition","result_kind":"health_report"}'
        ),
        lambda r: r["usage"].update(prompt_tokens=True),
    ],
)
async def test_live_adapter_rejects_unsafe_or_ambiguous_response(mutate):
    result = response()
    mutate(result)
    backend = OpenAICompatibleBackend(
        "https://llm.example/v1",
        "test-secret",
        "model",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=result)),
    )
    try:
        with pytest.raises(SkillError) as error:
            await backend.decide(state())
        assert "SECRET" not in str(error.value) and "test-secret" not in str(error.value)
    finally:
        await backend.aclose()


@pytest.mark.parametrize(
    "status,code",
    [
        (401, "backend_authentication_failed"),
        (429, "backend_rate_limited"),
        (503, "backend_unavailable"),
        (302, "backend_redirect_rejected"),
    ],
)
async def test_no_implicit_http_retries(status, code):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            status, text="SECRET upstream body", headers={"location": "https://elsewhere.example/"}
        )

    backend = OpenAICompatibleBackend(
        "https://llm.example/v1", "secret", "model", transport=httpx.MockTransport(handle)
    )
    try:
        with pytest.raises(SkillError) as error:
            await backend.decide(state())
        assert error.value.code == code and len(calls) == 1 and "SECRET" not in str(error.value)
    finally:
        await backend.aclose()


async def test_scripted_and_disabled_are_explicit(monkeypatch):
    monkeypatch.delenv("HEALTH_AGENT_BACKEND", raising=False)
    assert backend_from_env().mode == "disabled"
    backend = ScriptedDecisionBackend(lambda request: FinishAction(result_kind=request.objective))
    assert (await backend.decide(state())).decision.result_kind == "score_definition"
    assert backend.mode == "scripted"
    monkeypatch.setenv("HEALTH_AGENT_BACKEND", "scripted")
    with pytest.raises(SkillError):
        backend_from_env()


@pytest.mark.parametrize(
    "delegates", [[], ["rag"], ["data_analysis", "rag", "report_writer", "safety_verification"]]
)
async def test_provider_delegate_schema_matches_runtime_permissions(delegates):
    captured = []

    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=response())

    backend = OpenAICompatibleBackend(
        "https://llm.example/v1", "test-secret", "model", transport=httpx.MockTransport(handle)
    )
    try:
        request = state("planner").model_copy(update={"allowed_delegates": delegates})
        await backend.decide(request)
        tools = {t["function"]["name"]: t["function"] for t in captured[0]["tools"]}
        if delegates:
            assert tools["delegate"]["parameters"]["properties"]["agent"]["enum"] == sorted(
                delegates
            )
        else:
            assert "delegate" not in tools
    finally:
        await backend.aclose()


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.example/v1",
        "https://user:SECRET@llm.example/v1",
        "https://llm.example/v1?key=SECRET",
    ],
)
def test_provider_endpoint_is_admin_validated(url):
    with pytest.raises(SkillError) as error:
        OpenAICompatibleBackend(url, "secret", "model")
    assert error.value.code == "backend_configuration_error" and "SECRET" not in str(error.value)
