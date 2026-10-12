"""Small decision adapters with explicit limits and no automatic paid-call retries.

The live adapter uses the Chat Completions function-calling protocol. It sends one
bounded state snapshot per decision. Skill execution, permission checks, run state,
and final result publication belong to the runtime, not to this provider adapter.
"""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ValidationError

from smarthealth.agents.contracts import (
    ACTION_ADAPTER,
    AgentAction,
    DecisionReply,
    DecisionRequest,
    DelegateAction,
    FinishAction,
    RequestInputAction,
    UpgradeWorkflowAction,
)
from smarthealth.agents.prompts import build_messages
from smarthealth.config.settings import Settings
from smarthealth.runtime.errors import SkillError

MAX_CONTEXT_BYTES = 131_072
MAX_RESPONSE_BYTES = 65_536
MAX_ARGUMENT_BYTES = 32_768
MAX_TOOLS = 32
MAX_COMPLETION_TOKENS = 2_048
_CONTROL_NAMES = frozenset({"finish", "request_input", "delegate", "upgrade_workflow"})
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def _error(code: str, message: str) -> SkillError:
    # Callers only use constant messages; provider text and credentials stay private.
    return SkillError(code, message)


def _validated_request(request: DecisionRequest) -> DecisionRequest:
    try:
        return DecisionRequest.model_validate(request.model_dump(warnings=False))
    except Exception:
        raise _error("backend_invalid_request", "The decision state is invalid.") from None


def _validated_reply(value: Any) -> DecisionReply:
    try:
        if isinstance(value, DecisionReply):
            return DecisionReply.model_validate(value.model_dump(warnings=False))
        if isinstance(value, BaseModel):
            value = value.model_dump(warnings=False)
        if isinstance(value, dict) and "decision" in value:
            return DecisionReply.model_validate(value)
        return DecisionReply(decision=ACTION_ADAPTER.validate_python(value))
    except Exception:
        raise _error(
            "backend_invalid_decision", "The decision does not match its schema."
        ) from None


class DecisionBackend(ABC):
    """One decision per call; any retries must be visible to the run's budget."""

    mode: ClassVar[str]
    model_name: str | None = None

    @abstractmethod
    async def decide(self, request: DecisionRequest) -> DecisionReply:
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        raise NotImplementedError


class DisabledDecisionBackend(DecisionBackend):
    mode = "disabled"

    async def decide(self, request: DecisionRequest) -> DecisionReply:
        raise _error("backend_unavailable", "No live decision backend is configured.")

    async def aclose(self) -> None:
        pass


ScriptedDecision = Callable[
    [DecisionRequest],
    AgentAction | dict | DecisionReply | Awaitable[AgentAction | dict | DecisionReply],
]


class ScriptedDecisionBackend(DecisionBackend):
    """Deterministic test/example adapter. This is not an LLM or a production default.

    The callback selects an action from its request. This class keeps no conversation
    or sequence cursor that could accidentally mix concurrent runs.
    """

    mode = "scripted"

    def __init__(self, decision: ScriptedDecision):
        if not callable(decision):
            raise _error("backend_configuration_error", "A decision callback is required.")
        self._decision = decision
        self._closed = False

    async def decide(self, request: DecisionRequest) -> DecisionReply:
        if self._closed:
            raise _error("backend_unavailable", "The decision backend is closed.")
        validated = _validated_request(request)
        try:
            value = self._decision(validated)
            if inspect.isawaitable(value):
                value = await value
        except Exception:
            raise _error("backend_unavailable", "The scripted decision callback failed.") from None
        return _validated_reply(value)

    async def aclose(self) -> None:
        self._closed = True


def _control_tool(name: str, description: str, schema_model: type[BaseModel]) -> dict:
    schema = schema_model.model_json_schema()
    schema["properties"].pop("action")
    schema.pop("title", None)
    for prop in schema["properties"].values():
        prop.pop("default", None)
    # Strict control tools require all keys. Optional values, e.g. result_ref,
    # already use a nullable JSON schema and are sent explicitly as null.
    schema["required"] = list(schema["properties"])
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
            "strict": True,
        },
    }


def _tools_for(request: DecisionRequest) -> tuple[list[dict], frozenset[str]]:
    tools = request.tools.copy()
    if len(tools) > MAX_TOOLS:
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
            or not isinstance(schema, dict)
            or schema.get("type") != "object"
            or not isinstance(schema.get("properties"), dict)
        ):
            raise _error("backend_invalid_request", "The decision tool schema is invalid.")
        names.add(name)
    tools.extend(
        [
            _control_tool(
                "finish", "Request publication of a runtime-verified result.", FinishAction
            ),
            _control_tool(
                "request_input", "Pause this run for missing or unusable input.", RequestInputAction
            ),
        ]
    )
    if request.can_upgrade_workflow and request.workflow_type == "simple":
        tools.append(
            _control_tool(
                "upgrade_workflow",
                "Escalate this simple task once to a Planner workflow; budgets are shared.",
                UpgradeWorkflowAction,
            )
        )
    if request.agent == "planner" and request.allowed_delegates:
        delegation = _control_tool(
            "delegate",
            "Ask an allowed specialized agent to work on this task; its result returns to the Planner.",
            DelegateAction,
        )
        delegation["function"]["parameters"]["properties"]["agent"] = {
            "type": "string",
            "enum": sorted(set(request.allowed_delegates)),
        }
        tools.append(delegation)
    names.update(tool["function"]["name"] for tool in tools)
    return tools, frozenset(names)


def _endpoint(base_url: str) -> str:
    try:
        if (
            not isinstance(base_url, str)
            or not base_url
            or len(base_url) > 2048
            or any(character.isspace() or ord(character) < 32 for character in base_url)
        ):
            raise ValueError
        parts = urlsplit(base_url)
        if (
            parts.scheme not in {"https", "http"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
            or parts.port == 0
        ):
            raise ValueError
        if parts.scheme == "http" and parts.hostname != "localhost":
            if not ipaddress.ip_address(parts.hostname).is_loopback:
                raise ValueError
        # Constructing the URL now prevents unsafe URL errors from appearing later
        # in request logs. The administrator supplies an API root, typically /v1.
        return str(httpx.URL(base_url.rstrip("/") + "/chat/completions"))
    except Exception:
        raise _error(
            "backend_configuration_error", "The provider URL must use HTTPS or loopback HTTP."
        ) from None


def _strict_json(data: bytes | str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError("Invalid JSON constant")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    if isinstance(data, bytes):
        data = data.decode("utf-8")
    value = json.loads(data, parse_constant=reject_constant, object_pairs_hook=unique_object)
    # JSON exponents such as 1e999 can produce infinity without parse_constant.
    json.dumps(value, allow_nan=False)
    return value


class OpenAICompatibleBackend(DecisionBackend):
    """Bounded, non-streaming decision requests; SSE progress is owned by runtime.

    The URL and credentials are administrator configuration, never API request
    fields. Redirects and environment proxy discovery are disabled so credentials
    are sent only to that configured endpoint. No provider body is logged.
    """

    mode = "live"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 30,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._endpoint = _endpoint(base_url)
        if (
            not isinstance(api_key, str)
            or not 1 <= len(api_key) <= 4096
            or any(not 33 <= ord(character) <= 126 for character in api_key)
            or not isinstance(model, str)
            or _MODEL_NAME.fullmatch(model) is None
            or isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 120
        ):
            raise _error("backend_configuration_error", "The provider configuration is invalid.")
        self._model = model
        self.model_name = model
        self._timeout_seconds = timeout_seconds
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        }
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
            transport=transport if transport is not None else httpx.AsyncHTTPTransport(retries=0),
        )

    async def decide(self, request: DecisionRequest) -> DecisionReply:
        if self._client.is_closed:
            raise _error("backend_unavailable", "The decision backend is closed.")
        request = _validated_request(request)
        tools, callable_names = _tools_for(request)
        try:
            payload = json.dumps(
                {
                    "model": self._model,
                    "messages": build_messages(request),
                    "tools": tools,
                    "tool_choice": "required",
                    "parallel_tool_calls": False,
                    "max_completion_tokens": MAX_COMPLETION_TOKENS,
                    "n": 1,
                    "stream": False,
                    "store": False,
                },
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except Exception:
            raise _error(
                "backend_invalid_request", "The decision state cannot be encoded."
            ) from None
        if len(payload) > MAX_CONTEXT_BYTES:
            raise _error(
                "backend_context_too_large", "The decision context exceeds its size limit."
            )
        try:
            # httpx has per-operation timeouts; this also bounds the entire
            # response, including a provider that keeps sending tiny chunks.
            response_bytes = await asyncio.wait_for(
                self._post_bounded(payload), timeout=self._timeout_seconds
            )
        except SkillError:
            raise
        except (asyncio.TimeoutError, httpx.TimeoutException):
            raise _error(
                "backend_timeout", "The decision provider exceeded its time limit."
            ) from None
        except Exception:
            raise _error(
                "backend_transport_error", "The decision provider request failed."
            ) from None
        return self._parse_reply(response_bytes, callable_names)

    async def _post_bounded(self, payload: bytes) -> bytes:
        async with self._client.stream(
            "POST", self._endpoint, headers=self._headers, content=payload
        ) as response:
            status = response.status_code
            if 300 <= status < 400:
                raise _error("backend_redirect_rejected", "Provider redirects are disabled.")
            if status in {401, 403}:
                raise _error(
                    "backend_authentication_failed", "The provider rejected authentication."
                )
            if status == 429:
                raise _error("backend_rate_limited", "The provider rate limit was reached.")
            if status >= 500:
                raise _error("backend_unavailable", "The decision provider is unavailable.")
            if status != 200:
                raise _error(
                    "backend_request_failed", "The provider rejected the decision request."
                )
            encoding = response.headers.get("content-encoding", "identity").lower()
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if encoding not in {"", "identity"} or (
                content_type != "application/json" and not content_type.endswith("+json")
            ):
                raise _error(
                    "backend_invalid_response", "The provider response encoding is invalid."
                )
            length = response.headers.get("content-length")
            if length is not None:
                try:
                    declared_length = int(length)
                    if declared_length < 0:
                        raise ValueError
                except ValueError:
                    raise _error(
                        "backend_invalid_response", "The provider response length is invalid."
                    ) from None
                if declared_length > MAX_RESPONSE_BYTES:
                    raise _error(
                        "backend_response_too_large",
                        "The provider response exceeds its size limit.",
                    )
            body = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=8192):
                if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                    raise _error(
                        "backend_response_too_large",
                        "The provider response exceeds its size limit.",
                    )
                body.extend(chunk)
            return bytes(body)

    @staticmethod
    def _parse_reply(response_bytes: bytes, callable_names: frozenset[str]) -> DecisionReply:
        try:
            response = _strict_json(response_bytes)
            choices = response["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError
            choice = choices[0]
            message = choice["message"]
            if message.get("refusal") not in (None, ""):
                raise _error("backend_refusal", "The decision provider declined the request.")
            if (
                choice.get("finish_reason") != "tool_calls"
                or message.get("role") != "assistant"
                or message.get("content") not in (None, "")
            ):
                raise ValueError
            calls = message["tool_calls"]
            if not isinstance(calls, list) or len(calls) != 1:
                raise ValueError
            call = calls[0]
            if call.get("type") != "function":
                raise ValueError
            function = call["function"]
            name, argument_text = function["name"], function["arguments"]
            if not isinstance(name, str) or name not in callable_names:
                raise _error(
                    "backend_invalid_decision", "The provider selected an unavailable tool."
                )
            if (
                not isinstance(argument_text, str)
                or len(argument_text.encode("utf-8")) > MAX_ARGUMENT_BYTES
            ):
                raise ValueError
            arguments = _strict_json(argument_text)
            if not isinstance(arguments, dict):
                raise ValueError
            if name in _CONTROL_NAMES:
                if "action" in arguments:
                    raise ValueError
                action = {"action": name, **arguments}
            else:
                action = {"action": "call_skill", "skill": name, "arguments": arguments}
            decision = ACTION_ADAPTER.validate_python(action)
            usage = response.get("usage")
            if usage is not None and not isinstance(usage, dict):
                raise ValueError
            # Missing counters remain unavailable, never silently become zero.
            return DecisionReply(
                decision=decision,
                input_tokens=usage.get("prompt_tokens") if usage is not None else None,
                output_tokens=usage.get("completion_tokens") if usage is not None else None,
            )
        except SkillError:
            raise
        except ValidationError:
            raise _error(
                "backend_invalid_decision",
                "The provider decision or usage does not match its schema.",
            ) from None
        except Exception:
            raise _error(
                "backend_invalid_response", "The provider response is not a valid action."
            ) from None

    async def aclose(self) -> None:
        await self._client.aclose()


def backend_from_env(settings: Settings | None = None) -> DecisionBackend:
    """Explicit opt-in only. Scripted decisions remain test/example adapters."""
    try:
        config = settings if settings is not None else Settings.from_env()
    except ValueError:
        raise _error(
            "backend_configuration_error", "The provider configuration is invalid."
        ) from None
    if config.agent_backend == "disabled":
        return DisabledDecisionBackend()
    key = config.llm_api_key.get_secret_value() if config.llm_api_key else ""
    if not config.llm_base_url or not key or not config.llm_model:
        raise _error("backend_configuration_error", "The provider configuration is incomplete.")
    return OpenAICompatibleBackend(
        base_url=config.llm_base_url, api_key=key, model=config.llm_model
    )
