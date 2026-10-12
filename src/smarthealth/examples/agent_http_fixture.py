"""Loopback HTTP/SSE testing only: scripted decisions with real physiological Skills."""

import asyncio

from smarthealth.agents.backends import ScriptedDecisionBackend
from smarthealth.api.application import create_app
from smarthealth.examples.demo_policy import scripted_analysis
from smarthealth.runtime.events import RunEventStore
from smarthealth.runtime.memory.long_term import QdrantLongTermStore
from smarthealth.runtime.memory.memory_manager import MemoryManager


async def slow_scripted_decision(request):
    # Give a real HTTP subscriber time to disconnect while the run is active.
    await asyncio.sleep(0.1)
    return scripted_analysis(request)


app = create_app(
    decision_backend=ScriptedDecisionBackend(slow_scripted_decision),
    agent_event_store=RunEventStore(),
    api_token="local-sse-test",
    memory_manager=MemoryManager(long_term=QdrantLongTermStore()),
)
