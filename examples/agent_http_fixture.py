"""Loopback HTTP/SSE testing only: scripted decisions with real physiological Skills."""

import asyncio

from app.agents.backends import ScriptedDecisionBackend
from app.agents.events import RunEventStore
from app.main import create_app
from app.memory.long_term import QdrantLongTermStore
from app.memory.memory_manager import MemoryManager
from examples.demo_policy import scripted_analysis


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
