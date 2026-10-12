"""Loopback verification only; scripted role decisions with real E5/Qdrant/LightGBM.

Set HEALTH_RAG_MODE=local and the explicit model/index directories before starting.
This fixture is never selected by the production backend environment setting.
"""

import asyncio

from app.agents.backends import ScriptedDecisionBackend
from app.agents.events import RunEventStore
from app.harness.config import load_config
from app.main import create_app
from app.memory.long_term import QdrantLongTermStore
from app.memory.memory_manager import MemoryManager
from examples.report_policy import scripted_report


async def slow_scripted_report(request):
    await asyncio.sleep(0.1)
    return scripted_report(request)


app = create_app(
    decision_backend=ScriptedDecisionBackend(slow_scripted_report),
    agent_event_store=RunEventStore(),
    harness_config=load_config("configs/report"),
    api_token="local-report-test",
    memory_manager=MemoryManager(long_term=QdrantLongTermStore()),
)
