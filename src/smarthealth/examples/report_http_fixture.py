"""Loopback verification only; scripted role decisions with real E5/Qdrant/LightGBM.

Set HEALTH_RAG_MODE=local and the explicit model/index directories before starting.
This fixture is never selected by the production backend environment setting.
"""

import asyncio

from smarthealth.agents.backends import ScriptedDecisionBackend
from smarthealth.api.application import create_app
from smarthealth.examples.report_policy import scripted_report
from smarthealth.runtime.events import RunEventStore
from smarthealth.runtime.harness.config import load_config
from smarthealth.runtime.memory.long_term import QdrantLongTermStore
from smarthealth.runtime.memory.memory_manager import MemoryManager


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
