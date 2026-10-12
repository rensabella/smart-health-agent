"""Real local Qdrant + LightGBM; scripted decisions, synthetic input, no external services."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from uuid import uuid4

from smarthealth.agents.backends import ScriptedDecisionBackend
from smarthealth.agents.contracts import CreateRunRequest
from smarthealth.agents.executor import AgentRuntime
from smarthealth.agents.router import HealthRouter
from smarthealth.examples.demo_policy import scripted_analysis
from smarthealth.models.model_loader import load_model
from smarthealth.runtime.artifacts import ArtifactStore
from smarthealth.runtime.context import SkillContext
from smarthealth.runtime.events import RunEventStore
from smarthealth.runtime.memory.long_term import QdrantLongTermStore
from smarthealth.runtime.memory.memory_manager import MemoryManager
from smarthealth.runtime.memory.schemas import Preferences
from smarthealth.services.analysis_service import AnalysisService
from smarthealth.skills.bootstrap import create_registry


async def run(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("data/generated").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="memory-example-", dir="data/generated") as temporary:
        database = Path(temporary) / "qdrant"
        memory = MemoryManager(long_term=QdrantLongTermStore(path=database))
        service = AnalysisService(ArtifactStore(), load_model(Path("models_saved/lightgbm_v1.0")))
        registry = create_registry(service)
        contexts = []

        def policy(state):
            if state.objective == "score_definition":
                contexts.append(state.memory.model_dump(mode="json"))
            return scripted_analysis(state)

        runtime = AgentRuntime(
            service=service,
            registry=registry,
            backend=ScriptedDecisionBackend(policy),
            events=RunEventStore(),
            router=HealthRouter(),
            memory=memory,
        )
        owner, session = "synthetic_demo", uuid4().hex
        results = []
        try:
            await memory.set_preferences(owner, Preferences(language="en", detail="brief"))
            for _ in range(2):
                context = SkillContext(owner_id=owner, session_id=session)
                upload = service.create_upload(Path("data/sample/demo.csv").read_bytes(), context)
                memory.short_term.attach_upload(owner, session, context.input_version, upload)
                created = await runtime.create(
                    CreateRunRequest(
                        query="分析CSV数据",
                        session_id=session,
                        input_version=context.input_version,
                        upload_refs=[upload],
                    ),
                    owner_id=owner,
                    request_id="memory-example-analysis",
                )
                result = await runtime.wait(created.run_id, owner_id=owner, session_id=session)
                if result.status != "completed" or result.memory_status != "stored":
                    raise RuntimeError("Memory example analysis did not complete and persist")
                results.append(
                    {
                        "run_id": result.run_id,
                        "status": result.status,
                        "memory_status": result.memory_status,
                        "tool_calls": result.tool_calls,
                        "external_llm_requests": result.llm_requests,
                    }
                )
            created = await runtime.create(
                CreateRunRequest(
                    query="解释risk score",
                    session_id=session,
                    input_version=context.input_version,
                ),
                owner_id=owner,
                request_id="memory-example-definition",
            )
            await runtime.wait(created.run_id, owner_id=owner, session_id=session)
            short = memory.short_term.get(owner, session)
            trend = await memory.trends(owner)
            before = memory.long_term.list_analysis(owner)
            events = [
                event.model_dump(mode="json")
                for event in runtime.get_events(created.run_id, owner_id=owner, session_id=session)
            ]
        finally:
            await runtime.aclose()
            await registry.aclose()
            await memory.aclose()
            service.store.clear()
        reopened = MemoryManager(long_term=QdrantLongTermStore(path=database))
        try:
            assert reopened.long_term.list_analysis(owner) == before
            assert await reopened.preferences(owner) == Preferences(language="en", detail="brief")
            assert reopened.short_term.get(owner, session) is None
            assert reopened.long_term.list_analysis("different_owner") == []
            restored_count = len(reopened.long_term.list_analysis(owner))
            await reopened.delete_owner(owner)
            assert reopened.long_term.list_analysis(owner) == []
            assert await reopened.preferences(owner) is None
        finally:
            await reopened.aclose()
        evidence = {
            "mode": "scripted_decisions_real_lightgbm_real_local_qdrant",
            "fixture": "Two new analyses use the same synthetic CSV; these scores are not a clinical trend.",
            "runs": results,
            "session": short.model_dump(mode="json"),
            "definition_memory_context": contexts[-1],
            "trends": trend,
            "long_term_entries": [entry.model_dump(mode="json") for entry in before],
            "restart_restored_entries": restored_count,
            "short_term_cleared_on_restart": True,
            "cross_owner_empty": True,
            "delete_verified": True,
            "definition_trace": events,
        }
        (output_dir / "memory.json").write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False) + "\n"
        )
        return {
            "stored_analyses": len(before),
            "retained_turns": len(short.turns),
            "retrieved_summaries": len(contexts[-1]["historical_summaries"]),
            "restart_restored_entries": restored_count,
            "delete_verified": True,
            "external_llm_requests": 0,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/memory_example"))
    print(json.dumps(asyncio.run(run(parser.parse_args().output_dir)), indent=2))
