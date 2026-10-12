"""Real local E5/Qdrant/LightGBM with explicitly scripted multi-Agent decisions."""

import argparse
import asyncio
import json
from pathlib import Path

from app.agents.backends import ScriptedDecisionBackend
from app.agents.contracts import CreateRunRequest
from app.agents.events import RunEventStore
from app.agents.router import HealthRouter
from app.agents.runtime import AgentRuntime
from app.core.artifacts import ArtifactStore
from app.core.context import SkillContext
from app.harness.config import load_config
from app.harness.harness_validator import HarnessValidator
from app.memory.long_term import QdrantLongTermStore
from app.memory.memory_manager import MemoryManager
from app.models.model_loader import load_model
from app.rag.embeddings import E5ONNXEmbedder
from app.rag.retrieval import RetrievalService
from app.reports.service import ReportService
from app.services.analysis_service import AnalysisService
from examples.report_policy import scripted_report
from skills.bootstrap import create_registry


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    harness = HarnessValidator(load_config("configs/report"))
    store = ArtifactStore()
    predictor = load_model("models_saved/lightgbm_v1.0")
    retrieval = RetrievalService(args.index, E5ONNXEmbedder(args.model_dir))
    service = AnalysisService(
        store,
        predictor,
        retrieval=retrieval,
        reports=ReportService(store, predictor, validator=harness),
    )
    memory = MemoryManager(long_term=QdrantLongTermStore())
    registry = create_registry(service, harness=harness)
    runtime = AgentRuntime(
        service=service,
        registry=registry,
        backend=ScriptedDecisionBackend(scripted_report),
        events=RunEventStore(),
        router=HealthRouter(),
        memory=memory,
        harness=harness,
    )
    context = SkillContext(owner_id="synthetic_demo", session_id="report_demo", input_version="v1")
    try:
        ref = service.create_upload(Path("data/sample/demo.csv").read_bytes(), context)
        initial = await runtime.create(
            CreateRunRequest(
                query="分析合成CSV并生成有来源的健康分析报告和图表",
                session_id=context.session_id,
                input_version=context.input_version,
                upload_refs=[ref],
                include_visualizations=True,
            ),
            owner_id=context.owner_id,
            request_id="report-demo",
        )
        result = await runtime.wait(
            initial.run_id, owner_id=context.owner_id, session_id=context.session_id
        )
        events = runtime.get_events(
            result.run_id, owner_id=context.owner_id, session_id=context.session_id
        )
        (args.output / "run.json").write_text(result.model_dump_json(indent=2))
        (args.output / "trace.json").write_text(
            json.dumps([e.model_dump(mode="json") for e in events], indent=2)
        )
        if result.status != "completed":
            raise RuntimeError(f"Report demo failed: {result.status}/{result.error_code}")
        (args.output / "report.txt").write_text(result.final_result["report"]["body"])
        probe = retrieval.retrieve("EDA 皮电与活动背景")
        (args.output / "retrieval.json").write_text(probe.model_dump_json(indent=2))
        return {
            "status": result.status,
            "decisions": result.decision_count,
            "tool_calls": result.tool_calls,
            "delegations": result.delegation_count,
            "external_llm_calls": result.llm_requests,
            "embedding_id": probe.embedding_id,
            "review_mode": result.final_result["review_mode"],
            "memory_status": result.memory_status,
            "output": str(args.output),
        }
    finally:
        await runtime.aclose()
        await registry.aclose()
        retrieval.close()
        await memory.aclose()
        store.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/knowledge/v1"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/models/multilingual-e5-small"))
    parser.add_argument("--output", type=Path, default=Path("data/generated/report_example"))
    print(json.dumps(asyncio.run(run(parser.parse_args())), indent=2))
