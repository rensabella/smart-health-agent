"""Offline Router + ReAct + real-model example; all decisions are explicitly scripted."""

from __future__ import annotations

import argparse
import asyncio
import json
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
from smarthealth.services.analysis_service import AnalysisService
from smarthealth.skills.bootstrap import create_registry


async def run(model_dir: Path, csv_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    service = AnalysisService(ArtifactStore(), load_model(model_dir))
    registry = create_registry(service)
    runtime = AgentRuntime(
        service=service,
        registry=registry,
        backend=ScriptedDecisionBackend(scripted_analysis),
        events=RunEventStore(),
        router=HealthRouter(),
    )
    context = SkillContext(owner_id="synthetic_demo", session_id=uuid4().hex)
    upload = service.create_upload(csv_path.read_bytes(), context)
    results, traces = {}, {}

    async def execute(name, query, **kwargs):
        body = CreateRunRequest(
            query=query,
            session_id=context.session_id,
            input_version=context.input_version,
            **kwargs,
        )
        preview = runtime.preview_route(body, owner_id=context.owner_id)
        initial = await runtime.create(
            body, owner_id=context.owner_id, request_id=f"router-demo-{name}"
        )
        result = await runtime.wait(
            initial.run_id, owner_id=context.owner_id, session_id=context.session_id
        )
        results[name] = {
            "preview": preview.model_dump(mode="json"),
            "run": result.model_dump(mode="json"),
        }
        traces[name] = [
            e.model_dump(mode="json")
            for e in runtime.get_events(
                result.run_id, owner_id=context.owner_id, session_id=context.session_id
            )
        ]
        return result

    try:
        await execute("definition", "解释risk score")
        analysis = await execute("analysis", "分析CSV数据并绘制图表", upload_refs=[upload])
        if analysis.status != "completed":
            raise RuntimeError(f"Analysis failed: {analysis.error_code}")
        await execute("reuse", "显示上次结果和已有图表", source_run_id=analysis.run_id)
        await execute("health_reasoning", "这个用户为什么风险较高", upload_refs=[upload])
        await execute("missing_upload", "分析CSV数据")
        await execute("ambiguous", "继续")
        for name, data in (("runs.json", results), ("trace.json", traces)):
            (output_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        return {
            name: {
                "workflow_type": item["run"]["route"]["workflow_type"],
                "status": item["run"]["status"],
                "decisions": item["run"]["decision_count"],
                "tool_calls": item["run"]["tool_calls"],
                "llm_requests": item["run"]["llm_requests"],
            }
            for name, item in results.items()
        }
    finally:
        await runtime.aclose()
        await registry.aclose()
        service.store.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models_saved/lightgbm_v1.0"))
    parser.add_argument("--csv", type=Path, default=Path("data/sample/demo.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/router_example"))
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.model_dir, args.csv, args.output_dir)), indent=2))
