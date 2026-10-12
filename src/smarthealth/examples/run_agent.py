"""Exercise ReAct state, delegation and real Skills with a clearly scripted decision fixture."""

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


async def run(csv_path: Path, model_dir: Path, output_dir: Path, entry_agent="planner"):
    output_dir.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore()
    service = AnalysisService(store, load_model(model_dir))
    registry = create_registry(service)
    runtime = AgentRuntime(
        registry=registry,
        service=service,
        backend=ScriptedDecisionBackend(scripted_analysis),
        events=RunEventStore(output_dir / "traces"),
        router=HealthRouter(),
    )
    runtime.recover()
    context = SkillContext(owner_id="synthetic_demo", session_id=uuid4().hex)
    upload_ref = service.create_upload(csv_path.read_bytes(), context)
    request = CreateRunRequest(
        query="Analyze the synthetic data and show its window score plot.",
        session_id=context.session_id,
        input_version=context.input_version,
        upload_refs=[upload_ref],
        entry_agent=entry_agent,
        include_visualizations=True,
    )
    try:
        created = await runtime.create(
            request, owner_id=context.owner_id, request_id="scripted-agent-demo"
        )
        result = await runtime.wait(
            created.run_id, owner_id=context.owner_id, session_id=context.session_id
        )
        events = runtime.get_events(
            created.run_id, owner_id=context.owner_id, session_id=context.session_id
        )
        (output_dir / "run.json").write_text(
            result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "trace.json").write_text(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "backend_mode": "scripted",
                    "events": [e.model_dump(mode="json") for e in events],
                },
                indent=2,
            )
            + "\n"
        )
        if result.status != "completed":
            raise RuntimeError(f"Scripted example ended with {result.status}: {result.error_code}")
        for artifact in result.final_result["analysis"]["visualization"]:
            plot = store.get(artifact["artifact_ref"], kind="image", context=context)
            (output_dir / f"{artifact['kind']}.png").write_bytes(plot["content"])
        return {
            "status": result.status,
            "backend_mode": result.backend_mode,
            "llm_requests": result.llm_requests,
            "decisions": result.decision_count,
            "tool_calls": result.tool_calls,
            "delegations": result.delegation_count,
            "risk_score": result.final_result["analysis"]["prediction"]["risk_score"],
            "output_dir": str(output_dir),
        }
    finally:
        await runtime.aclose()
        await registry.aclose()
        store.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/sample/demo.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models_saved/lightgbm_v1.0"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/agent_example"))
    parser.add_argument("--entry-agent", choices=["planner", "data_analysis"], default="planner")
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(run(args.csv, args.model_dir, args.output_dir, args.entry_agent)), indent=2
        )
    )
