"""Per-mechanism ablation probes; each probe isolates exactly one guard seam."""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from smarthealth.agents.contracts import CreateRunRequest
from smarthealth.agents.executor import AgentRuntime
from smarthealth.agents.router import HealthRouter
from smarthealth.config.resources import resource_path
from smarthealth.config.settings import Settings
from smarthealth.evaluation.cases import load_cases
from smarthealth.evaluation.runner import (
    LiveBudget,
    TimedEvents,
    _service,
    backend_factory,
    evaluate_safety_case,
)
from smarthealth.models.model_loader import load_model
from smarthealth.retrieval.embeddings import E5ONNXEmbedder
from smarthealth.retrieval.files import read_json
from smarthealth.retrieval.retrieval import RetrievalService
from smarthealth.retrieval.schemas import KnowledgeChunk
from smarthealth.runtime.context import SkillContext
from smarthealth.runtime.errors import SkillError
from smarthealth.runtime.harness.config import load_config
from smarthealth.runtime.harness.harness_validator import HarnessValidator
from smarthealth.runtime.memory.long_term import QdrantLongTermStore
from smarthealth.runtime.memory.memory_manager import MemoryManager
from smarthealth.skills.bootstrap import create_registry


def _skill_error(call):
    try:
        call()
    except SkillError as exc:
        return exc.code
    return None


def probe_budget():
    """Harness check_call seams. The executor-side safety reservation is exercised by the suite."""
    probes = {}
    for baseline in ("full_guard", "no_budget"):
        harness = HarnessValidator(baseline=baseline)
        limits = harness.limits
        context = SkillContext(owner_id="ablation_probe", session_id="budget_probe")

        def call(skill, *, used=0, role_used=0):
            return _skill_error(
                lambda: harness.check_call(
                    "data_analysis",
                    skill,
                    simple=False,
                    used=used,
                    role_used=role_used,
                    context=context,
                )
            )

        probes[baseline] = {
            "global_budget_exhausted": call("csv_analysis", used=limits.max_tool_calls),
            "role_budget_exhausted": call(
                "csv_analysis", role_used=harness.role_tool_limit("data_analysis")
            ),
            "permission_denied_still_enforced": call("unrelated_skill"),
        }
    expected = {
        "full_guard": {
            "global_budget_exhausted": "budget_exceeded",
            "role_budget_exhausted": "budget_exceeded",
            "permission_denied_still_enforced": "permission_denied",
        },
        "no_budget": {
            "global_budget_exhausted": None,
            "role_budget_exhausted": None,
            "permission_denied_still_enforced": "permission_denied",
        },
    }
    return {
        "seam": "harness.check_call budgets; executor safety reservation is suite-level",
        "results": probes,
        "mechanism_observed": probes == expected,
    }


async def probe_review_gate(suite, predictor, chunks, content, make_backend):
    """Publishing boundary under full_guard vs guard_off on non-fabricated safety cases."""
    rows = []
    for baseline in ("full_guard", "guard_off"):
        for case in (c for c in suite.safety_cases if c.fabricated_evidence_id is None):
            result = await evaluate_safety_case(
                case, predictor, chunks, content, make_backend, baseline
            )
            rows.append(
                {
                    "case_id": case.id,
                    "baseline": baseline,
                    "expected_publish": case.expected.should_publish,
                    "published": result["published"],
                    "correct": result["correct"],
                    "review_enforced": result.get("review_enforced"),
                    "error_code": result["error_code"],
                }
            )
    pairs = {}
    for row in rows:
        pairs.setdefault(row["case_id"], {})[row["baseline"]] = row
    flipped, residual_published = [], []
    for case_id, pair in pairs.items():
        full, open_gate = pair["full_guard"], pair["guard_off"]
        if full["published"] != open_gate["published"]:
            flipped.append(case_id)
        elif open_gate["published"] and not open_gate["expected_publish"]:
            residual_published.append(case_id)
    return {
        "seam": "report publication verdict enforcement",
        "cases": rows,
        "flipped_to_published": sorted(flipped),
        "residual_published": sorted(residual_published),
        "mechanism_observed": bool(flipped),
    }


def _publication_code(harness, payload, objective, context, service):
    return _skill_error(
        lambda: harness.validate_publication(
            {"kind": "data_analysis", "analysis": payload},
            objective=objective,
            context=context,
            service=service,
        )
    )


async def _lineage_flow(harness, case, predictor, retrieval, content, make_backend):
    """Run the case to completion, then attempt publication from another input version."""
    service = _service(predictor, retrieval if case.retrieval_enabled else None, harness)
    registry = create_registry(service, harness=harness)
    memory = MemoryManager(long_term=QdrantLongTermStore())
    runtime = AgentRuntime(
        service=service,
        registry=registry,
        backend=make_backend(),
        events=TimedEvents(),
        router=HealthRouter(),
        memory=memory,
        harness=harness,
    )
    context = SkillContext(owner_id="synthetic_evaluation", session_id=case.id, input_version="v1")
    try:
        if case.input != "none":
            data = content if case.input == "demo_csv" else b"unexpected_column\ninvalid\n"
            upload_refs = [service.create_upload(data, context)]
        else:
            upload_refs = []
        request = CreateRunRequest(
            query=case.query,
            session_id=case.id,
            input_version="v1",
            upload_refs=upload_refs,
            include_visualizations=case.include_visualizations,
        )
        initial = await runtime.create(
            request, owner_id=context.owner_id, request_id=f"probe-{case.id}"
        )
        final = await runtime.wait(initial.run_id, owner_id=context.owner_id, session_id=case.id)
        payload = (final.final_result or {}).get("analysis")
        if final.status != "completed" or not payload:
            raise SkillError("evaluation_fixture_failed", "Probe run did not complete.")
        stale = SkillContext(owner_id=context.owner_id, session_id=case.id, input_version="v2")
        return _publication_code(harness, payload, "data_analysis", stale, service)
    finally:
        await runtime.aclose()
        await registry.aclose()
        await memory.aclose()
        service.store.clear()


async def probe_lineage(case, predictor, retrieval, content, make_backend):
    """Identical inputs under both baselines; only no_lineage republishes across versions."""
    config = load_config("configs/report" if case.harness_profile == "report" else "configs")
    results = {}
    for baseline in ("full_guard", "no_lineage"):
        harness = HarnessValidator(config, baseline=baseline)
        results[baseline] = {
            "stale_publication_code": await _lineage_flow(
                harness, case, predictor, retrieval, content, make_backend
            ),
        }
    return {
        "seam": "input-version lineage: artifact-store binding and publication boundary",
        "case_id": case.id,
        "results": results,
        "mechanism_observed": (
            results["full_guard"]["stale_publication_code"] == "stale_input"
            and results["no_lineage"]["stale_publication_code"] is None
        ),
    }


async def _run(args):
    meter = LiveBudget(limit=args.max_live_requests)
    make_backend = backend_factory(args.backend, meter)
    probe = make_backend()  # Validate backend configuration before loading local assets.
    await probe.aclose()
    suite = load_cases(args.cases, root=args.corpus_root)
    predictor = load_model(args.model)
    retrieval = RetrievalService(
        args.index, E5ONNXEmbedder(args.embedding_model), min_score=args.min_score
    )
    try:
        chunks = tuple(
            KnowledgeChunk.model_validate(c, strict=True)
            for c in read_json(args.index / "chunks.json", max_bytes=2 * 1024 * 1024)
        )
        lineage_case = next(c for c in suite.system_cases if c.id == args.lineage_case)
        content = args.csv.read_bytes()
        probes = {
            "budget": probe_budget(),
            "review_gate": await probe_review_gate(suite, predictor, chunks, content, make_backend),
            "lineage": await probe_lineage(
                lineage_case, predictor, retrieval, content, make_backend
            ),
        }
        return {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": args.backend,
            "probes": probes,
            "external_llm_requests": meter.used,
            "max_live_requests": meter.limit,
            "all_mechanisms_observed": all(p["mechanism_observed"] for p in probes.values()),
        }
    finally:
        retrieval.close()


def main():
    config = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=resource_path())
    parser.add_argument("--csv", type=Path, default=resource_path("data", "sample", "demo.csv"))
    parser.add_argument("--backend", choices=["scripted", "live"], default="scripted")
    parser.add_argument(
        "--cases", type=Path, default=resource_path("evaluation", "test_cases.json")
    )
    parser.add_argument("--model", type=Path, default=config.model_dir)
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=config.rag_model_dir or config.home / "data/models/multilingual-e5-small",
    )
    parser.add_argument(
        "--index", type=Path, default=config.rag_index_dir or config.home / "data/knowledge/v1"
    )
    parser.add_argument("--min-score", type=float, default=config.rag_min_score)
    parser.add_argument("--max-live-requests", type=int, default=128)
    parser.add_argument("--lineage-case", default="analysis_csv")
    parser.add_argument("--output", type=Path, default=Path("evaluation/ablation/probes.json"))
    args = parser.parse_args()
    try:
        report = asyncio.run(_run(args))
    except (SkillError, ValueError, OSError) as exc:
        code = exc.code if isinstance(exc, SkillError) else "evaluation_configuration_error"
        print(f"Ablation probes could not complete: {code}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "all_mechanisms_observed": report["all_mechanisms_observed"],
                "probes": {
                    name: data["mechanism_observed"] for name, data in report["probes"].items()
                },
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if report["all_mechanisms_observed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
