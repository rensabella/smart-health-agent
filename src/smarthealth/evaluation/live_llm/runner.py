"""Run reviewed synthetic tasks through the production AgentRuntime entry points."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from smarthealth.agents import prompts
from smarthealth.agents.backends import OpenAICompatibleBackend
from smarthealth.agents.contracts import CreateRunRequest
from smarthealth.agents.executor import AgentRuntime
from smarthealth.agents.router import HealthRouter
from smarthealth.cli.generate_demo_data import GENERATOR_VERSION, generate_dataset
from smarthealth.config.resources import resource_path
from smarthealth.config.settings import Settings
from smarthealth.evaluation.live_llm.budget import EvaluationBudget
from smarthealth.evaluation.live_llm.config import EvalConfig, load_config
from smarthealth.evaluation.live_llm.fault_injection import FaultInjector
from smarthealth.evaluation.live_llm.graders import grade_task
from smarthealth.evaluation.live_llm.metrics import aggregate
from smarthealth.evaluation.live_llm.report import render_markdown
from smarthealth.evaluation.live_llm.tasks import BenchmarkTask, load_tasks
from smarthealth.evaluation.live_llm.trace_collector import (
    ArtifactCollector,
    EvaluationTransport,
    ObservedBackend,
    SkillCollector,
    utcnow,
)
from smarthealth.models.model_loader import load_model
from smarthealth.pipeline.inference import analyze_csv
from smarthealth.reports.service import ReportService
from smarthealth.retrieval.corpus import load_corpus
from smarthealth.retrieval.embeddings import E5ONNXEmbedder
from smarthealth.retrieval.retrieval import RetrievalService
from smarthealth.retrieval.schemas import canonical_hash
from smarthealth.runtime.context import SkillContext
from smarthealth.runtime.errors import SkillError
from smarthealth.runtime.events import RunEventStore
from smarthealth.runtime.harness.config import load_config as load_harness
from smarthealth.runtime.harness.harness_validator import HarnessValidator
from smarthealth.runtime.limits import AgentLimits
from smarthealth.runtime.memory.long_term import QdrantLongTermStore
from smarthealth.runtime.memory.memory_manager import MemoryManager
from smarthealth.runtime.trace import InMemoryTraceSink
from smarthealth.services.analysis_service import AnalysisService
from smarthealth.skills.bootstrap import create_registry

INFRASTRUCTURE_ERRORS = frozenset(
    {
        "backend_timeout",
        "llm_timeout",
        "backend_transport_error",
        "backend_rate_limited",
        "backend_unavailable",
        "backend_authentication_failed",
        "backend_request_failed",
        "backend_configuration_error",
        "backend_redirect_rejected",
        "model_unavailable",
        "rag_unavailable",
        "rag_index_invalid",
        "rag_embedding_unavailable",
        "trace_unavailable",
        "internal_error",
        "evaluation_internal_error",
        "evaluation_configuration_error",
    }
)


def synthetic_input(seed: int) -> bytes:
    frame = generate_dataset(subjects=1, sessions=1, windows=10, seed=seed).drop(columns="label")
    frame["subject_id"], frame["session_id"] = "synthetic_eval", "synthetic_session"
    return frame.to_csv(index=False, float_format="%.8f").encode("utf-8")


def task_record(task, repetition, mode):
    return {
        "schema_version": "1.0",
        "task_id": task.task_id,
        "category": task.category,
        "repetition": repetition,
        "mode": mode,
        "status": "dry_run" if mode == "dry_run" else "executed",
        "task_hash": canonical_hash(task.model_dump(mode="json")),
        "snapshot": None,
        "events": [],
        "registry_events": [],
        "skill_calls": [],
        "provider_attempts": [],
        "decision_actions": [],
        "artifacts": [],
        "fault": {
            "kind": task.fault,
            "target_skill": task.fault_target,
            "triggered": False,
            "events": [],
        },
        "latency_ms": None,
        "reference_prediction": None,
        "max_execution_steps": task.max_execution_steps,
        "max_tool_calls": task.max_tool_calls,
    }


def new_backend(config, settings, transport):
    if settings.agent_backend != "openai_compatible" or not settings.llm_api_key:
        raise SkillError(
            "backend_configuration_error", "Explicit live provider configuration required."
        )
    return OpenAICompatibleBackend(
        settings.llm_base_url,
        settings.llm_api_key.get_secret_value(),
        config.model or settings.llm_model,
        timeout_seconds=config.decision_timeout_seconds,
        transport=transport,
        temperature=config.temperature,
        max_completion_tokens=config.max_completion_tokens,
        seed=config.seed,
    )


async def execute_task(
    task: BenchmarkTask,
    *,
    repetition: int,
    config: EvalConfig,
    settings: Settings,
    predictor,
    retrieval,
    content: bytes,
    budget: EvaluationBudget,
    reference_prediction=None,
    mode="live",
    make_backend=None,
    artifact_directory: Path | None = None,
) -> dict:
    """Test injection is explicit and cannot be labeled a live run.

    The CLI never accepts a backend factory. Infrastructure tests may supply one,
    but must use mode=infrastructure_test, which remains in every result record.
    """
    if mode not in {"live", "infrastructure_test"} or (
        make_backend and mode != "infrastructure_test"
    ):
        raise ValueError("Custom backends are limited to infrastructure tests")
    record = task_record(task, repetition, mode)
    if task.requires_rag and retrieval is None:
        record.update(status="skipped", skip_reason="real_rag_assets_unavailable")
        record["grades"] = grade_task(task, record)
        return record
    store = ArtifactCollector()
    runtime = registry = memory = backend = None
    transport = None
    initial = None
    started = time.perf_counter()
    injector = FaultInjector(
        task.fault if not task.fault.startswith("backend_") else "none", task.fault_target
    )
    context = SkillContext(
        owner_id="synthetic_live_evaluation",
        session_id=f"{task.task_id}_{repetition}",
        input_version="synthetic_v1",
    )
    trace_sink = InMemoryTraceSink()
    try:
        base = load_harness(
            resource_path("configs", "report") if task.requires_rag else resource_path("configs")
        )
        limits = AgentLimits(
            max_decisions=task.max_execution_steps,
            max_llm_requests=task.max_execution_steps,
            max_tool_calls=task.max_tool_calls,
            max_total_tokens=config.per_task_token_budget,
            run_timeout_seconds=task.timeout_seconds,
            decision_timeout_seconds=config.decision_timeout_seconds,
        )
        harness = HarnessValidator(base, limits=limits)
        service = AnalysisService(
            store,
            predictor,
            retrieval=injector.wrap_retrieval(retrieval),
            reports=ReportService(store, predictor, validator=harness),
        )
        original = create_registry(service, harness=harness, trace_sink=trace_sink)
        registry = SkillCollector(injector.wrap_registry(original), injector)
        if make_backend:
            inner = make_backend()
        else:
            transport = EvaluationTransport(task.fault)
            inner = new_backend(config, settings, transport)
        backend = ObservedBackend(inner, budget, original, transport)
        memory = MemoryManager(long_term=QdrantLongTermStore())
        runtime = AgentRuntime(
            service=service,
            registry=registry,
            backend=backend,
            events=RunEventStore(),
            router=HealthRouter(),
            memory=memory,
            harness=harness,
        )
        uploads = []
        if task.fixture != "none":
            value = content if task.fixture == "synthetic_csv" else b"unexpected_column\ninvalid\n"
            uploads.append(service.create_upload(value, context))
        request = CreateRunRequest(
            query=task.user_request,
            session_id=context.session_id,
            input_version=context.input_version,
            upload_refs=uploads,
            include_visualizations=task.include_visualizations,
        )
        started = time.perf_counter()
        # A single creation per repetition. No workflow replay after tool side effects.
        initial = await runtime.create(
            request,
            owner_id=context.owner_id,
            request_id=f"eval_{context.session_id}",
            idempotency_key=context.session_id,
        )
        try:
            final = await asyncio.wait_for(
                runtime.wait(
                    initial.run_id, owner_id=context.owner_id, session_id=context.session_id
                ),
                timeout=task.timeout_seconds + 1,
            )
        except asyncio.TimeoutError:
            runtime.cancel(initial.run_id, owner_id=context.owner_id, session_id=context.session_id)
            final = await runtime.wait(
                initial.run_id, owner_id=context.owner_id, session_id=context.session_id
            )
            record["runner_timeout"] = True
        record["snapshot"] = final.model_dump(mode="json")
        record["events"] = [
            e.model_dump(mode="json")
            for e in runtime.get_events(
                final.run_id, owner_id=context.owner_id, session_id=context.session_id
            )
        ]
        record["error_code"] = final.error_code
        if final.error_code in INFRASTRUCTURE_ERRORS:
            record["status"] = "infrastructure_error"
        elif final.error_code and final.error_code.startswith("evaluation_"):
            record.update(status="skipped", skip_reason=final.error_code)
        record["reference_prediction"] = reference_prediction
        record["effective_limits"] = runtime.limits.model_dump(mode="json")
        if artifact_directory is not None and final.status == "completed" and final.final_result:
            plots = final.final_result.get("analysis", {}).get("visualization", [])
            record["artifact_files"] = store.export_images(
                artifact_directory, [p["artifact_ref"] for p in plots], context
            )
    except SkillError as exc:
        record.update(status="infrastructure_error", error_code=exc.code)
    except asyncio.CancelledError:
        record.update(status="infrastructure_error", error_code="evaluation_interrupted")
    except Exception:
        # Exceptions can include paths, provider text and secrets. Stable code only.
        record.update(status="infrastructure_error", error_code="evaluation_internal_error")
    finally:
        record["latency_ms"] = (time.perf_counter() - started) * 1000

        async def close_safely(resource):
            try:
                await resource.aclose()
            except Exception:
                record.update(status="infrastructure_error", error_code="evaluation_cleanup_failed")

        if runtime:
            await close_safely(runtime)
        elif backend:
            await close_safely(backend)
        elif transport:
            await close_safely(transport)
        if registry:
            await close_safely(
                registry
            )  # Drain actual workers before exporting/clearing artifacts.
            record["skill_calls"] = registry.calls
        if memory:
            await close_safely(memory)
        if backend:
            record["provider_attempts"], record["decision_actions"] = (
                backend.attempts,
                backend.actions,
            )
        if initial:
            from dataclasses import replace

            record["registry_events"] = trace_sink.events_for(
                replace(context, run_id=initial.run_id)
            )
            # Read-only evaluation seam; production state and retention are unchanged.
            record["execution_spans"] = runtime._runs[initial.run_id].execution_trace.snapshot()
            if record["snapshot"] is None:
                record["snapshot"] = runtime.snapshot(
                    initial.run_id, owner_id=context.owner_id, session_id=context.session_id
                ).model_dump(mode="json")
                record["events"] = [
                    e.model_dump(mode="json")
                    for e in runtime.get_events(
                        initial.run_id, owner_id=context.owner_id, session_id=context.session_id
                    )
                ]
        record["artifacts"] = store.export()
        if transport and task.fault.startswith("backend_"):
            record["fault"].update(triggered=transport.triggered, events=transport.events)
        else:
            record["fault"].update(triggered=injector.triggered, events=injector.events)
        store.clear()
    record["grades"] = grade_task(task, record)
    return record


def select_tasks(suite, ids, max_tasks):
    if ids:
        if len(ids) != len(set(ids)) or set(ids) - {t.task_id for t in suite.tasks}:
            raise ValueError("Task IDs must be unique members of the packaged benchmark")
        lookup = {t.task_id: t for t in suite.tasks}
        return [lookup[i] for i in ids][:max_tasks]
    return suite.tasks[:max_tasks]


def fingerprint_file(path):
    return (
        hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if path and Path(path).is_file()
        else None
    )


def metadata(config, suite, suite_hash, selected, mode):
    return {
        "schema_version": "1.0",
        "mode": mode,
        "evaluation_date": utcnow(),
        "benchmark_version": suite.version,
        "benchmark_hash": suite_hash,
        "benchmark_task_count": len(suite.tasks),
        "selected_task_ids": [t.task_id for t in selected],
        "config": config.model_dump(mode="json"),
        "config_hash": canonical_hash(config.model_dump(mode="json")),
        "prompt_hash": canonical_hash([prompts.SYSTEM_PROMPT, prompts.REPORT_ROLE_PROMPTS]),
        "generator_version": GENERATOR_VERSION,
        "fixture_seed": config.fixture_seed,
        "python": platform.python_version(),
        "platform": platform.system(),
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in (
                "smart-health-agent",
                "httpx",
                "pydantic",
                "numpy",
                "scikit-learn",
                "qdrant-client",
            )
        },
        "comparison": {
            "available": False,
            "reason": "Guard-off and retry-only modes would change integral authorization and publication semantics.",
        },
        "limitations": [
            "Only synthetic fixtures; no clinical or production-quality claim.",
            "Router-only outcomes are not live LLM evidence; mocks remain infrastructure_test.",
            "Citation identity checks do not establish semantic entailment; no LLM judge used.",
            "No independently labeled conflicting-evidence cases or standalone single-Skill final objective.",
            "Provider seed/temperature support varies; a seed does not guarantee identical outputs.",
            "Conservative byte-based token reservation and operator prices are not billing guarantees.",
            "Whole tasks are never retried. Runtime retries transient decisions at most once.",
            "Duplicate-tool injection exposes actual repeated effects; no exactly-once guarantee.",
            "Provider is nonstreaming; first-token latency is unavailable.",
            "Latency measures in-process runtime tasks, not HTTP/network deployment throughput.",
        ],
    }


class ResultWriter:
    """New output directory per experiment; completed rows survive interrupted runs."""

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.directory = directory
        self.stream = (directory / "results.jsonl").open("x", encoding="utf-8")
        (directory / "results.jsonl").chmod(0o600)

    def append(self, record):
        self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def report(self, report):
        for name, content in (
            (
                "summary.json",
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            ),
            ("report.md", render_markdown(report)),
        ):
            path = self.directory / name
            temp = self.directory / (name + ".tmp")
            with temp.open("w", encoding="utf-8") as stream:
                stream.write(content)
            temp.chmod(0o600)
            temp.replace(path)

    def close(self):
        self.stream.close()


async def run(args):
    config = load_config(args.config)
    overrides = {
        key: getattr(args, key)
        for key in ("max_tasks", "repetitions", "concurrency", "model")
        if getattr(args, key) is not None
    }
    config = EvalConfig.model_validate({**config.model_dump(), **overrides})
    suite, suite_hash = load_tasks()
    tasks = select_tasks(suite, args.task, config.max_tasks)
    mode = "live" if args.live else "dry_run"
    # Dry-run never reads credentials, instantiates HTTP clients or loads model/index assets.
    settings = Settings.from_env() if args.live else None
    if args.live:
        config = EvalConfig.model_validate(
            {**config.model_dump(), "model": config.model or settings.llm_model}
        )
    report = metadata(config, suite, suite_hash, tasks, mode)
    report["records"] = []
    budget = EvaluationBudget(config)
    writer = ResultWriter(args.output)
    predictor = retrieval = None
    start = time.perf_counter()
    exit_code = 0

    def save(record):
        report["records"].append(record)
        writer.append(record)
        report.update(
            metrics=aggregate(report["records"]),
            budget=budget.snapshot(),
            elapsed_seconds=time.perf_counter() - start,
        )
        writer.report(report)

    try:
        if not args.live:
            for repetition in range(1, config.repetitions + 1):
                for task in tasks:
                    record = task_record(task, repetition, "dry_run")
                    record.update(
                        skip_reason="dry_run_no_execution",
                        expected=task.expected.model_dump(),
                        requires_rag=task.requires_rag,
                    )
                    record["grades"] = grade_task(task, record)
                    save(record)
            report["live_verification"] = "not_executed"
        else:
            # Validate provider configuration without a request, before preparing assets.
            probe_transport = EvaluationTransport()
            try:
                probe = new_backend(config, settings, probe_transport)
                await probe.aclose()
            finally:
                await probe_transport.aclose()
            report["provider"] = {
                "interface": config.provider,
                "model": config.model,
                "endpoint_host": urlsplit(settings.llm_base_url).hostname,
            }
            predictor = load_model(settings.model_dir)
            report["model_manifest_hash"] = fingerprint_file(settings.model_dir / "manifest.json")
            report["corpus_hash"] = canonical_hash(
                load_corpus(resource_path("knowledge", "corpus.json")).model_dump(mode="json")
            )
            if (
                any(t.requires_rag for t in tasks)
                and settings.rag_model_dir
                and settings.rag_index_dir
            ):
                retrieval = RetrievalService(
                    settings.rag_index_dir,
                    E5ONNXEmbedder(settings.rag_model_dir),
                    min_score=settings.rag_min_score,
                )
                report["retrieval_manifest"] = retrieval.manifest.model_dump(mode="json")
                report["retrieval_min_score"] = settings.rag_min_score
                if retrieval.manifest.corpus_hash != report["corpus_hash"]:
                    raise SkillError(
                        "evaluation_corpus_mismatch",
                        "Index must match the reviewed benchmark corpus.",
                    )
            content = synthetic_input(config.fixture_seed)
            report["fixture_sha256"] = hashlib.sha256(content).hexdigest()
            raw = analyze_csv(content, predictor)["prediction"]
            reference = {k: raw[k] for k in ("risk_score", "prediction", "window_count")}
            semaphore = asyncio.Semaphore(config.concurrency)

            async def one(task, repetition):
                async with semaphore:
                    result = await execute_task(
                        task,
                        repetition=repetition,
                        config=config,
                        settings=settings,
                        predictor=predictor,
                        retrieval=retrieval,
                        content=content,
                        budget=budget,
                        reference_prediction=reference,
                        artifact_directory=writer.directory
                        / "artifacts"
                        / f"{task.task_id}_{repetition}",
                    )
                    save(result)

            pending = [
                asyncio.create_task(one(task, rep))
                for rep in range(1, config.repetitions + 1)
                for task in tasks
            ]
            try:
                await asyncio.gather(*pending)
            finally:
                for job in pending:
                    if not job.done():
                        job.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            live_rows = [
                r
                for r in report["records"]
                if any(
                    p["network_dispatched"] and p["status"] == "success"
                    for p in r["provider_attempts"]
                )
            ]
            report["live_verification"] = (
                "valid_provider_responses_recorded"
                if live_rows
                else "no_verified_provider_responses"
            )
            if any(
                r["status"] != "executed" or not r["grades"].get("success")
                for r in report["records"]
            ):
                exit_code = 1
    except (SkillError, ValueError, OSError):
        # Failure details may contain secret input; stable classification only.
        report["setup_error"] = "evaluation_setup_failed"
        report["live_verification"] = "not_executed" if not report["records"] else "incomplete"
        for rep in range(1, config.repetitions + 1):
            for task in tasks:
                if not any(
                    r["task_id"] == task.task_id and r["repetition"] == rep
                    for r in report["records"]
                ):
                    row = task_record(task, rep, mode)
                    row.update(status="infrastructure_error", error_code="evaluation_setup_failed")
                    row["grades"] = grade_task(task, row)
                    save(row)
        exit_code = 2
    finally:
        if retrieval:
            retrieval.close()
        report.update(
            metrics=aggregate(report["records"]),
            budget=budget.snapshot(),
            elapsed_seconds=time.perf_counter() - start,
        )
        writer.report(report)
        writer.close()
    print(
        json.dumps(
            {
                "mode": mode,
                "recorded_tasks": len(report["records"]),
                "provider_requests_admitted": budget.snapshot()["admitted_requests"],
                "output": str(args.output),
                "exit_code": exit_code,
            }
        )
    )
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--live",
        action="store_true",
        help="Explicitly authorize provider calls for this invocation",
    )
    modes.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate tasks/config without running them (default)",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--task", action="append", help="Select a packaged task ID; repeat to select several"
    )
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--model")
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/live_runs") / time.strftime("%Y%m%d-%H%M%S")
    )
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run(args))
    except (ValueError, OSError, SkillError):
        print(
            "Evaluation configuration/output is invalid; no exception details are logged.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
