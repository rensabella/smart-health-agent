"""Application factory. Models are loaded during lifespan and never per request."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.agents.backends import backend_from_env
from app.agents.events import RunEventStore
from app.agents.router import HealthRouter
from app.agents.runtime import AgentRuntime
from app.api import agent, analysis, health, memory, predict
from app.core.artifacts import ArtifactStore
from app.core.errors import SkillError
from app.core.http_limits import UploadBodyLimitMiddleware
from app.harness.config import load_config
from app.harness.harness_validator import HarnessValidator
from app.memory.memory_manager import memory_from_env
from app.rag.retrieval import load_retrieval_from_env
from app.reports.service import ReportService
from app.services.analysis_service import AnalysisService
from skills.bootstrap import create_registry

logger = logging.getLogger("smart_health")
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models_saved" / "lightgbm_v1.0"


def create_app(
    model_dir=None,
    predictor=None,
    api_token=None,
    store=None,
    model_loader=None,
    decision_backend=None,
    agent_limits=None,
    agent_event_store=None,
    memory_manager=None,
    harness_config=None,
    retrieval_service=None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        # Fail configuration validation before starting models, workers or persistence.
        app.state.harness = HarnessValidator(
            harness_config if harness_config is not None else load_config(),
            limits=agent_limits,
        )
        logger.info(
            json.dumps(
                {
                    "event": "harness.loaded",
                    "harness_version": app.state.harness.version,
                    "harness_hash": app.state.harness.fingerprint,
                }
            )
        )
        app.state.store = store if store is not None else ArtifactStore()
        loaded = predictor
        if loaded is None:
            from app.models.model_loader import load_model

            loader = model_loader or load_model
            try:
                loaded = await asyncio.to_thread(
                    loader, model_dir or os.getenv("HEALTH_MODEL_DIR") or DEFAULT_MODEL
                )
            except Exception:
                logger.error(json.dumps({"event": "model.unavailable"}))
        app.state.api_token = api_token if api_token is not None else os.getenv("HEALTH_API_TOKEN")
        if app.state.api_token == "":
            raise ValueError("HEALTH_API_TOKEN must be omitted or nonempty")
        app.state.service = AnalysisService(app.state.store, loaded)
        app.state.service.retrieval = (
            retrieval_service
            if retrieval_service is not None
            else await asyncio.to_thread(load_retrieval_from_env)
        )
        app.state.service.reports = ReportService(
            app.state.store, loaded, validator=app.state.harness
        )
        app.state.registry = create_registry(app.state.service, harness=app.state.harness)
        app.state.memory = (
            memory_manager
            if memory_manager is not None
            else await asyncio.to_thread(memory_from_env)
        )
        limits = app.state.harness.limits
        events = (
            agent_event_store
            if agent_event_store is not None
            else RunEventStore(
                Path(
                    os.getenv(
                        "HEALTH_AGENT_TRACE_DIR",
                        str(DEFAULT_MODEL.parents[1] / "logs" / "agent_runs"),
                    )
                ),
                max_runs=limits.max_retained_runs,
                retention_seconds=limits.retention_seconds,
            )
        )
        app.state.agent_runtime = AgentRuntime(
            registry=app.state.registry,
            service=app.state.service,
            backend=decision_backend if decision_backend is not None else backend_from_env(),
            events=events,
            limits=limits,
            router=HealthRouter(),
            memory=app.state.memory,
            harness=app.state.harness,
        )
        app.state.agent_runtime.recover()

        async def cleanup():
            while True:
                await asyncio.sleep(30)
                app.state.store.cleanup()
                app.state.agent_runtime.cleanup()
                try:
                    await app.state.memory.cleanup()
                except SkillError:
                    logger.warning(
                        json.dumps(
                            {"event": "memory.cleanup_failed", "error_code": "memory_unavailable"}
                        )
                    )

        cleanup_task = asyncio.create_task(cleanup())
        try:
            yield
        finally:
            cleanup_task.cancel()
            await asyncio.gather(cleanup_task, return_exceptions=True)
            await app.state.agent_runtime.aclose()
            await app.state.registry.aclose()
            if app.state.service.retrieval is not None:
                await asyncio.to_thread(app.state.service.retrieval.close)
            await app.state.memory.aclose()
            app.state.store.clear()

    app = FastAPI(title="SmartHealth Analysis Tool", version="0.1.0", lifespan=lifespan)
    app.add_middleware(UploadBodyLimitMiddleware)

    def error_response(request, code, message, status):
        return JSONResponse(
            status_code=status,
            content={
                "schema_version": "1.0",
                "request_id": getattr(request.state, "request_id", uuid4().hex),
                "error": {"code": code, "message": message},
            },
        )

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = (
            supplied if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", supplied) else uuid4().hex
        )
        started = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            response = error_response(
                request, "internal_error", "The request could not be completed.", 500
            )
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        # Route templates avoid recording raw artifact IDs, query strings, or uploaded filenames.
        route = request.scope.get("route")
        logger.info(
            json.dumps(
                {
                    "event": "http.completed",
                    "request_id": request.state.request_id,
                    "run_id": getattr(request.state, "run_id", None),
                    "route": getattr(route, "path", "unknown"),
                    "status": response.status_code,
                    "latency_ms": round((time.monotonic() - started) * 1000, 3),
                    "latency_scope": "response_headers",
                    "prediction": getattr(request.state, "prediction", None),
                    "model_version": getattr(request.state, "model_version", None),
                },
                ensure_ascii=False,
            )
        )
        return response

    @app.exception_handler(SkillError)
    async def domain_error(request: Request, exc: SkillError):
        status = 422
        if exc.code in {
            "model_unavailable",
            "dependency_unavailable",
            "registry_closed",
            "trace_unavailable",
            "artifact_capacity_exceeded",
            "backend_unavailable",
            "runtime_unavailable",
            "run_capacity_exceeded",
            "memory_unavailable",
            "memory_capacity_exceeded",
            "skill_disabled",
            "skill_version_mismatch",
        }:
            status = 503
        elif exc.code == "file_too_large":
            status = 413
        elif exc.code == "unauthorized":
            status = 401
        elif exc.code == "permission_denied":
            status = 403
        elif exc.code in {"artifact_not_found", "run_not_found", "memory_not_found"}:
            status = 404
        elif exc.code == "artifact_expired":
            status = 410
        elif exc.code in {
            "idempotency_conflict",
            "session_busy",
            "invalid_parent_run",
            "replay_unavailable",
            "source_result_unavailable",
            "source_visualization_unavailable",
            "memory_conflict",
            "memory_state_conflict",
        }:
            status = 409
        elif exc.code == "timeout":
            status = 504
        elif exc.code in {
            "internal_error",
            "invalid_output",
            "prediction_failed",
            "harness_invalid_output",
        }:
            status = 500
        return error_response(request, exc.code, exc.message, status)

    @app.exception_handler(RequestValidationError)
    async def input_error(request: Request, exc: RequestValidationError):
        return error_response(
            request, "invalid_input", "Request parameters do not match the API schema.", 422
        )

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(analysis.router)
    app.include_router(agent.router)
    app.include_router(memory.router)
    return app


app = create_app()
