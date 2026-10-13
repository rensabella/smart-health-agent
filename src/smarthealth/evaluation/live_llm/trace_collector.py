"""Evaluation-only collectors. No raw provider payloads, CSV rows or secrets persist."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import pickle
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic import ValidationError

from smarthealth.agents.backends import DecisionBackend
from smarthealth.runtime.artifacts import ArtifactStore
from smarthealth.runtime.errors import SkillError


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class ArtifactCollector(ArtifactStore):
    """Keep bounded lineage manifests; all actual artifacts stay in the existing store."""

    def __init__(self):
        super().__init__()
        self.manifests: list[dict] = []
        self._manifest_lock = threading.Lock()

    def put(self, value, *, kind, context, source_ref=None, schema_version="1.0"):
        ref = super().put(
            value, kind=kind, context=context, source_ref=source_ref, schema_version=schema_version
        )
        # Hash only locally created trusted values. Nothing unpickles this digest.
        manifest = {
            "ref": ref,
            "kind": kind,
            "source_ref": source_ref,
            "input_version": context.input_version,
            "schema_version": schema_version,
            "sha256": hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest(),
        }
        if kind == "evidence":
            hits = value["result"]["hits"]
            manifest["evidence_ids"] = [h["evidence_id"] for h in hits]
            manifest["evidence_sources"] = {h["evidence_id"]: h["source"]["id"] for h in hits}
            manifest["evidence_details"] = {
                h["evidence_id"]: {
                    k: h[k] for k in ("source", "locator", "content_hash", "knowledge_version")
                }
                for h in hits
            }
            manifest["retrieval_status"] = value["result"]["status"]
        if kind in {"report_draft", "report_review"}:
            for key in (
                "draft_version",
                "draft_hash",
                "evidence_hash",
                "analysis_fingerprint",
                "policy_hash",
                "review_hash",
                "draft_ref",
                "evidence_ref",
                "safe",
            ):
                if key in value:
                    manifest[key] = value[key]
        if kind == "image":
            manifest["plot_type"] = value["kind"]
        with self._manifest_lock:
            self.manifests.append(manifest)
        return ref

    def export(self):
        with self._manifest_lock:
            return copy.deepcopy(self.manifests)

    def export_images(self, directory: Path, references: list[str], context):
        """Persist only PNGs referenced by a published synthetic result, never raw data."""
        if not references:
            return []
        directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        outputs = []
        for ref in references:
            value = self.get(ref, kind="image", context=context)
            content = value["content"]
            if value["mime_type"] != "image/png" or not content.startswith(b"\x89PNG"):
                raise SkillError("evaluation_artifact_invalid", "Expected a generated PNG.")
            name = f"{ref}.png"
            path = directory / name
            with path.open("xb") as stream:
                stream.write(content)
            path.chmod(0o600)
            outputs.append(
                {
                    "ref": ref,
                    "file": name,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "mime_type": "image/png",
                }
            )
        return outputs


class SkillCollector:
    def __init__(self, inner, injector):
        self.inner, self.injector = inner, injector
        self.calls: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def execute(self, name, payload, context):
        catalogue = self.inner._skills
        skill = catalogue.get(name)
        valid = False
        if skill is not None:
            try:
                skill.input_schema.model_validate(payload, strict=True)
                valid = True
            except (ValidationError, ValueError, TypeError):
                pass
        before = self.injector.triggered
        started = time.perf_counter()
        call = {
            "skill": name if skill else "unknown",
            "agent": context.agent_name,
            "arguments_valid": valid,
            "allowed": name in context.allowed_skills,
            "status": "error",
            "error_code": "cancelled",
            "output_refs": [],
            "injected": False,
            "start_time": utcnow(),
        }
        try:
            result = await self.inner.execute(name, payload, context)
            call.update(
                status=result.status,
                error_code=result.error.code if result.error else None,
                call_id=result.call_id,
            )
            if result.data:
                call["output_refs"] = [
                    v for k, v in result.data.items() if k.endswith("_ref") and isinstance(v, str)
                ]
                if name == "visualization":
                    call["output_refs"].extend(
                        a["artifact_ref"] for a in result.data.get("artifacts", [])
                    )
            return result
        finally:
            call.update(
                injected=not before and self.injector.triggered,
                end_time=utcnow(),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            self.calls.append(call)


class EvaluationTransport(httpx.AsyncBaseTransport):
    """One synthetic transport failure, otherwise the unchanged real HTTP transport.

    Injected responses never contain a fabricated model action. Every successful
    decision must come from the configured transport/provider.
    """

    def __init__(self, fault="none", inner=None):
        self.inner = inner or httpx.AsyncHTTPTransport(retries=0)
        self.fault = fault
        self.network_dispatches = 0
        self.triggered = False
        self.events = []

    async def handle_async_request(self, request):
        if not self.triggered and self.fault in {"backend_transient", "backend_invalid_json"}:
            self.triggered = True
            self.events.append(
                {"phase": "injected", "kind": self.fault, "target": "provider_transport"}
            )
            if self.fault == "backend_transient":
                return httpx.Response(503, json={"error": "evaluation_injected"})
            return httpx.Response(200, content=b"{", headers={"content-type": "application/json"})
        self.network_dispatches += 1
        return await self.inner.handle_async_request(request)

    async def aclose(self):
        await self.inner.aclose()


class ObservedBackend(DecisionBackend):
    """Record attempts, not reasoning. Existing runtime owns all retry decisions."""

    def __init__(self, inner, budget, registry, transport=None):
        self.inner, self.budget, self.registry, self.transport = inner, budget, registry, transport
        self.mode, self.model_name = inner.mode, inner.model_name
        self.attempts: list[dict] = []
        self.actions: list[dict] = []

    async def decide(self, request):
        start = time.perf_counter()
        row = {
            "agent": request.agent,
            "status": "error",
            "error_code": "cancelled",
            "start_time": utcnow(),
            "network_dispatched": False,
            "injected": False,
            "input_tokens": None,
            "output_tokens": None,
        }
        reservation = None
        network_before = self.transport.network_dispatches if self.transport else 0
        fault_before = self.transport.triggered if self.transport else False
        try:
            if self.mode == "live":
                # Uses exact same serialization as production. Never persist it.
                payload_bytes = self.inner.build_payload(request)
                reservation = self.budget.reserve(len(payload_bytes))
                row["reserved_tokens"] = reservation.tokens
            reply = await self.inner.decide(request)
            row.update(
                status="success",
                error_code=None,
                input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens,
            )
            action = reply.decision
            safe = {"action": action.action, "agent": request.agent, "injected": False}
            if action.action == "call_skill":
                skill = self.registry._skills.get(action.skill)
                safe["skill"] = action.skill if skill else "unknown"
                safe["allowed"] = action.skill in {t["function"]["name"] for t in request.tools}
                safe["arguments_valid"] = False
                if skill:
                    try:
                        skill.input_schema.model_validate(action.arguments, strict=True)
                        safe["arguments_valid"] = True
                    except (ValidationError, ValueError, TypeError):
                        pass
            self.actions.append(safe)
            return reply
        except SkillError as exc:
            row["error_code"] = exc.code
            raise
        except asyncio.CancelledError:
            raise
        except Exception:
            row["error_code"] = "evaluation_internal_error"
            raise SkillError("evaluation_internal_error", "Evaluation backend failed.") from None
        finally:
            if self.transport:
                row["network_dispatched"] = self.transport.network_dispatches > network_before
                row["injected"] = not fault_before and self.transport.triggered
            row.update(end_time=utcnow(), latency_ms=(time.perf_counter() - start) * 1000)
            self.attempts.append(row)
            if reservation:
                self.budget.settle(reservation, row["input_tokens"], row["output_tokens"])

    async def aclose(self):
        await self.inner.aclose()
