"""Memory policy and coordination; history cannot authorize tools or publish new facts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import defaultdict

from smarthealth.runtime.errors import SkillError
from smarthealth.runtime.memory.schemas import (
    ExecutionSummary,
    LongTermEntry,
    MemoryContext,
    MemorySearchRequest,
    Preferences,
    ScoreSummary,
)
from smarthealth.runtime.memory.short_term import LocalShortTermStore


class MemoryManager:
    def __init__(self, *, short_term=None, long_term, clock=time.time):
        self.short_term = short_term if short_term is not None else LocalShortTermStore(clock=clock)
        self.long_term = long_term
        self.clock = clock
        self._lock = asyncio.Lock()
        self._closed = False

    async def _io(self, method, *args, **kwargs):
        if self._closed:
            raise SkillError("memory_unavailable", "Memory storage is closed.")
        task = asyncio.create_task(asyncio.to_thread(method, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Drain a native write before releasing the lock. Deletion cannot race a late write.
            try:
                await task
            except Exception:
                pass
            raise
        except SkillError:
            raise
        except Exception:
            raise SkillError("memory_unavailable", "Memory storage is unavailable.") from None

    def begin(self, *, owner_id, session_id, input_version, run_id, upload_refs):
        return self.short_term.begin(owner_id, session_id, input_version, run_id, upload_refs)

    def is_current(self, *, owner_id, session_id, run_id):
        session = self.short_term.get(owner_id, session_id)
        return bool(
            session
            and session.execution
            and session.execution.run_id == run_id
            and session.execution.status in {"queued", "running", "verifying"}
        )

    def update(self, *, owner_id, session_id, lease, execution: ExecutionSummary):
        return self.short_term.update(owner_id, session_id, lease, execution)

    def finish_short(self, *, owner_id, session_id, lease, turn, result_run_id=None):
        return self.short_term.finish(owner_id, session_id, lease, turn, result_run_id)

    async def context(self, *, owner_id, session_id, query):
        """Retrieve once per run, using only server-scoped summaries and explicit preferences."""
        async with self._lock:
            session = self.short_term.get(owner_id, session_id)
            prefs = await self._io(self.long_term.get_preferences, owner_id)
            hits = await self._io(self.long_term.search, owner_id, query[:1000], limit=3)
            return MemoryContext(
                recent_turns=session.turns if session else [],
                preferences=prefs,
                historical_summaries=hits,
            )

    async def finish(self, *, owner_id, session_id, lease, turn, fresh_analysis: bool):
        """Called only after runtime publication. A deleted/expired lease cannot recreate memory."""
        async with self._lock:
            accepted = self.finish_short(
                owner_id=owner_id,
                session_id=session_id,
                lease=lease,
                turn=turn,
                result_run_id=turn.run_id if turn.analysis is not None else None,
            )
            if not accepted:
                return "skipped"
            if not fresh_analysis or turn.assistant_status != "completed" or turn.analysis is None:
                return "stored"
            now = self.clock()
            entry = LongTermEntry(
                kind="analysis",
                source_run_id=turn.run_id,
                created_at=now,
                expires_at=now + self.long_term.retention_seconds,
                analysis=turn.analysis,
            )
            await self._io(self.long_term.put_analysis, owner_id, entry)
            return "stored"

    async def preferences(self, owner_id):
        async with self._lock:
            return await self._io(self.long_term.get_preferences, owner_id)

    async def set_preferences(self, owner_id, preferences):
        preferences = Preferences.model_validate(preferences.model_dump())
        now = self.clock()
        entry = LongTermEntry(
            kind="preferences",
            created_at=now,
            expires_at=now + self.long_term.retention_seconds,
            preferences=preferences,
        )
        async with self._lock:
            await self._io(self.long_term.set_preferences, owner_id, entry)
        return preferences

    async def search(self, owner_id, request):
        request = MemorySearchRequest.model_validate(request.model_dump())
        async with self._lock:
            return await self._io(self.long_term.search, owner_id, request.query, request.limit)

    async def trends(self, owner_id, limit=100):
        """Chronological points grouped by exact model/config identity, never semantic top-k."""
        async with self._lock:
            entries = await self._io(self.long_term.list_analysis, owner_id, limit)
        grouped = defaultdict(list)
        protocols = {}
        for entry in entries:
            a = entry.analysis
            protocol = {
                key: getattr(a, key)
                for key in (
                    "model_name",
                    "model_version",
                    "model_fingerprint",
                    "input_profile",
                    "feature_schema_version",
                    "task_scope",
                    "window_seconds",
                    "aggregation",
                    "score_semantics",
                )
            }
            group_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
            protocols[group_id] = protocol
            grouped[group_id].append(
                {
                    "source_run_id": entry.source_run_id,
                    "analyzed_at": entry.created_at,
                    "risk_score": a.risk_score,
                    "risk_level": a.risk_level,
                    "window_count": a.window_count,
                }
            )
        return {
            "historical_only": True,
            "groups": [
                {"group_id": key, "protocol": protocols[key], "points": points}
                for key, points in grouped.items()
            ],
        }

    async def delete_owner(self, owner_id):
        async with self._lock:
            # Invalidate running leases first; an in-flight task may finish but cannot write back.
            self.short_term.delete_owner(owner_id)
            await self._io(self.long_term.delete_owner, owner_id)

    async def cleanup(self):
        self.short_term.cleanup()
        async with self._lock:
            await self._io(self.long_term.cleanup)

    async def aclose(self):
        async with self._lock:
            if not self._closed:
                try:
                    await self._io(self.long_term.close)
                finally:
                    self._closed = True


def score_summary(analysis, model_metadata):
    """Projection of a runtime-verified result. Do not copy its free-text or local evidence."""
    prediction = analysis["prediction"]
    model = prediction["model"]
    if (model["name"], model["version"]) != (model_metadata["name"], model_metadata["version"]):
        raise SkillError("invalid_memory", "The summary model provenance does not match.")
    fingerprint = hashlib.sha256(
        json.dumps(model_metadata, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()
    return ScoreSummary(
        prediction=prediction["prediction"],
        risk_score=prediction["risk_score"],
        risk_level=prediction["risk_level"],
        window_count=prediction["window_count"],
        model_name=model["name"],
        model_version=model["version"],
        model_fingerprint=fingerprint,
        input_profile=model_metadata["input_profile"],
        feature_schema_version=model_metadata["feature_schema_version"],
        task_scope=model_metadata["task_scope"],
        aggregation=prediction["aggregation"],
        window_seconds=model_metadata["processing_config"]["window_rows"]
        // model_metadata["processing_config"]["sample_rate_hz"],
        score_semantics=model_metadata["score_semantics"],
    )


def memory_from_env(settings=None):
    """Explicit local/ephemeral modes; never silently replace unavailable durable storage."""
    from smarthealth.config.settings import Settings
    from smarthealth.runtime.memory.long_term import QdrantLongTermStore

    try:
        config = settings if settings is not None else Settings.from_env()
        path = None if config.memory_mode == "ephemeral" else config.memory_dir
        session_ttl = config.memory_session_ttl_seconds
        retention = config.memory_retention_seconds
        short = LocalShortTermStore(ttl_seconds=session_ttl)
        long = QdrantLongTermStore(path=path, retention_seconds=retention)
    except SkillError:
        raise
    except (ValueError, TypeError):
        raise SkillError("memory_configuration_error", "Memory limits are invalid.") from None
    return MemoryManager(short_term=short, long_term=long)
