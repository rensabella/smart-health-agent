"""Bounded, scoped progress events with optional privacy-safe durable journals.

Only server-controlled action metadata belongs in this store. Skill outputs and
model text stay in the in-process runtime; journals cannot reconstruct reports.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from smarthealth.agents.contracts import TERMINAL_STATUSES, Identifier, Objective, Role, RunEvent
from smarthealth.agents.router import RouteDecision
from smarthealth.runtime.errors import SkillError
from smarthealth.skills.base_skill import SkillModel

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_CODE = re.compile(r"^[a-z][a-z0-9_.]{0,63}$")
_ARTIFACT = re.compile(r"^art_[0-9a-f]{32}$")
_HARNESS_VERSION = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_HARNESS_HASH = re.compile(r"^[0-9a-f]{64}$")
_HEADER_MARKER = "smart_health_agent_run"
_MAX_LINE_BYTES = 16 * 1024
_MAX_EVENT_FIELDS = 64
_MAX_REFS = 64
_MAX_CAPABILITIES = 16
_ROUTE_EVENTS = frozenset({"route.selected", "route.upgraded"})
_ROUTE_FIELDS = frozenset(RouteDecision.model_fields)
_COUNTERS = frozenset(
    {
        "decision_count",
        "llm_requests",
        "tool_calls",
        "delegation_count",
        "input_tokens",
        "output_tokens",
        "remaining_tool_calls",
        "remaining_decisions",
        "attempt",
        "memory_recent_turns",
        "memory_history_hits",
        "chunk_index",
    }
)
_CODE_FIELDS = frozenset(
    {
        "status",
        "error_code",
        "skill",
        "action",
        "reason",
        "result_kind",
        "backend_mode",
        "policy",
        "agent",
    }
)
_REF_FIELDS = frozenset(
    {
        "input_ref",
        "result_ref",
        "upload_ref",
        "dataset_ref",
        "window_set_ref",
        "feature_set_ref",
        "prediction_ref",
        "analysis_ref",
        "visualization_ref",
        "image_ref",
        "evidence_ref",
        "draft_ref",
        "review_ref",
    }
)
_IDENTIFIER_FIELDS = frozenset({"call_id", "skill_call_id", "delegate_step_id", "source_run_id"})
_BOOLEAN_FIELDS = frozenset(
    {
        "result_available",
        "include_visualizations",
        "token_usage_available",
        "preferences_available",
        "safe",
        "revision_needed",
    }
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


class _Header(SkillModel):
    record_type: Literal["smart_health_agent_run"] = _HEADER_MARKER
    schema_version: Literal["1.0"] = "1.0"
    run_id: Identifier
    owner_id: Identifier
    session_id: Identifier
    input_version: Identifier
    request_id: Identifier
    backend_mode: Literal["disabled", "scripted", "live"]
    decision_model: (
        Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")] | None
    ) = None
    objective: Objective
    entry_agent: Role
    created_at: Annotated[str, Field(max_length=64)]
    parent_run_id: Identifier | None = None
    source_run_id: Identifier | None = None
    route: RouteDecision | None = None
    harness_version: (
        Annotated[str, Field(pattern=_HARNESS_VERSION.pattern, min_length=1, max_length=64)] | None
    ) = None
    harness_hash: (
        Annotated[str, Field(pattern=_HARNESS_HASH.pattern, min_length=64, max_length=64)] | None
    ) = None

    @model_validator(mode="after")
    def validate_harness_metadata(self):
        if (self.harness_version is None) != (self.harness_hash is None):
            raise ValueError("Harness version and hash must be supplied together")
        return self

    @field_validator("route")
    @classmethod
    def validate_route_metadata(cls, value: RouteDecision | None) -> RouteDecision | None:
        if value is not None and not _valid_capabilities(value.required_capabilities):
            raise ValueError("Invalid route capabilities")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        _parse_time(value)
        return value


@dataclass
class _Run:
    header: _Header
    events: deque[RunEvent]
    latest_id: int = 0
    status: str = "queued"
    error_code: str | None = None
    updated_at: str = ""
    journal_bytes: int = 0
    counters: dict[str, int] = field(default_factory=dict)
    token_usage_available: bool = False
    route: RouteDecision | None = None
    workflow_upgrades: int = 0

    def __post_init__(self) -> None:
        self.route = self.header.route.model_copy(deep=True) if self.header.route else None


def _invalid_event() -> SkillError:
    return SkillError("invalid_trace_event", "Progress event contains unsupported metadata")


def _valid_capabilities(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) <= _MAX_CAPABILITIES
        and all(isinstance(item, str) and bool(_CODE.fullmatch(item)) for item in value)
    )


def _validate_data(data: dict | None) -> dict:
    """Reject unknown fields rather than accidentally persisting private strings."""
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise _invalid_event()
    clean: dict = {}
    for key, value in data.items():
        if key == "harness_version":
            valid = isinstance(value, str) and bool(_HARNESS_VERSION.fullmatch(value))
        elif key in {"harness_hash", "draft_hash", "report_hash"}:
            valid = isinstance(value, str) and bool(_HARNESS_HASH.fullmatch(value))
        elif key == "workflow_upgrades":
            valid = type(value) is int and 0 <= value <= 1
        elif key in _COUNTERS:
            valid = type(value) is int and 0 <= value <= 10**12
        elif key in _CODE_FIELDS:
            valid = value is None or isinstance(value, str) and bool(_CODE.fullmatch(value))
        elif key in _REF_FIELDS:
            valid = value is None or isinstance(value, str) and bool(_ARTIFACT.fullmatch(value))
        elif key in _IDENTIFIER_FIELDS:
            valid = value is None or isinstance(value, str) and bool(_IDENTIFIER.fullmatch(value))
        elif key in _BOOLEAN_FIELDS:
            valid = type(value) is bool
        elif key in {"version", "router_version"}:
            valid = (
                value is None
                or isinstance(value, str)
                and bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value))
            )
        elif key == "latency_ms":
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        elif key == "required_capabilities":
            valid = _valid_capabilities(value)
        elif key in _ROUTE_FIELDS:
            valid = value is None or isinstance(value, str) and bool(_CODE.fullmatch(value))
        elif key in {"input_fields", "output_fields"}:
            valid = (
                isinstance(value, list)
                and len(value) <= _MAX_EVENT_FIELDS
                and all(isinstance(item, str) and bool(_CODE.fullmatch(item)) for item in value)
            )
        elif key in {"refs", "artifact_refs", "output_refs"}:
            valid = (
                isinstance(value, list)
                and len(value) <= _MAX_REFS
                and all(isinstance(item, str) and bool(_ARTIFACT.fullmatch(item)) for item in value)
            )
        else:
            raise _invalid_event()
        if not valid:
            raise _invalid_event()
        clean[key] = value[:] if isinstance(value, list) else value
    return clean


def _validate_route_event(event: str, data: dict) -> RouteDecision | None:
    if event not in _ROUTE_EVENTS:
        return None
    # A route is one atomic decision. Partial updates could leave its objective,
    # role and workflow inconsistent after restart, so require the full schema.
    if set(data) != _ROUTE_FIELDS | {"workflow_upgrades"}:
        raise _invalid_event()
    expected_upgrades = 0 if event == "route.selected" else 1
    if data["workflow_upgrades"] != expected_upgrades:
        raise _invalid_event()
    try:
        return RouteDecision.model_validate({key: data[key] for key in _ROUTE_FIELDS})
    except (ValueError, TypeError, ValidationError):
        raise _invalid_event() from None


def _json_line(value: dict) -> bytes:
    encoded = (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    if len(encoded) > _MAX_LINE_BYTES:
        raise _invalid_event()
    return encoded


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


class RunEventStore:
    """Thread-safe replay store; journals are opt-in and scoped reads are mandatory.

    Terminal runs expire after seven days by default. At capacity, the oldest
    terminal run is removed; active runs are never evicted. ``latest_id`` and
    ``recover`` are trusted runtime operations, not public read endpoints.
    """

    def __init__(
        self,
        directory: Path | None = None,
        max_events: int = 512,
        max_runs: int = 128,
        *,
        retention_seconds: float = 7 * 24 * 3600,
        max_journal_bytes: int = 4 * 1024 * 1024,
    ):
        if max_events <= 0 or max_runs <= 0 or retention_seconds <= 0 or max_journal_bytes <= 0:
            raise ValueError("Event store limits must be positive")
        self.directory = Path(directory) if directory is not None else None
        self.max_events = max_events
        self.max_runs = max_runs
        self.retention_seconds = retention_seconds
        self.max_journal_bytes = max_journal_bytes
        self._runs: dict[str, _Run] = {}
        self._lock = threading.RLock()
        if self.directory is not None:
            try:
                if self.directory.is_symlink():
                    raise OSError("Journal directory is a symlink")
                self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                self.directory.chmod(0o700)
            except OSError:
                raise self._unavailable() from None

    @staticmethod
    def _unavailable() -> SkillError:
        return SkillError("trace_unavailable", "Agent trace storage is unavailable")

    def register(self, run_id: str, *, owner_id: str, session_id: str, metadata: dict) -> None:
        try:
            if set(metadata) & {
                "run_id",
                "owner_id",
                "session_id",
                "record_type",
                "schema_version",
            }:
                raise ValueError("Reserved metadata")
            header = _Header.model_validate(
                {
                    "created_at": _now().isoformat(),
                    **metadata,
                    "run_id": run_id,
                    "owner_id": owner_id,
                    "session_id": session_id,
                }
            )
        except (TypeError, ValueError, ValidationError):
            raise SkillError("invalid_run_metadata", "Run metadata is invalid") from None
        encoded = _json_line(header.model_dump(mode="json"))
        with self._lock:
            self._expire()
            if run_id in self._runs:
                raise SkillError("run_exists", "Run is already registered")
            self._make_room()
            run = _Run(header, deque(maxlen=self.max_events), updated_at=header.created_at)
            if self.directory is not None:
                self._write(run, encoded, create=True)
            self._runs[run_id] = run

    def append(
        self,
        run_id: str,
        event: str,
        *,
        agent: str | None = None,
        step_id: str | None = None,
        parent_step_id: str | None = None,
        data: dict | None = None,
    ) -> RunEvent:
        if not isinstance(event, str) or not _CODE.fullmatch(event):
            raise _invalid_event()
        if agent not in {
            None,
            "planner",
            "data_analysis",
            "rag",
            "report_writer",
            "safety_verification",
        }:
            raise _invalid_event()
        for value in (step_id, parent_step_id):
            if value is not None and (
                not isinstance(value, str) or not _IDENTIFIER.fullmatch(value)
            ):
                raise _invalid_event()
        clean = _validate_data(data)
        route = _validate_route_event(event, clean)
        with self._lock:
            run = self._find(run_id)
            self._validate_route_transition(run, event, route)
            result = RunEvent(
                event_id=run.latest_id + 1,
                run_id=run_id,
                timestamp=_now().isoformat(),
                event=event,
                agent=agent,
                step_id=step_id,
                parent_step_id=parent_step_id,
                data=clean,
            )
            encoded = _json_line(result.model_dump(mode="json"))
            if self.directory is not None:
                self._write(run, encoded)
            self._record(run, result)
            return result.model_copy(deep=True)

    def get_events(
        self, run_id: str, *, owner_id: str, session_id: str, after_id: int = 0
    ) -> list[RunEvent]:
        with self._lock:
            self._expire()
            run = self._find(run_id)
            if (owner_id, session_id) != (run.header.owner_id, run.header.session_id):
                raise SkillError("permission_denied", "Run belongs to another session or owner")
            if type(after_id) is not int or after_id < 0 or after_id > run.latest_id:
                raise SkillError("invalid_event_cursor", "Event cursor is outside this run")
            if run.events and after_id < run.events[0].event_id - 1:
                raise SkillError(
                    "replay_unavailable", "Requested event history is no longer retained"
                )
            return [
                event.model_copy(deep=True) for event in run.events if event.event_id > after_id
            ]

    def latest_id(self, run_id: str) -> int:
        """Internal runtime counter; API routes must authorize the run first."""
        with self._lock:
            return self._find(run_id).latest_id

    def forget(self, run_id: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            self._delete_journal(run)
            del self._runs[run_id]

    def recover(self) -> list[dict]:
        """Restore safe metadata, then fail nonterminal runs interrupted by restart.

        Run results are deliberately absent. A truncated final JSONL record is
        discarded; malformed complete records fail closed. Foreign files and
        symlinks are never modified. Call once before accepting new runs.
        """
        with self._lock:
            if self._runs:
                raise SkillError("recovery_conflict", "Recover before registering runs")
            if self.directory is None:
                return []
            recovered: dict[str, _Run] = {}
            try:
                with os.scandir(self.directory) as entries:
                    for entry in entries:
                        if not entry.is_file(follow_symlinks=False) or not entry.name.endswith(
                            ".jsonl"
                        ):
                            continue
                        run_id = entry.name[:-6]
                        if not _IDENTIFIER.fullmatch(run_id):
                            continue
                        run = self._read_journal(Path(entry.path), run_id)
                        if run is None:
                            continue
                        if self._is_expired(run):
                            self._delete_journal(run)
                            continue
                        recovered[run_id] = run
                        if len(recovered) > self.max_runs:
                            oldest_id = min(
                                recovered, key=lambda key: _parse_time(recovered[key].updated_at)
                            )
                            # All loaded runs are from a stopped process and can be evicted.
                            self._delete_journal(recovered.pop(oldest_id))
            except (OSError, ValueError, ValidationError, UnicodeError):
                raise self._unavailable() from None
            self._runs = recovered
            for run in tuple(self._runs.values()):
                if run.status not in TERMINAL_STATUSES:
                    self.append(
                        run.header.run_id,
                        "run.failed",
                        data={
                            "status": "failed",
                            "error_code": "process_restarted",
                            "result_available": False,
                        },
                    )
            return [self._recovered_metadata(run) for run in self._runs.values()]

    def _recovered_metadata(self, run: _Run) -> dict:
        metadata = run.header.model_dump(mode="json", exclude={"record_type", "schema_version"})
        if run.route is not None:
            metadata.update(
                route=run.route.model_dump(mode="json"),
                objective=run.route.objective,
                entry_agent=run.route.entry_agent,
            )
        return {
            **metadata,
            **run.counters,
            "status": run.status,
            "error_code": run.error_code,
            "last_event_id": run.latest_id,
            "updated_at": run.updated_at,
            "result_available": False,
            "token_usage_available": run.token_usage_available,
            "initial_workflow_type": (
                run.header.route.workflow_type if run.header.route is not None else None
            ),
            "workflow_upgrades": run.workflow_upgrades,
        }

    def _find(self, run_id: str) -> _Run:
        try:
            return self._runs[run_id]
        except (KeyError, TypeError):
            raise SkillError("run_not_found", "Run was not found") from None

    @staticmethod
    def _validate_route_transition(run: _Run, event: str, route: RouteDecision | None) -> None:
        if route is None:
            return
        if event == "route.selected":
            if run.workflow_upgrades or (run.route is not None and route != run.route):
                raise _invalid_event()
        elif (
            run.route is None
            or run.route.workflow_type != "simple"
            or route.workflow_type != "complex"
            or run.workflow_upgrades
        ):
            raise _invalid_event()

    @staticmethod
    def _record(run: _Run, event: RunEvent) -> None:
        route = _validate_route_event(event.event, event.data)
        RunEventStore._validate_route_transition(run, event.event, route)
        if route is not None:
            run.route = route.model_copy(deep=True)
            run.workflow_upgrades = event.data["workflow_upgrades"]
        run.events.append(event.model_copy(deep=True))
        run.latest_id = event.event_id
        run.updated_at = event.timestamp
        status = event.data.get("status")
        suffix = event.event.replace("_", ".").split(".")[-1]
        if status in {"queued", "running", "verifying", *TERMINAL_STATUSES}:
            run.status = status
        elif suffix in TERMINAL_STATUSES:
            run.status = suffix
        if "error_code" in event.data:
            run.error_code = event.data["error_code"]
        if "token_usage_available" in event.data:
            run.token_usage_available = event.data["token_usage_available"]
        for field_name in _COUNTERS:
            if field_name in event.data:
                run.counters[field_name] = event.data[field_name]

    def _path(self, run: _Run) -> Path:
        assert self.directory is not None
        return self.directory / f"{run.header.run_id}.jsonl"

    def _write(self, run: _Run, encoded: bytes, *, create: bool = False) -> None:
        if run.journal_bytes + len(encoded) > self.max_journal_bytes:
            raise self._unavailable()
        flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        flags |= os.O_CREAT | os.O_EXCL if create else 0
        try:
            descriptor = os.open(self._path(run), flags, 0o600)
            with os.fdopen(descriptor, "ab") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise OSError("Journal is not a regular file")
                os.fchmod(stream.fileno(), 0o600)
                start = stream.tell()
                try:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                except OSError:
                    # Restore the last committed boundary when the filesystem permits.
                    stream.truncate(start)
                    raise
        except OSError:
            raise self._unavailable() from None
        run.journal_bytes += len(encoded)

    def _delete_journal(self, run: _Run) -> None:
        if self.directory is None:
            return
        try:
            path = self._path(run)
            if path.is_symlink():
                raise OSError("Journal changed to a symlink")
            path.unlink(missing_ok=True)
        except OSError:
            raise self._unavailable() from None

    def _is_expired(self, run: _Run) -> bool:
        return (_now() - _parse_time(run.updated_at)).total_seconds() >= self.retention_seconds

    def _expire(self) -> None:
        for run_id, run in tuple(self._runs.items()):
            if run.status in TERMINAL_STATUSES and self._is_expired(run):
                self.forget(run_id)

    def _make_room(self) -> None:
        if len(self._runs) < self.max_runs:
            return
        terminal = [run for run in self._runs.values() if run.status in TERMINAL_STATUSES]
        if not terminal:
            raise SkillError("run_capacity_exceeded", "Run trace capacity is full")
        oldest = min(terminal, key=lambda run: _parse_time(run.updated_at))
        self.forget(oldest.header.run_id)

    def _read_journal(self, path: Path, run_id: str) -> _Run | None:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            header_line = stream.readline(_MAX_LINE_BYTES + 1)
            try:
                header_data = json.loads(header_line, object_pairs_hook=_unique_object)
            except (ValueError, UnicodeError):
                return None  # Not a recognizable journal; leave it untouched.
            if (
                not isinstance(header_data, dict)
                or header_data.get("record_type") != _HEADER_MARKER
            ):
                return None
            if len(header_line) > _MAX_LINE_BYTES or not header_line.endswith(b"\n"):
                raise self._unavailable()
            header = _Header.model_validate(header_data)
            if header.run_id != run_id:
                raise self._unavailable()
            payload = stream.read(self.max_journal_bytes + 1)
            if len(payload) + len(header_line) > self.max_journal_bytes:
                raise self._unavailable()
        run = _Run(header, deque(maxlen=self.max_events), updated_at=header.created_at)
        committed = len(header_line)
        for line in payload.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                break  # A crash may interrupt exactly the last append.
            if len(line) > _MAX_LINE_BYTES:
                raise self._unavailable()
            event = RunEvent.model_validate(json.loads(line, object_pairs_hook=_unique_object))
            if event.run_id != run_id or event.event_id != run.latest_id + 1:
                raise self._unavailable()
            _parse_time(event.timestamp)
            if not _CODE.fullmatch(event.event) or event.agent not in {
                None,
                "planner",
                "data_analysis",
                "rag",
                "report_writer",
                "safety_verification",
            }:
                raise self._unavailable()
            for identifier in (event.step_id, event.parent_step_id):
                if identifier is not None and not _IDENTIFIER.fullmatch(identifier):
                    raise self._unavailable()
            _validate_data(event.data)
            self._record(run, event)
            committed += len(line)
        run.journal_bytes = committed
        if committed != len(header_line) + len(payload):
            descriptor = os.open(path, os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "r+b") as stream:
                stream.truncate(committed)
                stream.flush()
                os.fsync(stream.fileno())
        path.chmod(0o600)
        return run
