"""Bounded API/SSE client independent of Streamlit and safe to unit test offline."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from smarthealth.agents.contracts import CreateRunRequest, RunEvent, RunSnapshot
from smarthealth.reports.schemas import ApprovedReport
from smarthealth.schemas.response import AnalyzeResponse, PredictResponse, UploadResponse

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_EVENT_BYTES = 64 * 1024
_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_REF = re.compile(r"^art_[0-9a-f]{32}$")
_CODE = re.compile(r"^[a-z][a-z0-9_.:-]{0,95}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


class ClientError(Exception):
    """Safe code only: do not include URLs, headers, tokens or provider messages."""

    def __init__(self, code: str, status: int | None = None):
        self.code = code if isinstance(code, str) and _CODE.fullmatch(code) else "request_failed"
        self.status = status
        super().__init__(self.code)


def _identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ClientError("invalid_identifier")
    return value


def _validated(schema, value):
    try:
        return schema.model_validate(value, strict=True)
    except (ValidationError, ValueError, TypeError):
        raise ClientError("invalid_response") from None


def _base_url(value: str) -> str:
    """Endpoint is administrator configuration, never a URL supplied by a report."""
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or any(ord(char) < 33 for char in value)
            or "\\" in value
            or parsed.port == 0
        ):
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise ClientError("invalid_api_configuration") from None
    return value.rstrip("/")


@dataclass(frozen=True)
class StreamMessage:
    event: str
    event_id: int | None
    payload: dict


def parse_sse(lines: Iterator[str]) -> Iterator[StreamMessage]:
    """Parse complete SSE frames only; a truncated frame never advances a cursor."""
    event, event_id, data, size = "message", None, [], 0
    for line in lines:
        size += len(line.encode("utf-8")) + 1
        if size > MAX_EVENT_BYTES:
            raise ClientError("stream_frame_too_large")
        if line == "":
            if data:
                try:
                    payload = json.loads("\n".join(data))
                except (ValueError, TypeError):
                    raise ClientError("invalid_stream") from None
                if not isinstance(payload, dict):
                    raise ClientError("invalid_stream")
                yield StreamMessage(event, event_id, payload)
            event, event_id, data, size = "message", None, [], 0
            continue
        if line.startswith(":"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            value = ""
        if value.startswith(" "):
            value = value[1:]
        if key == "event":
            if not _CODE.fullmatch(value):
                raise ClientError("invalid_stream")
            event = value
        elif key == "id":
            if not value.isascii() or not value.isdigit() or len(value) > 19:
                raise ClientError("invalid_stream")
            event_id = int(value)
        elif key == "data":
            data.append(value)


def _bounded_lines(response) -> Iterator[str]:
    pending = bytearray()
    for chunk in response.iter_bytes():
        pending.extend(chunk)
        while b"\n" in pending:
            line, _, rest = pending.partition(b"\n")
            if len(line) > MAX_EVENT_BYTES:
                raise ClientError("stream_frame_too_large")
            pending = bytearray(rest)
            try:
                yield line.rstrip(b"\r").decode("utf-8")
            except UnicodeError:
                raise ClientError("invalid_stream") from None
        if len(pending) > MAX_EVENT_BYTES:
            raise ClientError("stream_frame_too_large")


@dataclass
class EventBatch:
    messages: list[StreamMessage] = field(default_factory=list)
    disconnected: bool = False


class HealthApiClient:
    def __init__(self, base_url: str, token: str = "", *, transport=None):
        if not isinstance(token, str) or len(token) > 4096 or any(ord(c) < 33 for c in token):
            raise ClientError("invalid_token")
        self.base_url = _base_url(base_url)
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {token}"} if token else {},
            follow_redirects=False,
            trust_env=False,
            transport=transport,
            timeout=httpx.Timeout(200, connect=4, write=15, pool=4),
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._client.close()

    @staticmethod
    def _read(response, limit):
        chunks, size = [], 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > limit:
                raise ClientError("response_too_large")
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _raise_response(response, body):
        if 200 <= response.status_code < 300:
            return
        code = "request_failed"
        try:
            candidate = json.loads(body).get("error", {}).get("code")
            if isinstance(candidate, str) and _CODE.fullmatch(candidate):
                code = candidate
        except (ValueError, AttributeError, TypeError):
            pass
        raise ClientError(code, response.status_code)

    def _json(self, method, path, **kwargs):
        try:
            with self._client.stream(method, path, **kwargs) as response:
                body = self._read(response, MAX_RESPONSE_BYTES)
                self._raise_response(response, body)
                value = json.loads(body)
                if not isinstance(value, dict):
                    raise ClientError("invalid_response")
                return value
        except (httpx.HTTPError, ValueError, UnicodeError):
            raise ClientError("connection_failed") from None

    @staticmethod
    def _file(content):
        if not isinstance(content, bytes) or not content:
            raise ClientError("empty_file")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ClientError("file_too_large")
        return {"file": ("upload.csv", content, "text/csv")}

    def health(self):
        return self._json("GET", "/health")

    def upload(self, content: bytes, session_id: str) -> UploadResponse:
        return _validated(
            UploadResponse,
            self._json(
                "POST",
                "/uploads",
                params={"session_id": _identifier(session_id)},
                files=self._file(content),
            ),
        )

    def predict(self, content: bytes) -> PredictResponse:
        return _validated(
            PredictResponse, self._json("POST", "/predict", files=self._file(content))
        )

    def analyze(self, content: bytes, *, include_visualizations=True) -> AnalyzeResponse:
        return _validated(
            AnalyzeResponse,
            self._json(
                "POST",
                "/analyze",
                params={"include_visualizations": include_visualizations},
                files=self._file(content),
            ),
        )

    def create_run(self, request: CreateRunRequest, idempotency_key: str) -> RunSnapshot:
        payload = _validated(CreateRunRequest, request.model_dump()).model_dump(mode="json")
        return _validated(
            RunSnapshot,
            self._json(
                "POST",
                "/agent/runs",
                json=payload,
                headers={"Idempotency-Key": _identifier(idempotency_key)},
            ),
        )

    def snapshot(self, run_id, session_id) -> RunSnapshot:
        return _validated(
            RunSnapshot,
            self._json(
                "GET",
                f"/agent/runs/{_identifier(run_id)}",
                params={"session_id": _identifier(session_id)},
            ),
        )

    def cancel(self, run_id, session_id) -> RunSnapshot:
        return _validated(
            RunSnapshot,
            self._json(
                "POST",
                f"/agent/runs/{_identifier(run_id)}/cancel",
                params={"session_id": _identifier(session_id)},
            ),
        )

    def image(self, artifact_ref: str, session_id: str, input_version: str) -> bytes:
        # No returned URL is used: scheme, host, path and query all come from fixed contracts.
        if not isinstance(artifact_ref, str) or not _REF.fullmatch(artifact_ref):
            raise ClientError("invalid_identifier")
        try:
            with self._client.stream(
                "GET",
                f"/artifacts/{artifact_ref}",
                params={
                    "session_id": _identifier(session_id),
                    "input_version": _identifier(input_version),
                },
            ) as response:
                body = self._read(response, MAX_RESPONSE_BYTES)
                self._raise_response(response, body)
                if response.headers.get("Content-Type", "").split(";")[
                    0
                ] != "image/png" or not body.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ClientError("invalid_image")
                return body
        except httpx.HTTPError:
            raise ClientError("connection_failed") from None

    def events(self, run_id, session_id, *, after_id=0, window_seconds=0.5) -> EventBatch:
        if type(after_id) is not int or not 0 <= after_id < 10**19:
            raise ClientError("invalid_event_cursor")
        batch, started = EventBatch(), time.monotonic()
        try:
            with self._client.stream(
                "GET",
                f"/agent/runs/{_identifier(run_id)}/events",
                params={"session_id": _identifier(session_id)},
                headers={"Last-Event-ID": str(after_id)},
                timeout=httpx.Timeout(connect=3, read=0.3, write=3, pool=3),
            ) as response:
                if response.status_code != 200:
                    self._raise_response(response, self._read(response, MAX_EVENT_BYTES))
                if response.headers.get("Content-Type", "").split(";")[0] != "text/event-stream":
                    raise ClientError("invalid_stream")
                for message in parse_sse(_bounded_lines(response)):
                    batch.messages.append(message)
                    if len(batch.messages) >= 256 or time.monotonic() - started >= window_seconds:
                        break
        except httpx.ReadTimeout:
            pass  # Idle subscription: reopen from the last complete event on the next UI tick.
        except httpx.HTTPError:
            batch.disconnected = True
        return batch


def approved_report(snapshot: RunSnapshot) -> ApprovedReport | None:
    """A trace or a draft can never substitute for a completed publication snapshot."""
    if snapshot.status != "completed" or not snapshot.result_available or not snapshot.final_result:
        return None
    result = snapshot.final_result
    if result.get("kind") != "health_report":
        return None
    report = _validated(ApprovedReport, result.get("report"))
    publication = result.get("publication")
    if (
        report.safety.warnings
        or not isinstance(publication, dict)
        or publication.get("harness_hash") != report.safety.policy_hash
        or snapshot.harness_hash != report.safety.policy_hash
        or publication.get("draft_ref") != report.draft_ref
        or publication.get("review_ref") != report.review_ref
        or result.get("review_mode") not in {"llm_assisted", "scripted_test"}
        or len(report.body) > 102400
    ):
        raise ClientError("publication_not_verified")
    return report


@dataclass
class ProgressTracker:
    run_id: str
    cursor: int = 0
    publication_checked: bool = False
    content_available: bool = True
    report_hash: str | None = None
    chunks: dict[int, str] = field(default_factory=dict)
    rows: deque = field(default_factory=lambda: deque(maxlen=100))
    replay_gap: bool = False

    def apply(self, message: StreamMessage):
        if message.event == "stream.error":
            code = message.payload.get("code")
            raise ClientError(code if isinstance(code, str) else "stream_failed")
        event = _validated(RunEvent, message.payload)
        if (
            event.run_id != self.run_id
            or event.event != message.event
            or event.event_id != message.event_id
        ):
            raise ClientError("invalid_stream")
        if event.event_id <= self.cursor:
            return
        if event.event_id != self.cursor + 1:
            self.replay_gap = True
            self.publication_checked = False
            self.chunks.clear()
        if (
            event.event == "publication.checked"
            and event.data.get("result_kind") == "health_report"
        ):
            self.publication_checked = True
        elif event.event == "report.chunk":
            index, digest = event.data.get("chunk_index"), event.data.get("report_hash")
            if (
                type(index) is not int
                or not 0 <= index < 256
                or not isinstance(digest, str)
                or not _HASH.fullmatch(digest)
            ):
                raise ClientError("invalid_stream")
            if event.data.get("content_available") is not True:
                self.content_available = False
                self.chunks.clear()
            elif self.publication_checked and self.content_available:
                content = event.data.get("text")
                if not isinstance(content, str) or not 1 <= len(content) <= 400:
                    raise ClientError("invalid_stream")
                if self.report_hash not in {None, digest} or (
                    index in self.chunks and self.chunks[index] != content
                ):
                    raise ClientError("report_chunk_mismatch")
                self.report_hash = digest
                self.chunks[index] = content
        row = {"event_id": event.event_id, "event": event.event}
        for key, value in (
            ("agent", event.agent),
            ("skill", event.data.get("skill")),
            ("error", event.data.get("error_code")),
        ):
            if isinstance(value, str) and _CODE.fullmatch(value):
                row[key] = value
        self.rows.append(row)
        self.cursor = event.event_id

    def unavailable(self):
        self.content_available = False
        self.chunks.clear()

    def checked_prefix(self, report: ApprovedReport) -> str:
        if not self.publication_checked or not self.content_available or not self.chunks:
            return ""
        if self.report_hash != hashlib.sha256(report.body.encode()).hexdigest():
            raise ClientError("report_chunk_mismatch")
        prefix = []
        for index in range(len(self.chunks)):
            if index not in self.chunks:
                break
            chunk = self.chunks[index]
            if chunk != report.body[index * 400 : (index + 1) * 400]:
                raise ClientError("report_chunk_mismatch")
            prefix.append(chunk)
        return "".join(prefix)
