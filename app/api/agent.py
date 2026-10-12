"""Agent jobs outlive their SSE subscriptions; every read verifies session ownership."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse

from app.agents.contracts import TERMINAL_STATUSES, CreateRunRequest, Identifier, RunSnapshot
from app.agents.router import RouteDecision
from app.api.dependencies import require_access
from app.core.errors import SkillError

router = APIRouter(prefix="/agent", dependencies=[Depends(require_access)])


def _scope(request: Request, session_id: str):
    # Same single authenticated principal as the stable Tool API.
    return {"owner_id": "local_demo", "session_id": session_id}


@router.post("/route", response_model=RouteDecision)
async def preview_route(request: Request, body: CreateRunRequest):
    return request.app.state.agent_runtime.preview_route(body, owner_id="local_demo")


@router.post("/runs", status_code=202, response_model=RunSnapshot)
async def create_run(
    request: Request,
    body: CreateRunRequest,
    idempotency_key: Annotated[str | None, Header()] = None,
):
    result = await request.app.state.agent_runtime.create(
        body,
        owner_id="local_demo",
        request_id=request.state.request_id,
        idempotency_key=idempotency_key,
    )
    request.state.run_id = result.run_id
    return result


@router.get("/runs/{run_id}", response_model=RunSnapshot)
async def get_run(request: Request, run_id: Identifier, session_id: Identifier):
    result = request.app.state.agent_runtime.snapshot(run_id, **_scope(request, session_id))
    request.state.run_id = result.run_id
    return result


@router.post("/runs/{run_id}/cancel", response_model=RunSnapshot)
async def cancel_run(request: Request, run_id: Identifier, session_id: Identifier):
    result = request.app.state.agent_runtime.cancel(run_id, **_scope(request, session_id))
    request.state.run_id = result.run_id
    return result


@router.get("/runs/{run_id}/trace")
async def get_trace(request: Request, run_id: Identifier, session_id: Identifier):
    events = request.app.state.agent_runtime.get_events(run_id, **_scope(request, session_id))
    request.state.run_id = run_id
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "events": [e.model_dump(mode="json") for e in events],
    }


def _cursor(value):
    if value is None:
        return 0
    if not value.isascii() or not value.isdigit() or len(value) > 20:
        raise SkillError(
            "invalid_event_cursor", "Last-Event-ID must be a nonnegative event number."
        )
    return int(value)


@router.get("/runs/{run_id}/events")
async def run_events(
    request: Request,
    run_id: Identifier,
    session_id: Identifier,
    last_event_id: Annotated[str | None, Header()] = None,
):
    runtime = request.app.state.agent_runtime
    scope = _scope(request, session_id)
    cursor = _cursor(last_event_id)
    # Authorization and replay errors must occur before HTTP 200 headers are sent.
    runtime.get_events(run_id, **scope, after_id=cursor)
    runtime.snapshot(run_id, **scope)
    request.state.run_id = run_id

    async def stream():
        nonlocal cursor
        heartbeat = time.monotonic()
        try:
            while True:
                if await request.is_disconnected():
                    return
                try:
                    events = runtime.get_events(run_id, **scope, after_id=cursor)
                    for event in events:
                        cursor = event.event_id
                        if event.event == "report.chunk":
                            chunk = runtime.report_chunk(
                                run_id,
                                **scope,
                                index=event.data["chunk_index"],
                                report_hash=event.data["report_hash"],
                            )
                            event = event.model_copy(
                                update={
                                    "data": {
                                        **event.data,
                                        "content_available": chunk is not None,
                                        "text": chunk,
                                    }
                                }
                            )
                        yield f"id: {cursor}\nevent: {event.event}\ndata: {event.model_dump_json()}\n\n"
                    snapshot = runtime.snapshot(run_id, **scope)
                except SkillError as exc:
                    # A slow subscriber can fall behind the ring after headers were sent.
                    payload = {
                        "code": exc.code,
                        "snapshot_url": f"/agent/runs/{run_id}?session_id={session_id}",
                    }
                    yield f"event: stream.error\ndata: {json.dumps(payload)}\n\n"
                    return
                if snapshot.status in TERMINAL_STATUSES and cursor >= snapshot.last_event_id:
                    return
                if time.monotonic() - heartbeat >= 10:
                    yield ": keepalive\n\n"
                    heartbeat = time.monotonic()
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            # Cancellation belongs only to this connection; the run remains scheduled.
            raise

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
