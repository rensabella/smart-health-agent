"""Authenticated memory access. User identity and vector filters are never request fields."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response

from app.agents.contracts import CreateRunRequest
from app.api.dependencies import new_context, require_access
from app.core.errors import SkillError
from app.memory.schemas import (
    MemoryHit,
    MemoryIdentifier,
    MemorySearchRequest,
    Preferences,
    SessionMemory,
    TrendResponse,
)

router = APIRouter(prefix="/memory", dependencies=[Depends(require_access)])


@router.get("/sessions/{session_id}", response_model=SessionMemory)
async def get_session(request: Request, session_id: MemoryIdentifier):
    session = request.app.state.memory.short_term.get("local_demo", session_id)
    if session is None:
        raise SkillError("memory_not_found", "Session memory is unavailable or expired.")
    context = new_context(request, session_id=session_id, input_version=session.input_version)
    valid_uploads = []
    for ref in session.upload_refs:
        try:
            request.app.state.store.get(ref, kind="upload", context=context)
            valid_uploads.append(ref)
        except SkillError:
            pass
    session.upload_refs = valid_uploads
    if session.current_result_run_id:
        try:
            request.app.state.agent_runtime._source_result(
                CreateRunRequest(
                    query="显示上次结果",
                    session_id=session_id,
                    input_version=session.input_version,
                    source_run_id=session.current_result_run_id,
                ),
                "local_demo",
                context,
            )
        except SkillError:
            session.current_result_run_id = None
            session.prediction = None
    return session


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(request: Request, session_id: MemoryIdentifier):
    request.app.state.memory.short_term.delete_session("local_demo", session_id)
    return Response(status_code=204)


@router.get("/preferences", response_model=Preferences | None)
async def get_preferences(request: Request):
    return await request.app.state.memory.preferences("local_demo")


@router.put("/preferences", response_model=Preferences)
async def set_preferences(request: Request, body: Preferences):
    return await request.app.state.memory.set_preferences("local_demo", body)


@router.post("/search", response_model=list[MemoryHit])
async def search(request: Request, body: MemorySearchRequest):
    return await request.app.state.memory.search("local_demo", body)


@router.get("/trends", response_model=TrendResponse)
async def trends(request: Request, limit: Annotated[int, Query(ge=1, le=512)] = 100):
    return await request.app.state.memory.trends("local_demo", limit)


@router.delete("", status_code=204)
async def delete_all_memory(request: Request):
    await request.app.state.memory.delete_owner("local_demo")
    return Response(status_code=204)
