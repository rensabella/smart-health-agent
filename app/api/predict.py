"""Upload and deterministic prediction endpoints; no Agent or LLM work here."""

import asyncio

from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.api.dependencies import (
    new_context,
    prediction_steps,
    read_upload,
    require_access,
    require_analysis_skills,
)
from app.core.errors import SkillError
from app.schemas.response import PredictResponse, UploadResponse

router = APIRouter(dependencies=[Depends(require_access)])


@router.post("/uploads", response_model=UploadResponse)
async def upload(request: Request, file: UploadFile = File(...), session_id: str | None = None):
    content = await read_upload(file)
    context = new_context(request, session_id=session_id)
    ref = await asyncio.to_thread(request.app.state.service.create_upload, content, context)
    try:
        request.app.state.memory.short_term.attach_upload(
            context.owner_id, context.session_id, context.input_version, ref
        )
    except SkillError:
        request.app.state.store.delete(ref, context=context)
        raise
    return UploadResponse(
        request_id=request.state.request_id,
        session_id=context.session_id,
        input_version=context.input_version,
        upload_ref=ref,
        size_bytes=len(content),
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    if request.app.state.service.predictor is None:
        await file.close()
        raise SkillError("model_unavailable", "A trained model is required before prediction.")
    await require_analysis_skills(request, file)
    context = new_context(request)
    prediction, quality, warnings = await prediction_steps(
        request, await read_upload(file), context
    )
    return PredictResponse(
        **prediction,
        request_id=request.state.request_id,
        session_id=context.session_id,
        input_version=context.input_version,
        quality_report=quality,
        warnings=warnings,
    )
