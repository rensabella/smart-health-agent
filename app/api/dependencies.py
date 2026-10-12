"""Single-principal development API; caller headers never choose a user identity."""

from __future__ import annotations

import hmac
from uuid import uuid4

from fastapi import Request, UploadFile

from app.core.context import CallBudget, SkillContext
from app.core.errors import SkillError
from app.core.http_limits import MAX_UPLOAD_BYTES
from skills.bootstrap import DATA_SKILL_NAMES


def require_access(request: Request) -> None:
    token = request.app.state.api_token
    if token is not None:
        supplied = request.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, "Bearer " + token):
            raise SkillError("unauthorized", "A valid API bearer token is required.")
    elif not request.client or request.client.host not in {
        "127.0.0.1",
        "::1",
        "localhost",
        "testclient",
    }:
        raise SkillError("unauthorized", "Configure an API token for nonlocal access.")


def new_context(request: Request, *, session_id=None, input_version=None) -> SkillContext:
    try:
        context = SkillContext(
            owner_id="local_demo",
            session_id=session_id or uuid4().hex,
            run_id=uuid4().hex,
            input_version=input_version or uuid4().hex,
            allowed_skills=DATA_SKILL_NAMES,
            budget=CallBudget(max_calls=6),
        )
        request.state.run_id = context.run_id
        return context
    except ValueError:
        raise SkillError(
            "invalid_input", "Session and input version must be opaque identifiers."
        ) from None


async def read_upload(file: UploadFile) -> bytes:
    try:
        chunks, length = [], 0
        while chunk := await file.read(65536):
            length += len(chunk)
            if length > MAX_UPLOAD_BYTES:
                raise SkillError("file_too_large", "CSV upload exceeds the 5 MiB limit.")
            chunks.append(chunk)
        if not length:
            raise SkillError("invalid_csv", "The CSV upload is empty.")
        return b"".join(chunks)
    finally:
        await file.close()


async def invoke(request: Request, name: str, payload: dict, context: SkillContext) -> dict:
    result = await request.app.state.registry.execute(name, payload, context)
    if result.error:
        raise SkillError(result.error.code, result.error.message)
    return result.data


async def require_analysis_skills(
    request: Request, file: UploadFile, *, explain=False, plots=False
):
    names = {"csv_analysis", "signal_processing", "feature_extraction", "anomaly_prediction"}
    if explain:
        names.add("feature_explanation")
    if plots:
        names.add("visualization")
    try:
        request.app.state.registry.require_available(names)
    except SkillError:
        await file.close()
        raise


async def prediction_steps(request: Request, content: bytes, context: SkillContext):
    import asyncio

    upload = await asyncio.to_thread(request.app.state.service.create_upload, content, context)
    checked = await invoke(request, "csv_analysis", {"upload_ref": upload}, context)
    cleaned = await invoke(
        request, "signal_processing", {"dataset_ref": checked["dataset_ref"]}, context
    )
    features = await invoke(
        request, "feature_extraction", {"window_set_ref": cleaned["window_set_ref"]}, context
    )
    prediction = await invoke(
        request, "anomaly_prediction", {"feature_set_ref": features["feature_set_ref"]}, context
    )
    request.state.prediction = prediction["prediction"]
    request.state.model_version = prediction["model"]["version"]
    return prediction, cleaned["cleaning_report"], cleaned["warnings"]
