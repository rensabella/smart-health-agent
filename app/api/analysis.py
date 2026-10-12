"""Structured analysis with explicit synthetic scope and optional protected PNGs."""

import asyncio

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import Response

from app.api.dependencies import (
    invoke,
    new_context,
    prediction_steps,
    read_upload,
    require_access,
    require_analysis_skills,
)
from app.core.errors import SkillError
from app.schemas.response import AnalyzeResponse, Explanation, Prediction

router = APIRouter(dependencies=[Depends(require_access)])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request, file: UploadFile = File(...), include_visualizations: bool = False
):
    if request.app.state.service.predictor is None:
        await file.close()
        raise SkillError("model_unavailable", "A trained model is required before analysis.")
    await require_analysis_skills(request, file, explain=True, plots=include_visualizations)
    context = new_context(request)
    predicted, quality, warnings = await prediction_steps(request, await read_upload(file), context)
    explained = await invoke(
        request, "feature_explanation", {"prediction_ref": predicted["prediction_ref"]}, context
    )
    images = []
    if include_visualizations:
        visualized = await invoke(
            request, "visualization", {"analysis_ref": explained["analysis_ref"]}, context
        )
        images = visualized["artifacts"]
        for image in images:
            image["url"] += (
                f"?session_id={context.session_id}&input_version={context.input_version}"
            )
    prediction = Prediction.model_validate(
        {k: v for k, v in predicted.items() if k in Prediction.model_fields}
    )
    explanation = Explanation.model_validate(
        {k: v for k, v in explained.items() if k in Explanation.model_fields}
    )
    summary = (
        f"分析了 {prediction.window_count} 个 30 秒窗口；最高合成异常模型分数为 "
        f"{prediction.risk_score:.3f}，对应窗口 {prediction.highest_risk_window}（从 0 编号）。"
        "该分数用于合成异常模式演示，不表示临床患病概率或医学诊断。"
    )
    return AnalyzeResponse(
        request_id=request.state.request_id,
        session_id=context.session_id,
        input_version=context.input_version,
        analysis_ref=explained["analysis_ref"],
        summary=summary,
        risk_level=prediction.risk_level,
        prediction=prediction,
        explanation=explanation,
        visualization=images,
        quality_report=quality,
        warnings=warnings,
    )


@router.get("/artifacts/{ref}")
async def image_artifact(request: Request, ref: str, session_id: str, input_version: str):
    context = new_context(request, session_id=session_id, input_version=input_version)
    image = await asyncio.to_thread(request.app.state.store.get, ref, kind="image", context=context)
    return Response(image["content"], media_type="image/png", headers={"Cache-Control": "no-store"})
