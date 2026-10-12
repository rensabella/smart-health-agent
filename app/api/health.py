"""Liveness and model readiness are separate operational signals."""

from fastapi import APIRouter, Request

from app.core.errors import SkillError

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request):
    if request.app.state.service.predictor is None:
        raise SkillError(
            "model_unavailable", "Train or configure a compatible model before inference."
        )
    return {"status": "ready"}
