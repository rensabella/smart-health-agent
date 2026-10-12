"""Execute six real data Skills without HTTP or an LLM. Export synthetic demo artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.core.artifacts import ArtifactStore
from app.core.context import CallBudget, SkillContext
from app.models.model_loader import load_model
from app.services.analysis_service import AnalysisService
from skills.bootstrap import DATA_SKILL_NAMES, create_registry


async def run(csv_path: Path, model_dir: Path, output_dir: Path) -> dict:
    # Use only synthetic input here: this example intentionally saves its results to disk.
    store = ArtifactStore()
    predictor = load_model(model_dir)
    service = AnalysisService(store, predictor)
    registry = create_registry(service)
    context = SkillContext(
        owner_id="synthetic_demo",
        session_id="standalone_demo",
        allowed_skills=DATA_SKILL_NAMES,
        budget=CallBudget(max_calls=6),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = {"upload_ref": service.create_upload(csv_path.read_bytes(), context)}
        results = {}
        # Explicit service dependency order for this example; Agent decisions arrive in Step 4.
        for name, next_field in (
            ("csv_analysis", "dataset_ref"),
            ("signal_processing", "window_set_ref"),
            ("feature_extraction", "feature_set_ref"),
            ("anomaly_prediction", "prediction_ref"),
            ("feature_explanation", "analysis_ref"),
            ("visualization", None),
        ):
            result = await registry.execute(name, payload, context)
            if result.error:
                raise RuntimeError(f"{name}: {result.error.code}: {result.error.message}")
            results[name] = result.data
            if next_field:
                payload = {next_field: result.data[next_field]}
        for artifact in results["visualization"]["artifacts"]:
            plot = store.get(artifact["artifact_ref"], kind="image", context=context)
            (output_dir / f"{artifact['kind']}.png").write_bytes(plot["content"])
        trace = registry.trace_sink.export(context, output_dir / "traces")
        document = {
            "note": "Synthetic example; temporary references expire when this process exits.",
            "run_id": context.run_id,
            "tool_calls": context.budget.used,
            "results": results,
        }
        (output_dir / "analysis.json").write_text(
            json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (output_dir / "skill_manifest.json").write_text(
            json.dumps(registry.describe(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        prediction = results["anomaly_prediction"]
        return {
            "prediction": prediction["prediction"],
            "risk_score": prediction["risk_score"],
            "model": prediction["model"]["name"],
            "tool_calls": context.budget.used,
            "analysis": str(output_dir / "analysis.json"),
            "trace": str(trace),
        }
    finally:
        await registry.aclose()
        store.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/sample/demo.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models_saved/lightgbm_v1.0"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/example"))
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.csv, args.model_dir, args.output_dir)), indent=2))
