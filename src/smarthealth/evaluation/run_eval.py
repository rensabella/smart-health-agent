"""Run explicit scripted/live Agent, retrieval and claim-review evaluation modes."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from smarthealth.config.resources import resource_path
from smarthealth.config.settings import Settings
from smarthealth.evaluation.cases import load_cases
from smarthealth.evaluation.metrics import render_markdown
from smarthealth.evaluation.runner import LiveBudget, backend_factory, evaluate_suite
from smarthealth.models.model_loader import load_model
from smarthealth.retrieval.embeddings import E5ONNXEmbedder
from smarthealth.retrieval.files import read_json
from smarthealth.retrieval.retrieval import RetrievalService
from smarthealth.retrieval.schemas import KnowledgeChunk
from smarthealth.runtime.errors import SkillError


async def run(args):
    os.environ.setdefault("MPLCONFIGDIR", str(Path("data/generated/matplotlib-cache").resolve()))
    if not 1 <= args.max_live_requests <= 512:
        raise ValueError("Live request cap must be between 1 and 512")
    meter = LiveBudget(limit=args.max_live_requests)
    factory = backend_factory(args.backend, meter)
    probe = factory()  # Validate live configuration before loading expensive local assets.
    await probe.aclose()
    suite = load_cases(args.cases, root=args.corpus_root)
    predictor = load_model(args.model)
    retrieval = RetrievalService(
        args.index, E5ONNXEmbedder(args.embedding_model), min_score=args.min_score
    )
    try:
        chunks = tuple(
            KnowledgeChunk.model_validate(c, strict=True)
            for c in read_json(args.index / "chunks.json", max_bytes=2 * 1024 * 1024)
        )
        report = await evaluate_suite(
            suite,
            predictor,
            retrieval,
            chunks,
            content=args.csv.read_bytes(),
            corpus_root=args.corpus_root,
            mode=args.backend,
            meter=meter,
            make_backend=factory,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        markdown = args.output.with_suffix(".md")
        markdown.write_text(render_markdown(report))
        print(
            json.dumps(
                {"mode": report["mode"], "metrics": report["metrics"], "output": str(args.output)},
                indent=2,
            )
        )
        failed = (
            any(not c["passed"] for c in report["system_cases"])
            or any(not c["hit"] for c in report["retrieval_cases"])
            or any(not c["correct"] for c in report["safety_cases"])
        )
        return int(args.fail_on_regression and failed)
    finally:
        retrieval.close()


def main():
    config = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=resource_path())
    parser.add_argument("--csv", type=Path, default=resource_path("data", "sample", "demo.csv"))
    parser.add_argument("--backend", choices=["scripted", "live"], default="scripted")
    parser.add_argument(
        "--cases", type=Path, default=resource_path("evaluation", "test_cases.json")
    )
    parser.add_argument("--model", type=Path, default=config.model_dir)
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=config.rag_model_dir or config.home / "data/models/multilingual-e5-small",
    )
    parser.add_argument(
        "--index", type=Path, default=config.rag_index_dir or config.home / "data/knowledge/v1"
    )
    parser.add_argument("--min-score", type=float, default=config.rag_min_score)
    parser.add_argument("--max-live-requests", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("evaluation/evaluation_report.json"))
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except (SkillError, ValueError, OSError) as exc:
        code = exc.code if isinstance(exc, SkillError) else "evaluation_configuration_error"
        print(f"Evaluation could not complete: {code}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
