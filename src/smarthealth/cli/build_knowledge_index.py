"""Build a new immutable semantic knowledge index with pinned local model assets."""

import argparse
from pathlib import Path

from smarthealth.config.resources import resource_path
from smarthealth.config.settings import Settings
from smarthealth.retrieval.embeddings import E5ONNXEmbedder, download_model_cache
from smarthealth.retrieval.retrieval import build_index


def main():
    config = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=resource_path("knowledge", "corpus.json"))
    parser.add_argument(
        "--output", type=Path, default=config.rag_index_dir or config.home / "data/knowledge/v1"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=config.rag_model_dir or config.home / "data/models/multilingual-e5-small",
    )
    parser.add_argument("--download-model", action="store_true")
    args = parser.parse_args()
    if args.download_model:
        download_model_cache(args.model_dir)
    embedder = E5ONNXEmbedder(args.model_dir)
    try:
        print(build_index(args.corpus, args.output, embedder).model_dump_json(indent=2))
    finally:
        embedder.close()


if __name__ == "__main__":
    main()
