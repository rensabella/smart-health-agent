"""Build a new immutable semantic knowledge index with pinned local model assets."""

import argparse
from pathlib import Path

from app.rag.embeddings import E5ONNXEmbedder, download_model_cache
from app.rag.retrieval import build_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("knowledge/corpus.json"))
    parser.add_argument("--output", type=Path, default=Path("data/knowledge/v1"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/models/multilingual-e5-small"))
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
