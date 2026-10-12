"""Pinned pretrained E5 inference with local ONNX/tokenizer assets only."""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from app.core.errors import SkillError
from app.rag.files import checked_path, harden_directory, unavailable

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "fd1525a9fd15316a2d503bf26ab031a61d056e98"
EMBEDDING_ID = f"{MODEL_ID}@{MODEL_REVISION}:onnx_mean_l2_384_v1"
MODEL_FILES = {
    "model.onnx": (
        "onnx/model.onnx",
        470268510,
        "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
    ),
    "tokenizer.json": (
        "tokenizer.json",
        17082730,
        "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    ),
}
MAX_MODEL_TOKENS = 512
MAX_BATCH = 16


class Embedder(Protocol):
    embedding_id: str
    dimensions: int

    def token_count(self, text: str, *, prefix: str) -> int: ...
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...
    def encode_passages(self, texts: list[str]) -> np.ndarray: ...
    def close(self) -> None: ...


def verify_model_cache(directory: Path) -> Path:
    try:
        directory = checked_path(directory, directory=True)
        for name, (_, size, expected) in MODEL_FILES.items():
            path = checked_path(directory / name)
            if path.stat().st_size != size:
                raise unavailable("rag_model_invalid")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise unavailable("rag_model_invalid")
        return directory
    except Exception:
        raise unavailable("rag_model_invalid") from None


def download_model_cache(directory: Path):
    """Explicit build-time operation; fixed public URLs, no credentials or source fetching."""
    import httpx

    directory = checked_path(directory, directory=True, create=True)
    harden_directory(directory)
    for name, (remote, size, expected) in MODEL_FILES.items():
        destination = directory / name
        if destination.exists():
            # Existing assets are never silently replaced or trusted by file name.
            checked_path(destination)
            if destination.stat().st_size == size:
                digest = hashlib.sha256()
                with destination.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() == expected:
                    continue
            raise unavailable("rag_model_invalid")
        temporary = directory / f".{name}.download"
        if temporary.exists() or temporary.is_symlink():
            raise unavailable("rag_model_invalid")
        try:
            digest, total = hashlib.sha256(), 0
            # Redirects are needed for Hugging Face's signed CDN downloads; no
            # user-supplied URL or credentials enter this fixed artifact request.
            with httpx.Client(
                timeout=httpx.Timeout(60.0, connect=15.0), follow_redirects=True, trust_env=False
            ) as client:
                with client.stream(
                    "GET", f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{remote}"
                ) as response:
                    response.raise_for_status()
                    descriptor = os.open(
                        temporary,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                    )
                    with os.fdopen(descriptor, "wb") as stream:
                        for block in response.iter_bytes(1024 * 1024):
                            total += len(block)
                            if total > size:
                                raise unavailable("rag_model_invalid")
                            stream.write(block)
                            digest.update(block)
                        stream.flush()
                        os.fsync(stream.fileno())
            if total != size or digest.hexdigest() != expected:
                raise unavailable("rag_model_invalid")
            temporary.rename(destination)
        except Exception:
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
            raise unavailable("rag_model_download_failed") from None
    harden_directory(directory)
    return verify_model_cache(directory)


def normalized_vectors(value, rows: int, dimensions: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (rows, dimensions) or not np.isfinite(array).all():
        raise unavailable("rag_embedding_invalid")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise unavailable("rag_embedding_invalid")
    return array / norms


class E5ONNXEmbedder:
    embedding_id = EMBEDDING_ID
    dimensions = 384

    def __init__(self, model_dir: Path):
        self._lock = threading.RLock()
        self._session = None
        try:
            directory = verify_model_cache(model_dir)
            import onnxruntime as ort
            from tokenizers import Tokenizer

            ort.disable_telemetry_events()
            self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
            self._tokenizer.no_padding()
            self._tokenizer.no_truncation()
            self._pad_id = next(
                (
                    value
                    for name in ("<pad>", "[PAD]")
                    if (value := self._tokenizer.token_to_id(name)) is not None
                ),
                None,
            )
            if self._pad_id is None:
                raise unavailable("rag_model_invalid")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(directory / "model.onnx"),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._session.disable_fallback()
            self._input_names = {item.name for item in self._session.get_inputs()}
            if not {
                "input_ids",
                "attention_mask",
            } <= self._input_names or not self._input_names <= {
                "input_ids",
                "attention_mask",
                "token_type_ids",
            }:
                raise unavailable("rag_model_invalid")
        except SkillError:
            raise
        except Exception:
            raise unavailable("rag_embedding_unavailable") from None

    def token_count(self, text: str, *, prefix: str) -> int:
        if prefix not in {"query: ", "passage: "} or not isinstance(text, str) or len(text) > 4000:
            raise unavailable("rag_embedding_invalid")
        with self._lock:
            return len(self._tokenizer.encode(prefix + text).ids)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "query: ")

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "passage: ")

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        if (
            not isinstance(texts, list)
            or not 1 <= len(texts) <= MAX_BATCH
            or any(
                not isinstance(text, str) or not text.strip() or len(text) > 4000 for text in texts
            )
        ):
            raise unavailable("rag_embedding_invalid")
        with self._lock:
            try:
                if self._session is None:
                    raise unavailable()
                encoded = self._tokenizer.encode_batch([prefix + text for text in texts])
                length = max(len(item.ids) for item in encoded)
                if length > MAX_MODEL_TOKENS:
                    raise SkillError(
                        "rag_query_too_long", "Retrieval text exceeds the embedding token limit."
                    )
                inputs = np.full((len(texts), length), self._pad_id, dtype=np.int64)
                masks = np.zeros_like(inputs)
                types = np.zeros_like(inputs)
                for i, item in enumerate(encoded):
                    inputs[i, : len(item.ids)] = item.ids
                    masks[i, : len(item.ids)] = item.attention_mask
                    types[i, : len(item.ids)] = item.type_ids
                available = {"input_ids": inputs, "attention_mask": masks, "token_type_ids": types}
                hidden = np.asarray(
                    self._session.run(None, {name: available[name] for name in self._input_names})[
                        0
                    ],
                    dtype=np.float32,
                )
                if (
                    hidden.shape != (len(texts), length, self.dimensions)
                    or not np.isfinite(hidden).all()
                ):
                    raise unavailable("rag_embedding_invalid")
                pooled = (hidden * masks[..., None]).sum(axis=1) / masks.sum(axis=1, keepdims=True)
                return normalized_vectors(pooled, len(texts), self.dimensions)
            except SkillError:
                raise
            except Exception:
                raise unavailable("rag_embedding_unavailable") from None

    def close(self):
        with self._lock:
            self._session = None
