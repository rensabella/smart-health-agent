"""Bounded metadata reads and local index paths; no untrusted deserialization."""

import json
import os
import stat
from pathlib import Path

from smarthealth.runtime.errors import SkillError


def unavailable(code="rag_unavailable"):
    return SkillError(code, "Knowledge retrieval is unavailable or its local assets are invalid.")


def checked_path(path: Path, *, directory=False, create=False) -> Path:
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise unavailable()
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory:
        if not path.is_dir():
            raise unavailable()
    elif not path.is_file():
        raise unavailable()
    return path


def harden_directory(path: Path):
    path = checked_path(path, directory=True)
    path.chmod(0o700)
    for root, directories, files in os.walk(path, followlinks=False):
        for name in directories + files:
            entry = Path(root) / name
            mode = entry.lstat().st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)) or stat.S_ISLNK(mode):
                raise unavailable()
            entry.chmod(0o700 if stat.S_ISDIR(mode) else 0o600)


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON field")
        value[key] = item
    return value


def read_json(path: Path, *, max_bytes: int):
    path = checked_path(path)
    with path.open("rb") as stream:
        content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise unavailable()
    value = json.loads(
        content,
        object_pairs_hook=_unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Invalid numeric constant")),
    )

    def depth(item, level=0):
        if level > 16:
            raise unavailable()
        if isinstance(item, dict):
            for child in item.values():
                depth(child, level + 1)
        elif isinstance(item, list):
            for child in item:
                depth(child, level + 1)

    depth(value)
    return value
