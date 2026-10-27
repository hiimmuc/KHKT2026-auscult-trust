"""Atomic files, content hashes, and immutable artifact records."""

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256_file(path):
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def atomic_path(path):
    """Publish a complete file with an atomic rename; preserve the old file on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        yield Path(temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen record changed: {path}; choose a new run")
        return
    atomic_json(path, value)


def atomic_npz(path, **arrays):
    import numpy as np
    with atomic_path(path) as temporary:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)


@lru_cache(maxsize=100000)
def _hash(path, size, mtime, ctime):
    return sha256_file(path)


def file_hashes(paths):
    result = {}
    for path in sorted({Path(p).resolve() for p in paths}):
        st = path.stat()
        result[str(path)] = _hash(str(path), st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    return result


def verify_file(path, expected):
    if not expected or sha256_file(path) != expected:
        raise ValueError(f"Artifact integrity check failed: {path}; rebuild the artifact")
