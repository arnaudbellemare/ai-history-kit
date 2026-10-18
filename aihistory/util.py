"""Shared helpers: private files, hashing, timestamps, and compressed JSONL."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def short_id(prefix: str, value: object) -> str:
    return f"{prefix}_{sha256_text(str(value))[:20]}"


def private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def write_private_text(path: Path, text: str) -> None:
    """Write atomically with mode 0600."""
    private_dir(path.parent)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_private_json(path: Path, value) -> None:
    write_private_text(path, json.dumps(value, indent=1, ensure_ascii=False, sort_keys=True) + "\n")


@contextmanager
def private_jsonl(path: Path):
    """Yield a writer for JSON lines; the file appears atomically, mode 0600, on success."""
    private_dir(path.parent)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yield lambda value: handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def iso(value) -> str | None:
    """Normalize epoch seconds/milliseconds or ISO strings to an ISO-8601 UTC string."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 1e11 else value
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        except (OverflowError, OSError, ValueError):
            return None
    return str(value)


class ZstdUnavailable(RuntimeError):
    pass


def decompress_zstd(data: bytes) -> bytes:
    try:
        from compression import zstd  # Python 3.14+
    except ImportError:
        zstd = None
    if zstd is not None:
        return zstd.decompress(data)
    if shutil.which("zstd"):
        return subprocess.run(["zstd", "-dcq"], input=data, capture_output=True, check=True).stdout
    raise ZstdUnavailable("reading .zst files needs Python 3.14+ or the `zstd` command")


def read_lines(path: Path):
    """Text lines of a .jsonl or .jsonl.zst file."""
    if path.name.endswith(".zst"):
        yield from decompress_zstd(path.read_bytes()).decode("utf-8", "replace").splitlines()
        return
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line.rstrip("\n")


def json_lines(path: Path):
    for line in read_lines(path):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue
