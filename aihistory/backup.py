"""Copy raw history into a catalogued backup that never loses data.

Apps prune their own history (old Codex rollouts vanish; Claude Code can clean up
old sessions), so the backup is the archive. Rules:

- Nothing is moved or deleted. A copy counts only after a read-back digest matches.
- A longer version of a stored transcript replaces it (transcripts only grow).
- An older copy that is a prefix of the stored one is already contained.
- Any other difference is stored beside it as <name>.variant-<sha8><suffix>.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from .adapters import agent_trace
from .sources import CODEX_ROLLOUT, Source
from .util import CHUNK, private_dir, sha256_text, write_private_json

TRACE_FILES = (agent_trace.EVENTS, "meta.json", "annotations.jsonl")


def _hash(path: Path, prefix_bytes: int | None = None) -> tuple[str, str | None]:
    full, prefix, seen = hashlib.sha256(), hashlib.sha256(), 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            full.update(chunk)
            if prefix_bytes is not None and seen < prefix_bytes:
                prefix.update(chunk[: prefix_bytes - seen])
            seen += len(chunk)
    return full.hexdigest(), (prefix.hexdigest() if prefix_bytes is not None else None)


class Backup:
    def __init__(self, root: Path):
        self.root = private_dir(root)
        self.catalog_path = self.root / "CATALOG.json"
        self.catalog = json.loads(self.catalog_path.read_text()) if self.catalog_path.exists() else {"files": {}}
        self.counts: dict[str, int] = {}

    def _count(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    def save(self) -> None:
        self.catalog["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_private_json(self.catalog_path, self.catalog)

    def _entry(self, relative: str) -> dict | None:
        files = self.catalog["files"]
        path = self.root / relative
        if relative not in files and path.is_file():
            files[relative] = {"size": path.stat().st_size, "sha256": _hash(path)[0], "sources": []}
        return files.get(relative)

    def place(self, source: Path, relative: str, label: str) -> str:
        source_sha = _hash(source)[0]
        size = source.stat().st_size
        existing = self._entry(relative)
        if existing and existing["sha256"] == source_sha:
            outcome = "unchanged"
        elif existing and existing["size"] > size and _hash(self.root / relative, size)[1] == source_sha:
            outcome = "contained"
        else:
            outcome = None
        if outcome:
            existing["sources"] = sorted(set(existing["sources"]) | {label})
            self._count(outcome)
            return outcome

        pure = PurePosixPath(relative)
        variant = str(pure.with_name(f"{pure.stem}.variant-{source_sha[:8]}{pure.suffix}"))
        if variant in self.catalog["files"]:
            self._count("unchanged")
            return "unchanged"
        target = self.root / relative
        private_dir(target.parent)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        shutil.copyfile(source, temporary)
        os.chmod(temporary, 0o600)
        disk_sha, prefix_sha = _hash(temporary, existing["size"] if existing else None)
        if disk_sha != source_sha:
            temporary.unlink(missing_ok=True)
            self._count("changing_skipped")  # a live session grew mid-copy; the next run gets it
            return "changing_skipped"
        os.utime(temporary, (source.stat().st_atime, source.stat().st_mtime))
        if existing is None or (size > existing["size"] and prefix_sha == existing["sha256"]):
            os.replace(temporary, target)
            outcome, final = ("grown" if existing else "new"), relative
        else:
            os.replace(temporary, self.root / variant)
            outcome, final = "variant", variant
        previous = self.catalog["files"].get(final, {})
        self.catalog["files"][final] = {"size": size, "sha256": source_sha,
                                        "sources": sorted(set(previous.get("sources", [])) | {label})}
        self._count(outcome)
        return outcome

    def snapshot_sqlite(self, database: Path, label: str, area: str = "raw/codex/db") -> str:
        """Consistent copy of a live SQLite database, skipped when unchanged since the last one."""
        stat = database.stat()
        source_stat = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        prefix = f"{area}/{database.stem}/"
        if any(p.startswith(prefix) and e.get("source_stat") == source_stat for p, e in self.catalog["files"].items()):
            self._count("snapshot_unchanged")
            return "unchanged"
        relative = f"{prefix}{datetime.now(timezone.utc):%Y-%m-%d}/{database.name}"
        target = self.root / relative
        private_dir(target.parent)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        source = sqlite3.connect(f"file:{quote(str(database))}?mode=ro", uri=True)
        copy = sqlite3.connect(str(temporary))
        try:
            source.backup(copy)
            copy.execute("pragma journal_mode=delete")
            copy.commit()
            if copy.execute("pragma quick_check").fetchone()[0] != "ok":
                raise RuntimeError(f"snapshot of {database} failed quick_check")
        finally:
            copy.close()
            source.close()
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        self.catalog["files"][relative] = {"size": target.stat().st_size, "sha256": _hash(target)[0],
                                           "sources": [label], "source_stat": source_stat}
        self._count("snapshot_new")
        return "new"

    def verify(self) -> dict[str, str]:
        problems = {}
        for relative, entry in sorted(self.catalog["files"].items()):
            path = self.root / relative
            if not path.is_file():
                problems[relative] = "missing"
            elif _hash(path)[0] != entry["sha256"]:
                problems[relative] = "sha256 mismatch"
            elif relative.startswith("raw/agent-trace/") and PurePosixPath(relative).name.startswith("events"):
                chain = agent_trace.verify_chain(path)  # catches edits made before the copy, too
                if chain["status"] == "broken":
                    problems[relative] = f"hash chain broken at line {chain['line']}: {chain['reason']}"
        return problems


def back_up(sources: list[Source], root: Path) -> dict:
    backup = Backup(root)
    for source in sources:
        label = f"{source.kind}:{source.root}"
        for path in source.files:
            if source.kind == "claude-code":
                backup.place(path, f"raw/claude-code/projects/{path.relative_to(source.root).as_posix()}", label)
            elif source.kind == "codex":
                m = CODEX_ROLLOUT.match(path.name)
                backup.place(path, f"raw/codex/sessions/{m[1]}/{m[2]}/{m[3]}/{path.name}", label)
            elif source.kind == "codex-db":
                backup.snapshot_sqlite(path, label)
            elif source.kind == "hstry":
                backup.snapshot_sqlite(path, label, area="raw/hstry")
            elif source.kind == "agent-trace":
                area = f"raw/agent-trace/{sha256_text(str(source.root))[:8]}/{path.parent.relative_to(source.root).as_posix()}"
                for name in TRACE_FILES:
                    if (path.parent / name).is_file():
                        backup.place(path.parent / name, f"{area}/{name}", label)
            else:
                backup.place(path, f"raw/exports/{source.kind}/{path.name}", label)
        backup.save()
    return {"root": str(backup.root), "files": len(backup.catalog["files"]), "counts": backup.counts}


def backed_up_sources(root: Path) -> list[Source]:
    """The backup's own raw tree, as sources for normalization."""
    found = []
    claude = root / "raw" / "claude-code" / "projects"
    if claude.is_dir():
        found.append(Source("claude-code", claude, sorted(p for p in claude.rglob("*.jsonl") if p.is_file())))
    codex = root / "raw" / "codex" / "sessions"
    if codex.is_dir():
        found.append(Source("codex", codex, sorted(p for p in codex.rglob("rollout-*") if CODEX_ROLLOUT.match(p.name))))
    hstry = root / "raw" / "hstry"
    snapshots = sorted(hstry.glob("*/*/*.db")) if hstry.is_dir() else []
    if snapshots:  # the newest snapshot holds everything hstry had at that time
        found.append(Source("hstry", snapshots[-1].parent, [snapshots[-1]]))
    traces = root / "raw" / "agent-trace"
    for store in sorted(p for p in traces.iterdir() if p.is_dir()) if traces.is_dir() else []:
        if files := agent_trace.session_files(store):  # variants (events.variant-*.ndjson) stay put
            found.append(Source("agent-trace", store, files))
    exports = root / "raw" / "exports"
    for kind in ("chatgpt-export", "claude-export"):
        folder = exports / kind
        if folder.is_dir():
            found.append(Source(kind, folder, sorted(p for p in folder.iterdir() if p.is_file())))
    return found
