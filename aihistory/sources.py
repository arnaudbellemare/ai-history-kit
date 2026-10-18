"""Find AI history on this machine and describe its coverage."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

CODEX_ROLLOUT = re.compile(r"^rollout-(\d{4})-(\d{2})-(\d{2})T.*\.jsonl(?:\.zst)?$")
SKIP_NAMES = (".DS_Store",)


@dataclass
class Source:
    kind: str  # claude-code | codex | codex-db | chatgpt-export | claude-export | hstry | agent-trace
    root: Path
    files: list[Path] = field(default_factory=list)

    @property
    def bytes(self) -> int:
        return sum(p.stat().st_size for p in self.files if p.exists())


def _files(root: Path, pattern: str) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob(pattern) if p.is_file() and not p.is_symlink() and not p.name.startswith("._"))


def export_kind(path: Path) -> str | None:
    """chatgpt-export or claude-export for a conversations.json (or a zip holding one)."""
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                names = [n for n in archive.namelist() if n.endswith("conversations.json")]
                if not names:
                    return None
                head = archive.read(names[0])[:4000].decode("utf-8", "replace")
        else:
            head = path.read_text(encoding="utf-8", errors="replace")[:4000]
    except (OSError, zipfile.BadZipFile):
        return None
    if '"mapping"' in head:
        return "chatgpt-export"
    if '"chat_messages"' in head or '"sender"' in head:
        return "claude-export"
    return None


def _first_cwd(path: Path, max_lines: int) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for _, line in zip(range(max_lines), handle):
                try:
                    row = json.loads(line) if line.strip() else {}
                except json.JSONDecodeError:
                    continue
                cwd = (row.get("cwd") or (row.get("payload") or {}).get("cwd")) if isinstance(row, dict) else None
                if isinstance(cwd, str) and cwd:
                    return cwd
    except OSError:
        pass
    return None


def history_folders(home: Path) -> set[Path]:
    """Folders the Claude Code and Codex history says agents ran in: one session file per Claude
    project folder, and the first line of each uncompressed Codex rollout."""
    folders = set()
    claude = home / ".claude" / "projects"
    for project in sorted(p for p in claude.iterdir() if p.is_dir()) if claude.is_dir() else []:
        for path in sorted(project.glob("*.jsonl")):
            if cwd := _first_cwd(path, 50):
                folders.add(Path(cwd))
                break
    for folder in ("sessions", "archived_sessions"):
        for path in _files(home / ".codex" / folder, "rollout-*.jsonl"):
            if cwd := _first_cwd(path, 1):
                folders.add(Path(cwd))
    return folders


def agent_trace_stores(home: Path, explicit: list[Path] | None = None) -> list[Path]:
    """Stores named with --agent-traces, plus any .agent-traces folder in home or a history folder."""
    from .adapters import agent_trace

    stores = [agent_trace.open_store(path.expanduser()) for path in explicit or []]
    for folder in sorted({home, *history_folders(home)}):
        store = folder / agent_trace.STORE_NAME
        if store not in stores and agent_trace.session_files(store):
            stores.append(store)
    return stores


def discover(home: Path | None = None, exports: list[Path] | None = None, hstry: Path | None = None,
             agent_traces: list[Path] | None = None) -> list[Source]:
    from .adapters import agent_trace
    from .adapters import hstry as hstry_adapter

    home = home or Path.home()
    found = [Source("agent-trace", store, agent_trace.session_files(store)) for store in agent_trace_stores(home, agent_traces)]
    hstry_db = hstry or (home / hstry_adapter.DEFAULT_DB)
    if hstry is not None or hstry_db.is_file():
        hstry_adapter.open_db(hstry_db).close()  # fail early and clearly on a missing or foreign file
        found.append(Source("hstry", hstry_db.parent, [hstry_db]))
    claude = home / ".claude" / "projects"
    if claude.is_dir():
        found.append(Source("claude-code", claude, _files(claude, "*.jsonl")))
    for folder in ("sessions", "archived_sessions"):
        root = home / ".codex" / folder
        files = [p for p in _files(root, "rollout-*") if CODEX_ROLLOUT.match(p.name)]
        if files:
            found.append(Source("codex", root, files))
    databases = sorted((home / ".codex").glob("thread_history_*.sqlite")) if (home / ".codex").is_dir() else []
    if databases:
        found.append(Source("codex-db", home / ".codex", databases))
    for path in exports or []:
        kind = export_kind(path)
        if kind is None:
            raise SystemExit(f"{path} is not a ChatGPT or Claude.ai export (no conversations.json found)")
        found.append(Source(kind, path, [path]))
    return found


def file_day(source: Source, path: Path) -> date | None:
    if source.kind == "codex":
        match = CODEX_ROLLOUT.match(path.name)
        return date(int(match[1]), int(match[2]), int(match[3])) if match else None
    if source.kind == "claude-code":
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                for _, line in zip(range(20), handle):
                    stamp = json.loads(line).get("timestamp") if line.strip() else None
                    if stamp:
                        return date.fromisoformat(stamp[:10])
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if source.kind == "agent-trace":
        from .adapters.agent_trace import session_meta
        from .util import iso

        started = iso(session_meta(path.parent).get("started_at"))
        try:
            return date.fromisoformat(started[:10]) if started else None
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime).date()


def coverage(source: Source) -> dict:
    """Counts, date span, and days with no session file at all (informational)."""
    days = sorted({d for p in source.files if (d := file_day(source, p))})
    info = {"kind": source.kind, "root": str(source.root), "files": len(source.files), "bytes": source.bytes}
    if source.kind == "claude-code":
        info["subagent_files"] = sum(1 for p in source.files if p.name.startswith("agent-"))
    if source.kind == "codex":
        info["compressed_files"] = sum(1 for p in source.files if p.name.endswith(".zst"))
    if source.kind == "hstry":
        from .adapters.hstry import conversation_count

        info["conversations"] = sum(conversation_count(p) for p in source.files)
    if days and source.kind == "agent-trace":
        info["first_day"], info["last_day"] = days[0].isoformat(), days[-1].isoformat()
    if days and source.kind in ("claude-code", "codex"):
        info["first_day"], info["last_day"] = days[0].isoformat(), days[-1].isoformat()
        present, missing, day = set(days), [], days[0]
        while day <= days[-1]:
            if day not in present:
                missing.append(day.isoformat())
            day += timedelta(days=1)
        info["days_without_files"] = missing
    return info
