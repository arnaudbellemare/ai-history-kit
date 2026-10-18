"""hstry databases (https://github.com/byteowlz/hstry).

hstry imports history from many tools (Cursor, OpenCode, Gemini CLI, Aider, Goose,
Copilot, Cline, …) into one SQLite file. Reading it here adds those tools, with
this kit's attribution, redaction, and memory review on top.

hstry does not record who wrote a message, so attribution is heuristic: user
turns that look like injected harness text become `harness`, and the prompts of a
dispatched worker thread (fork_type 'thread') become `agent`. Conversations that
the native Claude Code or Codex readers already produced are skipped during
normalization; ones only hstry still has (deleted from disk since) are kept.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

from ..util import iso
from . import clip
from .claude_code import HARNESS_TEXT
from .codex import INJECTED

DEFAULT_DB = Path(".local") / "share" / "hstry" / "hstry.db"
NATIVE_ADAPTERS = {"claude-code": "claude-code", "codex": "codex"}
REQUIRED_TABLES = {"sources", "conversations", "messages"}


def open_db(path: Path) -> sqlite3.Connection:
    """Open read-only; raise ValueError with a clear message for a missing or foreign file."""
    if not path.is_file():
        raise ValueError(f"no hstry database at {path}")
    try:
        # hstry can leave committed data in WAL even after its CLI exits.
        # immutable=1 ignores WAL and can make a healthy database look corrupt.
        db = sqlite3.connect(f"file:{quote(str(path.resolve()))}?mode=ro", uri=True)
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    except sqlite3.DatabaseError as error:
        raise ValueError(f"{path} is not a readable hstry database ({error})") from None
    if not REQUIRED_TABLES <= tables:
        db.close()
        raise ValueError(f"{path} is not an hstry database (missing tables: {sorted(REQUIRED_TABLES - tables)})")
    db.row_factory = sqlite3.Row
    return db


def _columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def _parts(raw) -> list[dict]:
    try:
        parts = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return [p for p in parts if isinstance(p, dict)] if isinstance(parts, list) else []


def conversation_count(path: Path) -> int:
    db = open_db(path)
    try:
        return db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    finally:
        db.close()


def _tool_text(value) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def read_db(path: Path):
    db = open_db(path)
    try:
        conv_cols, has_tools = _columns(db, "conversations"), "tool_calls" in {
            r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        optional = [c for c in ("title", "updated_at", "fork_type") if c in conv_cols]
        rows = db.execute(
            f"SELECT c.id, c.external_id, c.created_at, {', '.join('c.' + c for c in optional) or 'NULL'}, s.adapter "
            "FROM conversations c JOIN sources s ON s.id = c.source_id ORDER BY c.created_at, c.id").fetchall()
        tools = {}
        if has_tools:
            for t in db.execute("SELECT message_id, tool_name, input, output FROM tool_calls"):
                tools.setdefault(t["message_id"], []).append(t)
        for row in rows:
            thread = "fork_type" in optional and row["fork_type"] == "thread"
            events = []
            for m in db.execute("SELECT id, role, content, parts_json, created_at FROM messages "
                                "WHERE conversation_id = ? ORDER BY idx", (row["id"],)):
                parts = _parts(m["parts_json"])
                text = m["content"] or "\n".join(p.get("text") or "" for p in parts if p.get("type") == "text")
                role = (m["role"] or "other").lower()
                if role == "user" and (HARNESS_TEXT.match(text) or INJECTED.match(text)):
                    role = "harness"
                elif role == "user" and thread:
                    role = "agent"
                stamp = iso(m["created_at"])
                canonical_tools = [p for p in parts if p.get("type") in ("tool_call", "tool_result")]
                # Tool-result content is a projection of its structured part, not
                # a second message. Keep one event with the original output.
                if text.strip() and not (role == "tool" and any(p.get("type") == "tool_result" for p in canonical_tools)):
                    events.append({"kind": "message", "role": role, "text": text, "timestamp": stamp})
                for part in parts:
                    if part.get("type") == "thinking" and (part.get("text") or "").strip():
                        events.append({"kind": "reasoning", "role": "assistant", "text": part["text"], "timestamp": stamp})
                    elif part.get("type") == "tool_call":
                        events.append({"kind": "tool_call", "role": "assistant", "tool": part.get("name") or "tool",
                                       "text": clip(_tool_text(part.get("input"))), "timestamp": stamp})
                    elif part.get("type") == "tool_result":
                        events.append({"kind": "tool_result", "role": "tool",
                                       "text": clip(_tool_text(part.get("output", text))), "timestamp": stamp})
                # New imports use parts_json; older stores can use tool_calls.
                # Prefer canonical parts when both representations are present.
                for tool in tools.get(m["id"], []) if not canonical_tools else []:
                    events.append({"kind": "tool_call", "role": "assistant", "tool": tool["tool_name"],
                                   "text": clip(tool["input"] or ""), "timestamp": stamp})
                    if tool["output"]:
                        events.append({"kind": "tool_result", "role": "tool", "text": clip(tool["output"]), "timestamp": stamp})
            if not events:
                continue
            yield {
                "source": f"hstry:{row['adapter']}",
                "identity": row["external_id"] or row["id"],
                "native": NATIVE_ADAPTERS.get(row["adapter"]),
                "title": row["title"] if "title" in optional else None,
                "created_at": iso(row["created_at"]),
                "updated_at": iso(row["updated_at"]) if "updated_at" in optional else None,
                "subagent": thread,
                "origin": f"hstry:{row['id']}",
                "events": events,
                "stats": {},
            }
    finally:
        db.close()
