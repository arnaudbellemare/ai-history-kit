"""A local full-text index over the corpus. By default it holds only text you wrote."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from . import OWNER_ROLE
from .normalize import read_corpus
from .util import private_dir

SCHEMA_SQL = """
CREATE TABLE conversations(
    rowid INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL, source TEXT NOT NULL,
    title TEXT, created_at TEXT, updated_at TEXT, subagent INTEGER NOT NULL, origin TEXT);
CREATE TABLE messages(
    rowid INTEGER PRIMARY KEY, conversation INTEGER NOT NULL REFERENCES conversations(rowid),
    role TEXT NOT NULL, kind TEXT NOT NULL, timestamp TEXT, text TEXT NOT NULL);
CREATE VIRTUAL TABLE messages_fts USING fts5(text, content='messages', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2');
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
"""


def build(out_dir: Path, roles: tuple[str, ...] = (OWNER_ROLE,)) -> dict:
    target = out_dir / "index.sqlite3"
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    private_dir(out_dir)
    temporary.unlink(missing_ok=True)
    db = sqlite3.connect(temporary)
    db.executescript(SCHEMA_SQL)
    conversations = messages = 0
    for record in read_corpus(out_dir):
        cursor = db.execute(
            "INSERT INTO conversations(conversation_id, source, title, created_at, updated_at, subagent, origin) VALUES(?,?,?,?,?,?,?)",
            (record["conversation_id"], record["source"], record.get("title"), record.get("created_at"),
             record.get("updated_at"), int(bool(record.get("subagent"))), record.get("origin")))
        conversations += 1
        for event in record["events"]:
            if event["role"] not in roles or event["kind"] not in ("message", "command"):
                continue
            row = db.execute("INSERT INTO messages(conversation, role, kind, timestamp, text) VALUES(?,?,?,?,?)",
                             (cursor.lastrowid, event["role"], event["kind"], event.get("timestamp"), event["text"]))
            db.execute("INSERT INTO messages_fts(rowid, text) VALUES(?,?)", (row.lastrowid, event["text"]))
            messages += 1
    db.execute("INSERT INTO meta VALUES('roles', ?)", (",".join(roles),))
    db.commit()
    db.close()
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
    return {"index": str(target), "conversations": conversations, "messages": messages, "roles": list(roles)}


def search(out_dir: Path, query: str, limit: int = 10) -> list[dict]:
    db = sqlite3.connect(f"file:{out_dir / 'index.sqlite3'}?mode=ro", uri=True)
    terms = [t for t in query.replace('"', " ").split() if t]
    if not terms:
        return []
    match = " ".join(f'"{t}"' for t in terms)
    rows = db.execute(
        """SELECT c.source, c.title, m.timestamp, m.role, snippet(messages_fts, 0, '[', ']', ' … ', 18)
           FROM messages_fts JOIN messages m ON m.rowid = messages_fts.rowid
           JOIN conversations c ON c.rowid = m.conversation
           WHERE messages_fts MATCH ? ORDER BY bm25(messages_fts) LIMIT ?""", (match, limit)).fetchall()
    return [{"source": s, "title": t, "timestamp": ts, "role": r, "snippet": snip} for s, t, ts, r, snip in rows]
