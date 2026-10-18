"""A small database shaped like hstry's (https://github.com/byteowlz/hstry).

Columns follow hstry-core migrations 001 (sources, conversations, messages,
tool_calls), 006 (sender_json), 007 (harness), and 012 (parent_conversation_id,
fork_type). Only the columns ai-history-kit reads are created.
"""

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE sources (id TEXT PRIMARY KEY, adapter TEXT NOT NULL, path TEXT);
CREATE TABLE conversations (id TEXT PRIMARY KEY, source_id TEXT NOT NULL, external_id TEXT, title TEXT,
    created_at INTEGER NOT NULL, updated_at INTEGER, harness TEXT, parent_conversation_id TEXT, fork_type TEXT);
CREATE TABLE messages (id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, idx INTEGER NOT NULL, role TEXT NOT NULL,
    content TEXT NOT NULL, parts_json JSON NOT NULL DEFAULT '[]', created_at INTEGER, sender_json TEXT, harness TEXT);
CREATE TABLE tool_calls (id TEXT PRIMARY KEY, message_id TEXT NOT NULL, tool_name TEXT NOT NULL, input JSON, output TEXT);
"""


def make_hstry_db(path: Path, fake_secret: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO sources VALUES (?,?,?)", [
        ("src-cursor", "cursor", "/x/cursor"), ("src-cc", "claude-code", "/x/claude"), ("src-oq", "opencode", "/x/oc")])
    conversations = [
        ("c-cursor", "src-cursor", "cur-1", "Editor setup", 1767225600, 1767225700, "cursor", None, None),
        ("c-dup", "src-cc", "sess-1", "already on disk", 1767225600, None, "claude", None, None),
        ("c-old", "src-cc", "old-session", "deleted from disk since", 1735689600, None, "claude", None, None),
        ("c-thread", "src-oq", "oq-2", "worker", 1767225600, None, "opencode", "c-cursor", "thread"),
    ]
    db.executemany("INSERT INTO conversations VALUES (?,?,?,?,?,?,?,?,?)", conversations)
    messages = [
        ("m1", "c-cursor", 0, "user", "I prefer vim keybindings.", "[]", 1767225601),
        ("m2", "c-cursor", 1, "user", "<system-reminder>Follow the project rules.</system-reminder>", "[]", 1767225602),
        ("m3", "c-cursor", 2, "assistant", "Done.", "[]", 1767225603),
        ("m4", "c-cursor", 3, "user", "", json.dumps([{"type": "text", "id": "p1", "text": "my token=" + fake_secret}]), 1767225604),
        ("m5", "c-dup", 0, "user", "I'm building a kite.", "[]", 1767225601),
        ("m6", "c-old", 0, "user", "I'm building a time machine.", "[]", 1735689601),
        ("m7", "c-thread", 0, "user", "I'm building a rocket.", "[]", 1767225601),
    ]
    db.executemany("INSERT INTO messages (id, conversation_id, idx, role, content, parts_json, created_at) VALUES (?,?,?,?,?,?,?)", messages)
    db.execute("INSERT INTO tool_calls VALUES ('t1', 'm3', 'edit_file', ?, 'ok')", (json.dumps({"path": "a.py"}),))
    db.commit()
    db.close()
    return path
