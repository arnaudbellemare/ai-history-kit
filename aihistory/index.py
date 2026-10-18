"""A local full-text index over the corpus. By default it holds only text you wrote.

Search tries, in order: every term (stemmed, so "migrations" finds "migration"), then any term,
then substrings ("ailboa" finds "sailboat"), then a spelling correction from the index's own
vocabulary. `show` reads a hit's conversation back from the corpus with every role labelled.
"""

from __future__ import annotations

import difflib
import json
import os
import sqlite3
from pathlib import Path
from urllib.parse import quote as url_quote

from . import OWNER_ROLE
from .normalize import read_corpus_offsets
from .query import Query, parse, quote, to_match
from .util import private_dir

SCHEMA_VERSION = "2"
TRIGRAM = sqlite3.sqlite_version_info >= (3, 34, 0)  # older SQLite falls back to a LIKE scan
SNIPPET_TOKENS = 18
SCHEMA_SQL = """
CREATE TABLE conversations(
    rowid INTEGER PRIMARY KEY, ref TEXT NOT NULL UNIQUE, conversation_id TEXT NOT NULL, source TEXT NOT NULL,
    title TEXT, created_at TEXT, updated_at TEXT, subagent INTEGER NOT NULL, origin TEXT, workspace TEXT,
    corpus_file TEXT NOT NULL, corpus_offset INTEGER NOT NULL);
CREATE TABLE messages(
    rowid INTEGER PRIMARY KEY, conversation INTEGER NOT NULL REFERENCES conversations(rowid),
    seq INTEGER NOT NULL, role TEXT NOT NULL, kind TEXT NOT NULL, timestamp TEXT, text TEXT NOT NULL);
CREATE VIRTUAL TABLE messages_fts USING fts5(text, content='messages', content_rowid='rowid',
    tokenize='porter unicode61 remove_diacritics 2', prefix='2 3');
CREATE VIRTUAL TABLE messages_vocab USING fts5vocab(messages_fts, 'row');
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
"""
TRIGRAM_SQL = "CREATE VIRTUAL TABLE messages_tri USING fts5(text, content='messages', content_rowid='rowid', tokenize='trigram')"


def build(out_dir: Path, roles: tuple[str, ...] = (OWNER_ROLE,)) -> dict:
    target = out_dir / "index.sqlite3"
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    private_dir(out_dir)
    temporary.unlink(missing_ok=True)
    db = sqlite3.connect(temporary)
    db.executescript(SCHEMA_SQL + (TRIGRAM_SQL + ";" if TRIGRAM else ""))
    conversations = messages = 0
    refs: dict[str, int] = {}
    for corpus_file, offset, record in read_corpus_offsets(out_dir):
        # One session split across folders is several records with one conversation_id; each part gets a ref.
        count = refs[record["conversation_id"]] = refs.get(record["conversation_id"], 0) + 1
        ref = record["conversation_id"] if count == 1 else f"{record['conversation_id']}.{count}"
        cursor = db.execute(
            "INSERT INTO conversations(ref, conversation_id, source, title, created_at, updated_at, subagent, origin, "
            "workspace, corpus_file, corpus_offset) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (ref, record["conversation_id"], record["source"], record.get("title"), record.get("created_at"),
             record.get("updated_at"), int(bool(record.get("subagent"))), record.get("origin"), record.get("workspace"),
             corpus_file, offset))
        conversations += 1
        for seq, event in enumerate(record["events"]):
            if event["role"] not in roles or event["kind"] not in ("message", "command"):
                continue
            row = db.execute("INSERT INTO messages(conversation, seq, role, kind, timestamp, text) VALUES(?,?,?,?,?,?)",
                             (cursor.lastrowid, seq, event["role"], event["kind"], event.get("timestamp"), event["text"]))
            db.execute("INSERT INTO messages_fts(rowid, text) VALUES(?,?)", (row.lastrowid, event["text"]))
            if TRIGRAM:
                db.execute("INSERT INTO messages_tri(rowid, text) VALUES(?,?)", (row.lastrowid, event["text"]))
            messages += 1
    db.executemany("INSERT INTO meta VALUES(?, ?)", [("roles", ",".join(roles)), ("schema_version", SCHEMA_VERSION),
                                                     ("trigram", str(int(TRIGRAM)))])
    db.commit()
    db.close()
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
    return {"index": str(target), "conversations": conversations, "messages": messages, "roles": list(roles)}


def _open(out_dir: Path) -> sqlite3.Connection:
    path = out_dir / "index.sqlite3"
    if not path.is_file():
        raise ValueError(f"no search index in {out_dir}; build it with `aihistory index`")
    db = sqlite3.connect(f"file:{url_quote(str(path.resolve()))}?mode=ro", uri=True)
    try:
        version = db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    except sqlite3.DatabaseError:
        version = None
    if not version or version[0] != SCHEMA_VERSION:
        db.close()
        raise ValueError("the search index was built by an older aihistory; rebuild it with `aihistory index`")
    return db


def _filter_sql(filters: dict) -> tuple[str, list]:
    sql, params = [], []
    if filters.get("source"):
        sql.append("(c.source = ? OR c.source LIKE ?)")
        params += [filters["source"], filters["source"] + ":%"]  # source:hstry matches hstry:cursor
    if filters.get("role"):
        sql.append("m.role = ?")
        params.append(filters["role"])
    if filters.get("workspace"):
        sql.append("c.workspace LIKE ?")
        params.append(f"%{filters['workspace']}%")
    if filters.get("after"):
        sql.append("m.timestamp >= ?")
        params.append(filters["after"])
    if filters.get("before"):
        sql.append("m.timestamp < ?")
        params.append(filters["before"])
    return "".join(f" AND {clause}" for clause in sql), params


HIT_COLUMNS = "c.ref, c.conversation_id, m.seq, c.source, c.title, m.timestamp, m.role, c.workspace"


def _hit(row, snippet: str) -> dict:
    ref, conversation_id, seq, source, title, timestamp, role, workspace = row
    return {"conversation": ref, "conversation_id": conversation_id, "seq": seq, "source": source, "title": title,
            "timestamp": timestamp, "role": role, "workspace": workspace, "snippet": snippet}


def _match(db, table: str, expression: str, filters: dict, limit: int) -> list[dict]:
    where, params = _filter_sql(filters)
    rows = db.execute(
        f"SELECT {HIT_COLUMNS}, snippet({table}, 0, '[', ']', ' … ', {SNIPPET_TOKENS}) FROM {table} "
        f"JOIN messages m ON m.rowid = {table}.rowid JOIN conversations c ON c.rowid = m.conversation "
        f"WHERE {table} MATCH ?{where} ORDER BY bm25({table}), m.timestamp DESC LIMIT ?",
        (expression, *params, limit)).fetchall()
    return [_hit(row[:-1], row[-1]) for row in rows]


def _excerpt(text: str, term: str, width: int = 90) -> str:
    at = text.lower().find(term.lower())
    if at < 0:
        return text[:width]
    start = max(0, at - width // 2)
    piece = text[start:at] + "[" + text[at:at + len(term)] + "]" + text[at + len(term):at + len(term) + width // 2]
    return ("… " if start else "") + piece.replace("\n", " ")


def _substring(db, query: Query, filters: dict, limit: int) -> list[dict]:
    terms = query.terms
    if TRIGRAM and all(len(t) >= 3 for t in terms):
        expression = " AND ".join(quote(t) for t in terms) + "".join(f" NOT {quote(t)}" for t in query.excluded)
        return _match(db, "messages_tri", expression, filters, limit)
    where, params = _filter_sql(filters)
    like = "".join(" AND m.text LIKE ? ESCAPE '\\'" for _ in terms) + "".join(" AND m.text NOT LIKE ? ESCAPE '\\'" for _ in query.excluded)
    escaped = ["%" + t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%" for t in terms + query.excluded]
    rows = db.execute(
        f"SELECT {HIT_COLUMNS}, m.text FROM messages m JOIN conversations c ON c.rowid = m.conversation "
        f"WHERE 1 = 1{like}{where} ORDER BY m.timestamp DESC LIMIT ?", (*escaped, *params, limit)).fetchall()
    return [_hit(row[:-1], _excerpt(row[-1], terms[0])) for row in rows]


def _corrections(db, query: Query) -> dict[str, str]:
    fixes = {}
    for term in query.terms:
        if " " in term or db.execute("SELECT 1 FROM messages_fts WHERE messages_fts MATCH ? LIMIT 1", (quote(term),)).fetchone():
            continue
        lowered = term.lower()
        nearby = [row[0] for row in db.execute(
            "SELECT term FROM messages_vocab WHERE length(term) BETWEEN ? AND ?", (len(lowered) - 2, len(lowered) + 2))]
        if close := difflib.get_close_matches(lowered, nearby, n=1, cutoff=0.75):
            fixes[term] = close[0]
    return fixes


def query(out_dir: Path, text: str, mode: str = "auto", limit: int = 10, **filters) -> dict:
    """Ranked hits for `text`. Modes: auto (fall back step by step), all, any, substring."""
    parsed = parse(text)
    merged = {**parsed.filters, **{k: v for k, v in filters.items() if v}}
    db = _open(out_dir)
    try:
        if not parsed.groups:
            return {"mode_used": None, "corrections": {}, "hits": []}
        steps = ["all", "any", "substring", "corrected"] if mode == "auto" else [mode]
        for step in steps:
            corrections = {}
            if step == "any" and mode == "auto" and len(parsed.groups) < 2:
                continue
            if step == "substring":
                hits = _substring(db, parsed, merged, limit)
            elif step == "corrected":
                corrections = _corrections(db, parsed)
                if not corrections:
                    continue
                fixed = Query([[(corrections.get(t, t), p) for t, p in group] for group in parsed.groups], parsed.excluded)
                hits = _match(db, "messages_fts", to_match(fixed), merged, limit)
            else:
                hits = _match(db, "messages_fts", to_match(parsed, step), merged, limit)
            if hits:
                return {"mode_used": step, "corrections": corrections, "hits": hits}
        return {"mode_used": None, "corrections": {}, "hits": []}
    finally:
        db.close()


def search(out_dir: Path, text: str, limit: int = 10) -> list[dict]:
    return query(out_dir, text, limit=limit)["hits"]


def show(out_dir: Path, ref: str, at: int | None = None, context: int | None = None) -> dict:
    """One indexed conversation, read back from the corpus; `at`/`context` keep only nearby events."""
    db = _open(out_dir)
    try:
        row = db.execute("SELECT conversation_id, source, title, corpus_file, corpus_offset FROM conversations "
                         "WHERE ref = ?", (ref,)).fetchone()
    finally:
        db.close()
    if not row:
        raise ValueError(f"no conversation {ref} in the index (use the `conversation` value from a search hit)")
    conversation_id, source, title, corpus_file, offset = row
    with open(out_dir / "corpus" / corpus_file, "rb") as handle:
        handle.seek(offset)
        record = json.loads(handle.readline())
    events = []
    for seq, event in enumerate(record["events"]):
        if at is not None and context is not None and abs(seq - at) > context:
            continue
        events.append({"seq": seq, "hit": seq == at, **{k: event.get(k) for k in ("kind", "role", "tool", "timestamp", "text")}})
    return {"conversation": ref, "conversation_id": conversation_id, "source": source, "title": title, "events": events}
