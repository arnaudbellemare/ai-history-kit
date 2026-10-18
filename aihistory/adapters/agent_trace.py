"""agent-trace stores (https://github.com/Siddhant-K-code/agent-trace, the `agent-strace` tool).

agent-strace hooks record sessions from Claude Code, Codex, Gemini CLI, Cursor, Copilot CLI and
others into `<project>/.agent-traces/<session>/`: a `meta.json` and an `events.ndjson` of
user_prompt, assistant_response, tool_call, tool_result, file_*, error and other events. Each
event line carries `prev_hash`, the SHA-256 of the previous stored line, so edits and gaps show.

The kit reads these files directly and never runs agent-strace. Attribution follows the kit's
rules: prompts that look like injected harness text become `harness`, and prompts of a session
with a parent session become `agent`. Hook traces keep the first 16 characters of the agent's
own session id, which is how duplicates of native Claude Code and Codex sessions are found.
Traces carry less than the native transcripts (no thinking; tool output cut at 1,000 chars).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import clip
from .claude_code import HARNESS_TEXT
from .codex import INJECTED
from ..util import iso

STORE_NAME = ".agent-traces"
EVENTS = "events.ndjson"
NATIVE_ID_PREFIX = 16  # agent-strace hooks keep this many characters of the agent's session id


def event_hash(line: str) -> str:
    """The prev_hash the next event carries: SHA-256 of this stored line, without its newline."""
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def session_files(store: Path) -> list[Path]:
    """Every session's events file; AGENT_STRACE_WORKSPACE nests sessions under workspaces/<id>/."""
    if not store.is_dir():
        return []
    return sorted(p for p in store.rglob(EVENTS) if p.is_file() and not p.is_symlink() and len(p.relative_to(store).parts) <= 4)


def open_store(path: Path) -> Path:
    """The store under `path` (a store itself or a folder holding .agent-traces); ValueError if none."""
    for store in (path, path / STORE_NAME):
        if session_files(store):
            return store
    raise ValueError(f"no agent-trace sessions in {path} (expected .agent-traces/<session>/{EVENTS})")


def _stored_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        return [line.strip() for line in handle if line.strip()]


def verify_chain(path: Path) -> dict:
    """Check every prev_hash. A store whose events carry no hashes at all is `unchained`, not broken."""
    lines = _stored_lines(path)
    previous, hashed = None, False
    for number, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return {"status": "broken", "events": len(lines), "line": number, "reason": "line is not JSON"}
        claimed = event.get("prev_hash") or ""
        hashed = hashed or bool(claimed)
        expected = "" if previous is None else event_hash(previous)
        if claimed != expected and (claimed or hashed):
            reason = "first event has a prev_hash" if previous is None else f"prev_hash does not match line {number - 1}"
            return {"status": "broken", "events": len(lines), "line": number, "reason": reason}
        previous = line
    return {"status": "ok" if hashed or len(lines) < 2 else "unchained", "events": len(lines)}


def session_meta(folder: Path) -> dict:
    try:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _text(data: dict, *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _payload(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _agent(meta: dict, events: list[dict]) -> str:
    start = next((e.get("data") or {} for e in events if e.get("event_type") == "session_start"), {})
    name = meta.get("agent_name") or start.get("agent_name") or start.get("provider") or "unknown"
    return "-".join(str(name).lower().split()) or "unknown"


def _native(agent: str) -> str | None:
    if "claude" in agent:
        return "claude-code"
    if "codex" in agent:
        return "codex"
    return None


def read_session(path: Path, store: Path) -> dict | None:
    lines = _stored_lines(path)
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    meta = session_meta(path.parent)
    agent = _agent(meta, rows)
    subagent = bool(meta.get("parent_session_id"))
    events, skill_calls = [], []
    for row in rows:
        kind, stamp = row.get("event_type"), iso(row.get("timestamp"))
        data = row.get("data") if isinstance(row.get("data"), dict) else {}
        if kind == "user_prompt":
            text = _text(data, "prompt", "text", "content")
            if not text:
                continue
            role = "harness" if HARNESS_TEXT.match(text) or INJECTED.match(text) else ("agent" if subagent else "user")
            event = {"kind": "message", "role": role, "text": text}
        elif kind in ("assistant_response", "llm_response"):
            text = _text(data, "text", "content", "response")
            if not text:
                continue
            event = {"kind": "message", "role": "assistant", "text": text}
        elif kind == "decision":
            text = _text(data, "reasoning", "decision", "text") or _payload(data)
            event = {"kind": "reasoning", "role": "assistant", "text": text}
        elif kind in ("tool_call", "file_read", "file_write"):
            tool = _text(data, "tool_name", "tool", "name") or kind
            arguments = data.get("arguments", data.get("input"))
            if tool == "Skill" and isinstance(arguments, dict):
                skill_calls.append(str(arguments.get("skill") or "").lstrip("/"))
            body = arguments if arguments is not None else {k: v for k, v in data.items() if k not in ("tool_name", "tool", "name")}
            event = {"kind": "tool_call", "role": "assistant", "tool": tool, "text": clip(_payload(body))}
        elif kind in ("tool_result", "error"):
            text = _text(data, "result", "output", "error") or _payload(data)
            event = {"kind": "tool_result", "role": "tool", "text": clip(("error: " if kind == "error" else "") + text)}
        else:
            continue  # session_start/end and llm_request (a replay of the conversation so far)
        event["timestamp"] = stamp
        events.append(event)
    if not events:
        return None
    stamps = [e["timestamp"] for e in events if e["timestamp"]]
    return {
        "source": f"agent-trace:{agent}",
        "identity": str(meta.get("session_id") or path.parent.name),
        "native": _native(agent),
        "title": None,
        "created_at": iso(meta.get("started_at")) or (stamps[0] if stamps else None),
        "updated_at": iso(meta.get("ended_at")) or (stamps[-1] if stamps else None),
        "subagent": subagent,
        "origin": f"agent-trace:{path.parent.relative_to(store).as_posix()}",
        "events": events,
        "stats": {"skill_calls": skill_calls, "chain": verify_chain(path)},
    }
