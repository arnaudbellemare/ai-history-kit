"""Synthetic agent-trace (agent-strace) stores: <project>/.agent-traces/<session>/{meta.json, events.ndjson}.

Each event line is compact JSON. Its prev_hash is the SHA-256 of the previous stored line, and
the first event has none, as agent-strace writes them.
"""

import hashlib
import json
from pathlib import Path

FIELDS = ("event_type", "timestamp", "event_id", "session_id", "parent_id", "duration_ms", "data",
          "prev_hash", "redacted", "tenant_id")
NATIVE_CLAUDE_ID = "11111111-2222-3333-4444-555555555555"  # a Claude Code session the native reader also has


def write_session(store: Path, session_id: str, agent: str, events, parent: str | None = None,
                  start: float = 1782900000.0) -> Path:
    folder = store / session_id
    folder.mkdir(parents=True)
    meta = {"session_id": session_id, "started_at": start, "agent_name": agent, "parent_session_id": parent}
    (folder / "meta.json").write_text(json.dumps({k: v for k, v in meta.items() if v is not None}), encoding="utf-8")
    lines, previous = [], ""
    for number, (kind, data) in enumerate(events):
        event = {"event_type": kind, "timestamp": start + number, "event_id": f"{session_id[:4]}{number:08d}",
                 "session_id": session_id, "data": data, "prev_hash": previous}
        line = json.dumps({k: event[k] for k in FIELDS if event.get(k) not in (None, "")}, separators=(",", ":"))
        lines.append(line)
        previous = hashlib.sha256(line.encode("utf-8")).hexdigest()
    (folder / "events.ndjson").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return folder


def make_trace_store(project: Path, fake_secret: str) -> Path:
    store = project / ".agent-traces"
    write_session(store, "gem0000000000001", "gemini-cli", [
        ("session_start", {"provider": "gemini"}),
        ("user_prompt", {"prompt": "I prefer two-space indents. token=" + fake_secret}),
        ("user_prompt", {"prompt": "<system-reminder>Follow the house style.</system-reminder>"}),
        ("tool_call", {"tool_name": "read_file", "arguments": {"path": "README.md"}}),
        ("tool_result", {"tool_name": "read_file", "result": "# readme"}),
        ("assistant_response", {"text": "Done: two spaces it is."}),
        ("session_end", {}),
    ])
    write_session(store, "gem0000000000002", "gemini-cli", [
        ("user_prompt", {"prompt": "I'm building a rocket."}),  # a brief the parent session dispatched
    ], parent="gem0000000000001")
    write_session(store, NATIVE_CLAUDE_ID[:16], "claude-code", [  # hooks keep the first 16 characters
        ("user_prompt", {"prompt": "already in the native transcript"}),
    ])
    write_session(store, "99999999-8888-77", "claude-code", [
        ("user_prompt", {"prompt": "only the trace still has this session"}),
    ])
    return store
