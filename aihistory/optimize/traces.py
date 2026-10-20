"""Write agent runs as agent-trace sessions, so every evaluation rollout can be replayed and checked.

The layout and hash chain follow agent-strace: `<store>/<session>/meta.json` and `events.ndjson`,
where each line is compact JSON and `prev_hash` is the SHA-256 of the previous stored line.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from ..adapters import clip
from ..adapters.agent_trace import EVENTS, event_hash
from ..util import append_private_line, private_dir, write_private_text

FIELDS = ("event_type", "timestamp", "event_id", "session_id", "parent_id", "duration_ms", "data",
          "prev_hash", "redacted", "tenant_id")


class TraceWriter:
    def __init__(self, store: Path, session_id: str, agent_name: str, parent: str | None = None):
        self.folder = private_dir(store / session_id)
        self.events_path = self.folder / EVENTS
        self.session_id, self.agent_name, self.parent = session_id, agent_name, parent
        self.started, self.previous, self.counts = time.time(), None, {}
        self._write_meta()

    def _write_meta(self, **extra) -> None:
        meta = {"session_id": self.session_id, "started_at": self.started, "agent_name": self.agent_name,
                "parent_session_id": self.parent, **extra}
        write_private_text(self.folder / "meta.json", json.dumps({k: v for k, v in meta.items() if v is not None}))

    def event(self, kind: str, data: dict, duration_ms: int | None = None) -> None:
        event = {"event_type": kind, "timestamp": time.time(), "event_id": uuid.uuid4().hex[:12],
                 "session_id": self.session_id, "duration_ms": duration_ms, "data": data,
                 "prev_hash": event_hash(self.previous) if self.previous is not None else ""}
        line = json.dumps({k: event[k] for k in FIELDS if event.get(k) not in (None, "")}, separators=(",", ":"))
        append_private_line(self.events_path, line)
        self.previous = line
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def annotate(self, row: dict) -> None:
        append_private_line(self.folder / "annotations.jsonl", json.dumps(row, ensure_ascii=False, sort_keys=True))

    def close(self, **extra) -> None:
        # Upstream SessionMeta rejects unknown keys such as exit_code. Keep kit-only
        # run details alongside scores in annotations, leaving metadata replayable.
        self._write_meta(ended_at=time.time(), tool_calls=self.counts.get("tool_call", 0))
        if extra:
            self.annotate({"run_result": extra})


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text") or "" for b in content or [] if isinstance(b, dict))


def from_claude_stream(lines, writer: TraceWriter) -> dict:
    """Convert `claude -p --output-format stream-json --verbose` output; returns cost and model."""
    summary = {"cost_usd": None, "model": None, "subtype": None}
    for line in lines:
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            summary["model"] = event.get("model")
            writer.event("session_start", {"provider": "claude-code", "model": event.get("model"),
                                           "skills_offered": len(event.get("skills") or [])})
        elif kind == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                btype = block.get("type")
                if btype == "tool_use":
                    writer.event("tool_call", {"tool_name": block.get("name"), "arguments": block.get("input") or {},
                                               "tool_use_id": block.get("id")})
                elif btype == "thinking" and (block.get("thinking") or "").strip():
                    writer.event("decision", {"reasoning": block["thinking"]})
                elif btype == "text" and (block.get("text") or "").strip():
                    writer.event("assistant_response", {"text": block["text"]})
        elif kind == "user":
            for block in (event.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    writer.event("tool_result", {"tool_use_id": block.get("tool_use_id"),
                                                 "result": clip(_result_text(block.get("content")), 1000)})
        elif kind == "result":
            summary["cost_usd"], summary["subtype"] = event.get("total_cost_usd"), event.get("subtype")
            writer.event("session_end", {"subtype": event.get("subtype"), "cost_usd": event.get("total_cost_usd"),
                                         "num_turns": event.get("num_turns")}, duration_ms=event.get("duration_ms"))
    return summary


def from_codex_jsonl(lines, writer: TraceWriter) -> dict:
    """Convert `codex exec --json` output (thread/turn/item events); returns token usage."""
    summary = {"usage": None}
    for line in lines:
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        kind = event.get("type")
        if kind == "thread.started":
            writer.event("session_start", {"provider": "codex", "thread_id": event.get("thread_id")})
        elif kind == "item.completed":
            item = event.get("item") or {}
            itype = item.get("type")
            if itype == "agent_message" and (item.get("text") or "").strip():
                writer.event("assistant_response", {"text": item["text"]})
            elif itype == "reasoning" and (item.get("text") or "").strip():
                writer.event("decision", {"reasoning": item["text"]})
            elif itype == "command_execution":
                writer.event("tool_call", {"tool_name": "shell", "arguments": {"command": item.get("command")}})
                writer.event("tool_result", {"tool_name": "shell", "result": clip(str(item.get("aggregated_output") or ""), 1000)})
            elif itype == "file_change":
                writer.event("file_write", {"changes": item.get("changes")})
            elif itype in ("mcp_tool_call", "web_search"):
                writer.event("tool_call", {"tool_name": item.get("tool") or itype,
                                           "arguments": item.get("arguments") or {"query": item.get("query")}})
        elif kind == "turn.completed":
            summary["usage"] = event.get("usage")
            writer.event("session_end", {"usage": event.get("usage")})
        elif kind in ("error", "turn.failed"):
            writer.event("error", {"error": str(event.get("message") or event.get("error") or kind)})
    return summary
