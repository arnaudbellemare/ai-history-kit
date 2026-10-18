"""Claude Code transcripts (~/.claude/projects/<project>/<session>.jsonl).

Claude Code stores text the owner never typed as "user" rows: skill bodies and
caveats (isMeta), compaction summaries, background-task notifications, system
reminders, and local command output. Those become role `harness`. In a subagent
transcript (agent-*.jsonl or isSidechain rows) the "user" turns are the parent
agent's brief, so they become role `agent`. Only text the owner typed stays `user`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import clip
from ..util import json_lines

HARNESS_TEXT = re.compile(
    r"\s*(?:<(?:task-notification|system-reminder|local-command-stdout|local-command-stderr|"
    r"local-command-caveat|bash-stdout|bash-stderr|side-sessions-event|ci-monitor-event)>"
    r"|\[Request interrupted by user|Caveat: The messages below were generated)"
)
SLASH = re.compile(r"<command-name>/?([^<]+)</command-name>")
SLASH_ARGS = re.compile(r"<command-args>(.*?)</command-args>", re.DOTALL)
SHELL_INPUT = re.compile(r"<bash-input>(.*?)</bash-input>", re.DOTALL)


def _blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b if isinstance(b, dict) else {"type": "text", "text": str(b)} for b in content or []]


def _user_event(row, text, subagent):
    """Classify one text block of a "user" row."""
    if row.get("isMeta") or row.get("isCompactSummary") or HARNESS_TEXT.match(text):
        return {"kind": "message", "role": "harness", "text": text}
    if subagent:
        return {"kind": "message", "role": "agent", "text": text}
    slash = SLASH.search(text)
    if slash:
        args = SLASH_ARGS.search(text)
        return {"kind": "command", "role": "user", "text": f"/{slash.group(1).strip()} {args.group(1).strip() if args else ''}".strip()}
    shell = SHELL_INPUT.search(text)
    if shell:
        return {"kind": "command", "role": "user", "text": "!" + shell.group(1).strip()}
    return {"kind": "message", "role": "user", "text": text}


def _tool_result_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict):
            parts.append(block.get("text") or ("[image]" if block.get("type") == "image" else ""))
    return "\n".join(p for p in parts if p)


def read_session(path: Path, root: Path) -> dict | None:
    subagent = path.name.startswith("agent-") or "subagents" in path.parts
    events, identity, title = [], None, None
    created = updated = None
    entrypoints = {}
    stats = {"thinking_blocks": 0, "thinking_with_text": 0, "skills_offered": 0, "skill_calls": []}
    for row in json_lines(path):
        kind = row.get("type")
        if kind in ("custom-title", "summary"):
            title = row.get("customTitle") or row.get("summary") or title
            continue
        attachment = row.get("attachment") or {}
        if attachment.get("type") == "skill_listing":
            stats["skills_offered"] = max(stats["skills_offered"], len(attachment.get("names") or []))
            continue
        if kind not in ("user", "assistant"):
            continue
        identity = identity or row.get("sessionId")
        subagent = subagent or bool(row.get("isSidechain"))
        if row.get("entrypoint"):
            entrypoints[row["entrypoint"]] = entrypoints.get(row["entrypoint"], 0) + 1
        stamp = row.get("timestamp")
        created = created or stamp
        updated = stamp or updated
        message = row.get("message") or {}
        for block in _blocks(message.get("content")):
            btype = block.get("type") or "text"
            if btype == "text":
                text = block.get("text") or ""
                if not text.strip():
                    continue
                event = _user_event(row, text, subagent) if kind == "user" else {"kind": "message", "role": "assistant", "text": text}
            elif btype == "thinking":
                stats["thinking_blocks"] += 1
                if not (block.get("thinking") or "").strip():
                    continue  # Claude 4.7+/5.x store an empty, signed block by default
                stats["thinking_with_text"] += 1
                event = {"kind": "reasoning", "role": "assistant", "text": block["thinking"]}
            elif btype == "tool_use":
                name = block.get("name") or "tool"
                payload = block.get("input") or {}
                if name == "Skill":
                    stats["skill_calls"].append(str(payload.get("skill") or "").lstrip("/"))
                event = {"kind": "tool_call", "role": "assistant", "tool": name, "text": clip(json.dumps(payload, ensure_ascii=False))}
            elif btype == "tool_result":
                event = {"kind": "tool_result", "role": "tool", "text": clip(_tool_result_text(block.get("content")))}
            elif btype == "image":
                event = {"kind": "attachment", "role": "user" if kind == "user" else "assistant", "text": "[image]"}
            else:
                continue
            event["timestamp"] = stamp
            events.append(event)
    if not events:
        return None
    return {
        "source": "claude-code",
        "identity": identity or path.stem,
        "title": title,
        "created_at": created,
        "updated_at": updated,
        "subagent": subagent,
        # `claude -p` / SDK runs are usually scripts; their prompts are kept and searchable
        # but don't become memories.
        "headless": bool(entrypoints) and max(entrypoints, key=entrypoints.get).startswith("sdk"),
        "origin": path.relative_to(root).as_posix(),
        "events": events,
        "stats": stats,
    }
