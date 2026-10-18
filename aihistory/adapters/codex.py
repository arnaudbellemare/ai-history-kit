"""Codex rollouts (~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl, and .jsonl.zst).

- Codex compresses older rollouts in place as .jsonl.zst; both are read.
- A thread's identity is its FIRST session_meta row; forked and spawned threads
  replay their parent's session_meta later in the file.
- Subagent threads (session_meta.source has "subagent", which includes the
  approval reviewer) contain prompts another agent wrote: role `agent`.
- Codex injects context into user turns as separate blocks (AGENTS.md
  instructions, environment context, skills, file lists): role `harness`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import clip
from ..util import json_lines

INJECTED = re.compile(
    r"\s*(?:<(?:environment_context|user_instructions|external_codex_apps_open_page|turn_aborted|"
    r"guardian_tool_descriptions|recommended_plugins|codex_internal_context|in-app-browser-context|"
    r"subagent_notification|skill|codex_delegation)\b|# AGENTS\.md instructions|# Files (?:mentioned|pasted))"
)


def _texts(content) -> list[str]:
    if isinstance(content, str):
        return [content]
    out = []
    for block in content or []:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            out.append(block["text"])
        elif isinstance(block, dict) and block.get("type") in ("input_image", "image"):
            out.append("[image]")
    return out


def read_rollout(path: Path, root: Path) -> dict | None:
    events, meta, subagent = [], None, False
    created = updated = None
    seen_user = set()
    for row in json_lines(path):
        payload = row.get("payload") or {}
        stamp = row.get("timestamp")
        created = created or stamp
        updated = stamp or updated
        rtype, ptype = row.get("type"), payload.get("type")
        if rtype == "session_meta":
            if meta is None:
                meta = payload
                source = payload.get("source")
                subagent = isinstance(source, dict) and "subagent" in source
                created = payload.get("timestamp") or created
            continue
        owner_role = "agent" if subagent else "user"
        if rtype == "event_msg" and ptype == "user_message":
            text = payload.get("message") or ""
            if text.strip() and text not in seen_user:
                seen_user.add(text)
                events.append({"kind": "message", "role": owner_role, "text": text, "timestamp": stamp})
        elif rtype == "response_item" and ptype == "message":
            role = payload.get("role") or "assistant"
            for text in _texts(payload.get("content")):
                if not text.strip():
                    continue
                if role == "user":
                    if INJECTED.match(text):
                        events.append({"kind": "message", "role": "harness", "text": text, "timestamp": stamp})
                    elif text not in seen_user:
                        seen_user.add(text)
                        events.append({"kind": "message", "role": owner_role, "text": text, "timestamp": stamp})
                else:
                    events.append({"kind": "message", "role": role, "text": text, "timestamp": stamp})
        elif rtype == "response_item" and ptype in ("function_call", "custom_tool_call", "web_search_call", "local_shell_call"):
            args = payload.get("arguments") or payload.get("input") or payload.get("action") or payload.get("query") or ""
            events.append({"kind": "tool_call", "role": "assistant", "tool": payload.get("name") or ptype,
                           "text": clip(args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)), "timestamp": stamp})
        elif rtype == "response_item" and ptype in ("function_call_output", "custom_tool_call_output", "local_shell_call_output"):
            output = payload.get("output") or ""
            events.append({"kind": "tool_result", "role": "tool",
                           "text": clip(output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)), "timestamp": stamp})
    if not events:
        return None
    return {
        "source": "codex",
        "identity": (meta or {}).get("id") or path.name.split(".jsonl")[0],
        "title": None,
        "created_at": created,
        "updated_at": updated,
        "subagent": subagent,
        "origin": path.relative_to(root).as_posix(),
        "workspace": (meta or {}).get("cwd"),
        "events": events,
        "stats": {},
    }
