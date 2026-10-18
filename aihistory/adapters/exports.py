"""Official exports: ChatGPT and Claude.ai (conversations.json, or the export zip)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from ..util import iso


def _load(path: Path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.endswith("conversations.json"))
            return json.loads(archive.read(name))
    return json.loads(path.read_text(encoding="utf-8"))


def _chatgpt_text(message) -> str:
    content = message.get("content") or {}
    parts = content.get("parts")
    if isinstance(parts, list):
        return "\n".join(p if isinstance(p, str) else "[attachment]" for p in parts if p)
    return content.get("text") or ""


def _chatgpt_branch(conversation):
    """Messages on the active branch (the path the user last saw)."""
    mapping = conversation.get("mapping") or {}
    node_id = conversation.get("current_node")
    if node_id in mapping:
        chain, seen = [], set()
        while node_id and node_id in mapping and node_id not in seen:
            seen.add(node_id)
            chain.append(mapping[node_id])
            node_id = mapping[node_id].get("parent")
        chain.reverse()
    else:
        chain = sorted(mapping.values(), key=lambda n: (n.get("message") or {}).get("create_time") or 0)
    return [n["message"] for n in chain if n.get("message")]


def read_chatgpt(path: Path):
    for conversation in _load(path):
        events = []
        for message in _chatgpt_branch(conversation):
            role = (message.get("author") or {}).get("role") or "unknown"
            text = _chatgpt_text(message)
            if not text.strip():
                continue
            events.append({"kind": "message", "role": role, "text": text, "timestamp": iso(message.get("create_time"))})
        if events:
            yield {"source": "chatgpt-export", "identity": conversation.get("id") or conversation.get("conversation_id"),
                   "title": conversation.get("title"), "created_at": iso(conversation.get("create_time")),
                   "updated_at": iso(conversation.get("update_time")), "subagent": False, "origin": path.name,
                   "events": events, "stats": {}}


def read_claude(path: Path):
    for conversation in _load(path):
        events = []
        for message in conversation.get("chat_messages") or conversation.get("messages") or []:
            sender = message.get("sender") or message.get("role") or "unknown"
            text = message.get("text") or "\n".join(
                b.get("text", "") for b in message.get("content") or [] if isinstance(b, dict))
            if not text.strip():
                continue
            events.append({"kind": "message", "role": "user" if sender in ("human", "user") else sender,
                           "text": text, "timestamp": message.get("created_at")})
        if events:
            yield {"source": "claude-export", "identity": conversation.get("uuid") or conversation.get("id"),
                   "title": conversation.get("name") or conversation.get("title"),
                   "created_at": conversation.get("created_at"), "updated_at": conversation.get("updated_at"),
                   "subagent": False, "origin": path.name, "events": events, "stats": {}}
