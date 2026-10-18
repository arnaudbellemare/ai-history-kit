"""Propose personal memories from what you wrote, for you to review.

Only `user` messages from non-subagent conversations count, so agent briefs,
injected skill bodies, and harness text can't become "your" memories. Within a
message, assistant turns quoted from a pasted transcript are skipped, and
"Remember: ..." counts only as a message's first sentence (skill bodies end with
"Remember:" lines addressed to a model). Nothing is accepted automatically: the
output is a REVIEW.md with checkboxes.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from . import OWNER_ROLE
from .normalize import read_corpus
from .redact import MARKER
from .util import sha256_text, write_private_json, write_private_text

SENTENCE = re.compile(r"(?<=[.!?。！？])\s+|[\r\n]+")
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
QUOTED_OTHER = re.compile(r"(?i)^\s*(?:\[\d+\]\s*)?(?:assistant|tool|system|claude|codex|chatgpt|gpt)\s*:")
QUOTED_OWNER = re.compile(r"(?i)^\s*(?:\[\d+\]\s*)?(?:user|human|me)\s*:")
INJECTION = re.compile(
    r"(?i)\b(?:ignore|disregard|override)\s+(?:all\s+)?(?:previous|prior|system|developer)\s+(?:instructions?|messages?|prompts?)\b"
    r"|\b(?:reveal|print|show|exfiltrate)\s+(?:the\s+)?(?:system\s+prompt|secrets?|credentials?)\b")
HIGH = re.compile(r"(?i)\b(?:address|bank|credit card|diagnosis|health|medical|passport|salary|ssn|social security|tax id|password|api key)\b")
RULES = (  # (rule, kind, fixed key or None, confidence, pattern)
    ("name", "fact", "identity.name", 0.95, r"(?:my name is|i am called|call me)\s+(.+)"),
    ("company", "fact", "identity.company", 0.9, r"my company is\s+(.+)"),
    ("works_at", "fact", "work.organization", 0.9, r"i work (?:at|for)\s+(.+)"),
    ("role", "fact", "work.role", 0.9, r"i(?:'m| am) (?:a|an|the)\s+(\w[\w -]{2,40}(?:engineer|developer|founder|designer|manager|owner|ceo|cto|student|researcher|consultant|lawyer|doctor))\b.*"),
    ("location", "fact", "identity.location", 0.85, r"i (?:live|am based) in\s+(.+)"),
    ("project", "project", None, 0.85, r"(?:i(?:'m| am)|we(?:'re| are))\s+(?:working on|building|developing|launching)\s+(.+)"),
    ("goal", "fact", "goals.current", 0.85, r"my (?:current |main )?goal is\s+(.+)"),
    ("prefer", "preference", None, 0.9, r"i (?:prefer|like it when|want you to always|always want)\s+(.+)"),
    ("dislike", "preference", None, 0.85, r"i (?:don't like it when|never want)\s+(.+)"),
    ("remember", "preference", "conversation.remember", 0.85, r"(?:remember|note|keep in mind)\s*[:：]\s*(.+)"),
    ("decided", "decision", None, 0.9, r"(?:i|we) (?:decided|chose|have decided) to\s+(.+)"),
    ("skill", "skill", None, 0.85, r"i(?:'m| am) (?:skilled|experienced|proficient|good) (?:at|in|with)\s+(.+)"),
    ("knows", "skill", None, 0.8, r"i (?:know how to|learned how to)\s+(.+)"),
    ("habit", "working_pattern", None, 0.85, r"i (?:always|usually|typically|tend to)\s+(.+)"),
)
COMPILED = [(r, k, key, c, re.compile(r"(?i)^" + p + r"$")) for r, k, key, c, p in RULES]
# Working rules ("never push to main without asking") and requests ("I want ...") are only
# memories when you repeat them across separate conversations; one-off task requests are not.
RECURRING = re.compile(r"(?i)^(?:please\s+)?(always|never|don'?t|do not|stop|avoid|make sure (?:to|you)|i want|i need you to)\b")
RECURRING_MIN_CONVERSATIONS = 3
FIRST_SENTENCE_ONLY = {"remember"}
STOP = {"and", "are", "for", "from", "have", "the", "this", "that", "use", "using", "with", "into", "about"}


def owner_text(text: str) -> str:
    """Drop quoted non-owner turns from a pasted transcript."""
    kept, quoting = [], False
    for line in text.splitlines():
        if QUOTED_OTHER.match(line):
            quoting = True
        elif QUOTED_OWNER.match(line):
            quoting = False
        if not quoting:
            kept.append(line)
    return "\n".join(kept)


def sentences(text: str):
    for part in SENTENCE.split(text):
        part = BULLET.sub("", part).strip()
        if 4 <= len(part) <= 400:
            yield part


def _clean(value: str) -> str:
    return value.strip().strip("\"'“”‘’ ").rstrip(".!?。！？ ").strip()


def _topic_key(kind: str, content: str) -> str:
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", content.lower()).split() if len(w) >= 3 and w not in STOP]
    return f"{kind}.{'.'.join(words[:3]) or 'general'}.{sha256_text(content)[:6]}"


def match(sentence: str, first: bool):
    if MARKER.search(sentence) or INJECTION.search(sentence) or sentence.lstrip().startswith(("<", "/", "!", "```")):
        return None
    for rule, kind, key, confidence, pattern in COMPILED:
        found = pattern.match(sentence.strip())
        if not found or (rule in FIRST_SENTENCE_ONLY and not first):
            continue
        content = _clean(found.group(1))
        if len(content) < 2 or len(content) > 240:
            return None
        return {"kind": kind, "key": key or _topic_key(kind, content), "content": content, "rule": rule,
                "confidence": confidence, "sensitivity": "high" if HIGH.search(content) else "normal"}
    return None


def _recurring(sentence: str, recurring: dict, record: dict, event: dict) -> None:
    opener = RECURRING.match(sentence)
    if not opener or not 15 <= len(sentence) <= 200 or MARKER.search(sentence) or INJECTION.search(sentence):
        return
    norm = " ".join(re.sub(r"[^a-z0-9 ]+", "", sentence.lower()).split())
    entry = recurring.setdefault(norm, {"text": sentence, "opener": opener.group(1).lower(), "occurrences": 0,
                                        "conversations": set(), "evidence": []})
    entry["occurrences"] += 1
    if record["conversation_id"] not in entry["conversations"] and len(entry["evidence"]) < 3:
        entry["evidence"].append({"source": record["source"], "when": (event.get("timestamp") or "")[:10], "excerpt": sentence[:300]})
    entry["conversations"].add(record["conversation_id"])


def propose(out_dir: Path) -> dict:
    found, recurring = {}, {}
    considered = 0
    for record in read_corpus(out_dir):
        if record.get("subagent") or record.get("headless"):
            continue
        for event in record["events"]:
            if event["role"] != OWNER_ROLE or event["kind"] != "message":
                continue
            considered += 1
            for index, sentence in enumerate(sentences(owner_text(event["text"]))):
                candidate = match(sentence, index == 0)
                if not candidate:
                    _recurring(sentence, recurring, record, event)
                    continue
                identity = (candidate["kind"], candidate["key"], candidate["content"].lower())
                entry = found.setdefault(identity, {**candidate, "occurrences": 0, "conversations": set(), "evidence": []})
                entry["occurrences"] += 1
                entry["conversations"].add(record["conversation_id"])
                if len(entry["evidence"]) < 3:
                    entry["evidence"].append({"source": record["source"], "when": (event.get("timestamp") or "")[:10],
                                              "excerpt": sentence[:300]})
    for entry in recurring.values():
        if len(entry["conversations"]) < RECURRING_MIN_CONVERSATIONS:
            continue
        content = _clean(entry["text"])
        kind = "project" if entry["opener"] in ("i want", "i need you to") else "preference"
        key = _topic_key("interests" if kind == "project" else "rules", content)
        found[(kind, key, content.lower())] = {"kind": kind, "key": key, "content": content, "rule": "recurring",
                                               "confidence": 0.6, "sensitivity": "high" if HIGH.search(content) else "normal",
                                               "occurrences": entry["occurrences"], "conversations": entry["conversations"],
                                               "evidence": entry["evidence"]}
    candidates = []
    for entry in found.values():
        entry["conversations"] = len(entry["conversations"])
        entry["id"] = "mem_" + sha256_text(f"{entry['kind']}\0{entry['key']}\0{entry['content']}")[:16]
        candidates.append(entry)
    by_key = defaultdict(list)
    for c in candidates:
        by_key[(c["kind"], c["key"])].append(c)
    for group in by_key.values():
        for c in group:
            c["conflicts_with"] = [o["id"] for o in group if o is not c]
    candidates.sort(key=lambda c: (c["kind"], -c["conversations"], -c["occurrences"], c["content"]))
    write_private_json(out_dir / "memories" / "candidates.json",
                       {"status": "pending_owner_review", "messages_considered": considered, "candidates": candidates})
    write_private_text(out_dir / "memories" / "REVIEW.md", review_markdown(candidates))
    return {"messages_considered": considered, "candidates": len(candidates),
            "review": str(out_dir / "memories" / "REVIEW.md")}


KIND_TITLES = {"fact": "Facts", "preference": "Preferences", "project": "Projects", "decision": "Decisions",
               "skill": "Skills", "working_pattern": "Working patterns"}


def review_markdown(candidates: list[dict]) -> str:
    lines = ["# Memory candidates — review before anything is saved", "",
             "Tick `[x]` to accept a memory. Edit nothing else on the line; the id after the box is how it's found.",
             "Every line comes from something you wrote. Check that each one is still true.", ""]
    for kind, title in KIND_TITLES.items():
        group = [c for c in candidates if c["kind"] == kind]
        if not group:
            continue
        lines += [f"## {title} ({len(group)})", ""]
        for c in group:
            flags = []
            if c["conflicts_with"]:
                flags.append("conflicts with another value for the same key")
            if c["sensitivity"] == "high":
                flags.append("sensitive")
            flag = f" — ⚠ {'; '.join(flags)}" if flags else ""
            lines.append(f"- [ ] `{c['id']}` **{c['content']}** ({c['conversations']} conversation(s)){flag}")
            for e in c["evidence"][:2]:
                lines.append(f"  > {e['excerpt']} — {e['source']}, {e['when'] or 'undated'}")
        lines.append("")
    if not candidates:
        lines.append("No explicit statements about yourself were found (sentences like \"I prefer …\", \"I'm building …\").")
    return "\n".join(lines) + "\n"


def accepted_ids(review_path: Path) -> list[str]:
    return re.findall(r"^- \[[xX]\] `(mem_[0-9a-f]+)`", review_path.read_text(encoding="utf-8"), re.MULTILINE)
