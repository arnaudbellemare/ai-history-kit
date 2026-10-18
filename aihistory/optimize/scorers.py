"""Score agent runs from kit events, whichever agent produced them.

A skill counts as loaded when the agent calls Claude Code's Skill tool for it, or reads its
`skills/<name>/SKILL.md` (how Codex and other agents load skills). The same function finds the
skill calls in your history (`mine`) and in evaluation rollouts (`eval`).
"""

from __future__ import annotations

import re

SKILL_ARGUMENT = re.compile(r'"skill"\s*:\s*"/?([^"]+)"')
SKILL_FILE = re.compile(r"skills[/\\]+([A-Za-z0-9][A-Za-z0-9._-]*)[/\\]+SKILL\.md")


def skills_loaded(events) -> list[str]:
    names = []
    for event in events:
        if event.get("kind") != "tool_call":
            continue
        pattern = SKILL_ARGUMENT if event.get("tool") == "Skill" else SKILL_FILE
        match = pattern.search(event.get("text") or "")
        if match and match.group(1) not in names:
            names.append(match.group(1))
    return names


def trigger_score(gold, loaded, targets) -> float:
    """1 when the agent loads only gold target skills (at least one), or no target when gold is empty."""
    called, gold = {s for s in loaded if s in targets}, set(gold)
    if not gold:
        return 0.0 if called else 1.0
    return 1.0 if called and called <= gold else 0.0


def side_info(example: dict, events, loaded) -> dict:
    """What a reflection model sees about one run: never the other prompts, never the label of others."""
    reasoning = " ".join(e["text"] for e in events if e.get("kind") == "reasoning")
    replies = [e["text"] for e in events if e.get("kind") == "message" and e.get("role") == "assistant"]
    tools = [e.get("tool") for e in events if e.get("kind") == "tool_call" and e.get("tool") != "Skill"]
    return {"Prompt": example["prompt"][:400], "Expected": example["gold"] or "no target skill",
            "Skills loaded": loaded, "Other tools": tools[:4], "Agent reasoning": reasoning[:600],
            "Agent final text": (replies[-1] if replies else "")[:300]}
