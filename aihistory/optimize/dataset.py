"""Evaluation prompts from your own history, labelled by you.

`mine` finds three kinds of candidates in the cleaned corpus, using only text you wrote (not
subagent briefs, headless runs, or injected harness text):
- your prompt just before a target skill was loaded (evidence: what the agent did then);
- your messages that mention a keyword for a target skill;
- a seeded random draw of your other messages, as likely negatives.

You decide the right answer for each in LABEL.md. Nothing is labelled by a model.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from ..memories import owner_text
from ..normalize import read_corpus
from ..util import write_private_json, write_private_text
from .agents import safe_name
from .scorers import skills_loaded

NONE = "none of these"
SECTION = re.compile(r"^## (c\d+)\s*$", re.MULTILINE)
BOX = re.compile(r"^- \[([ xX])\] (.+?)\s*$", re.MULTILINE)
SHOWN_CHARS = 1500


def _owner_messages(out_dir: Path):
    """(record, index, cleaned text) for each message you wrote, in conversations you had."""
    for record in read_corpus(out_dir):
        if record.get("subagent") or record.get("headless"):
            continue
        for index, event in enumerate(record["events"]):
            if event["role"] == "user" and event["kind"] in ("message", "command"):
                text = owner_text(event["text"]).strip()
                if text:
                    yield record, index, text


def mine(out_dir: Path, name: str, skills: list[str], keywords: dict[str, list[str]] | None = None,
         negatives: int = 40, per_skill: int = 40, seed: int = 7, run_dir: Path | None = None) -> Path:
    targets = [safe_name(s) for s in skills]
    keywords = {safe_name(k): [w.lower() for w in v] for k, v in (keywords or {}).items()}
    candidates: dict[str, dict] = {}

    def add(text, record, reason, evidence=None):
        entry = candidates.setdefault(text, {"text": text, "source": record["source"],
                                             "date": (record.get("created_at") or "")[:10], "evidence": [], "reasons": []})
        if evidence and evidence not in entry["evidence"]:
            entry["evidence"].append(evidence)
        if reason not in entry["reasons"]:
            entry["reasons"].append(reason)

    for record in read_corpus(out_dir):  # positives: what the agent loaded after your prompt
        if record.get("subagent") or record.get("headless"):
            continue
        last = None
        for event in record["events"]:
            if event["role"] == "user" and event["kind"] in ("message", "command"):
                last = owner_text(event["text"]).strip() or None
            elif event["kind"] == "tool_call" and last:
                for skill in skills_loaded([event]):
                    if skill in targets:
                        add(last, record, "called", skill)
    pool = []
    for record, _, text in _owner_messages(out_dir):
        lowered = text.lower()
        hits = [s for s, words in keywords.items() if any(re.search(rf"\b{re.escape(w)}", lowered) for w in words)]
        for skill in hits:
            if sum(1 for c in candidates.values() if f"mentions {skill}" in c["reasons"]) < per_skill:
                add(text, record, f"mentions {skill}")
        if not hits and 15 <= len(text) <= 600 and text not in candidates:
            pool.append((record, text))
    unique = list({text: (record, text) for record, text in pool}.values())
    for record, text in random.Random(seed).sample(unique, min(negatives, len(unique))):
        add(text, record, "random draw")

    from .experiment import create_run  # the run holds the dataset, arms, rollouts, and traces

    run_dir = run_dir or create_run(out_dir, name, targets={t: None for t in targets})
    ordered = [{"id": f"c{i:03d}", **c} for i, c in enumerate(candidates.values(), 1)]
    write_private_json(run_dir / "candidates.json", ordered)
    write_private_text(run_dir / "LABEL.md", label_markdown(ordered, targets, run_dir.name))
    return run_dir


def label_markdown(candidates: list[dict], targets: list[str], run: str) -> str:
    lines = ["# Label these prompts", "",
             "Each prompt is something you wrote. Tick every skill that would be a right choice for it, or "
             f"`{NONE}`. Boxes ticked from evidence show what the agent loaded at the time; fix them if that was wrong. "
             f"Prompts with nothing ticked are left out. Then run `aihistory optimize label --run {run}`.", ""]
    for c in candidates:
        lines += [f"## {c['id']}", f"- source: {c['source']}, {c['date'] or 'undated'}",
                  f"- why: {', '.join(c['reasons'])}" + (f"; evidence: called {', '.join(c['evidence'])}" if c["evidence"] else ""), ""]
        shown = c["text"] if len(c["text"]) <= SHOWN_CHARS else c["text"][:SHOWN_CHARS] + " …"
        lines += ["> " + line for line in shown.splitlines()] + [""]
        lines += [f"- [{'x' if t in c['evidence'] else ' '}] {t}" for t in targets] + [f"- [ ] {NONE}", ""]
    return "\n".join(lines)


def apply_labels(run_dir: Path, test: float = 0.4, val: float = 0.2, seed: int = 7) -> dict:
    """Turn LABEL.md into stratified train/val/test splits in dataset.json."""
    candidates = {c["id"]: c for c in json.loads((run_dir / "candidates.json").read_text(encoding="utf-8"))}
    text = (run_dir / "LABEL.md").read_text(encoding="utf-8")
    marks = list(SECTION.finditer(text))
    examples = []
    for number, mark in enumerate(marks):
        body = text[mark.end():marks[number + 1].start() if number + 1 < len(marks) else len(text)]
        ticked = [label for box, label in BOX.findall(body) if box.lower() == "x"]
        if not ticked or mark.group(1) not in candidates:
            continue
        gold = sorted(t for t in ticked if t != NONE)  # a ticked skill wins over a ticked "none"
        candidate = candidates[mark.group(1)]
        examples.append({"id": mark.group(1), "prompt": candidate["text"], "gold": gold,
                         "cls": "pos:" + "+".join(gold) if gold else "neg",
                         "provenance": f"{candidate['source']}:{candidate['date']}"})
    split = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for cls in sorted({e["cls"] for e in examples}):
        group = [e for e in examples if e["cls"] == cls]
        rng.shuffle(group)
        n_test, n_val = round(len(group) * test), round(len(group) * val)
        split["test"] += group[:n_test]
        split["val"] += group[n_test:n_test + n_val]
        split["train"] += group[n_test + n_val:]
    write_private_json(run_dir / "dataset.json", split)
    return split
