"""Evaluation prompts from your own history, with pre-filled labels you confirm.

`mine` finds candidates in the cleaned corpus, using only text you wrote (not subagent briefs,
headless runs, or injected harness text), and pre-fills a label for as many as it can. Every
pre-fill records where it came from:

- `history:accepted`: the agent loaded a target skill for your prompt and you carried on;
- `history:interrupted`, `history:corrected`: it loaded one and you interrupted or corrected it,
  so loading it was probably wrong;
- `history:missed-then-invoked`: it loaded nothing, then you invoked a target skill yourself;
- `history:invoked`: you invoked a target skill with a request, so the request is the prompt;
- `generated:<model>`: prompts a model wrote from the skill's instructions (`--generate`), some
  that should load it and near misses that should not;
- `judge:<model>`: a model's label for the rest (`--judge`). The generator and the judge see a
  skill's instructions with its frontmatter removed, never the description being tested.

You confirm labels in LABEL.md by ticking "I checked this". Only confirmed labels go into the
test split; unconfirmed pre-fills may still be used for training. Prompts that most need a look
come first: where sources disagree, where the judge was unsure, a random sample for checking
the pre-fills' accuracy, and generated prompts.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..memories import owner_text
from ..normalize import read_corpus
from ..util import write_private_json, write_private_text
from .agents import safe_name
from .scorers import skills_loaded

NONE = "none of these"
CHECKED = "I checked this"
SECTION = re.compile(r"^## (c\d+)\s*$", re.MULTILINE)
BOX = re.compile(r"^- \[([ xX])\] (.+?)\s*$", re.MULTILINE)
SHOWN_CHARS = 1500
INTERRUPT = "[Request interrupted by user"
CORRECTION = re.compile(r"(?i)^\s*(?:no\b|nope\b|wrong\b|not that\b|don'?t use\b|stop\b|that'?s not\b|why did you (?:use|load)\b)")
MISS_WINDOW = 3  # your messages before a /skill in which a plain prompt counts as a miss
JUDGE_BATCH = 10
UNSURE = 0.7
BODY_CHARS = 6000
PRIORITY = ("history:invoked", "history:missed-then-invoked", "history:interrupted", "history:corrected",
            "history:accepted", "judge:", "generated:")
GENERATE_PROMPT = """You are writing test prompts for one coding-agent skill, to check when an agent should load it.

What the skill does (its instructions):
<<<
{body}
>>>

Write {n} prompts a user might type that this skill should handle, in varied phrasings (casual and formal, some that never name the skill), and {n} near-miss prompts that share words or topics with it but need something else. Return only a JSON array of objects: {{"prompt": "...", "should_trigger": true or false}}."""
JUDGE_PROMPT = """You are labelling prompts for a test of which skill a coding agent loads. For each prompt, decide which of these skills the agent should load to handle it, judging from what each skill does. Several may fit, or none.

Skills:
{skills}

Prompts:
{prompts}

Return only a JSON array with one object per prompt: {{"key": "k0", "skills": ["skill-name", ...], "confidence": 0.0 to 1.0}}. Use [] when no skill fits."""


def skill_body(path: Path) -> str:
    """A skill's instructions without its frontmatter, so the description under test never leaks."""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        text = parts[2] if len(parts) == 3 else ""
    return text.strip()


def _json_array(text: str):
    for candidate in (text, *re.findall(r"```(?:json)?\s*(.*?)```", text or "", re.DOTALL)):
        try:
            value = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            start, end = (candidate or "").find("["), (candidate or "").rfind("]")
            if start < 0 or end <= start:
                continue
            try:
                value = json.loads(candidate[start:end + 1])
            except json.JSONDecodeError:
                continue
        if isinstance(value, list):
            return value
    return None


def _slash(text: str) -> tuple[str, str]:
    name, _, args = text[1:].partition(" ")
    return safe_name(name), args.strip()


def _missed_prompt(events: list[dict], owner: list[int], n: int, skills) -> str:
    """Your plain prompt just before the /skill at owner[n], if the agent loaded none of `skills` after it."""
    for back in reversed(owner[max(0, n - MISS_WINDOW):n]):
        if events[back]["kind"] != "message":
            continue
        if any(s in skills for s in skills_loaded(events[back:owner[n]])):
            return ""
        return owner_text(events[back]["text"]).strip()
    return ""


def _when(stamp) -> datetime | None:
    try:
        when = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def recent_misses(out_dir: Path, skills, since: datetime) -> dict[str, int]:
    """{skill: times} you invoked one of `skills` yourself since `since`, after the agent loaded none of them."""
    counts = Counter()
    for record in read_corpus(out_dir):
        if record.get("subagent") or record.get("headless"):
            continue
        events = record["events"]
        owner = [i for i, e in enumerate(events) if e["role"] == "user" and e["kind"] in ("message", "command")]
        for n, i in enumerate(owner):
            if events[i]["kind"] != "command" or not events[i]["text"].startswith("/"):
                continue
            skill, _ = _slash(events[i]["text"])
            when = _when(events[i].get("timestamp") or record.get("updated_at") or record.get("created_at"))
            if skill in skills and when and when >= since and _missed_prompt(events, owner, n, skills):
                counts[skill] += 1
    return dict(counts.most_common())


def mine(out_dir: Path, name: str, skills: list[str], keywords: dict[str, list[str]] | None = None,
         negatives: int = 40, per_skill: int = 40, seed: int = 7, run_dir: Path | None = None,
         skill_files: dict[str, Path] | None = None, generate: int = 0, generator=None, judge=None,
         audit: float = 0.1, max_calls: int = 30, yes: bool = False, on_plan=None) -> Path:
    targets = [safe_name(s) for s in skills]
    keywords = {safe_name(k): [w.lower() for w in v] for k, v in (keywords or {}).items()}
    candidates: dict[str, dict] = {}

    def add(text, record, reason, evidence=None, by=None, gold=None, **extra):
        entry = candidates.setdefault(text, {"text": text, "source": record["source"] if record else "generated",
                                             "date": ((record or {}).get("created_at") or "")[:10], "evidence": [],
                                             "reasons": [], "proposals": []})
        for skill in evidence or []:
            if skill not in entry["evidence"]:
                entry["evidence"].append(skill)
        if reason not in entry["reasons"]:
            entry["reasons"].append(reason)
        if by and not any(p["by"] == by and p["gold"] == sorted(gold) for p in entry["proposals"]):
            entry["proposals"].append({"by": by, "gold": sorted(gold), **extra})
        return entry

    for record in read_corpus(out_dir):  # what happened after your prompts, as label evidence
        if record.get("subagent") or record.get("headless"):
            continue
        events = record["events"]
        owner = [i for i, e in enumerate(events) if e["role"] == "user" and e["kind"] in ("message", "command")]
        for n, i in enumerate(owner):
            event = events[i]
            if event["kind"] == "command":
                skill, request = _slash(event["text"]) if event["text"].startswith("/") else ("", "")
                if skill not in targets:
                    continue
                if len(request) >= 12:
                    add(request, record, f"you invoked {skill}", by="history:invoked", gold=[skill])
                if text := _missed_prompt(events, owner, n, targets):
                    add(text, record, f"missed, then you invoked {skill}", by="history:missed-then-invoked", gold=[skill])
                continue
            text = owner_text(event["text"]).strip()
            following = owner[n + 1] if n + 1 < len(owner) else len(events)
            loaded = [s for s in skills_loaded(events[i:following]) if s in targets]
            if not text or not loaded:
                continue
            reaction = owner_text(events[following]["text"]) if following < len(events) else ""
            interrupted = INTERRUPT in reaction or any(INTERRUPT in e["text"] for e in events[i:following] if e["role"] == "harness")
            if interrupted:
                add(text, record, f"loaded {', '.join(loaded)}, then you interrupted", loaded, "history:interrupted", [])
            elif CORRECTION.match(reaction):
                add(text, record, f"loaded {', '.join(loaded)}, then you corrected it", loaded, "history:corrected", [])
            else:
                add(text, record, f"loaded {', '.join(loaded)}, you carried on", loaded, "history:accepted", loaded)

    pool = []
    for record in read_corpus(out_dir):
        if record.get("subagent") or record.get("headless"):
            continue
        for event in record["events"]:
            if event["role"] != "user" or event["kind"] != "message" or not (text := owner_text(event["text"]).strip()):
                continue
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

    bodies = {t: skill_body(p) for t, p in (skill_files or {}).items() if safe_name(t) in targets and Path(p).is_file()}
    bodies = {safe_name(t): b for t, b in bodies.items() if b}
    to_judge = [c for c in candidates.values() if not c["proposals"]] if judge and bodies else []
    calls = (len(bodies) if generate and generator else 0) + math.ceil(len(to_judge) / JUDGE_BATCH)
    if on_plan:
        on_plan(calls, len(to_judge))
    if calls > max_calls and not yes:
        raise ValueError(f"{calls} model calls planned (limit {max_calls}); add --yes to make them, or lower --generate")

    if generate and generator:
        for skill, body in bodies.items():
            items = _json_array(generator(GENERATE_PROMPT.format(body=body[:BODY_CHARS], n=generate))) or []
            for item in items[:4 * generate]:
                if isinstance(item, dict) and isinstance(item.get("prompt"), str) and item["prompt"].strip():
                    entry = add(item["prompt"].strip(), None, f"generated from {skill}'s instructions",
                                by=f"generated:{generator.label}", gold=[skill] if item.get("should_trigger") else [])
                    entry["generated"] = True

    skills_block = "\n\n".join(f"## {t}\n<<<\n{b[:BODY_CHARS]}\n>>>" for t, b in bodies.items())
    for start in range(0, len(to_judge), JUDGE_BATCH):
        batch = to_judge[start:start + JUDGE_BATCH]
        keyed = {f"k{n}": c for n, c in enumerate(batch)}
        lines = "\n".join(f"[{k}] " + " ".join(c["text"][:SHOWN_CHARS].split()) for k, c in keyed.items())
        for item in _json_array(judge(JUDGE_PROMPT.format(skills=skills_block, prompts=lines))) or []:
            if not isinstance(item, dict) or item.get("key") not in keyed or not isinstance(item.get("skills"), list):
                continue  # an unreadable answer abstains
            try:
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
            except (TypeError, ValueError):
                confidence = 0.5
            gold = sorted({safe_name(s) for s in item["skills"] if isinstance(s, str) and safe_name(s) in targets})
            entry = keyed[item["key"]]
            entry["proposals"].append({"by": f"judge:{judge.label}", "gold": gold, "confidence": confidence})

    from .experiment import create_run  # the run holds the dataset, arms, rollouts, and traces

    run_dir = run_dir or create_run(out_dir, name, targets={t: None for t in targets})
    ordered = _order_for_review(list(candidates.values()), audit, seed)
    ordered = [{"id": f"c{i:03d}", **c} for i, c in enumerate(ordered, 1)]
    write_private_json(run_dir / "candidates.json", ordered)
    write_private_text(run_dir / "LABEL.md", label_markdown(ordered, targets, run_dir.name))
    return run_dir


def _preferred(proposals: list[dict]) -> dict | None:
    for prefix in PRIORITY:
        for proposal in proposals:
            if proposal["by"].startswith(prefix):
                return proposal
    return None


def _order_for_review(candidates: list[dict], audit: float, seed: int) -> list[dict]:
    """Flag what most needs your eye, and put it first."""
    labelled = [c for c in candidates if c["proposals"] and not c.get("generated")]
    sample = random.Random(seed).sample(labelled, min(len(labelled), max(1, round(audit * len(labelled))))) if labelled and audit > 0 else []
    sampled = {id(c) for c in sample}
    for c in candidates:
        flags = []
        if len({tuple(p["gold"]) for p in c["proposals"]}) > 1:
            flags.append("sources disagree")
        if any(p["by"].startswith("judge:") and p.get("confidence", 1) < UNSURE for p in c["proposals"]):
            flags.append("the judge was unsure")
        if id(c) in sampled:
            flags.append("random check of the pre-fills")
        if c.get("generated"):
            flags.append("generated prompt: check its wording too")
        c["check_first"] = flags
    return sorted(candidates, key=lambda c: (not c["check_first"], not c["proposals"]))


def _describe(proposal: dict) -> str:
    label = ", ".join(proposal["gold"]) or NONE
    sure = f" (confidence {proposal['confidence']:.2f})" if "confidence" in proposal else ""
    return f"- pre-filled from {proposal['by']}: {label}{sure}"


def label_markdown(candidates: list[dict], targets: list[str], run: str) -> str:
    lines = ["# Check these prompts", "",
             "Each prompt is something you wrote, or one a model wrote from a skill's instructions (marked generated). "
             f"Tick every skill that would be a right choice, or `{NONE}`. Boxes are pre-filled where the kit had "
             "evidence; each pre-fill says where it came from. Fix what is wrong, then tick "
             f"`{CHECKED}`. Only checked prompts go into the test split; unchecked pre-fills may still be used for "
             "training. Prompts that most need a look come first. Prompts with nothing ticked are left out. "
             f"Then run `aihistory optimize label --run {run}`.", ""]
    for c in candidates:
        preferred = _preferred(c["proposals"])
        ticked = set(preferred["gold"]) if preferred else set(c["evidence"])
        why = f"- why: {', '.join(c['reasons'])}" + (f"; evidence: called {', '.join(c['evidence'])}" if c["evidence"] else "")
        lines += [f"## {c['id']}", f"- source: {c['source']}, {c['date'] or 'undated'}", why]
        lines += [_describe(p) for p in c["proposals"]]
        if c.get("check_first"):
            lines.append(f"- check first: {'; '.join(c['check_first'])}")
        shown = c["text"] if len(c["text"]) <= SHOWN_CHARS else c["text"][:SHOWN_CHARS] + " …"
        lines += [""] + ["> " + line for line in shown.splitlines()] + [""]
        lines += [f"- [{'x' if t in ticked else ' '}] {t}" for t in targets]
        lines += [f"- [{'x' if preferred and not preferred['gold'] else ' '}] {NONE}", f"- [ ] {CHECKED}", ""]
    return "\n".join(lines)


def apply_labels(run_dir: Path, test: float = 0.4, val: float = 0.2, seed: int = 7) -> dict:
    """LABEL.md into stratified splits. The test split takes only prompts you ticked as checked."""
    candidates = {c["id"]: c for c in json.loads((run_dir / "candidates.json").read_text(encoding="utf-8"))}
    text = (run_dir / "LABEL.md").read_text(encoding="utf-8")
    marks = list(SECTION.finditer(text))
    examples, audit = [], {}
    for number, mark in enumerate(marks):
        body = text[mark.end():marks[number + 1].start() if number + 1 < len(marks) else len(text)]
        ticked = [label for box, label in BOX.findall(body) if box.lower() == "x"]
        confirmed = CHECKED in ticked
        labels = [t for t in ticked if t != CHECKED]
        if not labels or mark.group(1) not in candidates:
            continue
        gold = sorted(t for t in labels if t != NONE)  # a ticked skill wins over a ticked "none"
        candidate = candidates[mark.group(1)]
        proposals = candidate.get("proposals") or []
        if confirmed:
            for proposal in proposals:  # how often each source's pre-fill survived your check
                tally = audit.setdefault(proposal["by"], {"kept": 0, "changed": 0})
                tally["kept" if proposal["gold"] == gold else "changed"] += 1
        examples.append({"id": mark.group(1), "prompt": candidate["text"], "gold": gold,
                         "cls": "pos:" + "+".join(gold) if gold else "neg", "confirmed": confirmed,
                         "label_sources": [p["by"] for p in proposals],
                         "provenance": f"{candidate['source']}:{candidate['date']}"})
    split = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for cls in sorted({e["cls"] for e in examples}):
        group = [e for e in examples if e["cls"] == cls]
        rng.shuffle(group)
        sure = [e for e in group if e["confirmed"]]
        n_test = min(len(sure), round(len(group) * test))
        split["test"] += sure[:n_test]
        rest = sure[n_test:] + [e for e in group if not e["confirmed"]]
        n_val = min(len(rest), round(len(group) * val))
        split["val"] += rest[:n_val]
        split["train"] += rest[n_val:]
    write_private_json(run_dir / "dataset.json", {**split, "label_audit": audit})
    return split
