"""REPORT.md: what history exists, what is missing, and what to do next."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .normalize import read_corpus
from .util import write_private_text


def _size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def corpus_stats(out_dir: Path) -> dict:
    stats = Counter()
    offered, called = Counter(), Counter()
    for record in read_corpus(out_dir):
        stats[f"conversations:{record['source']}"] += 1
        stats["subagent_conversations"] += bool(record.get("subagent"))
        stats["headless_conversations"] += bool(record.get("headless"))
        s = record.get("stats") or {}
        stats["thinking_blocks"] += s.get("thinking_blocks", 0)
        stats["thinking_with_text"] += s.get("thinking_with_text", 0)
        if s.get("skills_offered"):
            stats["sessions_offered_skills"] += 1
            offered["max"] = max(offered["max"], s["skills_offered"])
        called.update(s.get("skill_calls") or [])
        for e in record["events"]:
            stats[f"role:{e['role']}"] += 1
    return {"counts": stats, "skills_called": called, "max_skills_offered": offered["max"]}


def write(out_dir: Path, scan: list[dict], steps: dict) -> Path:
    lines = ["# Your AI history", ""]
    lines += ["## What was found", "", "| Source | Files | Size | First day | Last day | Notes |", "|---|---|---|---|---|---|"]
    for s in scan:
        notes = []
        if s.get("subagent_files"):
            notes.append(f"{s['subagent_files']} subagent transcripts")
        if s.get("compressed_files"):
            notes.append(f"{s['compressed_files']} compressed (.zst)")
        if s.get("days_without_files"):
            notes.append(f"{len(s['days_without_files'])} days with no files")
        lines.append(f"| {s['kind']} | {s['files']} | {_size(s['bytes'])} | {s.get('first_day', '—')} | {s.get('last_day', '—')} | {'; '.join(notes)} |")
    gaps = [(s["kind"], s["days_without_files"]) for s in scan if s.get("days_without_files")]
    if gaps:
        lines += ["", "**Days with no session files.** These may be days off, or files an app has since deleted. Back up regularly; apps prune old history.", ""]
        for kind, days in gaps:
            lines.append(f"- {kind}: {', '.join(days[:30])}{' …' if len(days) > 30 else ''}")
    if "backup" in steps:
        b = steps["backup"]
        lines += ["", "## Backup", "", f"`{b['root']}`: {b['files']} files catalogued in `CATALOG.json`; this run: {b['counts']}.",
                  "The raw backup still contains any secrets in your transcripts. Keep it on an encrypted disk (see README)."]
    if "normalize" in steps:
        n = steps["normalize"]
        lines += ["", "## Cleaned corpus", ""]
        for kind, s in n["sources"].items():
            lines.append(f"- `{s['file']}`: {s['conversations']} conversations, {s['events']} events; roles {s['roles']}")
        if "hstry" in n["sources"]:
            h = n["sources"]["hstry"]
            lines.append(f"- hstry: {h['already_read_natively']} conversations skipped because the Claude Code or Codex reader "
                         "already had them. For other tools, who-wrote-what is a best guess: hstry doesn't record it, so only "
                         "known harness text and dispatched worker threads are separated from your words.")
        if n["redactions"]:
            lines.append(f"- Redacted before writing: {n['redactions']}")
        if n["warning_count"]:
            lines.append(f"- {n['warning_count']} files could not be read: {n['warnings'][:3]}")
    try:
        c = corpus_stats(out_dir)
    except FileNotFoundError:
        c = None
    if c:
        counts = c["counts"]
        total = counts["thinking_blocks"]
        lines += ["", "## What the transcripts can and can't tell you", ""]
        if total:
            lines.append(f"- Thinking: {counts['thinking_with_text']} of {total} thinking blocks ({counts['thinking_with_text'] / total:.0%}) kept any text. Recent Claude models store thinking empty by default, so don't judge skills or rules from it.")
        if counts["sessions_offered_skills"]:
            used = sum(c["skills_called"].values())
            lines.append(f"- Skills: {counts['sessions_offered_skills']} sessions were offered up to {c['max_skills_offered']} skills; {used} skill calls to {len(c['skills_called'])} distinct skills. Most called: {', '.join(f'{k} ({v})' for k, v in c['skills_called'].most_common(8)) or 'none'}.")
        lines.append(f"- Text you wrote: {counts['role:user']} messages. Agent-written prompts (subagents): {counts['role:agent']}. Harness-injected text: {counts['role:harness']}. Only your own messages feed memories.")
        if counts["headless_conversations"]:
            lines.append(f"- {counts['headless_conversations']} headless `claude -p` sessions are searchable but don't feed memories (their prompts usually come from scripts).")
    if "index" in steps:
        lines += ["", "## Search", "", f"{steps['index']['messages']} of your messages are searchable: `aihistory search \"words\"`."]
    if "memories" in steps:
        m = steps["memories"]
        lines += ["", "## Memory candidates", "", f"{m['candidates']} candidates from {m['messages_considered']} of your messages.",
                  f"Review them in `{m['review']}`: tick `[x]` on the ones that are true, then `aihistory agent-me push` (dry run) and `--apply`."]
    path = out_dir / "REPORT.md"
    write_private_text(path, "\n".join(lines) + "\n")
    return path
