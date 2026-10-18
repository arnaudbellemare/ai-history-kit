"""A fake home directory with the traps real Claude Code and Codex history contains."""

import json
from pathlib import Path

FAKE_SECRET = "abcdefgh" + "12345678"


def _jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def user(text, **extra):
    return {"type": "user", "sessionId": "sess-1", "timestamp": "2026-07-01T10:00:00Z",
            "message": {"role": "user", "content": text}, **extra}


def assistant(*blocks):
    return {"type": "assistant", "sessionId": "sess-1", "timestamp": "2026-07-01T10:00:05Z",
            "message": {"role": "assistant", "content": list(blocks)}}


def make_home(home: Path) -> Path:
    project = home / ".claude" / "projects" / "-Users-someone-proj"
    _jsonl(project / "sess-1.jsonl", [
        {"type": "attachment", "attachment": {"type": "skill_listing", "names": ["demo", "other"], "content": "- demo: x"}},
        # A fake credential, assembled at runtime so secret scanners don't flag the fixture.
        user("My name is Alex. I prefer short answers. my " + "api_key" + "=" + FAKE_SECRET + " is old."),
        user([{"type": "text", "text": "I usually cite primary sources."}], isMeta=True),  # a skill body
        user("<task-notification>I completed the audit.</task-notification>"),
        user("This session is being continued. I decided to rewrite everything.", isCompactSummary=True),
        user("<command-name>/goal</command-name><command-args>ship the beta</command-args>"),
        user("Here is the thread:\n[1] assistant: Yes. I completed a source-backed design.\n[2] user: thanks\nI'm building a kite."),
        user(f"Notes are in {home}/notes.md"),
        assistant({"type": "thinking", "thinking": "", "signature": "sig"},
                  {"type": "tool_use", "id": "t1", "name": "Skill", "input": {"skill": "demo"}},
                  {"type": "text", "text": "I prefer tabs."}),
        user([{"type": "tool_result", "tool_use_id": "t1", "content": "done"}]),
    ])
    _jsonl(project / "sess-1" / "subagents" / "agent-abc.jsonl", [
        user("I'm building a cost-aware model selector."),  # a brief Claude wrote to a subagent
    ])
    # A headless `claude -p` run started by a script: its prompt is the script's, not a conversation.
    _jsonl(project / "headless-run.jsonl", [
        {**user("I prefer purple everywhere."), "sessionId": "sess-headless", "entrypoint": "sdk-cli"},
    ])
    # The same session id in a second project folder (a session that moved into a worktree).
    _jsonl(home / ".claude" / "projects" / "-Users-someone-proj-worktree" / "sess-1.jsonl", [
        user("I'm building a kite."),
    ])
    codex = home / ".codex" / "sessions"
    _jsonl(codex / "2026/10/06/rollout-2026-10-06T08-00-00-top.jsonl", [
        {"type": "session_meta", "payload": {"id": "thread-top", "source": "vscode", "timestamp": "2026-10-06T08:00:00Z"}},
        {"type": "response_item", "timestamp": "2026-10-06T08:00:01Z", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "# AGENTS.md instructions for the repo\nI always run the linter."},
            {"type": "input_text", "text": "<environment_context>\n<cwd>/tmp</cwd>\n</environment_context>"},
            {"type": "input_text", "text": "I always write tests first."}]}},
        {"type": "response_item", "timestamp": "2026-10-06T08:00:02Z", "payload": {"type": "message", "role": "assistant",
                                                                                   "content": [{"type": "output_text", "text": "OK"}]}},
    ])
    _jsonl(codex / "2026/10/07/rollout-2026-10-07T08-00-00-guardian.jsonl", [
        {"type": "session_meta", "payload": {"id": "thread-guardian", "source": {"subagent": "other"}}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "Remember: You're a technical quality auditor."}]}},
    ])
    _jsonl(home / ".codex" / "archived_sessions" / "rollout-2026-09-01T08-00-00-old.jsonl", [
        {"type": "session_meta", "payload": {"id": "thread-archived", "source": "vscode"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "I'm building a sailboat."}},
    ])
    for number in range(3):  # a working rule the owner repeats across conversations
        _jsonl(codex / f"2026/10/0{number + 1}/rollout-2026-10-0{number + 1}T08-00-00-rule{number}.jsonl", [
            {"type": "session_meta", "payload": {"id": f"thread-rule-{number}", "source": "vscode"}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": f"Fix bug {number}. Never push to main without asking."}},
        ])
    _jsonl(codex / "2026/10/08/rollout-2026-10-08T08-00-00-fork.jsonl", [
        {"type": "session_meta", "payload": {"id": "thread-child", "source": "vscode", "forked_from_id": "thread-top"}},
        {"type": "session_meta", "payload": {"id": "thread-top", "source": "vscode"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "continue"}},
    ])
    return home
