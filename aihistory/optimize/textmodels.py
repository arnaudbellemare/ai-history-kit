"""Text-in, text-out models that write and judge eval prompts for `optimize mine`.

Each sends what it is given to its provider: your prompts for judging, and your skills'
instructions. `mine` counts the calls first and asks before going over --max-calls.

- ClaudeText: `claude -p` with no tools, MCP servers, slash commands, or saved session.
- CodexText: `codex exec --ephemeral` in a read-only sandbox in an empty folder. To judge prompts
  for a Claude Code evaluation, prefer this one: a judge from another model family is less likely
  to share the agent's habits.
- FakeText: for tests and dry runs; records every prompt it receives.
"""

from __future__ import annotations

import json
import subprocess
import tempfile

from .agents import _clean_env


class FakeText:
    label = "fake"

    def __init__(self, responder=None):
        self.responder = responder or (lambda prompt: "[]")
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responder(prompt)


class ClaudeText:
    label = "claude"

    def __init__(self, model: str = "sonnet"):
        from .gepa_train import ClaudeReflector  # gepa itself is only imported by train()

        self.model, self._call = model, ClaudeReflector(model)

    def __call__(self, prompt: str) -> str:
        return self._call(prompt)


class CodexText:
    label = "codex"

    def __init__(self, model: str | None = None, timeout: int = 600):
        self.model, self.timeout = model, timeout

    def __call__(self, prompt: str) -> str:
        command = ["codex", "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config",
                   "--sandbox", "read-only"] + (["--model", self.model] if self.model else []) + ["--", prompt]
        with tempfile.TemporaryDirectory(prefix="aihistory-judge-") as folder:
            try:
                proc = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=self.timeout,
                                      env=_clean_env())
            except FileNotFoundError:
                raise ValueError("`codex` is not installed or not on PATH") from None
            except subprocess.TimeoutExpired:
                return ""
        replies = []
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = event.get("item") or {}
            if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                replies.append(item.get("text") or "")
        return replies[-1] if replies else ""


def make_text_model(name: str, model: str | None = None):
    if name == "claude":
        return ClaudeText(model or "sonnet")
    if name == "codex":
        return CodexText(model)
    if name == "fake":
        return FakeText()
    raise ValueError(f"unknown model {name}; use claude, codex, or fake")
