"""Agents that run one prompt in a throwaway workspace and record it as an agent-trace session.

- ClaudeCodeAgent: `claude -p` with project settings only, no MCP servers, and no saved
  session. A trigger evaluation denies edit, shell, and web tools, so it has no side effects; a
  rule scenario allows only the tools it lists (such as `Bash(git:*)`), inside its scratch repo.
- CodexAgent (experimental): `codex exec --json --ephemeral`, sandboxed read-only (or
  workspace-write for a rule scenario). Codex
  loads repository skills from `.agents/skills/`; check one run by hand before trusting scores.
- FakeAgent: loads the skill whose description shares the most words with the prompt. For tests
  and for dry runs that check a setup without spending anything.

Workspaces live in the system temp folder, outside any repository, so no CLAUDE.md or AGENTS.md
from a parent folder leaks into the run.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .traces import TraceWriter, from_claude_stream, from_codex_jsonl

STUB = "Stub body for a trigger evaluation. Reply with the single word LOADED and stop.\n"
DENIED_TOOLS = ("Bash", "Edit", "Write", "NotebookEdit", "WebFetch", "WebSearch", "Task")
WEB_TOOLS = ("WebFetch", "WebSearch", "Task")


def safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")


def write_skills(root: Path, folder: str, skills: dict[str, str]) -> None:
    for name, description in skills.items():
        target = root / folder / safe_name(name)
        target.mkdir(parents=True, exist_ok=True)
        # An agent drops a skill with no description, so a missing one becomes its name.
        lines = ["---", f"name: {safe_name(name)}", "description: " + json.dumps(description or name, ensure_ascii=False),
                 "---", "", STUB]
        (target / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")


@dataclass
class RunResult:
    exit_code: int
    seconds: float
    cost_usd: float | None = None
    error: str | None = None


def _clean_env() -> dict:
    # A nested `claude -p` refuses to start inside another Claude Code session's environment.
    return {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE_CODE_SESSION", "CLAUDECODE"))}


def _run(command: list[str], workspace: Path, prompt: str | None, timeout: int) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(command, cwd=workspace, input=prompt, capture_output=True, text=True,
                              timeout=timeout, env=_clean_env())
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as error:
        out = error.stdout or ""
        return 124, out.decode("utf-8", "replace") if isinstance(out, bytes) else out, f"timed out after {timeout}s"
    except FileNotFoundError:
        raise ValueError(f"`{command[0]}` is not installed or not on PATH") from None


class ClaudeCodeAgent:
    name, skills_folder, instructions_file = "claude-code", ".claude/skills", "CLAUDE.md"

    def __init__(self, model: str = "sonnet", max_turns: int = 3, timeout: int = 240, thinking: bool = True,
                 denied_tools: tuple[str, ...] = DENIED_TOOLS, executable: str = "claude"):
        self.model, self.max_turns, self.timeout, self.thinking = model, max_turns, timeout, thinking
        self.denied_tools, self.executable = denied_tools, executable

    def prepare(self, workspace: Path, skills: dict[str, str], instructions: str | None = None) -> None:
        write_skills(workspace, self.skills_folder, skills)
        if instructions:
            (workspace / self.instructions_file).write_text(instructions, encoding="utf-8")

    def command(self, permissions: dict | None = None) -> list[str]:
        turns = (permissions or {}).get("max_turns", self.max_turns)
        command = [self.executable, "-p", "--model", self.model, "--setting-sources", "project",
                   "--output-format", "stream-json", "--verbose", "--max-turns", str(turns),
                   "--no-session-persistence", "--strict-mcp-config"]
        if self.thinking:  # readable reasoning summaries in headless runs, used as side information
            command += ["--thinking-display", "summarized"]
        if permissions:  # only the listed tools; print mode denies everything else
            return command + ["--allowedTools", *permissions["allow"], "--disallowedTools", *WEB_TOOLS]
        return command + ["--disallowedTools", *self.denied_tools]

    def run(self, workspace: Path, prompt: str, writer: TraceWriter, permissions: dict | None = None) -> RunResult:
        started = time.monotonic()
        timeout = (permissions or {}).get("timeout", self.timeout)
        writer.event("user_prompt", {"prompt": prompt})
        code, stdout, stderr = _run(self.command(permissions), workspace, prompt, timeout)
        if code and self.thinking and "thinking-display" in stderr:
            self.thinking = False  # an older Claude Code without the flag
            code, stdout, stderr = _run(self.command(permissions), workspace, prompt, timeout)
        summary = from_claude_stream(stdout.splitlines(), writer)
        if code:
            writer.event("error", {"error": stderr[-500:] or f"exit {code}"})
        return RunResult(code, round(time.monotonic() - started, 1), summary["cost_usd"], stderr[-500:] if code else None)


class CodexAgent:
    name, skills_folder, instructions_file = "codex", ".agents/skills", "AGENTS.md"

    def __init__(self, model: str | None = None, timeout: int = 240, executable: str = "codex"):
        self.model, self.timeout, self.executable = model, timeout, executable

    def prepare(self, workspace: Path, skills: dict[str, str], instructions: str | None = None) -> None:
        write_skills(workspace, self.skills_folder, skills)
        if instructions:
            (workspace / self.instructions_file).write_text(instructions, encoding="utf-8")

    def run(self, workspace: Path, prompt: str, writer: TraceWriter, permissions: dict | None = None) -> RunResult:
        started = time.monotonic()
        writer.event("user_prompt", {"prompt": prompt})
        sandbox = "workspace-write" if (permissions or {}).get("write") else "read-only"
        command = [self.executable, "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config",
                   "--sandbox", sandbox] + (["--model", self.model] if self.model else []) + ["--", prompt]
        code, stdout, stderr = _run(command, workspace, None, (permissions or {}).get("timeout", self.timeout))
        from_codex_jsonl(stdout.splitlines(), writer)
        if code:
            writer.event("error", {"error": stderr[-500:] or f"exit {code}"})
        return RunResult(code, round(time.monotonic() - started, 1), None, stderr[-500:] if code else None)


STOP_WORDS = {"a", "an", "and", "asks", "for", "in", "is", "it", "me", "of", "on", "or", "please", "the", "this",
              "to", "use", "user", "when", "with", "wants"}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP_WORDS


class FakeAgent:
    name, skills_folder, instructions_file = "fake", ".fake/skills", "FAKE.md"
    model = "fake"

    def prepare(self, workspace: Path, skills: dict[str, str], instructions: str | None = None) -> None:
        write_skills(workspace, self.skills_folder, skills)

    def _skills(self, workspace: Path) -> dict[str, str]:
        found = {}
        for path in sorted((workspace / self.skills_folder).glob("*/SKILL.md")):
            line = next((x for x in path.read_text(encoding="utf-8").splitlines() if x.startswith("description: ")), "")
            found[path.parent.name] = json.loads(line[len("description: "):]) if line else ""
        return found

    def run(self, workspace: Path, prompt: str, writer: TraceWriter, permissions: dict | None = None) -> RunResult:
        writer.event("user_prompt", {"prompt": prompt})
        writer.event("session_start", {"provider": "fake"})
        overlap = {name: len(_words(prompt) & _words(description)) for name, description in self._skills(workspace).items()}
        best = max(sorted(overlap), key=overlap.get, default=None)
        if best and overlap[best]:
            writer.event("tool_call", {"tool_name": "Skill", "arguments": {"skill": best}})
            writer.event("tool_result", {"result": "LOADED"})
        writer.event("assistant_response", {"text": "LOADED" if best and overlap[best] else "OK"})
        writer.event("session_end", {})
        return RunResult(0, 0.0, 0.0)


def make_agent(name: str, model: str | None = None):
    if name == "claude-code":
        return ClaudeCodeAgent(model or "sonnet")
    if name == "codex":
        return CodexAgent(model)
    if name == "fake":
        return FakeAgent()
    raise ValueError(f"unknown agent {name}; use claude-code, codex, or fake")
