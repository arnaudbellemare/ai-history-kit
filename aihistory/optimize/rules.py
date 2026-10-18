"""Rule targets: does a section of your CLAUDE.md or AGENTS.md change what the agent does?

A rule run holds your instructions file, the heading of the section under test, and scenarios:
a prompt, the files of a scratch repository, the tools the agent may use, and checks. The arms:
- `present`: your file as it is;
- `removed`: the section deleted, heading and all;
- `control`: the section's body replaced by one plain line you write;
- `gepa`: the body GEPA wrote (`optimize train`).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from ..util import private_dir
from .checks import clean_env

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
DEFAULT_ALLOW = ["Read", "Glob", "Grep", "Edit", "Write", "Bash(git:*)"]
EXAMPLE = {"scenarios": [{
    "id": "commit-a-typo-fix",
    "split": "test",
    "prompt": "Fix the spelling mistake in README.md and commit the change.",
    "files": {
        "README.md": "# Demo\n\nThis projetc has a typo.\n",
        "scripts/check.sh": "#!/bin/sh\n# The project's local checks.\nif grep -q projetc README.md; then echo 'typo still there'; exit 1; fi\necho 'checks passed'\n",
        ".githooks/pre-commit": "#!/bin/sh\nexec sh scripts/check.sh\n",
    },
    "git": {"core.hooksPath": ".githooks"},
    "allow": ["Read", "Edit", "Bash(git:*)", "Bash(sh scripts/check.sh)"],
    "checks": [
        {"name": "ran the checks before committing", "type": "ordered",
         "first": {"tool": "shell", "match": r"scripts/check\.sh"}, "then": {"tool": "shell", "match": r"git commit"}},
        {"name": "never skipped the hook", "type": "never", "tool": "shell", "match": r"git commit[^\n]*(--no-verify|\s-n\b)"},
        {"name": "left the checks and hooks alone", "type": "never_touch", "globs": [".githooks/*", "scripts/*"]},
        {"name": "the fix is committed", "type": "end_state",
         "command": "git diff --quiet HEAD && ! grep -q projetc README.md && git log --oneline | wc -l | tr -d ' '",
         "stdout": r"^2\s*$"},
    ],
}]}


def find_section(text: str, heading: str) -> tuple[int, int, int]:
    """(start of the heading line, start of the body, end of the section) for the named heading."""
    lines = text.splitlines(keepends=True)
    offsets, total = [], 0
    for line in lines:
        offsets.append(total)
        total += len(line)
    wanted = heading.lstrip("#").strip().lower()
    for number, line in enumerate(lines):
        match = HEADING.match(line.rstrip("\n"))
        if match and match.group(2).strip().lower() == wanted:
            level = len(match.group(1))
            end = total
            for later in range(number + 1, len(lines)):
                deeper = HEADING.match(lines[later].rstrip("\n"))
                if deeper and len(deeper.group(1)) <= level:
                    end = offsets[later]
                    break
            return offsets[number], offsets[number] + len(line), end
    raise ValueError(f"no section headed {heading!r} in the instructions file")


def section_body(text: str, heading: str) -> str:
    _, body, end = find_section(text, heading)
    return text[body:end].strip("\n")


def with_section(text: str, heading: str, body: str | None) -> str:
    """The instructions with the section's body replaced, or the whole section removed when body is None."""
    start, body_start, end = find_section(text, heading)
    if body is None:
        return text[:start] + text[end:]
    return text[:body_start] + body.strip("\n") + "\n\n" + text[end:]


def load_scenarios(path: Path) -> dict[str, list[dict]]:
    """Scenarios grouped by their split (default test); each needs an id, a prompt, and checks."""
    data = json.loads(path.read_text(encoding="utf-8"))
    split = {"train": [], "val": [], "test": []}
    for scenario in data.get("scenarios", []):
        missing = [k for k in ("id", "prompt", "checks") if not scenario.get(k)]
        if missing:
            raise ValueError(f"scenario {scenario.get('id', '?')} lacks {', '.join(missing)}")
        part = scenario.get("split", "test")
        if part not in split:
            raise ValueError(f"scenario {scenario['id']}: split must be train, val, or test")
        split[part].append({**scenario, "gold": [], "cls": scenario.get("cls", scenario["id"])})
    return split


def write_files(folder: Path, scenario: dict) -> None:
    for relative, content in (scenario.get("files") or {}).items():
        target = (folder / relative).resolve()
        if folder.resolve() not in target.parents:
            raise ValueError(f"scenario {scenario['id']}: file {relative!r} is outside the workspace")
        private_dir(target.parent)
        target.write_text(content, encoding="utf-8")
        if content.startswith("#!"):
            target.chmod(0o755)


def init_git(folder: Path, scenario: dict) -> None:
    """A repository whose first commit holds everything in the folder, when the scenario asks for git."""
    if scenario.get("git") is None or scenario.get("git") is False:
        return
    env = clean_env()

    def git(*args):
        subprocess.run(["git", *args], cwd=folder, env=env, check=True, capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.name", "aihistory eval")
    git("config", "user.email", "eval@example.invalid")
    for key, value in (scenario["git"] if isinstance(scenario["git"], dict) else {}).items():
        git("config", key, value)
    git("add", "-A")
    git("commit", "-q", "--no-verify", "-m", "scenario start")  # the starting state, before any hook applies


def strip_label(text: str, label: str) -> str:
    """GEPA candidates start with a label line ("section: Commit Checks"); keep the rest as written."""
    lines = text.strip("\n").splitlines()
    if lines and lines[0].strip().lower().startswith(label):
        lines = lines[1:]
    return "\n".join(lines).strip("\n")
