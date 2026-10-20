"""`aihistory optimize review`: confirm pre-filled labels one prompt at a time.

It walks LABEL.md in its own order (prompts that most need a look come first) and skips prompts
you already checked, unless --all is given. For each prompt:

    Enter or y   accept the ticked label and mark the prompt checked
    1, 2, ...    toggle that skill (ticking a skill clears "none of these")
    n            set "none of these"
    s            skip without checking
    u            undo the previous answer and go back to it
    q            quit

Every answer is written to LABEL.md at once (atomically, mode 0600), so quitting or Ctrl-C never
loses work. Only the answered prompt's checkbox lines change.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

from ..util import write_private_text
from .dataset import CHECKED, NONE, SHOWN_CHARS

HEADING = re.compile(r"(?m)^(?=## c\d+$)")
# One checkbox line, without its newline. dataset.BOX ends in \s*, which can swallow the newline.
LINE_BOX = re.compile(r"(?m)^- \[([ xX])\] (.+?)[ \t]*$")
HELP = "Enter accept · 1-{n} toggle a skill · n none of these · s skip · u undo · q quit"


def split_sections(text: str) -> tuple[str, list[str]]:
    """The page's header and its prompt sections, so that header + "".join(sections) == text."""
    parts = HEADING.split(text)
    return parts[0], parts[1:]


def boxes(section: str) -> list[tuple[str, bool]]:
    return [(label, mark.lower() == "x") for mark, label in LINE_BOX.findall(section)]


def set_boxes(section: str, ticked: set[str], checked: bool) -> str:
    """Rewrite only the checkbox lines of one prompt."""
    def replace(match):
        label = match.group(2)
        on = checked if label == CHECKED else label in ticked
        return f"- [{'x' if on else ' '}] {label}"
    return LINE_BOX.sub(replace, section)


def _describe(section: str, position: int, total: int, skills: list[str], ticked: set[str]) -> list[str]:
    lines = section.splitlines()
    ident = lines[0][3:].strip()
    flags = next((line[len("- check first: "):] for line in lines if line.startswith("- check first: ")), "")
    shown = [f"[{position}/{total}] {ident}" + (f"   check first: {flags}" if flags else "")]
    shown += ["  " + line[2:] for line in lines if line.startswith("- pre-filled from ")]
    prompt = "\n".join(line[2:] for line in lines if line.startswith("> "))
    prompt = prompt if len(prompt) <= SHOWN_CHARS else prompt[:SHOWN_CHARS] + " …"
    for paragraph in prompt.splitlines() or [""]:
        wrapped = textwrap.wrap(paragraph, 96) or [""]
        shown += ["  > " + piece for piece in wrapped]
    choices = [f"{n} [{'x' if s in ticked else ' '}] {s}" for n, s in enumerate(skills, 1)]
    shown.append("  " + "   ".join(choices + [f"[{'x' if NONE in ticked else ' '}] {NONE}"]))
    return shown


def review(run_dir: Path, ask=None, show=print, everything: bool = False) -> dict:
    ask = ask or (lambda prompt: input(prompt))  # looked up at call time, so tests can patch it
    path = run_dir / "LABEL.md"
    header, sections = split_sections(path.read_text(encoding="utf-8"))
    queue = [i for i, s in enumerate(sections) if everything or not dict(boxes(s)).get(CHECKED)]
    history: list[tuple[int, int, str]] = []  # (queue position, section index, section before the answer)
    position, answered = 0, 0

    def save():
        write_private_text(path, header + "".join(sections))

    while position < len(queue):
        index = queue[position]
        current = boxes(sections[index])
        skills = [label for label, _ in current if label not in (NONE, CHECKED)]
        ticked = {label for label, on in current if on and label != CHECKED}
        for line in _describe(sections[index], position + 1, len(queue), skills, ticked):
            show(line)
        try:
            key = ask(HELP.format(n=len(skills)) + " > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if key in ("", "y"):
            if not ticked:
                show(f"  Tick a skill or press n for {NONE} first.")
                continue
            history.append((position, index, sections[index]))
            sections[index] = set_boxes(sections[index], ticked, checked=True)
            save()
            answered += 1
            position += 1
        elif key.isdigit() and 1 <= int(key) <= len(skills):
            ticked = (ticked ^ {skills[int(key) - 1]}) - {NONE}
            sections[index] = set_boxes(sections[index], ticked, checked=False)
            save()
        elif key == "n":
            sections[index] = set_boxes(sections[index], {NONE}, checked=False)
            save()
        elif key == "s":
            position += 1
        elif key == "u":
            if not history:
                show("  Nothing to undo yet.")
                continue
            position, previous, before = history.pop()
            sections[previous] = before
            save()
            answered -= 1
        elif key == "q":
            break
        else:
            show("  " + HELP.format(n=len(skills)))
    remaining = [s for s in sections if not dict(boxes(s)).get(CHECKED)]
    return {"checked_this_session": answered,
            "checked": len(sections) - len(remaining),
            "unchecked": len(remaining),
            "unchecked_flagged": sum("- check first: " in s for s in remaining)}
