"""What a run measures. `evaluate`, `report`, and `train` work through a task, so skill triggers
and rule sections share the guards, traces, statistics, and GEPA setup.

A task turns an arm into a payload (what the agent gets), sets up a workspace for one prompt,
scores the run from its trace events, and, for GEPA, maps candidates to payloads and arms.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

from . import gskill
from .checks import run_checks, snapshot
from .rules import DEFAULT_ALLOW, init_git, section_body, strip_label, with_section, write_files
from .scorers import side_info, skills_loaded, trigger_score

TRIGGER_TEMPLATE = """You are improving the trigger description of one coding-agent skill. The agent sees a menu of skills (a name and a description each) and, from the user's prompt alone, decides whether to load one. A description that is too broad takes prompts it should not (false triggers); one that is too narrow misses prompts it should take.

Current text (the first line names the skill; keep that line unchanged):
```
<curr_param>
```

Results from recent runs: the prompt, the skills that would be right, what the agent loaded, its reasoning when available, and the score:
```
<side_info>
```

Write an improved description in one to three plain sentences, under 450 characters. Say which requests it handles and which nearby requests it should not take. Generalize; do not quote the prompts. Return the full text, with its first line unchanged, inside a single ``` block."""

RULE_TEMPLATE = """You are improving one section of the instructions file (CLAUDE.md or AGENTS.md) a coding agent reads before it works. The section should make the agent follow a working rule in situations like the scenarios below, without stopping it from finishing the task.

Current section (the first line names it; keep that line unchanged):
```
<curr_param>
```

Results from recent runs: the scenario, the checks that failed, the agent's tool calls, its reasoning when available, and the score:
```
<side_info>
```

Rewrite the section's text. Keep it short and concrete: what to do, in what order, and what never to do. Generalize; do not name the scenarios. Return the full text, with its first line unchanged, inside a single ``` block."""


def strip_header(text: str) -> str:
    """A skill description from a GEPA candidate: drop the `skill:` line, join the rest into one line."""
    lines = [line.strip() for line in text.strip().splitlines()]
    if lines and lines[0].lower().startswith("skill:"):
        lines = lines[1:]
    return " ".join(line for line in lines if line)


class TriggerTask:
    kind, base_arm, template = "triggers", "original", TRIGGER_TEMPLATE

    def __init__(self, config: dict):
        self.config, self.targets = config, set(config["targets"])
        self._shared: dict[str, Path] = {}
        self._lock = threading.Lock()

    def payload(self, arm: str) -> dict:
        from .experiment import arm_descriptions

        return arm_descriptions(self.config, arm)

    def permissions(self, example: dict) -> dict | None:
        return None  # the agent's defaults for a trigger evaluation: no edits, shell, or web

    def setup(self, agent, payload: dict, example: dict) -> tuple[Path, dict | None]:
        from .experiment import arm_key

        key = arm_key(payload)
        with self._lock:  # one workspace per set of descriptions; trigger runs change nothing in it
            if key not in self._shared:
                self._shared[key] = Path(tempfile.mkdtemp(prefix="aihistory-eval-"))
                agent.prepare(self._shared[key], {**self.config.get("decoys", {}), **payload})
            return self._shared[key], None

    def score(self, example: dict, events: list[dict], workspace: Path, before) -> tuple[float, dict]:
        loaded = skills_loaded(events)
        return trigger_score(example["gold"], loaded, self.targets), {"loaded": loaded}

    def release(self, workspace: Path) -> None:
        pass

    def finish(self) -> None:
        for folder in self._shared.values():
            shutil.rmtree(folder, ignore_errors=True)
        self._shared.clear()

    def seed_candidate(self) -> dict[str, str]:
        original = self.payload(self.base_arm)
        return {t: f"skill: {t}\n{original[t]}" for t in self.config["targets"]}

    def candidate_payload(self, candidate: dict) -> dict:
        return {**self.payload(self.base_arm), **{t: strip_header(text) for t, text in candidate.items()}}

    def candidate_arm(self, candidate: dict) -> dict:
        return {t: strip_header(text) for t, text in candidate.items()}

    def side_info(self, example: dict, events: list[dict], details: dict) -> dict:
        info = side_info(example, events, details.get("loaded", []))
        return {k: v for k, v in info.items() if k not in ("Prompt", "Expected")}

    def columns(self, rows: list[dict]) -> list[tuple[str, list[float]]]:
        return [("Should trigger", [r["score"] for r in rows if r["gold"]]),
                ("No false trigger", [r["score"] for r in rows if not r["gold"]])]


class RuleTask:
    kind, base_arm, template = "rule", "present", RULE_TEMPLATE

    def __init__(self, config: dict):
        self.config, self.text, self.heading = config, config["instructions"], config["section"]

    def payload(self, arm: str) -> dict:
        arms = self.config["arms"]
        if arm not in arms:
            raise ValueError(f"no arm `{arm}` in this run (arms: {', '.join(arms)})")
        if arm == "present":
            return {"instructions": self.text}
        return {"instructions": with_section(self.text, self.heading, arms[arm].get("section"))}

    def permissions(self, example: dict) -> dict:
        return {"allow": example.get("allow") or DEFAULT_ALLOW, "write": True,
                "max_turns": example.get("max_turns", 20), "timeout": example.get("timeout", 600)}

    def setup(self, agent, payload: dict, example: dict) -> tuple[Path, dict]:
        folder = Path(tempfile.mkdtemp(prefix="aihistory-rule-"))  # a fresh repository for every run
        write_files(folder, example)
        agent.prepare(folder, {}, payload["instructions"])
        init_git(folder, example)
        return folder, snapshot(folder)

    def score(self, example: dict, events: list[dict], workspace: Path, before) -> tuple[float, dict]:
        results = run_checks(example["checks"], events, workspace, before or {})
        process = all(r["passed"] for r in results if r["kind"] == "process")
        outcome = all(r["passed"] for r in results if r["kind"] == "outcome")
        failed = [f"{r['name']}: {r['detail']}" if r["detail"] else r["name"] for r in results if not r["passed"]]
        return (1.0 if process and outcome else 0.0), {"process": process, "outcome": outcome, "failed": failed}

    def release(self, workspace: Path) -> None:
        shutil.rmtree(workspace, ignore_errors=True)

    def finish(self) -> None:
        pass

    def seed_candidate(self) -> dict[str, str]:
        return {"rule": f"section: {self.heading}\n{section_body(self.text, self.heading)}"}

    def candidate_payload(self, candidate: dict) -> dict:
        return {"instructions": with_section(self.text, self.heading, strip_label(candidate["rule"], "section:"))}

    def candidate_arm(self, candidate: dict) -> dict:
        return {"section": strip_label(candidate["rule"], "section:")}

    def side_info(self, example: dict, events: list[dict], details: dict) -> dict:
        calls = [f"{e.get('tool')}: {e['text'][:160]}" for e in events if e.get("kind") == "tool_call"]
        reasoning = " ".join(e["text"] for e in events if e.get("kind") == "reasoning")
        return {"Scenario": example["prompt"][:300], "Failed checks": details.get("failed", []),
                "Tool calls": calls[:12], "Agent reasoning": reasoning[:600]}

    def columns(self, rows: list[dict]) -> list[tuple[str, list[float]]]:
        return [("Process checks", [1.0 if r.get("process") else 0.0 for r in rows]),
                ("Outcome checks", [1.0 if r.get("outcome") else 0.0 for r in rows])]


GSKILL_TEMPLATE = """You are helping a coding agent learn a skill file (CLAUDE.md or AGENTS.md) for fixing bugs in one repository. Each task gives the agent a short bug report (how many tests now fail, and one failure excerpt); the agent must fix the source without touching tests. A run passes only when every test that passed before passes again.

Current skill (it may be empty):
```
<curr_param>
```

Results from recent runs: the task, how it ended, the tests still failing, the agent's tool calls, and the score:
```
<side_info>
```

Write an improved skill: concrete, reusable guidance about this repository and about working through such tasks (where code lives, how to run and read the tests, what finishing means). Generalize; do not describe individual bugs or how they were made. Return the full skill inside a single ``` block."""


class GskillTask:
    kind, base_arm, template = "gskill", "baseline", GSKILL_TEMPLATE

    def __init__(self, config: dict):
        self.config, self.base = config, Path(config["base"])
        self.venv = Path(config["venv"]) if config.get("venv") else None
        self.base_pass = set(config["base_pass"])

    def payload(self, arm: str) -> dict:
        arms = self.config["arms"]
        if arm not in arms:
            raise ValueError(f"no arm `{arm}` in this run (arms: {', '.join(arms)})")
        return {"skill": (arms[arm].get("skill") or "").strip()}

    def permissions(self, example: dict) -> dict:
        return {"allow": gskill.allowed_tools(self.config["test_command"]), "write": True, "max_turns": 40, "timeout": 480}

    def setup(self, agent, payload: dict, example: dict) -> tuple[Path, None]:
        folder = Path(tempfile.mkdtemp(prefix="aihistory-gskill-run-"))
        gskill.copy_tree(self.base, folder, self.venv)
        for relative, source in example["mutated_files"].items():
            (folder / relative).write_text(source, encoding="utf-8")
        agent.prepare(folder, {}, gskill.skill_document(payload["skill"]))
        return folder, None

    def score(self, example: dict, events: list[dict], workspace: Path, before) -> tuple[float, dict]:
        result = gskill.grade(workspace, example, self.base, self.config["test_command"], self.base_pass, self.config["tests"])
        return (1.0 if result["status"] == "all_passed" else 0.0), result

    def release(self, workspace: Path) -> None:
        shutil.rmtree(workspace, ignore_errors=True)

    def finish(self) -> None:
        pass

    def seed_candidate(self) -> dict[str, str]:
        return {"skill": self.payload(self.base_arm)["skill"]}

    def candidate_payload(self, candidate: dict) -> dict:
        return {"skill": (candidate.get("skill") or "").strip()}

    def candidate_arm(self, candidate: dict) -> dict:
        return {"skill": (candidate.get("skill") or "").strip()}

    def side_info(self, example: dict, events: list[dict], details: dict) -> dict:
        calls = [f"{e.get('tool')}: {e['text'][:160]}" for e in events if e.get("kind") == "tool_call"]
        replies = [e["text"] for e in events if e.get("kind") == "message" and e.get("role") == "assistant"]
        return {"Task": example["prompt"][:600], "Status": details.get("status"), "Still failing": details.get("still_failing"),
                "Regressions": details.get("regressions"), "Test output": details.get("test_output", "")[:600],
                "Tool calls": calls[:45], "Agent final text": (replies[-1] if replies else "")[:400]}

    def columns(self, rows: list[dict]) -> list[tuple[str, list[float]]]:
        return [("All tests pass", [r["score"] for r in rows]),
                ("Regressions", [1.0 if r.get("status") == "p2p_regression" else 0.0 for r in rows])]


def make_task(config: dict):
    kind = config.get("kind", "triggers")
    return {"rule": RuleTask, "gskill": GskillTask}.get(kind, TriggerTask)(config)
