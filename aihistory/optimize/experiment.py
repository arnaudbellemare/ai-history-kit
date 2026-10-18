"""Optimization runs: arms of skill descriptions, evaluated by an agent, compared against a control.

A run lives in `<out>/optimize/<name>/` (mode 0700): config.json (targets, decoys, arms),
candidates.json + LABEL.md + dataset.json, rollouts.jsonl, traces/ (one agent-trace session per
agent run), and REPORT.md.

Guards:
- every evaluation prints its run count first, and above --max-runs needs --yes;
- an arm is evaluated on the test split once; new descriptions need a new arm name;
- a report needs a hand-written control arm, because earlier trials found a one-line control
  explained most of what an optimizer gained.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from ..adapters import agent_trace
from ..util import append_private_line, private_dir, sha256_text, write_private_json, write_private_text
from .agents import safe_name
from .stats import majority_correct, paired
from .traces import TraceWriter

_lock = threading.Lock()


def skill_roots(home: Path, cwd: Path | None = None) -> list[Path]:
    roots = [home / ".claude" / "skills", home / ".agents" / "skills", home / ".codex" / "skills"]
    if cwd:
        roots += [cwd / ".claude" / "skills", cwd / ".agents" / "skills"]
    plugins = home / ".claude" / "plugins"
    if plugins.is_dir():
        roots += sorted({p.parent.parent for p in plugins.rglob("skills/*/SKILL.md")})
    return roots


def _description(skill_file: Path) -> str | None:
    try:
        parts = skill_file.read_text(encoding="utf-8").split("---")
    except OSError:
        return None
    if len(parts) < 3:
        return None
    lines = parts[1].splitlines()
    for number, line in enumerate(lines):
        if line.startswith("description:"):
            value = line[len("description:"):].strip()
            if value in (">", "|", ">-", "|-"):  # a YAML block: the indented lines that follow
                block = []
                for following in lines[number + 1:]:
                    if following and not following[0].isspace():
                        break
                    block.append(following.strip())
                return " ".join(b for b in block if b) or None
            if value.startswith('"'):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value.strip('"')
            return value.strip("'") or None
    return None


def find_skills(roots: list[Path]) -> dict[str, str]:
    """{name: description} for every SKILL.md under the roots; the first of a name wins."""
    found = {}
    for root in roots:
        for path in sorted(root.glob("*/SKILL.md")) if root.is_dir() else []:
            name = safe_name(path.parent.name)
            if name not in found and (description := _description(path)):
                found[name] = description
    return found


def run_path(out_dir: Path, name: str) -> Path:
    return out_dir / "optimize" / safe_name(name)


def load_config(run_dir: Path) -> dict:
    path = run_dir / "config.json"
    if not path.is_file():
        raise ValueError(f"no optimization run at {run_dir}; start one with `aihistory optimize mine`")
    return json.loads(path.read_text(encoding="utf-8"))


def save_config(run_dir: Path, config: dict) -> None:
    write_private_json(run_dir / "config.json", config)


def create_run(out_dir: Path, name: str, targets: dict[str, str | None], decoys: dict[str, str] | None = None) -> Path:
    run_dir = private_dir(run_path(out_dir, name))
    config = load_config(run_dir) if (run_dir / "config.json").is_file() else {"arms": {}}
    names = [safe_name(t) for t in targets]
    config.update({"name": run_dir.name, "targets": names,
                   "decoys": {k: v for k, v in (decoys or {}).items() if safe_name(k) not in names}})
    originals = {safe_name(t): d for t, d in targets.items() if d}
    if len(originals) == len(names) and "original" not in config["arms"]:
        config["arms"]["original"] = originals
    save_config(run_dir, config)
    return run_dir


def set_arm(run_dir: Path, arm: str, descriptions: dict[str, str]) -> None:
    config = load_config(run_dir)
    descriptions = {safe_name(k): v for k, v in descriptions.items()}
    if unknown := sorted(set(descriptions) - set(config["targets"])):
        raise ValueError(f"not a target of this run: {', '.join(unknown)} (targets: {', '.join(config['targets'])})")
    config["arms"][arm] = descriptions
    save_config(run_dir, config)


def arm_descriptions(config: dict, arm: str) -> dict[str, str]:
    """An arm's description for every target; an arm may change only some, the rest stay original."""
    if arm not in config["arms"]:
        raise ValueError(f"no arm `{arm}` in this run (arms: {', '.join(config['arms']) or 'none'})")
    effective = {**config["arms"].get("original", {}), **config["arms"][arm]}
    if missing := [t for t in config["targets"] if not effective.get(t)]:
        raise ValueError(f"arm `{arm}` has no description for {', '.join(missing)}; set it with `aihistory optimize arm`")
    return {t: effective[t] for t in config["targets"]}


def arm_key(descriptions: dict[str, str]) -> str:
    return sha256_text(json.dumps(descriptions, sort_keys=True))[:12]


def load_split(run_dir: Path, split: str) -> list[dict]:
    path = run_dir / "dataset.json"
    if not path.is_file():
        raise ValueError("no dataset yet; label LABEL.md and run `aihistory optimize label`")
    data = json.loads(path.read_text(encoding="utf-8")).get(split) or []
    if not data:
        raise ValueError(f"the {split} split is empty; label more prompts in LABEL.md")
    return data


def rollouts(run_dir: Path) -> list[dict]:
    path = run_dir / "rollouts.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def agent_label(agent) -> str:
    return f"{agent.name}:{getattr(agent, 'model', None) or 'default'}"


def evaluate(run_dir: Path, agent, split: str, arms: list[str], runs: int = 3, workers: int = 4,
             max_runs: int = 200, yes: bool = False, on_plan=None) -> dict:
    from .tasks import make_task

    config, examples = load_config(run_dir), load_split(run_dir, split)
    task, label, rows = make_task(config), agent_label(agent), rollouts(run_dir)
    done = {(r["split"], r["arm"], r["key"], r["agent"], r["id"], r["attempt"]) for r in rows if r.get("score") is not None}
    arms_info, jobs = {}, []
    for arm in arms:
        payload = task.payload(arm)
        key = arm_key(payload)
        earlier = {r["key"] for r in rows if r["split"] == "test" and r["arm"] == arm and r["agent"] == label}
        if split == "test" and earlier and key not in earlier:
            raise ValueError(f"arm `{arm}` was already evaluated on the test split with other text. "
                             "The test split is used once per arm: give new text a new arm name, or try it on val.")
        arms_info[arm] = (key, payload)
        jobs += [(arm, example, attempt) for example in examples for attempt in range(runs)
                 if (split, arm, key, label, example["id"], attempt) not in done]
    if on_plan:
        on_plan(len(jobs))
    if len(jobs) > max_runs and not yes:
        raise ValueError(f"{len(jobs)} agent runs planned (limit {max_runs}); add --yes to run them, or lower --runs")
    store = run_dir / "traces"

    def work(job):
        arm, example, attempt = job
        key, payload = arms_info[arm]
        session = sha256_text(f"{run_dir.name}|{split}|{arm}|{key}|{label}|{example['id']}|{attempt}")[:16]
        shutil.rmtree(store / session, ignore_errors=True)  # an interrupted earlier attempt
        writer = TraceWriter(store, session, agent.name)
        workspace, before = task.setup(agent, payload, example)
        try:
            result = agent.run(workspace, example["prompt"], writer, permissions=task.permissions(example))
            writer.close(exit_code=result.exit_code)
            record = agent_trace.read_session(writer.events_path, store)
            score, details = (None, {}) if result.exit_code else task.score(
                example, record["events"] if record else [], workspace, before)  # errors are retried, not scored
        finally:
            task.release(workspace)
        writer.annotate({"arm": arm, "example": example["id"], "split": split, "score": score})
        row = {"split": split, "arm": arm, "key": key, "agent": label, "id": example["id"], "cls": example["cls"],
               "gold": example["gold"], "attempt": attempt, "score": score, **details,
               "cost_usd": result.cost_usd, "seconds": result.seconds, "session": session, "error": result.error}
        with _lock:
            append_private_line(run_dir / "rollouts.jsonl", json.dumps(row, ensure_ascii=False, sort_keys=True))
        return score

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            scores = list(pool.map(work, jobs))
    finally:
        task.finish()
    return {"ran": len(jobs), "errors": sum(1 for s in scores if s is None), "split": split, "agent": label}


def _rate(scores: list[float]) -> str:
    return f"{sum(scores) / len(scores):.2f}" if scores else "—"


def report(run_dir: Path, split: str = "test") -> str:
    from .tasks import make_task

    config = load_config(run_dir)
    task = make_task(config)
    rows = [r for r in rollouts(run_dir) if r["split"] == split and r["arm"] != "gepa-search"]
    scored = [r for r in rows if r.get("score") is not None]
    if not scored:
        raise ValueError(f"no {split} results yet; run `aihistory optimize eval` first")
    latest = {}
    for r in scored:  # an arm re-evaluated on val with new text: report its latest version
        latest[(r["arm"], r["agent"])] = r["key"]
    groups = {g: [r for r in scored if (r["arm"], r["agent"]) == g and r["key"] == k] for g, k in latest.items()}
    if not any(arm == "control" for arm, _ in groups):
        hint = ("--skill NAME=\"Use when …\"" if task.kind == "triggers" else "--text \"one plain line\"")
        raise ValueError(f"a report needs a control arm: `aihistory optimize arm --run {run_dir.name} control {hint}`, then evaluate it")
    title = {"triggers": "Skill trigger evaluation", "gskill": "gskill evaluation"}.get(
        task.kind, f"Rule evaluation: section \"{config.get('section')}\"")
    headers = [name for name, _ in task.columns([])]
    lines = [f"# {title}: {run_dir.name} ({split} split)", "",
             "| Arm | Agent | Runs | Score | " + " | ".join(headers) + " | Right by majority | Cost (USD) |",
             "|---|---|---|---|" + "---|" * len(headers) + "---|---|"]
    per_prompt = {}
    for (arm, agent), group in sorted(groups.items()):
        prompts = {}
        for r in group:
            prompts.setdefault(r["id"], []).append(r["score"])
        per_prompt[(arm, agent)] = prompts
        cost = sum(r["cost_usd"] or 0 for r in group)
        values = " | ".join(_rate(scores) for _, scores in task.columns(group))
        lines.append(f"| {arm} | {agent} | {len(group)} | {_rate([r['score'] for r in group])} | {values} | "
                     f"{sum(1 for s in prompts.values() if majority_correct(s))}/{len(prompts)} | {cost:.2f} |")
    lines += ["", "## Paired comparison (McNemar, by majority of runs per prompt)", "",
              "| Arms | Agent | Prompts | Only first right | Only second right | p |", "|---|---|---|---|---|---|"]
    keys = sorted(groups)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[1] == b[1]:
                pair = paired(per_prompt[a], per_prompt[b])
                lines.append(f"| {a[0]} vs {b[0]} | {a[1]} | {pair['prompts']} | {pair['only_a']} | {pair['only_b']} | {pair['p']} |")
    if task.kind == "rule":
        failures = {}
        for group in groups.values():
            for r in group:
                for failure in r.get("failed") or []:
                    failures[(r["arm"], failure)] = failures.get((r["arm"], failure), 0) + 1
        if failures:
            lines += ["", "## Failed checks", ""] + [f"- {arm}: {name} ({count}×)" for (arm, name), count in sorted(failures.items())]
    if task.kind == "gskill" and (shapes := gskill_shapes(config)):
        lines += ["", f"Warning: the `gepa` skill describes the bug generator ({', '.join(shapes)}). It may have learned how "
                  "these tasks were made rather than anything about your repository; read it before you trust its score."]
    if errors := len(rows) - len(scored):
        lines += ["", f"{errors} runs ended in an agent error and are not scored; `eval` retries them."]
    lines += ["", "Read this with care: a small test split rarely shows a significant difference, and in earlier trials "
              "a one-line control explained most of what an optimizer gained.",
              "", f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Traces: `{run_dir / 'traces'}`."]
    text = "\n".join(lines) + "\n"
    write_private_text(run_dir / "REPORT.md", text)
    return text


def gskill_shapes(config: dict) -> list[str]:
    from .gskill import generator_shapes

    return generator_shapes((config["arms"].get("gepa") or {}).get("skill", ""))


def create_rule_run(out_dir: Path, name: str, instructions: Path, section: str, scenarios: Path) -> Path:
    """A rule run: your instructions file, the section under test, and scenarios split train/val/test."""
    from .rules import load_scenarios, section_body

    text = instructions.read_text(encoding="utf-8")
    section_body(text, section)  # the section must exist
    split = load_scenarios(scenarios)
    run_dir = private_dir(run_path(out_dir, name))
    config = load_config(run_dir) if (run_dir / "config.json").is_file() else {"arms": {}}
    if config.get("kind", "rule") != "rule":
        raise ValueError(f"{run_dir.name} is a skill-trigger run; pick another --run name")
    config.update({"kind": "rule", "name": run_dir.name, "instructions": text, "section": section,
                   "instructions_file": instructions.name})
    config["arms"].setdefault("present", {})
    config["arms"].setdefault("removed", {"section": None})
    save_config(run_dir, config)
    write_private_json(run_dir / "dataset.json", split)
    return run_dir


def set_text_arm(run_dir: Path, arm: str, text: str | None) -> None:
    """A rule run's section text, or a gskill run's skill, for one arm."""
    config = load_config(run_dir)
    kind = config.get("kind", "triggers")
    fixed = {"rule": ("present", "removed"), "gskill": ("baseline",)}.get(kind)
    if fixed is None:
        raise ValueError("--text sets a rule run's section or a gskill run's skill; for skill descriptions use --skill NAME=text")
    if arm in fixed:
        raise ValueError(f"`{arm}` is fixed in a {kind} run")
    config["arms"][arm] = {"section": text} if kind == "rule" else {"skill": text or ""}
    save_config(run_dir, config)


set_rule_arm = set_text_arm


def parse_assignments(values: list[str]) -> dict[str, str]:
    """NAME=text pairs from the command line."""
    out = {}
    for value in values or []:
        name, sep, text = value.partition("=")
        if not sep or not re.match(r"^[A-Za-z0-9:._-]+$", name):
            raise ValueError(f"expected NAME=text, got {value!r}")
        out[name] = text
    return out
