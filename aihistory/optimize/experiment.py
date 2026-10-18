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
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from ..adapters import agent_trace
from ..util import append_private_line, private_dir, sha256_text, write_private_json, write_private_text
from .agents import safe_name
from .scorers import skills_loaded, trigger_score
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
    config, examples = load_config(run_dir), load_split(run_dir, split)
    label, rows = agent_label(agent), rollouts(run_dir)
    done = {(r["split"], r["arm"], r["key"], r["agent"], r["id"], r["attempt"]) for r in rows if r.get("score") is not None}
    arms_info, jobs = {}, []
    for arm in arms:
        descriptions = arm_descriptions(config, arm)
        key = arm_key(descriptions)
        earlier = {r["key"] for r in rows if r["split"] == "test" and r["arm"] == arm and r["agent"] == label}
        if split == "test" and earlier and key not in earlier:
            raise ValueError(f"arm `{arm}` was already evaluated on the test split with other descriptions. "
                             "The test split is used once per arm: give new descriptions a new arm name, or try them on val.")
        arms_info[arm] = (key, descriptions)
        jobs += [(arm, example, attempt) for example in examples for attempt in range(runs)
                 if (split, arm, key, label, example["id"], attempt) not in done]
    if on_plan:
        on_plan(len(jobs))
    if len(jobs) > max_runs and not yes:
        raise ValueError(f"{len(jobs)} agent runs planned (limit {max_runs}); add --yes to run them, or lower --runs")
    targets, store = set(config["targets"]), run_dir / "traces"
    workspaces = {}

    def work(job):
        arm, example, attempt = job
        key, _ = arms_info[arm]
        session = sha256_text(f"{run_dir.name}|{split}|{arm}|{key}|{label}|{example['id']}|{attempt}")[:16]
        shutil.rmtree(store / session, ignore_errors=True)  # an interrupted earlier attempt
        writer = TraceWriter(store, session, agent.name)
        result = agent.run(workspaces[arm], example["prompt"], writer)
        writer.close(exit_code=result.exit_code)
        record = agent_trace.read_session(writer.events_path, store)
        loaded = skills_loaded(record["events"] if record else [])
        score = None if result.exit_code else trigger_score(example["gold"], loaded, targets)  # errors are retried
        writer.annotate({"arm": arm, "example": example["id"], "split": split, "score": score})
        row = {"split": split, "arm": arm, "key": key, "agent": label, "id": example["id"], "cls": example["cls"],
               "gold": example["gold"], "attempt": attempt, "score": score, "loaded": loaded,
               "cost_usd": result.cost_usd, "seconds": result.seconds, "session": session, "error": result.error}
        with _lock:
            append_private_line(run_dir / "rollouts.jsonl", json.dumps(row, ensure_ascii=False, sort_keys=True))
        return score

    try:
        for arm, (_, descriptions) in arms_info.items():
            workspaces[arm] = Path(tempfile.mkdtemp(prefix="aihistory-eval-"))
            agent.prepare(workspaces[arm], {**config.get("decoys", {}), **descriptions})
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            scores = list(pool.map(work, jobs))
    finally:
        for workspace in workspaces.values():
            shutil.rmtree(workspace, ignore_errors=True)
    return {"ran": len(jobs), "errors": sum(1 for s in scores if s is None), "split": split, "agent": label}


def _rate(rows) -> str:
    return f"{sum(r['score'] for r in rows) / len(rows):.2f}" if rows else "—"


def report(run_dir: Path, split: str = "test") -> str:
    rows = [r for r in rollouts(run_dir) if r["split"] == split]
    scored = [r for r in rows if r.get("score") is not None]
    if not scored:
        raise ValueError(f"no {split} results yet; run `aihistory optimize eval` first")
    latest = {}
    for r in scored:  # an arm re-evaluated on val with new descriptions: report its latest version
        latest[(r["arm"], r["agent"])] = r["key"]
    groups = {g: [r for r in scored if (r["arm"], r["agent"]) == g and r["key"] == k] for g, k in latest.items()}
    if not any(arm == "control" for arm, _ in groups):
        raise ValueError("a report needs a control arm: write one plain line per skill with "
                         "`aihistory optimize arm --run RUN control --skill NAME=\"Use when …\"` and evaluate it")
    lines = [f"# Skill trigger evaluation: {run_dir.name} ({split} split)", "",
             "| Arm | Agent | Runs | Accuracy | Should trigger | No false trigger | Prompts right (majority) | Cost (USD) |",
             "|---|---|---|---|---|---|---|---|"]
    per_prompt = {}
    for (arm, agent), group in sorted(groups.items()):
        prompts = {}
        for r in group:
            prompts.setdefault(r["id"], []).append(r["score"])
        per_prompt[(arm, agent)] = prompts
        cost = sum(r["cost_usd"] or 0 for r in group)
        lines.append(f"| {arm} | {agent} | {len(group)} | {_rate(group)} | {_rate([r for r in group if r['gold']])} | "
                     f"{_rate([r for r in group if not r['gold']])} | "
                     f"{sum(1 for s in prompts.values() if majority_correct(s))}/{len(prompts)} | {cost:.2f} |")
    lines += ["", "## Paired comparison (McNemar, by majority of runs per prompt)", "",
              "| Arms | Agent | Prompts | Only first right | Only second right | p |", "|---|---|---|---|---|---|"]
    keys = sorted(groups)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[1] == b[1]:
                pair = paired(per_prompt[a], per_prompt[b])
                lines.append(f"| {a[0]} vs {b[0]} | {a[1]} | {pair['prompts']} | {pair['only_a']} | {pair['only_b']} | {pair['p']} |")
    if errors := len(rows) - len(scored):
        lines += ["", f"{errors} runs ended in an agent error and are not scored; `eval` retries them."]
    lines += ["", "Read this with care: a small test split rarely shows a significant difference, and in earlier trials "
              "a one-line control explained most of what an optimizer gained.",
              "", f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Traces: `{run_dir / 'traces'}`."]
    text = "\n".join(lines) + "\n"
    write_private_text(run_dir / "REPORT.md", text)
    return text


def parse_assignments(values: list[str]) -> dict[str, str]:
    """NAME=text pairs from the command line."""
    out = {}
    for value in values or []:
        name, sep, text = value.partition("=")
        if not sep or not re.match(r"^[A-Za-z0-9:._-]+$", name):
            raise ValueError(f"expected NAME=text, got {value!r}")
        out[name] = text
    return out
