"""`aihistory optimize train`: GEPA rewrites the target skills' descriptions.

GEPA (`gepa.optimize_anything`, the optional `[optimize]` extra) proposes new descriptions from
what the agent did on training prompts and keeps the ones that score best on validation
prompts. The test split is never loaded here. The best descriptions become the arm `gepa`, to be
evaluated against the original and your control with `optimize eval --arms original,control,gepa`.

Rewrites come from the kit's own reflector: `claude -p` with no tools, MCP servers, or saved
session. gepa's default reflection model, reached through litellm, is never used.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

from ..adapters import agent_trace
from ..util import append_private_line, private_dir, sha256_text, write_private_json
from .agents import _clean_env
from .experiment import agent_label, arm_descriptions, arm_key, load_config, load_split, set_arm
from .scorers import side_info, skills_loaded, trigger_score
from .traces import TraceWriter

TEMPLATE = """You are improving the trigger description of one coding-agent skill. The agent sees a menu of skills (a name and a description each) and, from the user's prompt alone, decides whether to load one. A description that is too broad takes prompts it should not (false triggers); one that is too narrow misses prompts it should take.

Current text (the first line names the skill; keep that line unchanged):
```
<curr_param>
```

Results from recent runs: the prompt, the skills that would be right, what the agent loaded, its reasoning when available, and the score:
```
<side_info>
```

Write an improved description in one to three plain sentences, under 450 characters. Say which requests it handles and which nearby requests it should not take. Generalize; do not quote the prompts. Return the full text, with its first line unchanged, inside a single ``` block."""


def strip_header(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines()]
    if lines and lines[0].lower().startswith("skill:"):
        lines = lines[1:]
    return " ".join(line for line in lines if line)


class ClaudeReflector:
    """GEPA language model: prompt in, text out, through `claude -p` with nothing else loaded."""

    def __init__(self, model: str, log: Path | None = None, timeout: int = 600):
        self.model, self.log, self.timeout = model, log, timeout

    def __call__(self, prompt) -> str:
        if not isinstance(prompt, str):
            prompt = "\n\n".join(str(m.get("content", "")) for m in prompt)
        command = ["claude", "-p", "--model", self.model, "--output-format", "json", "--tools", "",
                   "--setting-sources", "project", "--strict-mcp-config", "--disable-slash-commands",
                   "--no-session-persistence"]
        with tempfile.TemporaryDirectory(prefix="aihistory-reflect-") as cwd:
            proc = subprocess.run(command, input=prompt, capture_output=True, text=True, timeout=self.timeout,
                                  cwd=cwd, env=_clean_env())
        try:
            text = json.loads(proc.stdout).get("result", "")
        except json.JSONDecodeError:
            text = proc.stdout
        if self.log:
            append_private_line(self.log, json.dumps({"prompt_chars": len(prompt), "response": text, "exit_code": proc.returncode}))
        return text


def train(run_dir: Path, agent, budget: int = 100, runs: int = 2, reflector=None, reflection_model: str = "sonnet",
          workers: int = 4, max_runs: int = 200, yes: bool = False, on_plan=None, log=None) -> dict:
    try:
        from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, TrackingConfig, optimize_anything
    except ImportError:
        raise ValueError("`optimize train` needs the optional gepa package: pip install 'ai-history-kit[optimize]' "
                         "(Python 3.10 to 3.14)") from None
    config = load_config(run_dir)
    train_set, val_set = load_split(run_dir, "train"), load_split(run_dir, "val")
    original = arm_descriptions(config, "original")
    planned = budget * runs  # a GEPA metric call is one example, run `runs` times
    if on_plan:
        on_plan(planned)
    if planned > max_runs and not yes:
        raise ValueError(f"up to {planned} agent runs planned (budget {budget} × {runs} runs); "
                         f"limit {max_runs}: add --yes to run them, or lower --budget or --runs")
    targets, label, store = set(config["targets"]), agent_label(agent), run_dir / "traces"
    split_of = {e["id"]: "train" for e in train_set} | {e["id"]: "val" for e in val_set}
    lock, workspaces = threading.Lock(), {}

    def workspace(descriptions: dict) -> Path:
        key = arm_key(descriptions)
        with lock:
            if key not in workspaces:
                workspaces[key] = Path(tempfile.mkdtemp(prefix="aihistory-train-"))
                agent.prepare(workspaces[key], {**config.get("decoys", {}), **descriptions})
            return workspaces[key]

    def evaluator(candidate: dict, example: dict):
        descriptions = {**original, **{t: strip_header(text) for t, text in candidate.items()}}
        key, folder = arm_key(descriptions), workspace(descriptions)
        scores, runs_seen = [], []
        for attempt in range(runs):
            session = sha256_text(f"{run_dir.name}|search|{key}|{label}|{example['id']}|{attempt}|{uuid.uuid4()}")[:16]
            writer = TraceWriter(store, session, agent.name)
            result = agent.run(folder, example["prompt"], writer)
            writer.close(exit_code=result.exit_code)
            record = agent_trace.read_session(writer.events_path, store)
            events = record["events"] if record else []
            loaded = skills_loaded(events)
            score = 0.0 if result.exit_code else trigger_score(example["gold"], loaded, targets)
            scores.append(score)
            info = side_info(example, events, loaded)
            runs_seen.append({k: v for k, v in info.items() if k not in ("Prompt", "Expected")} | {"Score": score})
            row = {"split": split_of.get(example["id"], "train"), "arm": "gepa-search", "key": key, "agent": label,
                   "id": example["id"], "cls": example["cls"], "gold": example["gold"], "attempt": attempt,
                   "score": score, "loaded": loaded, "cost_usd": result.cost_usd, "seconds": result.seconds,
                   "session": session, "error": result.error}
            with lock:
                append_private_line(run_dir / "rollouts.jsonl", json.dumps(row, ensure_ascii=False, sort_keys=True))
        return sum(scores) / len(scores), {"Prompt": example["prompt"][:400],
                                           "Expected": example["gold"] or "no target skill", "Runs": runs_seen}

    gepa_dir = private_dir(run_dir / "gepa")

    class _Logger:  # GEPA's progress: always to gepa/log.txt, and to `log` (the CLI prints it)
        def log(self, message: str) -> None:
            append_private_line(gepa_dir / "log.txt", str(message))
            if log:
                log(message)

    seed = {t: f"skill: {t}\n{original[t]}" for t in config["targets"]}
    settings = GEPAConfig(
        engine=EngineConfig(max_metric_calls=budget, run_dir=str(gepa_dir), seed=42,
                            parallel=workers > 1, max_workers=max(1, workers), cache_evaluation=True,
                            cache_evaluation_storage="memory", use_cloudpickle=False, display_progress_bar=False,
                            candidate_selection_strategy="pareto"),
        reflection=ReflectionConfig(reflection_lm=reflector or ClaudeReflector(reflection_model, run_dir / "reflections.jsonl"),
                                    reflection_minibatch_size=3, skip_perfect_score=True, perfect_score=1.0,
                                    reflection_prompt_template=TEMPLATE),
        tracking=TrackingConfig(logger=_Logger()))
    try:
        result = optimize_anything(seed_candidate=seed, evaluator=evaluator, dataset=train_set, valset=val_set, config=settings)
    finally:
        for folder in workspaces.values():
            subprocess.run(["rm", "-rf", str(folder)], check=False)
    best = {t: strip_header(text) for t, text in result.best_candidate.items()}
    scores = list(result.val_aggregate_scores)
    summary = {"best_idx": result.best_idx, "best": best, "val_aggregate_scores": scores,
               "candidates": [{t: strip_header(x) for t, x in c.items()} for c in result.candidates],
               "total_metric_calls": getattr(result, "total_metric_calls", None)}
    write_private_json(run_dir / "train_result.json", summary)
    set_arm(run_dir, "gepa", best)
    return {"best_idx": result.best_idx, "seed_val_score": scores[0], "best_val_score": scores[result.best_idx],
            "candidates": len(summary["candidates"]), "metric_calls": summary["total_metric_calls"]}
