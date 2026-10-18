"""`aihistory optimize train`: GEPA rewrites the text under test.

For a skill-trigger run that is the target skills' descriptions; for a rule run, the section of
your instructions file. GEPA (`gepa.optimize_anything`, the optional `[optimize]` extra) proposes
rewrites from what the agent did on training prompts and keeps what scores best on validation
prompts. The test split is never loaded here. The best rewrite becomes the arm `gepa`, to be
evaluated against the base arm and your control with `optimize eval`.

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
from .experiment import agent_label, arm_key, load_config, load_split, save_config
from .tasks import make_task, strip_header  # noqa: F401  (strip_header: kept importable from here)
from .traces import TraceWriter


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
    task = make_task(config)
    train_set, val_set = load_split(run_dir, "train"), load_split(run_dir, "val")
    task.payload(task.base_arm)  # the base arm must be complete before anything runs
    planned = budget * runs  # a GEPA metric call is one example, run `runs` times
    if on_plan:
        on_plan(planned)
    if planned > max_runs and not yes:
        raise ValueError(f"up to {planned} agent runs planned (budget {budget} × {runs} runs); "
                         f"limit {max_runs}: add --yes to run them, or lower --budget or --runs")
    label, store = agent_label(agent), run_dir / "traces"
    split_of = {e["id"]: "train" for e in train_set} | {e["id"]: "val" for e in val_set}
    lock = threading.Lock()

    def evaluator(candidate: dict, example: dict):
        payload = task.candidate_payload(candidate)
        key = arm_key(payload)
        scores, runs_seen = [], []
        for attempt in range(runs):
            session = sha256_text(f"{run_dir.name}|search|{key}|{label}|{example['id']}|{attempt}|{uuid.uuid4()}")[:16]
            writer = TraceWriter(store, session, agent.name)
            workspace, before = task.setup(agent, payload, example)
            try:
                result = agent.run(workspace, example["prompt"], writer, permissions=task.permissions(example))
                writer.close(exit_code=result.exit_code)
                record = agent_trace.read_session(writer.events_path, store)
                events = record["events"] if record else []
                score, details = (0.0, {}) if result.exit_code else task.score(example, events, workspace, before)
            finally:
                task.release(workspace)
            scores.append(score)
            runs_seen.append({**task.side_info(example, events, details), "Score": score})
            row = {"split": split_of.get(example["id"], "train"), "arm": "gepa-search", "key": key, "agent": label,
                   "id": example["id"], "cls": example["cls"], "gold": example["gold"], "attempt": attempt,
                   "score": score, **details, "cost_usd": result.cost_usd, "seconds": result.seconds,
                   "session": session, "error": result.error}
            with lock:
                append_private_line(run_dir / "rollouts.jsonl", json.dumps(row, ensure_ascii=False, sort_keys=True))
        return sum(scores) / len(scores), {"Prompt": example["prompt"][:400],
                                           "Expected": example["gold"] or ("no target skill" if task.kind == "triggers" else "every check passes"),
                                           "Runs": runs_seen}

    gepa_dir = private_dir(run_dir / "gepa")

    class _Logger:  # GEPA's progress: always to gepa/log.txt, and to `log` (the CLI prints it)
        def log(self, message: str) -> None:
            append_private_line(gepa_dir / "log.txt", str(message))
            if log:
                log(message)

    settings = GEPAConfig(
        engine=EngineConfig(max_metric_calls=budget, run_dir=str(gepa_dir), seed=42,
                            parallel=workers > 1, max_workers=max(1, workers), cache_evaluation=True,
                            cache_evaluation_storage="memory", use_cloudpickle=False, display_progress_bar=False,
                            candidate_selection_strategy="pareto"),
        reflection=ReflectionConfig(reflection_lm=reflector or ClaudeReflector(reflection_model, run_dir / "reflections.jsonl"),
                                    reflection_minibatch_size=3, skip_perfect_score=True, perfect_score=1.0,
                                    reflection_prompt_template=task.template),
        tracking=TrackingConfig(logger=_Logger()))
    try:
        result = optimize_anything(seed_candidate=task.seed_candidate(), evaluator=evaluator, dataset=train_set,
                                   valset=val_set, config=settings)
    finally:
        task.finish()
    best = task.candidate_arm(result.best_candidate)
    scores = list(result.val_aggregate_scores)
    summary = {"best_idx": result.best_idx, "best": best, "val_aggregate_scores": scores,
               "candidates": [task.candidate_arm(c) for c in result.candidates],
               "total_metric_calls": getattr(result, "total_metric_calls", None)}
    write_private_json(run_dir / "train_result.json", summary)
    config = load_config(run_dir)
    config["arms"]["gepa"] = best
    save_config(run_dir, config)
    return {"best_idx": result.best_idx, "seed_val_score": scores[0], "best_val_score": scores[result.best_idx],
            "candidates": len(summary["candidates"]), "metric_calls": summary["total_metric_calls"]}
