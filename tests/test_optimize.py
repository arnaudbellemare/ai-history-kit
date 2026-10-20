import json
import sys
import tempfile
import unittest
from pathlib import Path

from aihistory.adapters import agent_trace
from aihistory.optimize import agents, dataset, experiment, scorers, traces
from tests.fixtures import make_home
from tests.test_kit import run

CLAUDE_STREAM = [  # trimmed `claude -p --output-format stream-json --verbose` output
    {"type": "system", "subtype": "init", "model": "claude-sonnet", "skills": ["demo", "other"]},
    {"type": "assistant", "message": {"content": [
        {"type": "thinking", "thinking": "The user wants a demo; the demo skill fits."},
        {"type": "tool_use", "id": "t1", "name": "Skill", "input": {"skill": "demo"}}]}},
    {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "LOADED"}]}},
    {"type": "assistant", "message": {"content": [{"type": "text", "text": "LOADED"}]}},
    {"type": "result", "subtype": "success", "total_cost_usd": 0.01, "duration_ms": 900, "num_turns": 2},
]
CODEX_JSONL = [  # trimmed `codex exec --json` output
    {"type": "thread.started", "thread_id": "thread-1"},
    {"type": "item.completed", "item": {"id": "i0", "type": "reasoning", "text": "A demo is wanted."}},
    {"type": "item.completed", "item": {"id": "i1", "type": "command_execution",
                                        "command": "cat .agents/skills/demo/SKILL.md", "aggregated_output": "LOADED",
                                        "exit_code": 0}},
    {"type": "item.completed", "item": {"id": "i2", "type": "agent_message", "text": "LOADED"}},
    {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}},
]


def example(id_, prompt, gold, cls):
    return {"id": id_, "prompt": prompt, "gold": gold, "cls": cls, "provenance": "test"}


class OptimizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.home = make_home(self.root / "home")
        self.out = self.root / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def small_run(self, arms):
        run_dir = experiment.create_run(self.out, "t", targets={"demo": "Builds demos."}, decoys={"other": "Unrelated work."})
        experiment.set_arm(run_dir, "original", {"demo": "Builds demos."})
        for name, descriptions in arms.items():
            experiment.set_arm(run_dir, name, descriptions)
        data = [example("p1", "please build a demo of the parser", ["demo"], "pos:demo"),
                example("p2", "make me a quick demo video", ["demo"], "pos:demo"),
                example("n1", "rename this variable", [], "neg"),
                example("n2", "the old demo folder can go; delete it", [], "neg")]  # mentions demo, wants no skill
        (run_dir / "dataset.json").write_text(json.dumps({"train": [], "val": [], "test": data}))
        return run_dir

    def test_trigger_dataset_comes_from_owner_messages_only(self):
        self.assertEqual(run("--home", str(self.home), "--out", str(self.out), "run"), 0)
        run_dir = dataset.mine(self.out, "triggers", ["demo"], negatives=50, seed=1)  # draws them all
        label = (run_dir / "LABEL.md").read_text(encoding="utf-8")
        self.assertIn("evidence: called demo", label)
        self.assertIn("Notes are in $HOME/notes.md", label, "the owner's prompt just before the Skill call")
        for not_owner in ("cost-aware model selector", "purple everywhere", "<task-notification>", "I usually cite"):
            self.assertNotIn(not_owner, label, "subagent, headless and harness text never becomes an eval prompt")
        (run_dir / "LABEL.md").write_text(label.replace("- [ ] none of these", "- [x] none of these"), encoding="utf-8")
        split = dataset.apply_labels(run_dir, seed=1)
        ids = [[e["id"] for e in split[name]] for name in ("train", "val", "test")]
        flat = [i for part in ids for i in part]
        self.assertEqual(len(flat), len(set(flat)), "a prompt is in one split only")
        self.assertIn(["demo"], [e["gold"] for part in split.values() for e in part])

    def test_rollouts_are_agent_trace_sessions_that_verify(self):
        run_dir = self.small_run({"control": {"demo": "Use when the user asks for a demo."}})
        experiment.evaluate(run_dir, agents.FakeAgent(), "test", ["original", "control"], runs=1)
        sessions = agent_trace.session_files(run_dir / "traces")
        self.assertEqual(len(sessions), 8)
        for path in sessions:
            self.assertEqual(agent_trace.verify_chain(path)["status"], "ok", path)
        rows = [json.loads(line) for line in (run_dir / "rollouts.jsonl").read_text().splitlines()]
        control = {r["id"]: r["score"] for r in rows if r["arm"] == "control"}
        self.assertEqual(control, {"p1": 1.0, "p2": 1.0, "n1": 1.0, "n2": 0.0}, "loading demo for n2 is a false trigger")

    def test_claude_and_codex_output_score_alike(self):
        loaded = {}
        for name, convert, lines in (("claude", traces.from_claude_stream, CLAUDE_STREAM),
                                     ("codex", traces.from_codex_jsonl, CODEX_JSONL)):
            writer = traces.TraceWriter(self.root / "store", f"{name}-session-0001", name)
            writer.event("user_prompt", {"prompt": "show me a demo"})
            convert((json.dumps(line) for line in lines), writer)
            writer.close()
            record = agent_trace.read_session(writer.events_path, self.root / "store")
            loaded[name] = scorers.skills_loaded(record["events"])
        self.assertEqual(loaded, {"claude": ["demo"], "codex": ["demo"]})

    def test_a_report_needs_a_control_arm(self):
        run_dir = self.small_run({})
        experiment.evaluate(run_dir, agents.FakeAgent(), "test", ["original"], runs=1)
        with self.assertRaisesRegex(ValueError, "control"):
            experiment.report(run_dir, "test")
        experiment.set_arm(run_dir, "control", {"demo": "Use when the user asks for a demo."})
        experiment.evaluate(run_dir, agents.FakeAgent(), "test", ["control"], runs=1)
        text = experiment.report(run_dir, "test")
        self.assertIn("McNemar", text)
        self.assertTrue((run_dir / "REPORT.md").is_file())

    def test_a_run_larger_than_the_limit_needs_yes(self):
        run_dir = self.small_run({})
        with self.assertRaisesRegex(ValueError, "8 agent runs"):
            experiment.evaluate(run_dir, agents.FakeAgent(), "test", ["original"], runs=2, max_runs=5)
        self.assertFalse((run_dir / "rollouts.jsonl").exists(), "nothing ran")
        experiment.evaluate(run_dir, agents.FakeAgent(), "test", ["original"], runs=2, max_runs=5, yes=True)

    def test_the_test_split_is_not_rerun_with_a_changed_arm(self):
        run_dir = self.small_run({})
        experiment.evaluate(run_dir, agents.FakeAgent(), "test", ["original"], runs=1)
        experiment.set_arm(run_dir, "original", {"demo": "Builds demos and videos."})
        with self.assertRaisesRegex(ValueError, "already evaluated on the test split"):
            experiment.evaluate(run_dir, agents.FakeAgent(), "test", ["original"], runs=1)

    def test_the_cli_runs_a_dry_run_end_to_end(self):
        run_dir = self.small_run({"control": {"demo": "Use when the user asks for a demo."}})
        self.assertEqual(run("--out", str(self.out), "optimize", "eval", "--run", run_dir.name, "--agent", "fake",
                             "--arms", "original,control", "--runs", "1"), 0)
        self.assertEqual(run("--out", str(self.out), "optimize", "report", "--run", run_dir.name), 0)
        self.assertIn("| control | fake:fake |", (run_dir / "REPORT.md").read_text())


try:
    import gepa  # noqa: F401  (optional: pip install 'ai-history-kit[optimize]')
except ImportError:
    gepa = None


@unittest.skipUnless(gepa, "needs the optional gepa package (Python 3.10 to 3.14)")
class TrainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "out"
        self.run_dir = experiment.create_run(self.out, "g", targets={"demo": "Builds demos."},
                                             decoys={"other": "Unrelated work."})
        split = {"train": [example("p1", "please build a demo of the parser", ["demo"], "pos:demo"),
                           example("p3", "a short demo for the sales call", ["demo"], "pos:demo"),
                           example("n1", "rename this variable", [], "neg")],
                 "val": [example("p2", "make me a quick demo video", ["demo"], "pos:demo"),
                         example("n3", "fix the flaky login check", [], "neg")],
                 "test": [example("t1", "HELD-OUT-MARKER demo request", ["demo"], "pos:demo")]}
        (self.run_dir / "dataset.json").write_text(json.dumps(split))
        self.reflections = []

    def tearDown(self):
        self.tmp.cleanup()

    def reflector(self, prompt):
        self.reflections.append(prompt)
        return "```\nskill: demo\nUse when the user asks for a demo.\n```"

    def test_gepa_tunes_an_arm_with_the_kits_own_reflector(self):
        from aihistory.optimize import gepa_train

        result = gepa_train.train(self.run_dir, agents.FakeAgent(), budget=12, runs=1, reflector=self.reflector, workers=1)
        self.assertTrue(self.reflections, "the kit's reflector was asked for a rewrite")
        self.assertNotIn("litellm", sys.modules, "gepa's default reflection model (through litellm) is never used")
        self.assertEqual(experiment.load_config(self.run_dir)["arms"]["gepa"], {"demo": "Use when the user asks for a demo."})
        self.assertGreater(result["best_val_score"], result["seed_val_score"])

    def test_training_never_sees_the_test_split(self):
        from aihistory.optimize import gepa_train

        gepa_train.train(self.run_dir, agents.FakeAgent(), budget=12, runs=1, reflector=self.reflector, workers=1)
        self.assertFalse(any("HELD-OUT-MARKER" in prompt for prompt in self.reflections))
        rows = experiment.rollouts(self.run_dir)
        self.assertTrue(rows)
        self.assertEqual({r["split"] for r in rows}, {"train", "val"})
        self.assertNotIn("t1", {r["id"] for r in rows})

    def test_a_training_budget_over_the_limit_needs_yes(self):
        from aihistory.optimize import gepa_train

        with self.assertRaisesRegex(ValueError, "agent runs planned"):
            gepa_train.train(self.run_dir, agents.FakeAgent(), budget=100, runs=3, reflector=self.reflector, max_runs=200)
        self.assertFalse(self.reflections)


FAKE_CLAUDE = """#!/bin/sh
cat > /dev/null
{body}
"""
MAX_TURNS_STREAM = "\n".join(json.dumps(line) for line in [
    {"type": "system", "subtype": "init", "model": "claude-haiku", "skills": ["demo"]},
    {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Skill", "input": {"skill": "demo"}}]}},
    {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t2", "name": "Grep", "input": {"pattern": "x"}}]}},
    {"type": "result", "subtype": "error_max_turns", "total_cost_usd": 0.001, "num_turns": 4},
])


class ClaudeRunnerTests(unittest.TestCase):
    """The Claude Code runner, driven by a fake `claude` that prints a canned stream."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run_dir = experiment.create_run(self.root / "out", "c", targets={"demo": "Builds demos."})
        data = [example("p1", "please build a demo", ["demo"], "pos:demo")]
        (self.run_dir / "dataset.json").write_text(json.dumps({"train": [], "val": [], "test": data}))

    def fake(self, body):
        path = self.root / "claude"
        path.write_text(FAKE_CLAUDE.format(body=body))
        path.chmod(0o755)
        return agents.ClaudeCodeAgent(model="haiku", thinking=False, executable=str(path))

    def test_hitting_the_turn_limit_is_scored_not_retried(self):
        agent = self.fake(f"cat <<'STREAM'\n{MAX_TURNS_STREAM}\nSTREAM\nexit 1")
        result = experiment.evaluate(self.run_dir, agent, "test", ["original"], runs=1, workers=1)
        row, = experiment.rollouts(self.run_dir)
        self.assertEqual(result["errors"], 0, "the skill decision was made before the turn limit; score it")
        self.assertEqual((row["score"], row["loaded"], row["error"]), (1.0, ["demo"], None))

    def test_a_real_failure_stays_an_error(self):
        agent = self.fake("echo 'Not logged in' >&2\nexit 1")
        result = experiment.evaluate(self.run_dir, agent, "test", ["original"], runs=1, workers=1)
        row, = experiment.rollouts(self.run_dir)
        self.assertEqual(result["errors"], 1)
        self.assertIsNone(row["score"], "an infrastructure failure is retried, never scored")
        self.assertIn("Not logged in", row["error"])


if __name__ == "__main__":
    unittest.main()
