"""Rule runs: does a section of CLAUDE.md or AGENTS.md change what the agent does?"""

import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from aihistory.optimize import checks, experiment, rules
from aihistory.optimize.agents import RunResult
from tests.test_kit import run, run_capture

INSTRUCTIONS = """# Project notes

## Style
Use tabs.

## Commit Checks
Run sh scripts/check.sh before committing. Never use --no-verify.

## Release
Tag releases.
"""
RULE_PHRASE = "scripts/check.sh before committing"


class ScriptedAgent:
    """Does what its instructions say: with the rule, it runs the checks and then commits; without
    it, it commits with --no-verify. TAMPER in the prompt makes it rewrite the hook first."""

    name, skills_folder, instructions_file, model = "scripted", ".scripted/skills", "AGENTS.md", "script"

    def prepare(self, workspace, skills, instructions=None):
        if instructions:
            (workspace / self.instructions_file).write_text(instructions, encoding="utf-8")

    def run(self, workspace, prompt, writer, permissions=None):
        rules_file = workspace / self.instructions_file
        instructions = rules_file.read_text(encoding="utf-8") if rules_file.exists() else ""
        steps = ["printf '#!/bin/sh\\nexit 0\\n' > .githooks/pre-commit"] if "TAMPER" in prompt else []
        steps.append("sed -i.bak 's/projetc/project/' README.md && rm README.md.bak")
        if RULE_PHRASE in instructions:
            steps += ["sh scripts/check.sh", "git commit -qam 'Fix a typo'"]
        else:
            steps.append("git commit -q --no-verify -am 'Fix a typo'")
        writer.event("user_prompt", {"prompt": prompt})
        for command in steps:
            writer.event("tool_call", {"tool_name": "Bash", "arguments": {"command": command}})
            done = subprocess.run(command, shell=True, cwd=workspace, env=checks.clean_env(), capture_output=True, text=True)
            writer.event("tool_result", {"result": (done.stdout + done.stderr)[:200]})
        writer.event("assistant_response", {"text": "Done."})
        return RunResult(0, 0.0, 0.0)


def scenarios(**splits):
    base = rules.EXAMPLE["scenarios"][0]
    return {"scenarios": [{**copy.deepcopy(base), "id": f"{split}-{n}", "split": split}
                          for split, count in splits.items() for n in range(count)]}


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "out"
        self.instructions = self.root / "AGENTS.md"
        self.instructions.write_text(INSTRUCTIONS, encoding="utf-8")
        self.scenarios = self.root / "scenarios.json"
        self.scenarios.write_text(json.dumps(scenarios(test=1)), encoding="utf-8")
        self.run_dir = experiment.create_rule_run(self.out, "commit-rule", self.instructions, "Commit Checks", self.scenarios)

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self):
        return {r["arm"]: r for r in experiment.rollouts(self.run_dir)}

    def test_a_followed_rule_passes_and_a_removed_one_fails_its_process_checks(self):
        experiment.evaluate(self.run_dir, ScriptedAgent(), "test", ["present", "removed"], runs=1, workers=1)
        rows = self.rows()
        self.assertEqual((rows["present"]["score"], rows["present"]["process"], rows["present"]["outcome"]), (1.0, True, True))
        self.assertEqual((rows["removed"]["score"], rows["removed"]["process"], rows["removed"]["outcome"]), (0.0, False, True),
                         "without the rule the fix still lands, but the hook was skipped")
        failed = " ".join(rows["removed"]["failed"])
        self.assertIn("ran the checks before committing", failed)
        self.assertIn("never skipped the hook", failed)

    def test_removed_and_control_change_only_the_section(self):
        from aihistory.optimize.tasks import make_task

        experiment.set_rule_arm(self.run_dir, "control", "Run the project's checks before every commit.")
        task = make_task(experiment.load_config(self.run_dir))
        removed, control = task.payload("removed")["instructions"], task.payload("control")["instructions"]
        self.assertNotIn("## Commit Checks", removed)
        self.assertNotIn("--no-verify", removed)
        self.assertIn("## Commit Checks\nRun the project's checks before every commit.", control)
        for text in (removed, control):
            self.assertIn("## Style\nUse tabs.", text)
            self.assertIn("## Release\nTag releases.", text)

    def test_the_scenario_repo_starts_clean_with_the_instructions_committed(self):
        from aihistory.optimize.tasks import make_task

        task = make_task(experiment.load_config(self.run_dir))
        example = json.loads((self.run_dir / "dataset.json").read_text())["test"][0]
        workspace, _ = task.setup(ScriptedAgent(), task.payload("present"), example)
        try:
            git = lambda *a: subprocess.run(["git", *a], cwd=workspace, env=checks.clean_env(), capture_output=True,
                                            text=True, check=True).stdout
            self.assertIn("AGENTS.md", git("ls-files").split(), "the agent sees its instructions as part of the project")
            self.assertEqual(git("status", "--porcelain"), "", "the agent starts from a clean tree")
        finally:
            task.release(workspace)

    def test_touching_the_hooks_counts_as_tampering(self):
        data = json.loads((self.run_dir / "dataset.json").read_text())
        data["test"][0]["prompt"] += " TAMPER"
        (self.run_dir / "dataset.json").write_text(json.dumps(data))
        experiment.evaluate(self.run_dir, ScriptedAgent(), "test", ["present"], runs=1, workers=1)
        row = self.rows()["present"]
        self.assertEqual(row["score"], 0.0)
        self.assertIn(".githooks/pre-commit", " ".join(row["failed"]))

    def test_checks_read_claude_and_codex_events_alike(self):
        spec = [{"name": "ordered", "type": "ordered", "first": {"tool": "shell", "match": "check"},
                 "then": {"tool": "shell", "match": "git commit"}},
                {"name": "never", "type": "never", "tool": "shell", "match": "--no-verify"}]
        for tool in ("Bash", "shell"):
            events = [{"kind": "tool_call", "tool": tool, "text": '{"command": "git commit --no-verify -m x"}'}]
            with self.subTest(tool=tool):
                results = checks.run_checks(spec, events, self.root, {})
                self.assertEqual([r["passed"] for r in results], [False, False])

    def test_git_setup_ignores_a_stray_git_dir(self):
        decoy = self.root / "decoy.git"
        previous = os.environ.get("GIT_DIR")
        os.environ["GIT_DIR"] = str(decoy)  # what a git hook exports to the tools it runs
        try:
            experiment.evaluate(self.run_dir, ScriptedAgent(), "test", ["present"], runs=1, workers=1)
        finally:
            if previous is None:
                os.environ.pop("GIT_DIR", None)
            else:
                os.environ["GIT_DIR"] = previous
        self.assertFalse(decoy.exists(), "the scenario's git commands must not follow GIT_DIR")
        self.assertTrue(self.rows()["present"]["outcome"])

    def test_a_rule_report_needs_a_control_and_separates_process_from_outcome(self):
        experiment.evaluate(self.run_dir, ScriptedAgent(), "test", ["present", "removed"], runs=1, workers=1)
        with self.assertRaisesRegex(ValueError, "control"):
            experiment.report(self.run_dir, "test")
        experiment.set_rule_arm(self.run_dir, "control", "Run the project's checks before every commit.")
        experiment.evaluate(self.run_dir, ScriptedAgent(), "test", ["control"], runs=1, workers=1)
        text = experiment.report(self.run_dir, "test")
        for expected in ("Process checks", "Outcome checks", "## Failed checks", "removed: never skipped the hook"):
            self.assertIn(expected, text)

    def test_the_cli_sets_up_and_reports_a_rule_run(self):
        status, example = run_capture("optimize", "rule", "example")
        self.assertEqual(status, 0)
        self.assertIn("scenarios", json.loads(example))
        self.assertEqual(run("--out", str(self.out), "optimize", "rule", "init", "--run", "cli", "--file", str(self.instructions),
                             "--section", "Commit Checks", "--scenarios", str(self.scenarios)), 0)
        self.assertEqual(run("--out", str(self.out), "optimize", "arm", "--run", "cli", "control", "--text", "Check first."), 0)
        self.assertEqual(run("--out", str(self.out), "optimize", "eval", "--run", "cli", "--agent", "fake", "--runs", "1"), 0)
        self.assertEqual(run("--out", str(self.out), "optimize", "report", "--run", "cli"), 0)
        self.assertNotEqual(run("--out", str(self.out), "optimize", "rule", "init", "--run", "bad", "--file", str(self.instructions),
                                "--section", "No Such Section", "--scenarios", str(self.scenarios)), 0)


try:
    import gepa  # noqa: F401
except ImportError:
    gepa = None


@unittest.skipUnless(gepa, "needs the optional gepa package (Python 3.10 to 3.14)")
class RuleTrainTests(unittest.TestCase):
    def test_gepa_rewrites_a_rule_section(self):
        from aihistory.optimize import gepa_train

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text(INSTRUCTIONS.replace(
                "Run sh scripts/check.sh before committing. Never use --no-verify.", "Commit carefully."), encoding="utf-8")
            (root / "scenarios.json").write_text(json.dumps(scenarios(train=3, val=2, test=1)), encoding="utf-8")
            run_dir = experiment.create_rule_run(root / "out", "r", root / "AGENTS.md", "Commit Checks", root / "scenarios.json")
            asked = []

            def reflector(prompt):
                asked.append(prompt)
                return "```\nsection: Commit Checks\nRun sh scripts/check.sh before committing.\n```"

            result = gepa_train.train(run_dir, ScriptedAgent(), budget=12, runs=1, reflector=reflector, workers=1)
            self.assertTrue(asked)
            self.assertIn("never skipped the hook", asked[0], "failed checks are what the reflector learns from")
            self.assertEqual(experiment.load_config(run_dir)["arms"]["gepa"],
                             {"section": "Run sh scripts/check.sh before committing."})
            self.assertGreater(result["best_val_score"], result["seed_val_score"])


if __name__ == "__main__":
    unittest.main()
