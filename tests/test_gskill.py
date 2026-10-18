"""gskill for Python repositories: tasks from mutations, disjoint splits, grading, and training."""

import ast
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from aihistory.optimize import experiment, gskill
from aihistory.optimize.agents import FakeAgent, RunResult
from aihistory.optimize.tasks import make_task
from tests.gskill_fixture import BREAKS, make_repo
from tests.test_kit import run


def restore(workspace: Path, task: dict, repo: Path) -> None:
    for relative in task["mutated_files"]:
        shutil.copy2(repo / relative, workspace / relative)


class GskillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):  # making tasks runs the suite once per candidate, so build them once
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.repo, cls.command = make_repo(root / "repo")
        cls.run_dir = gskill.init_run(root / "out", "g", cls.repo, cls.command, k=2, candidates=60, workers=4)
        cls.config = experiment.load_config(cls.run_dir)
        cls.split = json.loads((cls.run_dir / "dataset.json").read_text())
        cls.bugs = json.loads((cls.run_dir / "bugs.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_mutations_that_break_tests_become_tasks(self):
        self.assertGreaterEqual(len(self.bugs), 6)
        self.assertEqual(self.config["base_fail"], [])
        for bug in self.bugs:
            with self.subTest(bug=bug["bug"]):
                self.assertTrue(1 <= len(bug["FAIL_TO_PASS"]) <= 20)
                ast.parse(bug["mutated_source"])
                self.assertFalse(bug["file"].startswith("tests/"), "test files are never mutated")
        self.assertIn("The task is done when every test that passed before this change passes again.",
                      next(t for part in self.split.values() for t in part)["prompt"])

    def test_each_bug_lives_in_one_split_and_a_task_spans_files(self):
        splits_of = {}
        for name, tasks in self.split.items():
            for task in tasks:
                self.assertEqual(len(task["mutated_files"]), 2, "two bugs, from different files")
                for bug in task["components"]:
                    splits_of.setdefault(bug, set()).add(name)
        self.assertTrue(splits_of)
        self.assertTrue(all(len(names) == 1 for names in splits_of.values()), "no bug appears in two splits")

    def test_every_split_gets_enough_bugs_for_a_task(self):
        fractions = {"test": 0.4, "val": 0.2, "train": 0.4}
        self.assertEqual(gskill.pool_sizes(7, 2, fractions), {"test": 3, "val": 2, "train": 2})
        self.assertEqual(gskill.pool_sizes(30, 3, fractions), {"test": 12, "val": 6, "train": 12})
        self.assertEqual(sum(gskill.pool_sizes(9, 3, fractions).values()), 9)
        self.assertTrue(all(n >= 3 for n in gskill.pool_sizes(9, 3, fractions).values()))

    def test_combined_bugs_always_come_from_different_files(self):
        import random

        pool = [{"bug": f"a{n}", "file": "calc/ops.py"} for n in range(5)] + [{"bug": "b0", "file": "calc/text.py"}]
        combos = gskill.draw_combos(pool, 2, random.Random(1))
        self.assertTrue(combos)
        self.assertTrue(all(len({b["file"] for b in combo}) == 2 for combo in combos))

    def test_grading_restores_the_tests_and_names_the_outcome(self):
        task = next(t for part in self.split.values() for t in part)
        failing = " ".join(task["FAIL_TO_PASS"])
        broken_file, original, broken, _ = next(b for b in BREAKS if b[3] not in failing)

        def edit_tests(workspace):
            (workspace / "tests" / "test_calc.py").write_text("import unittest\n", encoding="utf-8")
            first = next(iter(task["mutated_files"]))
            with open(workspace / first, "a", encoding="utf-8") as handle:
                handle.write("# looked at this\n")

        def regress(workspace):
            restore(workspace, task, self.repo)
            path = workspace / broken_file
            path.write_text(path.read_text(encoding="utf-8").replace(original, broken), encoding="utf-8")

        actions = {"all_passed": lambda w: restore(w, task, self.repo), "f2p_failed": edit_tests,
                   "bugged_file_untouched": lambda w: None, "p2p_regression": regress}
        for expected, act in actions.items():
            with self.subTest(expected=expected):
                runner = make_task(self.config)
                workspace, before = runner.setup(FakeAgent(), {"skill": ""}, task)
                try:
                    act(workspace)
                    score, details = runner.score(task, [], workspace, before)
                finally:
                    runner.release(workspace)
                self.assertEqual(details["status"], expected)
                self.assertEqual(score, 1.0 if expected == "all_passed" else 0.0)

    def test_a_skill_that_describes_the_generator_is_flagged(self):
        self.assertGreaterEqual(len(gskill.generator_shapes("Check for off-by-one constants and negated if conditions.")), 2)
        self.assertEqual(gskill.generator_shapes("Run the tests with unittest; the package lives in calc/."), [])

    def test_the_cli_makes_tasks_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            self.assertEqual(run("--out", str(out), "optimize", "gskill", "--run", "c", "--repo", str(self.repo),
                                 "--test-cmd", self.command, "--k", "2", "--candidates", "40"), 0)
            self.assertEqual(run("--out", str(out), "optimize", "arm", "--run", "c", "control", "--text",
                                 "Run the whole suite before you stop."), 0)
            self.assertEqual(run("--out", str(out), "optimize", "eval", "--run", "c", "--agent", "fake", "--runs", "1",
                                 "--split", "train"), 0)
            self.assertEqual(run("--out", str(out), "optimize", "report", "--run", "c", "--split", "train"), 0)
            self.assertIn("| baseline | fake:fake |", (out / "optimize" / "c" / "REPORT.md").read_text())
        self.assertEqual(self.repo.joinpath("calc", "ops.py").read_text(encoding="utf-8").count("return a + b"), 1,
                         "the repository itself is never changed")


class FixerAgent:
    """Restores every mutated file when its skill says to fix every failing test; otherwise only the first."""

    name, skills_folder, instructions_file, model = "fixer", ".fixer/skills", "AGENTS.md", "script"

    def __init__(self, repo: Path):
        self.repo = repo

    def prepare(self, workspace, skills, instructions=None):
        if instructions:
            (workspace / self.instructions_file).write_text(instructions, encoding="utf-8")

    def run(self, workspace, prompt, writer, permissions=None):
        skill = workspace / self.instructions_file
        thorough = skill.exists() and "every failing test" in skill.read_text(encoding="utf-8")
        writer.event("user_prompt", {"prompt": prompt})
        changed = [p for p in sorted(self.repo.rglob("*.py")) if (workspace / p.relative_to(self.repo)).read_text()
                   != p.read_text()]
        for path in changed if thorough else changed[:1]:
            relative = path.relative_to(self.repo).as_posix()
            writer.event("tool_call", {"tool_name": "Edit", "arguments": {"file_path": relative}})
            shutil.copy2(path, workspace / relative)
        writer.event("assistant_response", {"text": "Fixed."})
        return RunResult(0, 0.0, 0.0)


try:
    import gepa  # noqa: F401
except ImportError:
    gepa = None


@unittest.skipUnless(gepa, "needs the optional gepa package (Python 3.10 to 3.14)")
class GskillTrainTests(unittest.TestCase):
    def test_gepa_learns_a_repository_skill(self):
        from aihistory.optimize import gepa_train

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, command = make_repo(root / "repo")
            run_dir = gskill.init_run(root / "out", "t", repo, command, k=2, candidates=60, workers=4)
            asked = []

            def reflector(prompt):
                asked.append(prompt)
                return "```\nFix every failing test before you stop, not only the one in the report.\n```"

            result = gepa_train.train(run_dir, FixerAgent(repo), budget=10, runs=1, reflector=reflector, workers=1)
            self.assertTrue(asked)
            self.assertIn("f2p_failed", asked[0], "the reflector sees how runs ended")
            self.assertIn("every failing test", experiment.load_config(run_dir)["arms"]["gepa"]["skill"])
            self.assertGreater(result["best_val_score"], result["seed_val_score"])


if __name__ == "__main__":
    unittest.main()
