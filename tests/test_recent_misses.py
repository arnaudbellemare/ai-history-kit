"""A monthly routine: rebuild from a backup kept elsewhere, then mine prompts for the skills you had to invoke yourself."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from aihistory.normalize import read_corpus
from tests.fixtures import assistant
from tests.test_kit import run, run_capture
from tests.test_label_proposals import said, skill_load, slash

NOW = datetime.now(timezone.utc)


def at(row, days_ago, seconds=0):
    stamp = NOW - timedelta(days=days_ago) + timedelta(seconds=seconds)
    return {**row, "timestamp": stamp.isoformat().replace("+00:00", "Z")}


def reply(text, sid):
    return {**assistant({"type": "text", "text": text}), "sessionId": sid}


SESSIONS = {
    # The agent loaded nothing, so you invoked /demo yourself: a miss, two days ago.
    "recent-miss": [(said("walk me through the product tour", "recent-miss"), 2), (reply("Sure.", "recent-miss"), 2),
                    (slash("demo", "", "recent-miss"), 2)],
    # The same for /deploy, but three months ago.
    "old-miss": [(said("push this build to production", "old-miss"), 90), (reply("OK.", "old-miss"), 90),
                 (slash("deploy", "", "old-miss"), 90)],
    # The agent did load deploy before you invoked it: not a miss.
    "loaded": [(said("ship the release now", "loaded"), 1), (skill_load("deploy", "loaded"), 1),
               (slash("deploy", "", "loaded"), 1)],
    # /compact is a command, not one of your skills.
    "builtin": [(said("this conversation is getting long", "builtin"), 1), (reply("Yes.", "builtin"), 1),
                (slash("compact", "", "builtin"), 1)],
}


class RecentMissTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home, self.out = self.root / "home", self.root / "out"
        project = self.home / ".claude" / "projects" / "-Users-someone-proj"
        project.mkdir(parents=True)
        for sid, rows in SESSIONS.items():
            lines = [at(row, days, seconds=n) for n, (row, days) in enumerate(rows)]
            (project / f"{sid}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
        self.skills = self.root / "skills"
        for name in ("demo", "deploy"):
            path = self.skills / name / "SKILL.md"
            path.parent.mkdir(parents=True)
            path.write_text(f"---\nname: {name}\ndescription: Use for {name}.\n---\n\nHow to {name}.\n", encoding="utf-8")

    def mine(self, *args):
        self.assertEqual(run("--home", str(self.home), "--out", str(self.out), "run"), 0)
        return run_capture("--home", str(self.home), "--out", str(self.out), "optimize", "mine", "--run", "misses",
                           "--skills-dir", str(self.skills), *args)

    def test_missed_since_targets_the_skills_you_invoked_yourself_after_a_recent_miss(self):
        status, printed = self.mine("--missed-since", "30")
        self.assertEqual(status, 0, printed)
        run_dir = self.out / "optimize" / "misses"
        targets = json.loads((run_dir / "config.json").read_text())["targets"]
        self.assertEqual(targets, ["demo"], "deploy's miss is three months old; its recent use was loaded; /compact isn't a skill")
        candidates = {c["text"]: c for c in json.loads((run_dir / "candidates.json").read_text())}
        miss = candidates.get("walk me through the product tour")
        self.assertIsNotNone(miss, "the prompt the agent missed is offered for labelling")
        self.assertEqual([(p["by"], p["gold"]) for p in miss["proposals"]], [("history:missed-then-invoked", ["demo"])])

    def test_a_month_without_misses_mines_nothing(self):
        status, printed = self.mine("--missed-since", "1")
        self.assertEqual(status, 0, printed)
        self.assertIn("nothing to mine", printed)
        self.assertFalse((self.out / "optimize" / "misses").exists(), "no run is created when no skill missed")

    def test_run_reads_a_backup_kept_elsewhere_without_copying_it(self):
        drive = self.root / "drive"
        self.assertEqual(run("--home", str(self.home), "--out", str(drive), "backup"), 0)
        empty = self.root / "empty-home"
        empty.mkdir()
        status = run("--home", str(empty), "--out", str(self.out), "--from-backup", str(drive), "run")
        self.assertEqual(status, 0)
        said_texts = {e["text"] for r in read_corpus(self.out) for e in r["events"] if e["role"] == "user"}
        self.assertIn("walk me through the product tour", said_texts, "the corpus comes from the backup, not the empty home")
        self.assertFalse((self.out / "raw").exists() or (self.out / "CATALOG.json").exists(), "nothing is copied")

    def test_a_folder_without_a_backup_fails_clearly(self):
        status, printed = run_capture("--home", str(self.home), "--out", str(self.out), "--from-backup",
                                      str(self.root / "nowhere"), "run")
        self.assertEqual(status, 1)
        self.assertIn("No backup found", printed)


if __name__ == "__main__":
    unittest.main()
