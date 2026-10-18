"""Scripted Claude prompts stay searchable, but cannot become personal evidence."""

import json
import tempfile
import unittest
from pathlib import Path

from aihistory import index
from aihistory.normalize import read_corpus
from aihistory.optimize import dataset
from tests.fixtures import assistant, user
from tests.test_kit import run


class HeadlessTests(unittest.TestCase):
    def test_sdk_evidence_survives_mixed_entrypoints_and_missing_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            home, out = Path(tmp) / "home", Path(tmp) / "out"
            project = home / ".claude/projects/project"
            project.mkdir(parents=True)
            variants = {
                "sdk": [user("I prefer violet scripts.", entrypoint="sdk-cli")],
                "mixed": [user("I prefer amber scripts.", entrypoint="sdk-cli"),
                          user("continue", entrypoint="cli"), user("continue again", entrypoint="cli")],
                "late": [user("I prefer teal scripts."),
                         {**assistant({"type": "text", "text": "OK"}), "entrypoint": "sdk-py"}],
                "interactive": [user("I prefer copper prose.", entrypoint="cli")],
                "legacy": [user("I prefer silver prose.")],
            }
            for name, rows in variants.items():
                if name in {"sdk", "mixed", "late"}:
                    rows.append(assistant({"type": "tool_use", "id": "s1", "name": "Skill", "input": {"skill": "demo"}}))
                (project / f"{name}.jsonl").write_text(
                    "".join(json.dumps({**row, "sessionId": name}) + "\n" for row in rows), encoding="utf-8")
            self.assertEqual(run("--home", str(home), "--out", str(out), "run"), 0)
            records = {Path(r["origin"]).stem: r for r in read_corpus(out)}
            self.assertEqual({name: r["headless"] for name, r in records.items()},
                             {name: name in {"sdk", "mixed", "late"} for name in variants})
            memories = json.loads((out / "memories/candidates.json").read_text())["candidates"]
            contents = {c["content"] for c in memories}
            self.assertTrue({"copper prose", "silver prose"} <= contents)
            mined = dataset.mine(out, "headless", ["demo"], keywords={"demo": ["prefer"]}, negatives=50)
            prompts = [c["text"] for c in json.loads((mined / "candidates.json").read_text())]
            for color in ("violet", "amber", "teal"):
                self.assertTrue(index.query(out, color)["hits"], f"{color}: scripted prompt stays searchable")
                self.assertFalse(any(color in c for c in contents), f"{color}: no scripted personal memory")
                self.assertFalse(any(color in p for p in prompts), f"{color}: no scripted optimizer prompt")
            self.assertTrue(list((out / "raw/claude-code/projects/project").glob("*.jsonl")))
            self.assertIn("3 headless", (out / "REPORT.md").read_text())
