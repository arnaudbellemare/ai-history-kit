"""Pre-filled labels for skill-trigger prompts: from your history, generated prompts, and a judge.

Every proposal records where it came from, and only labels you confirmed reach the test split.
"""

import json
import re
import tempfile
import unittest
from pathlib import Path

from aihistory.optimize import dataset
from aihistory.optimize.textmodels import FakeText
from tests.fixtures import assistant, user
from tests.test_kit import run

DESCRIPTION = "DESCRIPTION-UNDER-TEST: use for demos."
BODY = "Build demo walkthroughs: script the steps, record the screen, add captions."


def skill_load(name, sid):
    return {**assistant({"type": "tool_use", "id": f"t-{sid}", "name": "Skill", "input": {"skill": name}}), "sessionId": sid}


def said(text, sid):
    return {**user(text), "sessionId": sid}


def slash(name, args, sid):
    return said(f"<command-name>/{name}</command-name><command-args>{args}</command-args>", sid)


SESSIONS = {
    "accepted": ["show me the onboarding demo please", "load:demo", "great, thanks"],
    "interrupted": ["demo of the new parser would be cool", "load:demo", "[Request interrupted by user]"],
    "corrected": ["make a demo reel for the launch", "load:demo", "no, not that skill, just write a script"],
    "missed": ["walk me through the product tour", "reply:Sure, here is the tour.", "/demo:"],
    "explicit": ["/demo:build the onboarding walkthrough for new users"],
    "other": ["rename the variables in this module", "reply:Done.", "what does this regex match exactly?"],
}


def make_history(home: Path) -> None:
    project = home / ".claude" / "projects" / "-Users-someone-proj"
    project.mkdir(parents=True)
    for sid, steps in SESSIONS.items():
        rows = []
        for step in steps:
            if step.startswith("load:"):
                rows.append(skill_load(step[5:], sid))
            elif step.startswith("reply:"):
                rows.append({**assistant({"type": "text", "text": step[6:]}), "sessionId": sid})
            elif step.startswith("/"):
                name, _, args = step[1:].partition(":")
                rows.append(slash(name, args, sid))
            else:
                rows.append(said(step, sid))
        (project / f"{sid}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def make_skill(root: Path) -> Path:
    path = root / "skills" / "demo" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(f"---\nname: demo\ndescription: {DESCRIPTION}\n---\n\n{BODY}\n", encoding="utf-8")
    return path


def generator_reply(prompt):
    return "```json\n" + json.dumps([
        {"prompt": "could you put together a walkthrough video of signup?", "should_trigger": True},
        {"prompt": "record the screen while I click through checkout", "should_trigger": True},
        {"prompt": "write the release notes for the demo build", "should_trigger": False},
        {"prompt": "what does demo mean in our branch names?", "should_trigger": False},
    ]) + "\n```"


def judge_reply(prompt):
    keys = re.findall(r"^\[(k\d+)\] (.*)$", prompt, re.MULTILINE)
    return json.dumps([{"key": key, "skills": ["demo"] if "demo" in text or "tour" in text else [],
                        "confidence": 0.9 if "demo" in text else 0.55} for key, text in keys])


class LabelProposalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.home, self.out = root / "home", root / "out"
        make_history(self.home)
        self.skill = make_skill(root)
        self.assertEqual(run("--home", str(self.home), "--out", str(self.out), "run"), 0)

    def mine(self, **kwargs):
        options = {"negatives": 10, "seed": 1, "skill_files": {"demo": self.skill}, **kwargs}
        run_dir = dataset.mine(self.out, "p", ["demo"], **options)
        candidates = json.loads((run_dir / "candidates.json").read_text())
        return run_dir, {c["text"]: c for c in candidates}

    def test_history_rules_pre_fill_labels_with_their_source(self):
        _, found = self.mine()
        proposals = {text: [(p["by"], p["gold"]) for p in c["proposals"]] for text, c in found.items() if c["proposals"]}
        self.assertEqual(proposals, {
            "show me the onboarding demo please": [("history:accepted", ["demo"])],
            "demo of the new parser would be cool": [("history:interrupted", [])],
            "make a demo reel for the launch": [("history:corrected", [])],
            "walk me through the product tour": [("history:missed-then-invoked", ["demo"])],
            "build the onboarding walkthrough for new users": [("history:invoked", ["demo"])],
        })

    def test_generated_prompts_come_from_the_skill_body_not_its_description(self):
        model = FakeText(generator_reply)
        _, found = self.mine(generate=2, generator=model)
        self.assertEqual(len(model.prompts), 1)
        self.assertIn(BODY, model.prompts[0])
        self.assertNotIn("DESCRIPTION-UNDER-TEST", model.prompts[0], "the generator must not see the text under test")
        generated = {t: c["proposals"][0]["gold"] for t, c in found.items() if c.get("generated")}
        self.assertEqual(generated, {"could you put together a walkthrough video of signup?": ["demo"],
                                     "record the screen while I click through checkout": ["demo"],
                                     "write the release notes for the demo build": [],
                                     "what does demo mean in our branch names?": []})
        self.assertTrue(all(c["proposals"][0]["by"] == "generated:fake" for c in found.values() if c.get("generated")))

    def test_the_judge_labels_the_rest_without_seeing_descriptions(self):
        model = FakeText(judge_reply)
        _, found = self.mine(judge=model)
        self.assertTrue(model.prompts)
        self.assertTrue(all(BODY in p and "DESCRIPTION-UNDER-TEST" not in p for p in model.prompts))
        judged = {t: c["proposals"] for t, c in found.items() if any(p["by"] == "judge:fake" for p in c["proposals"])}
        self.assertIn("rename the variables in this module", judged)
        self.assertNotIn("show me the onboarding demo please", judged, "history-labelled prompts are not re-judged")
        self.assertEqual(judged["rename the variables in this module"][0]["gold"], [])
        self.assertEqual(judged["rename the variables in this module"][0]["confidence"], 0.55)

    def test_an_unreadable_judge_reply_abstains(self):
        _, found = self.mine(judge=FakeText(lambda prompt: "I cannot answer in JSON, sorry."))
        self.assertFalse(any(p["by"].startswith("judge:") for c in found.values() for p in c["proposals"]))

    def test_only_labels_you_confirmed_reach_the_test_split(self):
        run_dir, found = self.mine(judge=FakeText(judge_reply))
        label = (run_dir / "LABEL.md").read_text()
        self.assertIn("- [ ] I checked this", label)
        sections = re.split(r"(?m)^(?=## c\d+$)", label)
        checked = {"show me the onboarding demo please", "walk me through the product tour",
                   "build the onboarding walkthrough for new users"}  # every other prompt stays unchecked
        for number, section in enumerate(sections):
            if not any(f"> {text}" in section for text in checked):
                continue
            if "show me the onboarding demo please" in section:  # you overrule this pre-fill
                section = section.replace("- [x] demo", "- [ ] demo").replace(f"- [ ] {dataset.NONE}", f"- [x] {dataset.NONE}")
            sections[number] = section.replace("- [ ] I checked this", "- [x] I checked this")
        (run_dir / "LABEL.md").write_text("".join(sections))
        split = dataset.apply_labels(run_dir, test=0.5, val=0.2, seed=3)
        self.assertTrue(split["test"], "some confirmed prompts are in the test split")
        self.assertTrue(all(e["confirmed"] for e in split["test"]), "the test split holds only labels you confirmed")
        unconfirmed = [e for part in ("train", "val") for e in split[part] if not e["confirmed"]]
        self.assertTrue(unconfirmed, "unconfirmed pre-fills may still train")
        audit = json.loads((run_dir / "dataset.json").read_text())["label_audit"]
        self.assertEqual(audit["history:accepted"], {"kept": 0, "changed": 1}, "your override is counted against its source")

    def test_the_report_shows_how_often_you_kept_each_source(self):
        from aihistory.optimize import agents, experiment

        run_dir, _ = self.mine()
        label = (run_dir / "LABEL.md").read_text().replace("- [ ] I checked this", "- [x] I checked this")
        (run_dir / "LABEL.md").write_text(label)
        dataset.apply_labels(run_dir, test=0.6, val=0.0, seed=1)
        experiment.set_arm(run_dir, "original", {"demo": "Builds demos."})
        experiment.set_arm(run_dir, "control", {"demo": "Use when the user asks for a demo."})
        experiment.evaluate(run_dir, agents.FakeAgent(), "test", ["original", "control"], runs=1, workers=1)
        report = experiment.report(run_dir, "test")
        self.assertIn("## Pre-filled labels you checked", report)
        self.assertIn("| history:accepted | 1 | 0 |", report)

    def test_model_calls_over_the_limit_need_yes(self):
        model = FakeText(judge_reply)
        with self.assertRaisesRegex(ValueError, "model calls planned"):
            self.mine(judge=model, generate=2, generator=model, max_calls=1)
        self.assertEqual(model.prompts, [], "nothing is sent before the limit is accepted")

    def test_the_cli_pre_fills_and_asks_for_confirmation(self):
        status = run("--home", str(self.home), "--out", str(self.out), "optimize", "mine", "--run", "cli",
                     "--skill", "demo", "--skills-dir", str(self.skill.parent.parent), "--judge", "fake")
        self.assertEqual(status, 0)
        label = (self.out / "optimize" / "cli" / "LABEL.md").read_text()
        self.assertIn("I checked this", label)
        self.assertIn("pre-filled from history:accepted", label)


if __name__ == "__main__":
    unittest.main()
