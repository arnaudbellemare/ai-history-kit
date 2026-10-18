"""`aihistory optimize review`: confirm pre-filled labels one prompt at a time from the keyboard."""

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aihistory.optimize import dataset, review
from tests.test_kit import run, run_capture
from tests.test_label_proposals import make_history, make_skill

ACCEPTED = "show me the onboarding demo please"  # pre-filled: demo
INTERRUPTED = "demo of the new parser would be cool"  # pre-filled: none of these
MISSED = "walk me through the product tour"  # pre-filled: demo


class Keyboard:
    """Answers by which prompt is on screen; prompts without a script are skipped."""

    def __init__(self, script=None, default="s", sequence=None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.default, self.screen, self.seen = default, [], []
        self.sequence = list(sequence) if sequence is not None else None  # keys in order, whatever is shown

    def show(self, text=""):
        self.screen.append(str(text))

    def current(self):
        for line in reversed(self.screen):
            if line.lstrip().startswith("> "):
                return line.lstrip()[2:].strip()
        return None

    def ask(self, prompt=""):
        text = self.current()
        if not self.seen or self.seen[-1] != text:
            self.seen.append(text)
        if self.sequence is not None:
            key = self.sequence.pop(0) if self.sequence else "q"
            if isinstance(key, BaseException):
                raise key
            return key
        keys = self.script.get(text)
        if keys:
            key = keys.pop(0)
            if isinstance(key, BaseException):
                raise key
            return key
        return self.default


def section(text, prompt):
    return next(s for s in re.split(r"(?m)^(?=## c\d+$)", text) if f"> {prompt}" in s)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.home, self.out = root / "home", root / "out"
        make_history(self.home)
        skill = make_skill(root)
        self.assertEqual(run("--home", str(self.home), "--out", str(self.out), "run"), 0)
        self.run_dir = dataset.mine(self.out, "r", ["demo"], negatives=10, seed=1, skill_files={"demo": skill})
        self.label = self.run_dir / "LABEL.md"

    def review(self, keyboard, **kwargs):
        return review.review(self.run_dir, ask=keyboard.ask, show=keyboard.show, **kwargs)

    def checked(self, prompt):
        return "- [x] I checked this" in section(self.label.read_text(), prompt)

    def test_enter_accepts_the_pre_fill_and_marks_it_checked(self):
        self.review(Keyboard({ACCEPTED: [""]}))
        self.assertTrue(self.checked(ACCEPTED))
        self.assertIn("- [x] demo", section(self.label.read_text(), ACCEPTED))
        self.assertFalse(self.checked(MISSED), "a skipped prompt stays unchecked")

    def test_a_number_toggles_a_skill_and_n_means_none(self):
        self.review(Keyboard({INTERRUPTED: ["1", ""], ACCEPTED: ["n", ""]}))
        interrupted, accepted = section(self.label.read_text(), INTERRUPTED), section(self.label.read_text(), ACCEPTED)
        self.assertIn("- [x] demo", interrupted)
        self.assertIn(f"- [ ] {dataset.NONE}", interrupted, "ticking a skill clears none of these")
        self.assertIn("- [ ] demo", accepted)
        self.assertIn(f"- [x] {dataset.NONE}", accepted)
        self.assertTrue(self.checked(INTERRUPTED) and self.checked(ACCEPTED))

    def test_a_prompt_with_no_label_cannot_be_accepted_blank(self):
        unlabelled = "rename the variables in this module"  # a random draw: nothing pre-filled
        keyboard = Keyboard({unlabelled: ["", "n", ""]})
        self.review(keyboard)
        self.assertTrue(any("Tick a skill or press n" in line for line in keyboard.screen))
        self.assertIn(f"- [x] {dataset.NONE}", section(self.label.read_text(), unlabelled))
        self.assertTrue(self.checked(unlabelled))

    def test_quitting_or_an_interrupt_keeps_earlier_answers(self):
        quit_early = Keyboard(sequence=["", "q"])
        self.review(quit_early)
        first, second = quit_early.seen[:2]
        self.assertTrue(self.checked(first))
        self.assertFalse(self.checked(second))
        interrupted = Keyboard(sequence=["", KeyboardInterrupt()])
        self.review(interrupted)
        answered, cut_off = interrupted.seen[:2]
        self.assertTrue(self.checked(answered), "an answer is saved before the next prompt is shown")
        self.assertFalse(self.checked(cut_off))

    def test_checked_prompts_are_skipped_unless_all(self):
        self.review(Keyboard({ACCEPTED: [""]}))
        again = Keyboard()
        self.review(again)
        self.assertNotIn(ACCEPTED, again.seen)
        everything = Keyboard()
        self.review(everything, everything=True)
        self.assertIn(ACCEPTED, everything.seen)

    def test_u_undoes_the_previous_answer(self):
        before = self.label.read_text()
        keyboard = Keyboard(sequence=["", "u", "s", "q"])  # accept, undo it, skip it, quit
        self.review(keyboard)
        first = keyboard.seen[0]
        self.assertEqual(keyboard.seen[:3], [first, keyboard.seen[1], first], "undo goes back to the previous prompt")
        self.assertEqual(section(self.label.read_text(), first), section(before, first), "undo restores it as it was")

    def test_the_file_round_trips_except_for_the_answered_boxes(self):
        before = self.label.read_text()
        self.review(Keyboard({ACCEPTED: [""]}))
        after = self.label.read_text()
        changed = [(a, b) for a, b in zip(before.splitlines(), after.splitlines()) if a != b]
        self.assertEqual(changed, [("- [ ] I checked this", "- [x] I checked this")])
        self.assertEqual(len(before.splitlines()), len(after.splitlines()))

    def test_reviewed_prompts_go_into_the_test_split(self):
        reviewed = [ACCEPTED, INTERRUPTED, MISSED, "build the onboarding walkthrough for new users"]
        self.review(Keyboard({p: [""] for p in reviewed}))
        split = dataset.apply_labels(self.run_dir, test=0.5, val=0.0, seed=1)
        self.assertTrue(split["test"])
        self.assertTrue({e["prompt"] for e in split["test"]} <= set(reviewed))

    def test_the_cli_reviews_and_prints_what_is_left(self):
        with mock.patch("builtins.input", side_effect=["", "q"]):
            status, output = run_capture("--out", str(self.out), "optimize", "review", "--run", "r")
        self.assertEqual(status, 0)
        self.assertIn("1 checked this session", output)
        self.assertIn("optimize label --run r", output)
        candidates = json.loads((self.run_dir / "candidates.json").read_text())
        self.assertEqual(self.label.read_text().count("- [x] I checked this"), 1, len(candidates))


if __name__ == "__main__":
    unittest.main()
