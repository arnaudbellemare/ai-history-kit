"""A tiny Python package with unittest tests, for gskill task generation."""

import shlex
import sys
from pathlib import Path

OPS = '''def add(a, b):
    return a + b


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def is_even(number):
    return number % 2 == 0


def both(a, b):
    return a and b
'''
TEXT = '''def shout(text):
    return text.upper() + "!"


def initials(name):
    parts = name.split()
    return "".join(part[0] for part in parts if len(part) > 0)


def is_blank(text):
    if text.strip() == "":
        return True
    return False
'''
TESTS = '''import unittest

from calc.ops import add, both, clamp, is_even
from calc.text import initials, is_blank, shout


class OpsTests(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_clamp_low(self):
        self.assertEqual(clamp(-5, 0, 10), 0)

    def test_clamp_high(self):
        self.assertEqual(clamp(50, 0, 10), 10)

    def test_clamp_inside(self):
        self.assertEqual(clamp(0, 0, 10), 0)
        self.assertEqual(clamp(10, 0, 10), 10)
        self.assertEqual(clamp(5, 0, 10), 5)

    def test_is_even(self):
        self.assertTrue(is_even(4))
        self.assertFalse(is_even(3))

    def test_both(self):
        self.assertFalse(both(True, False))
        self.assertTrue(both(True, True))


class TextTests(unittest.TestCase):
    def test_shout(self):
        self.assertEqual(shout("hi"), "HI!")

    def test_initials(self):
        self.assertEqual(initials("Ada Lovelace"), "AL")

    def test_is_blank(self):
        self.assertTrue(is_blank("  "))
        self.assertFalse(is_blank("x"))
'''
# (file, original line, broken line, the test that catches it): used to cause a regression on purpose
BREAKS = [("calc/ops.py", "    return a + b\n", "    return a * b\n", "test_add"),
          ("calc/text.py", '    return text.upper() + "!"\n', "    return text.upper()\n", "test_shout"),
          ("calc/ops.py", "    return number % 2 == 0\n", "    return True\n", "test_is_even")]


def make_repo(root: Path) -> tuple[Path, str]:
    for relative, content in {"calc/__init__.py": "", "calc/ops.py": OPS, "calc/text.py": TEXT,
                              "tests/__init__.py": "", "tests/test_calc.py": TESTS}.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root, f"{shlex.quote(sys.executable)} -m unittest -v"
