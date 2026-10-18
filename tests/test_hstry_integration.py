"""Optional checks against a real hstry executable and its version-matched adapters."""

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

from aihistory import index
from aihistory.adapters import hstry
from aihistory.normalize import read_corpus
from tests.fixtures import FAKE_SECRET, user
from tests.test_kit import run

HSTRY_BIN = os.environ.get("HSTRY_BIN")
HSTRY_ADAPTERS = os.environ.get("HSTRY_ADAPTERS")


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@unittest.skipUnless(HSTRY_BIN and HSTRY_ADAPTERS, "set HSTRY_BIN and HSTRY_ADAPTERS for real hstry checks")
class HstryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home, self.out = self.root / "home", self.root / "out"
        self.home.mkdir()
        self.db = self.home / ".local/share/hstry/hstry.db"
        self.claude = self.home / ".claude/projects"
        self.pi = self.home / ".pi/agent/sessions"
        bun = shutil.which("bun")
        self.assertIsNotNone(bun, "this check uses Bun to run hstry's release adapters")
        self.env = {"HOME": str(self.home), "PATH": str(Path(bun).parent) + os.pathsep + os.defpath,
                    "XDG_CONFIG_HOME": str(self.root / "config"), "XDG_DATA_HOME": str(self.home / ".local/share"),
                    "XDG_STATE_HOME": str(self.root / "state"), "XDG_CACHE_HOME": str(self.root / "cache"),
                    "HSTRY_NO_SERVICE": "1", "HSTRY_NO_API": "1", "DO_NOT_TRACK": "1", "NO_COLOR": "1"}
        self.config = self.root / "config.toml"
        self.config.write_text(f'database = {json.dumps(str(self.db))}\n'
                               f'adapter_paths = [{json.dumps(str(Path(HSTRY_ADAPTERS).resolve()))}]\n'
                               'adapter_repos = []\njs_runtime = "bun"\nsources = []\nworkspaces = []\n'
                               '[service]\nenabled = false\n[api]\nenabled = false\n', encoding="utf-8")
        for session, prompt in (("native", "I prefer copper prose."), ("hstry-only", "I prefer silver prose.")):
            jsonl(self.claude / "project" / f"{session}.jsonl",
                  [{**user(prompt), "sessionId": session, "uuid": session, "cwd": str(self.root / "project")}])
        self.write_pi()
        self.import_source(self.claude, "claude-code")
        self.import_source(self.pi, "pi")
        (self.claude / "project/hstry-only.jsonl").unlink()

    def import_source(self, source, adapter):
        result = subprocess.run([HSTRY_BIN, "--config", str(self.config), "--json", "import", str(source),
                                 "--adapter", adapter], env=self.env, cwd=self.root,
                                capture_output=True, text=True, check=False, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout)
        self.assertTrue(body["ok"], body)
        self.assertGreater(body["result"]["conversations"], 0, body)

    def write_pi(self, session="pi-example", prompt=None):
        stamp = "2026-07-01T10:00:00Z"
        rows = [{"type": "session", "version": 3, "id": session, "timestamp": stamp, "cwd": str(self.root / "project")}]
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt or "I prefer violet margins. token=" + FAKE_SECRET}]},
            {"role": "user", "content": [{"type": "text", "text": "<system-reminder>I prefer injected rules.</system-reminder>"}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Inspect the example file."},
                {"type": "text", "text": "I prefer assistant rules."},
                {"type": "toolCall", "id": "t1", "name": "read_file", "arguments": {"path": "example.py"}}]},
            {"role": "toolResult", "toolCallId": "t1", "toolName": "read_file",
             "content": [{"type": "text", "text": "file contents token=" + FAKE_SECRET}]},
        ]
        for i, msg in enumerate(messages):
            rows.append({"type": "message", "id": f"m{i}", "parentId": f"m{i-1}" if i else None,
                         "timestamp": stamp, "message": msg})
        jsonl(self.pi / "project" / f"{session}.jsonl", rows)

    def run_kit(self, out=None, no_backup=False):
        args = ["--home", str(self.home), "--out", str(out or self.out)]
        if no_backup:
            args.append("--no-backup")
        self.assertEqual(run(*args, "run"), 0)
        return list(read_corpus(out or self.out))

    def test_real_import_backup_dedup_attribution_redaction_and_tool_parts(self):
        self.assertEqual(hstry.conversation_count(self.db), 3)
        self.import_source(self.pi, "pi")
        self.assertEqual(hstry.conversation_count(self.db), 3, "hstry re-import is idempotent")
        original = hashlib.sha256(self.db.read_bytes()).hexdigest()
        records = self.run_kit()
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(), original, "kit reads hstry without writing it")
        self.assertEqual({r["source"] for r in records}, {"claude-code", "hstry:claude-code", "hstry:pi"})
        imported_claude, = [r for r in records if r["source"] == "hstry:claude-code"]
        self.assertIn("silver prose", imported_claude["events"][0]["text"], "only hstry retains the deleted session")
        pi, = [r for r in records if r["source"] == "hstry:pi"]
        events = pi["events"]
        self.assertIn(("harness", "<system-reminder>I prefer injected rules.</system-reminder>"),
                      [(e["role"], e["text"]) for e in events])
        self.assertIn("2026-07-01T10:00:00Z", [e["timestamp"] for e in events])
        self.assertIn("reasoning", [e["kind"] for e in events])
        calls = [e for e in events if e["kind"] == "tool_call"]
        self.assertEqual([e["tool"] for e in calls], ["read_file"])
        self.assertEqual(json.loads(calls[0]["text"]), {"path": "example.py"})
        results = [e for e in events if e["kind"] == "tool_result"]
        self.assertEqual(len(results), 1)
        self.assertIn("file contents", results[0]["text"])
        self.assertNotIn(FAKE_SECRET, json.dumps(records))
        memories = json.loads((self.out / "memories/candidates.json").read_text())["candidates"]
        contents = {m["content"] for m in memories}
        self.assertTrue({"copper prose", "silver prose", "violet margins"} <= contents)
        self.assertFalse({"injected rules", "assistant rules"} & contents)
        self.assertEqual(run("--out", str(self.out), "verify"), 0)
        self.assertTrue(index.query(self.out, "violet")["hits"])
        snapshot, = (self.out / "raw/hstry").glob("*/*/*.db")
        with sqlite3.connect(snapshot) as db:
            self.assertEqual(db.execute("PRAGMA quick_check").fetchone()[0], "ok")
            self.assertEqual(db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0], 3)
            self.assertIn(FAKE_SECRET, db.execute("SELECT content FROM messages WHERE content LIKE '%token=%'").fetchone()[0])
        direct = self.run_kit(self.root / "direct", no_backup=True)
        self.assertEqual([(r["source"], r["events"]) for r in direct], [(r["source"], r["events"]) for r in records])

    def test_live_wal_imports_are_visible_and_refresh_the_backup(self):
        self.run_kit()
        keeper = sqlite3.connect(self.db)
        self.addCleanup(keeper.close)
        keeper.execute("BEGIN")
        keeper.execute("SELECT COUNT(*) FROM conversations").fetchone()  # pin the pre-import snapshot
        before = self.db.stat()
        self.write_pi("pi-new", "I prefer amber margins.")
        self.import_source(self.pi, "pi")
        after = self.db.stat()
        self.assertEqual((before.st_size, before.st_mtime_ns), (after.st_size, after.st_mtime_ns),
                         "the new hstry import lives only in WAL while an old reader is open")
        self.assertTrue(Path(str(self.db) + "-wal").stat().st_size)
        self.assertEqual(hstry.conversation_count(self.db), 4, "read-only scans must include committed WAL data")
        for no_backup, out in ((True, self.root / "direct"), (False, self.out)):
            records = self.run_kit(out, no_backup=no_backup)
            self.assertTrue(any("amber margins" in e["text"] for r in records for e in r["events"]),
                            f"new WAL messages must survive both import paths: no_backup={no_backup}")
