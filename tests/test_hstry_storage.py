"""hstry storage regressions reproduced by the real release integration check."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from aihistory.adapters import hstry
from aihistory.backup import Backup
from tests.fixtures import FAKE_SECRET
from tests.hstry_fixture import make_hstry_db


class HstryStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = make_hstry_db(self.root / "hstry.db", FAKE_SECRET)

    def test_scans_and_backup_refresh_include_committed_wal_messages(self):
        writer = sqlite3.connect(self.path)
        self.addCleanup(writer.close)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        backup = Backup(self.root / "out")
        backup.snapshot_sqlite(self.path, "hstry", area="raw/hstry")
        before = self.path.stat()
        writer.execute("INSERT INTO conversations (id,source_id,external_id,created_at) VALUES ('new','src-cursor','new',1767225800)")
        writer.execute("INSERT INTO messages (id,conversation_id,idx,role,content) VALUES ('new','new',0,'user','new WAL message')")
        writer.commit()
        after = self.path.stat()
        self.assertEqual((before.st_size, before.st_mtime_ns), (after.st_size, after.st_mtime_ns))
        self.assertEqual(hstry.conversation_count(self.path), 5)
        self.assertTrue(any(e["text"] == "new WAL message" for r in hstry.read_db(self.path) for e in r["events"]))
        backup.snapshot_sqlite(self.path, "hstry", area="raw/hstry")
        snapshot, = (backup.root / "raw/hstry").glob("*/*/*.db")
        self.assertEqual(hstry.conversation_count(snapshot), 5)
        self.assertEqual(backup.verify(), {})
        self.assertEqual(backup.snapshot_sqlite(self.path, "hstry", area="raw/hstry"), "unchanged")

    def test_canonical_tool_parts_are_read_without_duplicating_legacy_rows(self):
        parts = [{"id": "p1", "type": "tool_call", "toolCallId": "t1", "name": "edit_file", "input": {"path": "a.py"}},
                 {"id": "p2", "type": "tool_result", "toolCallId": "t1", "name": "edit_file", "output": "ok"}]
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE messages SET parts_json=? WHERE id='m3'", (json.dumps(parts),))
        for legacy in (True, False):
            with self.subTest(legacy_table_rows=legacy):
                if not legacy:
                    with sqlite3.connect(self.path) as db:
                        db.execute("DELETE FROM tool_calls")
                record = next(r for r in hstry.read_db(self.path) if r["source"] == "hstry:cursor")
                calls = [e for e in record["events"] if e["kind"] == "tool_call"]
                results = [e for e in record["events"] if e["kind"] == "tool_result"]
                self.assertEqual([(e["tool"], json.loads(e["text"])) for e in calls], [("edit_file", {"path": "a.py"})])
                self.assertEqual([e["text"] for e in results], ["ok"])

    def test_canonical_tool_result_is_one_event_with_structured_output(self):
        parts = [{"id": "p1", "type": "tool_result", "toolCallId": "t2", "output": {"ok": True, "count": 2}}]
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE messages SET role='tool',content='projected output',parts_json=? WHERE id='m4'", (json.dumps(parts),))
        record = next(r for r in hstry.read_db(self.path) if r["source"] == "hstry:cursor")
        results = [e for e in record["events"] if e["role"] == "tool" and e["text"] != "ok"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["kind"], "tool_result")
        self.assertEqual(json.loads(results[0]["text"]), {"ok": True, "count": 2})
