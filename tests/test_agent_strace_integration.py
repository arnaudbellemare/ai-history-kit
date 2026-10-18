"""Optional contract checks against real agent-strace; see docs/agent-strace-compatibility.md.

No hooks are installed. Only synthetic payloads and temporary stores are used.
"""

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aihistory.adapters import agent_trace as kit_trace
from aihistory.normalize import read_corpus
from aihistory.optimize import traces
from tests.test_kit import run
from tests.test_optimize import CLAUDE_STREAM, CODEX_JSONL

UPSTREAM = importlib.util.find_spec("agent_trace")


@unittest.skipUnless(UPSTREAM, "needs real agent-strace on PYTHONPATH (see docs/agent-strace-compatibility.md)")
class AgentStraceIntegrationTests(unittest.TestCase):
    def setUp(self):
        from agent_trace.audit import verify_chain
        from agent_trace.store import TraceStore

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store_type, self.verify_upstream = TraceStore, verify_chain
        # Do not inherit collector URLs, tenancy, attribution, or hook configuration.
        self.env = {"PATH": os.defpath, "HOME": str(self.root / "home"), "DO_NOT_TRACK": "1",
                    "PYTHONPATH": str(Path(UPSTREAM.origin).parent.parent),
                    "AGENT_TRACE_DIR": str(self.root / ".agent-traces")}

    def hook(self, event, payload):
        result = subprocess.run([sys.executable, "-m", "agent_trace.cli", "hook", "--provider", "gemini", event],
                                input=json.dumps({"session_id": "gemini-session-00000001", **payload}),
                                text=True, capture_output=True, env=self.env, cwd=self.root, timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(result.stderr, result.stderr)  # hooks can swallow errors with exit 0

    def test_real_hook_capture_imports_backs_up_redacts_and_verifies(self):
        self.hook("session-start", {"source": "startup"})
        self.hook("user-prompt", {"prompt": "I prefer violet margins. token=" + "abcdefgh" + "12345678"})
        self.hook("pre-tool", {"tool_name": "read_file", "tool_input": {"path": "README.md"}, "tool_use_id": "t1"})
        self.hook("post-tool", {"tool_name": "read_file", "tool_response": "# example", "tool_use_id": "t1"})
        self.hook("stop", {"last_assistant_message": "Read the example."})
        self.hook("session-end", {})
        store, out = self.root / ".agent-traces", self.root / "out"
        session = "gemini-session-00000001"[:16]
        upstream = self.store_type(store, use_workspace_env=False)
        self.assertTrue(self.verify_upstream(upstream, session).ok)
        self.assertEqual(run("--home", self.env["HOME"], "--out", str(out), "--agent-traces", str(store), "run"), 0)
        record, = read_corpus(out)
        self.assertEqual(record["source"], "agent-trace:gemini-cli")
        self.assertEqual(record["stats"]["chain"], {"status": "ok", "events": 6})
        self.assertEqual([e["role"] for e in record["events"]], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(record["events"][1]["tool"], "read_file")
        self.assertEqual(record["events"][2]["text"], "# example")
        self.assertNotIn("abcdefgh" + "12345678", json.dumps(record))
        self.assertEqual(run("--out", str(out), "verify"), 0)
        path = store / session / "events.ndjson"
        path.write_text(path.read_text().replace("violet", "orange"), encoding="utf-8")
        self.assertFalse(self.verify_upstream(upstream, session).ok)
        self.assertEqual(kit_trace.verify_chain(path)["status"], "broken")

    def test_kit_rollouts_open_and_verify_in_real_agent_strace(self):
        from agent_trace.replay import replay_session

        store = self.root / "exports"
        upstream = self.store_type(store, use_workspace_env=False)
        for name, convert, stream in (("claude-code", traces.from_claude_stream, CLAUDE_STREAM),
                                      ("codex", traces.from_codex_jsonl, CODEX_JSONL)):
            with self.subTest(agent=name):
                writer = traces.TraceWriter(store, name, name)
                writer.event("user_prompt", {"prompt": "show a demo"})
                convert(map(json.dumps, stream), writer)
                writer.close(exit_code=0)  # exactly how eval and train close rollouts
                meta = upstream.load_meta(name)
                self.assertEqual(meta.agent_name, name)
                self.assertIsNotNone(meta.ended_at)
                events = upstream.load_events(name)
                self.assertGreater(len(events), 3)
                self.assertEqual(events[0].data["prompt"], "show a demo")
                self.assertTrue(self.verify_upstream(upstream, name).ok)
                output = io.StringIO()
                replay_session(upstream, name, out=output)
                self.assertIn("show a demo", output.getvalue())
                annotations = [json.loads(line) for line in (writer.folder / "annotations.jsonl").read_text().splitlines()]
                self.assertIn({"run_result": {"exit_code": 0}}, annotations)

    def test_real_workspace_and_child_sessions_keep_attribution(self):
        from agent_trace.models import EventType, SessionMeta, TraceEvent

        store = self.root / "nested"
        upstream = self.store_type(store, workspace_id="example")
        for session, parent in (("root", ""), ("child", "root")):
            upstream.create_session(SessionMeta(session_id=session, agent_name="cursor", parent_session_id=parent))
            upstream.append_event(session, TraceEvent(EventType.USER_PROMPT, data={"prompt": "I prefer ruby scripts."}))
            upstream.append_event(session, TraceEvent(EventType.ASSISTANT_RESPONSE, data={"text": "OK"}))
        paths = kit_trace.session_files(store)
        self.assertEqual(len(paths), 2)
        records = {p.parent.name: kit_trace.read_session(p, store) for p in paths}
        self.assertEqual(records["root"]["events"][0]["role"], "user")
        self.assertEqual(records["child"]["events"][0]["role"], "agent")
        self.assertTrue(all(r["stats"]["chain"]["status"] == "ok" for r in records.values()))
