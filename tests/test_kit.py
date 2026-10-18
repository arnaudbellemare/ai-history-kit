import io
import json
import shutil
import stat
import tempfile
import threading
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from aihistory import agentme, cli, index
from aihistory.backup import Backup
from aihistory.normalize import read_corpus
from tests.fixtures import FAKE_SECRET, make_home
from tests.hstry_fixture import make_hstry_db

try:
    from compression import zstd  # Python 3.14+
except ImportError:
    zstd = None


def run(*args) -> int:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return cli.main(list(args))


class KitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = make_home(Path(self.tmp.name) / "home")
        self.out = Path(self.tmp.name) / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def run_all(self, *extra):
        self.assertEqual(run("--home", str(self.home), "--out", str(self.out), *extra, "run"), 0)

    def events(self, source):
        return [(e["role"], e["text"]) for r in read_corpus(self.out) if r["source"] == source for e in r["events"]]

    def memories(self):
        return [c["content"] for c in json.loads((self.out / "memories" / "candidates.json").read_text())["candidates"]]

    def test_only_what_the_owner_wrote_becomes_a_memory(self):
        self.run_all()
        found = self.memories()
        for mine in ("Alex", "short answers", "a kite", "write tests first"):
            self.assertIn(mine, found)
        for not_mine in ("cite primary sources", "the audit", "rewrite everything", "a source-backed design", "purple everywhere",
                         "a cost-aware model selector", "You're a technical quality auditor", "tabs", "run the linter"):
            self.assertNotIn(not_mine, found, f"{not_mine!r} was not written by the owner")
        self.assertIn("- [ ] `mem_", (self.out / "memories" / "REVIEW.md").read_text())

    def test_every_codex_folder_lands_in_the_corpus(self):
        for extra in ((), ("--no-backup",)):
            shutil.rmtree(self.out, ignore_errors=True)
            self.run_all(*extra)
            texts = [text for _, text in self.events("codex")]
            self.assertIn("I'm building a sailboat.", texts, f"archived_sessions must not overwrite sessions {extra}")
            self.assertIn("I always write tests first.", texts, f"sessions must not overwrite archived_sessions {extra}")

    def test_a_rule_repeated_across_conversations_becomes_a_candidate(self):
        self.run_all()
        candidates = json.loads((self.out / "memories" / "candidates.json").read_text())["candidates"]
        rule = [c for c in candidates if c["content"] == "Never push to main without asking"]
        self.assertEqual(len(rule), 1)
        self.assertEqual((rule[0]["kind"], rule[0]["conversations"]), ("preference", 3))
        self.assertNotIn("Fix bug 0", self.memories(), "a one-off request is not a memory")

    def test_claude_code_roles_separate_owner_harness_and_agent_text(self):
        self.run_all()
        roles = dict((text, role) for role, text in self.events("claude-code"))
        self.assertEqual(roles["I usually cite primary sources."], "harness")
        self.assertEqual(roles["<task-notification>I completed the audit.</task-notification>"], "harness")
        self.assertEqual(roles["I'm building a cost-aware model selector."], "agent")
        self.assertEqual(roles["/goal ship the beta"], "user")

    def test_codex_identity_injected_context_and_subagents(self):
        self.run_all()
        records = {r["origin"].split("/")[-1]: r for r in read_corpus(self.out) if r["source"] == "codex"}
        top = [(e["role"], e["text"][:20]) for e in records["rollout-2026-10-06T08-00-00-top.jsonl"]["events"]]
        self.assertEqual([r for r, _ in top], ["harness", "harness", "user", "assistant"])
        self.assertTrue(records["rollout-2026-10-07T08-00-00-guardian.jsonl"]["subagent"])
        self.assertNotEqual(records["rollout-2026-10-08T08-00-00-fork.jsonl"]["conversation_id"],
                            records["rollout-2026-10-06T08-00-00-top.jsonl"]["conversation_id"],
                            "a fork keeps its own identity, not its parent's")

    def test_secrets_and_home_paths_are_redacted_and_files_are_private(self):
        self.run_all()
        corpus = (self.out / "corpus" / "claude-code.jsonl").read_text()
        self.assertNotIn("abcdefgh12345678", corpus)
        self.assertNotIn(str(self.home), corpus)
        self.assertIn("$HOME/notes.md", corpus)
        for path in (self.out / "corpus").glob("*.jsonl"):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_credentials_named_with_any_prefix_are_redacted(self):
        from aihistory.redact import Redactor

        redactor = Redactor("/nonexistent")
        for name in ("token", "GITHUB_TOKEN", "client_secret", "DATABASE_PASSWORD", "STRIPE_API_KEY",
                     "AWS_SECRET_ACCESS_KEY"):
            with self.subTest(name=name):
                self.assertEqual(redactor.text(f"{name}={FAKE_SECRET}"), f"{name}=[redacted]",
                                 "credential value survived redaction")

    def test_settings_that_only_mention_tokens_or_secrets_are_kept(self):
        from aihistory.redact import Redactor

        redactor = Redactor("/nonexistent")
        for harmless in ("max_tokens=128000000", "token_count: 123456789", "tokens=12345678",
                         "tokenizer=cl100k_base", "password_reset_url=https://example.com/reset"):
            with self.subTest(text=harmless):
                self.assertEqual(redactor.text(harmless), harmless, "a non-secret setting was redacted")

    def test_report_and_search(self):
        self.run_all()
        report = (self.out / "REPORT.md").read_text()
        self.assertIn("claude-code", report)
        self.assertIn("demo (1)", report, "skill calls are counted")
        hits = index.search(self.out, "kite")
        self.assertTrue(hits and all(h["role"] == "user" for h in hits))

    def test_backup_never_loses_a_version(self):
        self.run_all()
        session = self.home / ".claude/projects/-Users-someone-proj/sess-1.jsonl"
        original = session.read_text()
        stored = self.out / "raw/claude-code/projects/-Users-someone-proj/sess-1.jsonl"
        session.write_text(original + json.dumps({"type": "user", "message": {"content": "more"}}) + "\n")
        self.run_all()
        self.assertEqual(stored.read_text(), session.read_text(), "a grown transcript replaces its prefix")
        session.write_text(original)
        self.run_all()
        self.assertNotEqual(stored.read_text(), original, "an older prefix is already contained, not a downgrade")
        session.write_text("something else entirely\n")
        self.run_all()
        variants = list(stored.parent.glob("sess-1.variant-*.jsonl"))
        self.assertEqual(len(variants), 1, "a different copy is kept beside the original")
        self.assertEqual(Backup(self.out).verify(), {})

    def test_exports_are_read(self):
        chatgpt = Path(self.tmp.name) / "chatgpt.zip"
        with zipfile.ZipFile(chatgpt, "w") as archive:
            archive.writestr("conversations.json", json.dumps([{"id": "c1", "title": "t", "current_node": "b", "mapping": {
                "a": {"id": "a", "parent": None, "message": {"author": {"role": "user"}, "content": {"parts": ["I work at Acme."]}}},
                "b": {"id": "b", "parent": "a", "message": {"author": {"role": "assistant"}, "content": {"parts": ["Nice."]}}},
                "x": {"id": "x", "parent": "a", "message": {"author": {"role": "assistant"}, "content": {"parts": ["abandoned"]}}}}}]))
        claude = Path(self.tmp.name) / "claude.json"
        claude.write_text(json.dumps([{"uuid": "u1", "name": "n", "chat_messages": [
            {"sender": "human", "text": "My company is Kiteworks."}, {"sender": "assistant", "text": "Hi"}]}]))
        self.run_all("--export", str(chatgpt), "--export", str(claude))
        self.assertNotIn(("assistant", "abandoned"), self.events("chatgpt-export"))
        self.assertIn("Acme", self.memories())
        self.assertIn("Kiteworks", self.memories())

    @unittest.skipUnless(zstd or shutil.which("zstd"), "needs Python 3.14+ or the zstd command")
    def test_compressed_codex_rollouts_are_read(self):
        rows = [{"type": "session_meta", "payload": {"id": "thread-zst", "source": "vscode"}},
                {"type": "event_msg", "payload": {"type": "user_message", "message": "I prefer dark mode."}}]
        data = "".join(json.dumps(r) + "\n" for r in rows).encode()
        target = self.home / ".codex/sessions/2026/10/09/rollout-2026-10-09T08-00-00-z.jsonl.zst"
        target.parent.mkdir(parents=True)
        if zstd:
            target.write_bytes(zstd.compress(data))
        else:
            import subprocess
            target.write_bytes(subprocess.run(["zstd", "-q", "-c"], input=data, capture_output=True, check=True).stdout)
        self.run_all()
        self.assertIn("dark mode", self.memories())


class HstryTests(unittest.TestCase):
    """Reading an hstry database adds every tool hstry supports."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = make_home(Path(self.tmp.name) / "home")
        self.db = make_hstry_db(self.home / ".local/share/hstry/hstry.db", FAKE_SECRET)
        self.out = Path(self.tmp.name) / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def records(self):
        return [r for r in read_corpus(self.out) if r["source"].startswith("hstry:")]

    def test_default_database_is_found_and_read_with_kit_attribution(self):
        self.assertEqual(run("--home", str(self.home), "--out", str(self.out), "run"), 0)
        cursor = next(r for r in self.records() if r["source"] == "hstry:cursor")
        roles = [(e["role"], e["text"][:24]) for e in cursor["events"] if e["kind"] == "message"]
        self.assertIn(("harness", "<system-reminder>Follow "), roles, "harness text in an hstry user turn is not the owner's")
        self.assertIn("edit_file", [e.get("tool") for e in cursor["events"]])
        corpus = (self.out / "corpus" / "hstry.jsonl").read_text()
        self.assertNotIn(FAKE_SECRET, corpus, "hstry stores secrets as-is; the kit redacts them")
        candidates = [c["content"] for c in json.loads((self.out / "memories" / "candidates.json").read_text())["candidates"]]
        self.assertIn("vim keybindings", candidates)
        self.assertNotIn("a rocket", candidates, "a dispatched worker thread's prompt is not the owner's")

    def test_sessions_the_native_reader_has_are_skipped_but_history_only_hstry_kept_is_not(self):
        self.assertEqual(run("--home", str(self.home), "--out", str(self.out), "run"), 0)
        titles = {r.get("title") for r in self.records() if r["source"] == "hstry:claude-code"}
        self.assertEqual(titles, {"deleted from disk since"})

    def test_an_explicit_path_and_bad_databases(self):
        moved = make_hstry_db(Path(self.tmp.name) / "elsewhere.db", FAKE_SECRET)
        self.db.unlink()
        self.assertEqual(run("--home", str(self.home), "--out", str(self.out), "--from-hstry", str(moved), "run"), 0)
        self.assertTrue(self.records())
        self.assertEqual(run("--home", str(self.home), "--out", str(self.out), "--from-hstry", str(self.tmp.name) + "/missing.db", "scan"), 2)
        broken = Path(self.tmp.name) / "broken.db"
        broken.write_bytes(b"not a database at all" * 100)
        self.assertEqual(run("--home", str(self.home), "--out", str(self.out), "--from-hstry", str(broken), "scan"), 2)


class AgentMeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "out"
        home = make_home(Path(self.tmp.name) / "home")
        self.assertEqual(run("--home", str(home), "--out", str(self.out), "run"), 0)
        self.posts = []
        posts = self.posts

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self._reply([])

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                posts.append((self.headers["Authorization"], body))
                self._reply({"id": f"e{len(posts)}", "status": "pending", **body})

            def _reply(self, value):
                data = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.api = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def test_only_ticked_memories_are_sent_and_only_with_apply(self):
        review = self.out / "memories" / "REVIEW.md"
        lines = review.read_text().splitlines()
        target = next(i for i, line in enumerate(lines) if "**Alex**" in line)
        lines[target] = lines[target].replace("- [ ]", "- [x]")
        review.write_text("\n".join(lines) + "\n")

        dry = agentme.push(self.out, self.api, "tok")
        self.assertEqual(len(dry["would_submit"]), 1)
        self.assertEqual(self.posts, [], "a dry run sends nothing")

        agentme.push(self.out, self.api, "tok", apply=True)
        self.assertEqual(len(self.posts), 1)
        auth, body = self.posts[0]
        self.assertEqual((auth, body["kind"], body["key"], body["content"]), ("Bearer tok", "fact", "identity.name", "Alex"))

    def test_non_local_agent_me_urls_are_refused(self):
        for url in ("https://agent-me.example.com", "http://10.0.0.5:8000", "http://127.0.0.1:8000/api"):
            with self.assertRaises(ValueError, msg=url):
                agentme.loopback(url)


if __name__ == "__main__":
    unittest.main()
