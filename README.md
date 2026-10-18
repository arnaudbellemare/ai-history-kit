# ai-history-kit

**Own your AI history.** One command backs up your Claude Code, Codex, ChatGPT, and Claude.ai conversations, plus anything hstry or agent-trace recorded. It cleans them so your own words are separated from text that agents and tools wrote, redacts secrets, makes everything searchable, and drafts personal memories for you to review. Everything stays on your machine.

```bash
pipx install git+https://github.com/arnaudbellemare/ai-history-kit
# or: uv tool install git+https://github.com/arnaudbellemare/ai-history-kit
aihistory run
```

Then open `~/AI-History/REPORT.md`.

No dependencies beyond Python 3.10+. Nothing is uploaded anywhere.

## Why

- **Apps delete your history.** Coding agents prune old sessions, and Codex compresses and drops old rollouts. If you want your history in a year, keep your own copy now.
- **Most "user" text in a transcript isn't yours.** Agents write prompts for subagents. Harnesses inject skill instructions, system reminders, task reports, and AGENTS.md files as "user" messages. Mine that naively and your personal memory fills with words an agent wrote. This kit labels every message by who actually wrote it.
- **Transcripts are full of secrets.** API keys, tokens, and passwords pasted into chats are redacted before anything is written to the cleaned corpus.

## Three steps

1. **`aihistory run`** finds your history and backs it up to `~/AI-History/raw`. It cleans it into `~/AI-History/corpus`, builds a search index, and writes `REPORT.md`.
2. **Review `~/AI-History/memories/REVIEW.md`.** Each line is something you said about yourself ("I prefer…", "I'm building…", "My name is…"), with the quote it came from. Tick `[x]` on the ones that are true.
3. **Optional: send them to [Agent-Me](https://github.com/jzjzzzzzzz/agent-me)**, an open-source personal AI twin.

   ```bash
   aihistory agent-me push --token-file ~/agent-me/private/personal.env           # dry run
   aihistory agent-me push --token-file ~/agent-me/private/personal.env --apply
   ```

   Only the ticked memories are sent, only to a local Agent-Me, and only as *pending* entries. You still confirm each one inside Agent-Me.

## Add your ChatGPT and Claude.ai history

Request your data export from each app. ChatGPT: Settings → Data controls → Export. Claude.ai: Settings → Privacy → Export data. Then:

```bash
aihistory --export ~/Downloads/chatgpt-export.zip --export ~/Downloads/claude-export.zip run
```

## Search

```bash
aihistory search "postgres migration"
```

By default only your own messages are indexed. Add `--index-assistant` to `run` or `index` to include replies.

## What's in `~/AI-History`

| Path | What it is |
|---|---|
| `REPORT.md` | What was found, date coverage and gaps, skill and thinking stats, next steps |
| `raw/` + `CATALOG.json` | Verified copies of the original files. Re-running adds new and grown sessions and never deletes |
| `corpus/*.jsonl` | Cleaned conversations, one file per source, secrets redacted |
| `index.sqlite3` | Full-text index for `aihistory search` |
| `memories/REVIEW.md`, `candidates.json` | Memory candidates waiting for your review |

All files are created readable only by you (mode 0600).

## Who wrote what

Every event in the corpus has a `role`:

| Role | Meaning |
|---|---|
| `user` | You typed it. Only this feeds search and memories by default |
| `assistant` | The model's reply |
| `tool` | Tool output |
| `harness` | Injected by the app: skill bodies, system reminders, task notifications, compaction summaries, AGENTS.md/CLAUDE.md, environment context |
| `agent` | A prompt another agent wrote: subagent briefs, Codex subagent and reviewer threads |
| `developer`, `system` | Provider-level instructions |

Pasted transcripts are handled too. Inside your own message, lines quoted from an assistant (`assistant: …`, `[12] assistant: …`) are not treated as your words.

## Keep it safe

The cleaned corpus is redacted, but **`raw/` keeps the original files, secrets included**. Put `~/AI-History` on an encrypted disk, or point `--out` at one.

- **macOS:** create an encrypted volume, then pass `--out` to it.

  ```bash
  hdiutil create -size 200g -type SPARSEBUNDLE -fs APFS -volname "AI History" -encryption AES-256 ~/AI-History.sparsebundle
  open ~/AI-History.sparsebundle     # enter your passphrase; tick "Remember in keychain"
  aihistory --out "/Volumes/AI History" run
  ```

- **Linux:** a LUKS or VeraCrypt volume.
- **Windows:** a BitLocker drive.

Check the backup at any time with `aihistory verify`.

## Back up every day

The backup only adds what's new, so run it on a schedule. On macOS or Linux, run `crontab -e` and add:

```
30 3 * * * aihistory --out ~/AI-History backup >> ~/AI-History/backup.log 2>&1
```

## What it reads

| Source | Where |
|---|---|
| Claude Code | `~/.claude/projects/**/*.jsonl`, including subagent transcripts |
| Codex | `~/.codex/sessions` and `archived_sessions`, `rollout-*.jsonl` and compressed `.jsonl.zst` (needs Python 3.14+ or the `zstd` command) |
| Codex thread database | `~/.codex/thread_history_*.sqlite`, backed up as a consistent snapshot (not parsed) |
| ChatGPT / Claude.ai | Official export zip or `conversations.json` via `--export` |
| [hstry](https://github.com/byteowlz/hstry) | `~/.local/share/hstry/hstry.db` when it exists, or `--from-hstry DB`: adds every tool hstry imports (Cursor, OpenCode, Gemini CLI, Aider, Goose, Copilot, Cline, …) |
| [agent-trace](https://github.com/Siddhant-K-code/agent-trace) | `.agent-traces/` in your home folder or any project folder your Claude Code or Codex history names, or `--agent-traces DIR` (repeatable): adds the tools its hooks record (Gemini CLI, Cursor, Copilot CLI, …) |

**About hstry.** If you already use hstry, ai-history-kit reads its database read-only and backs it up as a consistent snapshot.
- **Duplicates:** Claude Code and Codex conversations that this kit already read from disk are skipped. Ones only hstry still has, because the app has since deleted them, are kept.
- **Attribution is a best guess for the other tools.** hstry doesn't record who wrote a message, so only known harness text and dispatched worker threads are separated from your words.
- **Secrets:** hstry stores them as-is, and they're redacted here like everywhere else.

**About agent-trace.** If you record sessions with [agent-strace](https://github.com/Siddhant-K-code/agent-trace), ai-history-kit reads its trace files directly. It never runs agent-strace.
- **Duplicates:** agent-strace hooks keep the first 16 characters of the agent's session id. Traces of Claude Code and Codex sessions this kit already read from disk are skipped; traces of sessions the app has since deleted are kept.
- **Less detail than the native files:** traces have no thinking, and tool output is cut at 1,000 characters.
- **Tampering shows:** each event carries the hash of the one before it. `aihistory verify` and `REPORT.md` name any session whose chain is broken.
- **Telemetry:** agent-strace sends anonymous usage telemetry by default. Set `DO_NOT_TRACK=1` or run `agent-strace telemetry disable` to turn it off.

## Limits, honestly

- **Memories come from two kinds of evidence only:**
  - explicit statements: "My name is…", "I prefer…", "I'm building…", "I decided to…", "I usually…";
  - working rules or requests you repeat in 3 or more separate conversations, such as "Never push to main without asking".

  The kit doesn't guess traits from patterns. Coding-agent sessions rarely contain "I prefer…"-style sentences; ChatGPT and Claude.ai exports usually contain many more. Review everything; frequency is evidence, not truth.
- **Headless `claude -p` runs and subagent transcripts don't feed memories.** Their prompts usually come from scripts or other agents. They're still backed up and searchable.
- **Thinking text is mostly empty.** Recent Claude models store an empty, signed thinking block by default. `REPORT.md` shows how much was kept. Don't use it to judge why an agent did something.
- **Tool inputs and outputs are truncated** to 2,000 characters in the corpus. The raw backup keeps everything.
- **"Days without files" in the report are informational.** They may be days off, or history an app deleted before you backed it up.

## Development

```bash
python3 -m unittest -v
```

The tests build a fake home directory with the traps real histories contain: skill bodies, subagent briefs, Codex reviewer threads, injected AGENTS.md blocks, quoted assistant turns, secrets, forked threads, sessions split across folders, compressed rollouts, hstry databases, and agent-trace stores with dispatched sessions and a tampered event. They check that only your own words become memories.

## License

MIT
