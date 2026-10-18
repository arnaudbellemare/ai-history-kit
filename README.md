# ai-history-kit

**Own your AI history.** One command backs up your Claude Code, Codex, ChatGPT, and Claude.ai conversations, plus anything hstry or agent-trace recorded. It cleans them so your own words are separated from text that agents and tools wrote, redacts secrets, makes everything searchable, and drafts personal memories for you to review. Everything stays on your machine.

```bash
pipx install git+https://github.com/arnaudbellemare/ai-history-kit
# or: uv tool install git+https://github.com/arnaudbellemare/ai-history-kit
aihistory run
```

Then open `~/AI-History/REPORT.md`.

No dependencies beyond Python 3.10+. Nothing is uploaded anywhere. The one exception is `aihistory optimize eval`, which runs your own coding agent on prompts you choose, so those prompts go to that agent's provider, as they do when you use it.

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
aihistory search postgres migration          # every word; "migrations" matches too
aihistory search '"write tests"' -kite       # an exact phrase, without "kite"
aihistory search 'deploy OR release' 'migrat*'
aihistory search source:codex after:2026-09-01 before:2026-10-01 worktree
aihistory search --json "pre-commit"         # for scripts and agents
```

- **Fallbacks:** if nothing has every word, search shows messages with any of them, then messages that contain the text as a substring ("ailboa" finds "sailboat"). Last, it tries a spelling correction from your own vocabulary ("postgress" finds "postgres").
- **Filters:** `source:`, `role:`, `workspace:` (the folder the agent ran in), `after:` and `before:`. Each also works as a flag, such as `--source codex`.
- **Context:** each hit ends with `(CONVERSATION --at N)`. `aihistory show CONVERSATION --at N` prints the messages around it, each labelled by who wrote it (`user`, `assistant`, `harness`, `agent`, `tool`).

By default only your own messages are indexed. Add `--index-assistant` to `run` or `index` to include replies. After upgrading, run `aihistory index` once to rebuild the index.

If search across many tools is your main need, [hstry](https://github.com/byteowlz/hstry) is stronger: query modes, more filters, and an HTTP and MCP interface. This index searches only the words you wrote, labelled.

## Test what steers your agent (optional)

Does your agent load the right skill for the prompts you actually write, and does a rule in your CLAUDE.md or AGENTS.md change what it does? `aihistory optimize` measures both, always against a hand-written control.

```bash
cd ~/my-project    # project skills are found from the folder you run in
aihistory optimize mine  --run triggers --skill deploy --skill review --keywords deploy=release,ship
#   label ~/AI-History/optimize/triggers/LABEL.md (tick the right skill for each prompt)
aihistory optimize label --run triggers
aihistory optimize arm   --run triggers control --skill deploy="Use when the user asks to deploy or release."
aihistory optimize eval  --run triggers --agent fake        # checks the setup; spends nothing
aihistory optimize eval  --run triggers --split test        # Claude Code (Sonnet), 3 runs per prompt
aihistory optimize report --run triggers
```

To let GEPA rewrite the descriptions, install the optional extra (gepa 0.1.4 runs on Python 3.10 to 3.14), train on the train and validation splits, then compare all three arms on the test split:

```bash
uv tool install --python 3.14 "ai-history-kit[optimize] @ git+https://github.com/arnaudbellemare/ai-history-kit"
aihistory optimize train --run triggers --budget 60 --runs 2     # the best rewrite becomes the arm `gepa`
aihistory optimize eval  --run triggers --arms original,control,gepa
aihistory optimize report --run triggers
```

GEPA sees only the train and validation prompts. Its rewrites come from `claude -p` with no tools or MCP servers (`--reflection-model`, default Sonnet), never from gepa's default model. A training run makes up to `budget × runs` agent runs, plus one reflection call per rewrite.

- **Prompts come from you.** `mine` offers three kinds of message you wrote: the one just before a target skill was loaded, ones that mention your keywords, and a random draw as likely negatives. Subagent briefs, headless runs and injected text are never offered. You label every prompt; no model does.
- **The agent sees your real menu.** Your other skills are installed as stub decoys in a throwaway workspace, so a target skill competes the way it does in your sessions. A run counts as right when the agent loads only correct skills, or none when none is right.
- **Claude Code runs with no side effects:** project settings only, no MCP servers, no saved session, and edit, shell and web tools denied. Codex (`--agent codex`, `codex exec --ephemeral` in a read-only sandbox) is experimental: check one run by hand before trusting its scores.
- **Every run is an agent-trace session** under `optimize/<run>/traces/`, scored from its events. Claude Code and Codex are judged the same way, and you can read back exactly what the agent did. The format follows agent-strace's source; the kit hasn't yet been checked against agent-strace itself.
- **Guards:**
  - every `eval` prints how many agent runs it plans, and more than `--max-runs` (default 200) needs `--yes`;
  - an arm is evaluated on the test split once, so new descriptions need a new arm name;
  - `report` needs a control arm.
- **Expect small effects.** In earlier trials, a one-line hand-written control explained most of what an automatic optimizer gained. With a few dozen test prompts, few differences are statistically significant. The report gives McNemar p-values so you can see that.

### Test a rule in your CLAUDE.md or AGENTS.md

Does a section of your instructions change what the agent does? A rule run gives the agent scenarios in a scratch repository, with that section present, removed, or replaced by your one-line control. It then checks what the agent did.

```bash
aihistory optimize rule example > scenarios.json   # a commit-hook scenario to adapt: prompt, files, checks
aihistory optimize rule init --run commit-rule --file AGENTS.md --section "Commit Checks" --scenarios scenarios.json
aihistory optimize arm   --run commit-rule control --text "Run the project's checks before every commit."
aihistory optimize eval  --run commit-rule --agent fake    # checks the setup; spends nothing
aihistory optimize eval  --run commit-rule                  # arms present, removed, control
aihistory optimize report --run commit-rule
```

- **Each run starts fresh:** a new scratch repository holding the scenario's files and your instructions file in its first commit.
- **Checks are deterministic** and read the run's trace and the repository afterwards:
  - `ordered`: run A before B;
  - `never`: never run C;
  - `never_touch`: leave these files alone;
  - `end_state`: a command that must succeed afterwards.

  The report separates process checks (how the agent worked) from outcome checks (what it left behind), and lists every failed check. The agent's reasoning is never scored.
- **`optimize train` works on rule runs too.** GEPA rewrites the section from the failed checks, and the result is the arm `gepa`.
- **Safety: the agent runs real commands here.** Claude Code may use only the tools a scenario lists in `allow` (for example `Bash(git commit:*)`), in a temporary folder outside your repositories. Git and the checks run without inherited `GIT_*` variables. Name git subcommands rather than allowing `Bash(git:*)`, which would also permit `git config --global` or `git -C` on another repository. If a scenario allows broad commands, run the evaluation in a container or VM.

## What's in `~/AI-History`

| Path | What it is |
|---|---|
| `REPORT.md` | What was found, date coverage and gaps, skill and thinking stats, next steps |
| `raw/` + `CATALOG.json` | Verified copies of the original files. Re-running adds new and grown sessions and never deletes |
| `corpus/*.jsonl` | Cleaned conversations, one file per source, secrets redacted |
| `index.sqlite3` | Full-text index for `aihistory search` and `show` |
| `memories/REVIEW.md`, `candidates.json` | Memory candidates waiting for your review |
| `optimize/<run>/` | Optional: labelled prompts, arms, agent-trace sessions of every evaluation run, and its report |

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
