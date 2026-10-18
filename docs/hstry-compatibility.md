# hstry compatibility

Checked on 2026-10-18 with the official Apple Silicon hstry **0.5.27** release
binary and its matching adapters at commit
`21d0fd8dec1cc29ae058a303c0d0164e4c1f513d` ([release](https://github.com/byteowlz/hstry/releases/tag/v0.5.27)).
The downloaded archive `hstry-v0.5.27-aarch64-apple-darwin.tar.gz` matched its
published SHA-256:
`19f6c18e3d411d0577fe6e0a4da5d79bc5829b2c8cba63fc17ac3e9b70c1c457`.
Adapters ran with Bun 1.3.14. Installation and all data stayed in temporary
directories; no personal histories, credentials or global configuration were used.

## What was exercised

The integration suite calls the actual `hstry import` CLI on synthetic Claude Code
and Pi transcript files. hstry creates the full database schema itself. Checks cover:

- Repeat imports remain idempotent.
- Native Claude sessions are deduplicated; sessions deleted from native storage
  remain available through hstry.
- Pi adds an extra tool source with user, assistant, harness, thinking, tool-call
  and tool-result events, including structured parts and millisecond timestamps.
- Secrets remain in the raw backup and are redacted from the corpus. Only owner
  preferences become memory candidates; owner messages remain searchable.
- Both live reads (`--no-backup`) and SQLite backup snapshots include committed
  WAL data. An import while an older database reader stays open must refresh the
  backup even when the main database's size and modification time are unchanged.
- Backups pass both SQLite `quick_check` and the kit's catalog verification.

This validates the tested release and these two adapters. It does not establish
coverage for every adapter hstry supports or every historical schema version.

## Fixes found by the real check

1. Opening a live hstry database with SQLite `immutable=1` ignored its WAL and
   reported a healthy database as malformed. The adapter now uses `mode=ro`.
2. Backup freshness checked only the main file's size and mtime, skipping new
   imports held in WAL. It now checks the WAL's size and mtime too.
3. Real imports stored tool calls and results in `messages.parts_json` while
   leaving `tool_calls` empty. The reader now handles canonical tool parts,
   retains the older table fallback, and avoids duplicate events when both exist.

The new regressions were observed failing before their fixes. Three storage
regressions also run in the normal suite without installing hstry.

## Reproduce

Download and extract the binary for your platform from the pinned release above.
Have Bun on PATH, then run from the kit checkout:

```sh
upstream=$(mktemp -d)
git clone --depth 1 --branch v0.5.27 https://github.com/byteowlz/hstry.git "$upstream/hstry"
HSTRY_BIN=/absolute/path/to/extracted/hstry \
HSTRY_ADAPTERS="$upstream/hstry/adapters" \
  python3 -m unittest tests.test_hstry_integration -v
python3 -m unittest tests.test_hstry_storage -v
```

Expect **2 integration tests and 3 storage tests, OK, no skips**. The integration
suite skips in a normal discovery run unless both `HSTRY_BIN` and
`HSTRY_ADAPTERS` are set. When configured, missing tools or failed imports fail
the check. hstry and Bun are not kit runtime dependencies.
