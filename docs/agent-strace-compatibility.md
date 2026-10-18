# agent-strace compatibility

Checked on 2026-10-18 against [agent-strace](https://github.com/Siddhant-K-code/agent-trace)
0.95.0, commit `df2cb2c0798e2b9bc793252b7ae1d5efcc8cd6ae`, using its real Python
package from source. No global installation, hook setup, model calls or personal
history is needed. Telemetry is disabled with `DO_NOT_TRACK=1`.

The optional integration suite checks:

- Real Gemini hook CLI captures through the kit's backup, import, redaction and
  verification path, using synthetic hook payloads in a temporary home.
- Claude and Codex optimizer traces through agent-strace's metadata reader,
  event reader, hash verifier and terminal replay.
- Real workspace stores and child-session attribution.
- An edited event rejected by both tools' hash verifiers.

This checks storage interoperability. It does not exercise a live Gemini process
or install hooks into any agent's configuration.

## Reproduce

From the kit checkout, with Python 3.10 or later:

```sh
upstream=$(mktemp -d)
git clone https://github.com/Siddhant-K-code/agent-trace.git "$upstream/agent-trace"
git -C "$upstream/agent-trace" checkout df2cb2c0798e2b9bc793252b7ae1d5efcc8cd6ae
DO_NOT_TRACK=1 PYTHONPATH="$upstream/agent-trace/src" \
  python3 -m unittest tests.test_agent_strace_integration -v
```

Expect **3 tests, OK, no skips**. A normal `unittest discover` skips these three
checks when the optional `agent_trace` package is unavailable. agent-strace is
not a kit runtime dependency. The headless-session regression runs in the normal
suite (`python3 -m unittest tests.test_headless -v`).

## Compatibility fix

Real agent-strace rejected the kit's former `meta.json` with
`TypeError: SessionMeta.__init__() got an unexpected keyword argument 'exit_code'`.
New traces keep upstream metadata fields in `meta.json` and preserve kit-only
close details in `annotations.jsonl`, as `{"run_result":{"exit_code":0}}`.
Existing traces are not rewritten; regenerate old optimizer traces if they need
to open in agent-strace.
