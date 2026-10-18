"""aihistory: back up, clean, search, and mine your AI history. Everything stays local.

    aihistory run                     # everything below, into ~/AI-History
    aihistory scan                    # what exists, date coverage, gaps
    aihistory backup                  # verified, catalogued copy of raw history
    aihistory normalize               # redacted corpus with owner/agent/harness roles
    aihistory index                   # local full-text index of what you wrote
    aihistory search "postgres migration"
    aihistory memories                # memory candidates -> memories/REVIEW.md
    aihistory agent-me push [--apply] # send ticked memories to a local Agent-Me
    aihistory verify                  # re-check every backed-up file

An hstry database (~/.local/share/hstry/hstry.db, or --from-hstry DB) is read too, which
adds every tool hstry imports. So are agent-trace stores (.agent-traces in any folder your
history names, or --agent-traces DIR), which add the tools its hooks record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .backup import Backup, back_up, backed_up_sources
from .sources import coverage, discover


def _out(args) -> Path:
    return Path(args.out).expanduser()


def _discover(args):
    return discover(args.home, args.export, args.from_hstry, args.agent_traces)


def _sources(args, prefer_backup: bool):
    out = _out(args)
    if prefer_backup and (out / "CATALOG.json").exists() and not args.no_backup:
        live_exports = [s for s in _discover(args) if s.kind.endswith("-export")]
        return backed_up_sources(out) + [s for s in live_exports if not (out / "raw" / "exports" / s.kind).exists()]
    return _discover(args)


def cmd_scan(args):
    scan = [coverage(s) for s in _discover(args)]
    if not scan:
        print("No AI history found. Looked in ~/.claude/projects and ~/.codex; add exports with --export FILE.")
        return 1
    for s in scan:
        if s["kind"] == "hstry":
            print(f"{'hstry':15} {s['conversations']:6} conversations in {s['root']}")
            continue
        if s["kind"] == "agent-trace":
            print(f"{'agent-trace':15} {s['files']:6} sessions {s.get('first_day', ''):10} → {s.get('last_day', ''):10}  in {s['root']}")
            continue
        print(f"{s['kind']:15} {s['files']:6} files  {s.get('first_day', ''):10} → {s.get('last_day', ''):10}  "
              f"{len(s.get('days_without_files', []))} days without files")
    return 0


def cmd_run(args):
    from . import index, memories, normalize, report

    out = _out(args)
    found = _discover(args)
    if not found:
        print("No AI history found. Looked in ~/.claude/projects and ~/.codex; add exports with --export FILE.")
        return 1
    scan = [coverage(s) for s in found]
    steps = {}
    listing = ", ".join("{} ({} files)".format(s["kind"], s["files"]) for s in scan)
    print(f"Found: {listing}")
    if not args.no_backup:
        print("Backing up raw history (verified copies, nothing deleted)…")
        steps["backup"] = back_up(found, out)
        found = _sources(args, prefer_backup=True)
    print("Cleaning: separating your words from agent and harness text, redacting secrets…")
    steps["normalize"] = normalize.normalize(found, out, args.home)
    print("Building the search index…")
    steps["index"] = index.build(out, roles=("user", "assistant") if args.index_assistant else ("user",))
    print("Finding memory candidates…")
    steps["memories"] = memories.propose(out)
    path = report.write(out, scan, steps)
    print(f"\nDone. Read {path}\nReview memories: {steps['memories']['review']}")
    return 0


def cmd_backup(args):
    print(json.dumps(back_up(_discover(args), _out(args)), indent=1))
    return 0


def cmd_normalize(args):
    from .normalize import normalize

    print(json.dumps(normalize(_sources(args, prefer_backup=True), _out(args), args.home), indent=1))
    return 0


def cmd_index(args):
    from .index import build

    print(json.dumps(build(_out(args), roles=("user", "assistant") if args.index_assistant else ("user",)), indent=1))
    return 0


def cmd_search(args):
    from .index import search

    for hit in search(_out(args), " ".join(args.query), args.limit):
        print(f"{(hit['timestamp'] or '')[:10]}  {hit['source']:14} {hit['snippet']}")
    return 0


def cmd_memories(args):
    from .memories import propose

    print(json.dumps(propose(_out(args)), indent=1))
    return 0


def cmd_agent_me(args):
    from .agentme import push, read_token

    token = read_token(Path(args.token_file).expanduser() if args.token_file else None)
    result = push(_out(args), args.api, token, apply=args.apply)
    print(json.dumps(result, indent=1))
    if not args.apply:
        print("Dry run. Add --apply to send these as pending entries; confirm them inside Agent-Me.")
    return 0


def cmd_verify(args):
    problems = Backup(_out(args)).verify()
    print(json.dumps({"ok": not problems, "problems": problems}, indent=1))
    return 0 if not problems else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="aihistory", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--out", default="~/AI-History", help="output folder (default ~/AI-History)")
    parser.add_argument("--home", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--export", type=Path, action="append", default=[],
                        help="a ChatGPT or Claude.ai export (zip or conversations.json); repeatable")
    parser.add_argument("--from-hstry", type=Path, default=None, metavar="DB",
                        help="an hstry database to read (default: ~/.local/share/hstry/hstry.db when it exists)")
    parser.add_argument("--agent-traces", type=Path, action="append", default=[], metavar="DIR",
                        help="an agent-trace store, or a folder holding .agent-traces; repeatable "
                             "(default: any .agent-traces in your home or a folder your history names)")
    parser.add_argument("--no-backup", action="store_true", help="skip the raw backup and read history in place")
    parser.add_argument("--index-assistant", action="store_true", help="also index assistant replies")
    sub = parser.add_subparsers(dest="command")
    for name in ("run", "scan", "backup", "normalize", "index", "memories", "verify"):
        sub.add_parser(name)
    search = sub.add_parser("search")
    search.add_argument("query", nargs="+")
    search.add_argument("--limit", type=int, default=10)
    agent = sub.add_parser("agent-me", help="send ticked memories to a local Agent-Me")
    agent_sub = agent.add_subparsers(dest="agent_command", required=True)
    agent_push = agent_sub.add_parser("push")
    agent_push.add_argument("--api", default="http://127.0.0.1:8000")
    agent_push.add_argument("--token-file", help="Agent-Me's private/personal.env (or set AGENT_ME_TOKEN)")
    agent_push.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    handlers = {"run": cmd_run, "scan": cmd_scan, "backup": cmd_backup, "normalize": cmd_normalize, "index": cmd_index,
                "search": cmd_search, "memories": cmd_memories, "agent-me": cmd_agent_me, "verify": cmd_verify, None: cmd_run}
    try:
        return handlers[args.command](args)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"aihistory: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
