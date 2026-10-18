"""aihistory: back up, clean, search, and mine your AI history. Everything stays local.

    aihistory run                     # everything below, into ~/AI-History
    aihistory scan                    # what exists, date coverage, gaps
    aihistory backup                  # verified, catalogued copy of raw history
    aihistory normalize               # redacted corpus with owner/agent/harness roles
    aihistory index                   # local full-text index of what you wrote
    aihistory search "postgres migration"   # see `aihistory search -h` for the query syntax
    aihistory show CONVERSATION --at SEQ   # a search hit with the messages around it
    aihistory memories                # memory candidates -> memories/REVIEW.md
    aihistory agent-me push [--apply] # send ticked memories to a local Agent-Me
    aihistory verify                  # re-check every backed-up file
    aihistory optimize mine|review|label|arm|eval|report|train --run NAME   # test skill descriptions

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
    from .index import query

    result = query(_out(args), " ".join(args.query), mode=args.mode, limit=args.limit, source=args.source,
                   role=args.role, workspace=args.workspace, after=args.after, before=args.before)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return 0
    if result["corrections"]:
        print("No matches as typed; showing " + ", ".join(f"{k} → {v}" for k, v in result["corrections"].items()))
    elif result["mode_used"] in ("any", "substring"):
        print({"any": "No message has every word; showing messages with any of them.",
               "substring": "No whole-word matches; showing messages that contain the text."}[result["mode_used"]])
    for hit in result["hits"]:
        print(f"{(hit['timestamp'] or '')[:10]}  {hit['source']:14} {hit['snippet']}  ({hit['conversation']} --at {hit['seq']})")
    if not result["hits"]:
        print("No matches.")
    return 0


def cmd_show(args):
    from .index import show

    context = args.context if args.context is not None else (3 if args.at is not None else None)
    conversation = show(_out(args), args.conversation, args.at, context)
    if args.json:
        print(json.dumps(conversation, ensure_ascii=False, indent=1))
        return 0
    print(f"# {conversation['source']}  {conversation['title'] or ''}  ({conversation['conversation']})")
    for event in conversation["events"]:
        label = event["role"] + (f":{event['tool']}" if event.get("tool") else "")
        print(f"{'>' if event['hit'] else ' '} [{label}] {event['text']}")
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


def cmd_optimize(args):
    from .optimize import agents, dataset, experiment

    out = _out(args)
    if args.optimize_command == "rule":
        from .optimize import rules

        if args.rule_command == "example":
            print(json.dumps(rules.EXAMPLE, indent=1))
            return 0
        run_dir = experiment.create_rule_run(out, args.run, Path(args.file).expanduser(), args.section,
                                             Path(args.scenarios).expanduser())
        split = json.loads((run_dir / "dataset.json").read_text())
        print(f"Rule run {run_dir.name}: scenarios {', '.join(f'{k} {len(v)}' for k, v in split.items())}. "
              f"Arms present and removed are set; add a control with "
              f"`aihistory optimize arm --run {args.run} control --text \"one plain line\"`.")
        return 0
    if args.optimize_command == "gskill":
        from .optimize import gskill

        run_dir = gskill.init_run(out, args.run, Path(args.repo).expanduser(), args.test_cmd,
                                  Path(args.venv).expanduser() if args.venv else None, args.tests or None,
                                  args.source or None, k=args.k, candidates=args.candidates, max_f2p=args.max_f2p,
                                  workers=args.workers, on_progress=print)
        split = json.loads((run_dir / "dataset.json").read_text())
        print(f"gskill run {run_dir.name}: tasks {', '.join(f'{k} {len(v)}' for k, v in split.items())}. "
              f"Add a control with `aihistory optimize arm --run {args.run} control --text \"one line\"`.")
        return 0
    run_dir = experiment.run_path(out, args.run)
    if args.optimize_command == "mine":
        home = args.home or Path.home()
        roots = experiment.skill_roots(home, Path.cwd()) + [Path(d).expanduser() for d in args.skills_dir]
        menu = experiment.find_skills(roots)
        targets = {agents.safe_name(s): menu.get(agents.safe_name(s)) for s in args.skill}
        experiment.create_run(out, args.run, targets, decoys=None if args.no_decoys else menu)
        keywords = {k: v.split(",") for k, v in experiment.parse_assignments(args.keywords).items()}
        from .optimize.textmodels import make_text_model

        def plan(calls, judged):
            if calls:
                print(f"{calls} model calls planned: {'generating prompts from skill instructions; ' if args.generate else ''}"
                      f"{f'sending {judged} prompts you wrote to {args.judge} for judging' if judged else 'no judging'}.")

        dataset.mine(out, args.run, args.skill, keywords, negatives=args.negatives, run_dir=run_dir,
                     skill_files=experiment.find_skill_files(roots), generate=args.generate,
                     generator=make_text_model(args.generator, args.generator_model) if args.generate else None,
                     judge=make_text_model(args.judge, args.judge_model) if args.judge != "none" else None,
                     audit=args.audit, max_calls=args.max_calls, yes=args.yes, on_plan=plan)
        missing = [t for t, d in targets.items() if not d]
        print(f"Wrote {run_dir / 'LABEL.md'}: check the pre-filled labels, tick `I checked this` on each prompt you "
              f"checked, then `aihistory optimize label --run {args.run}`.")
        if missing:
            print(f"No SKILL.md found for {', '.join(missing)}; set their current text with "
                  f"`aihistory optimize arm --run {args.run} original --skill NAME=...`.")
    elif args.optimize_command == "review":
        from .optimize import review

        result = review.review(run_dir, everything=args.all)
        print(f"\n{result['checked_this_session']} checked this session; {result['checked']} checked in all, "
              f"{result['unchecked']} still unchecked ({result['unchecked_flagged']} of them flagged). "
              f"When you're done, run `aihistory optimize label --run {args.run}`.")
    elif args.optimize_command == "label":
        split = dataset.apply_labels(run_dir, test=args.test, val=args.val)
        audit = json.loads((run_dir / "dataset.json").read_text()).get("label_audit") or {}
        print(json.dumps({"splits": {name: len(part) for name, part in split.items()},
                          "confirmed": sum(e["confirmed"] for part in split.values() for e in part),
                          "pre-fills you checked, kept or changed by source": audit}))
    elif args.optimize_command == "arm":
        if bool(args.skill) == (args.text is not None):
            raise ValueError("give --skill NAME=text (skill triggers) or --text (a rule section), not both")
        if args.text is not None:
            experiment.set_text_arm(run_dir, args.arm, args.text)
        else:
            experiment.set_arm(run_dir, args.arm, experiment.parse_assignments(args.skill))
        print(f"Arm `{args.arm}` set.")
    elif args.optimize_command == "eval":
        agent = agents.make_agent(args.agent, args.model)
        kind = experiment.load_config(run_dir).get("kind", "triggers")
        arms = args.arms or {"rule": "present,removed,control", "gskill": "baseline,control"}.get(kind, "original,control")
        result = experiment.evaluate(run_dir, agent, args.split, arms.split(","), runs=args.runs,
                                     workers=args.workers, max_runs=args.max_runs, yes=args.yes,
                                     on_plan=lambda n: print(f"{n} agent runs planned with {experiment.agent_label(agent)}."))
        print(json.dumps(result))
    elif args.optimize_command == "report":
        print(experiment.report(run_dir, args.split))
    elif args.optimize_command == "train":
        from .optimize import gepa_train

        agent = agents.make_agent(args.agent, args.model)
        result = gepa_train.train(run_dir, agent, budget=args.budget, runs=args.runs,
                                  reflection_model=args.reflection_model, workers=args.workers,
                                  max_runs=args.max_runs, yes=args.yes, log=print,
                                  on_plan=lambda n: print(f"Up to {n} agent runs planned with {experiment.agent_label(agent)}."))
        print(json.dumps(result))
        print(f"The best descriptions are now the arm `gepa`. Compare them on the test split: "
              f"aihistory optimize eval --run {args.run} --arms original,control,gepa")
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
    from .query import __doc__ as query_syntax

    search = sub.add_parser("search", description=query_syntax, formatter_class=argparse.RawDescriptionHelpFormatter)
    search.add_argument("query", nargs="+")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--mode", choices=("auto", "all", "any", "substring"), default="auto",
                        help="auto tries every word, then any word, then substrings, then a spelling fix")
    for name in ("source", "role", "workspace", "after", "before"):
        search.add_argument(f"--{name}", help=f"same as {name}: in the query")
    search.add_argument("--json", action="store_true", help="machine-readable hits")
    show = sub.add_parser("show", help="a conversation from the index, every message labelled by who wrote it")
    show.add_argument("conversation", help="the `conversation` value of a search hit")
    show.add_argument("--at", type=int, help="the hit's message number (seq); marked with >")
    show.add_argument("--context", type=int, help="messages to show on each side of --at (default 3)")
    show.add_argument("--json", action="store_true")
    agent = sub.add_parser("agent-me", help="send ticked memories to a local Agent-Me")
    agent_sub = agent.add_subparsers(dest="agent_command", required=True)
    agent_push = agent_sub.add_parser("push")
    agent_push.add_argument("--api", default="http://127.0.0.1:8000")
    agent_push.add_argument("--token-file", help="Agent-Me's private/personal.env (or set AGENT_ME_TOKEN)")
    agent_push.add_argument("--apply", action="store_true")
    optimize = sub.add_parser("optimize", help="test (and tune) skill descriptions against prompts you wrote")
    optimize_sub = optimize.add_subparsers(dest="optimize_command", required=True)
    mine = optimize_sub.add_parser("mine", help="candidate prompts from your history -> LABEL.md")
    mine.add_argument("--skill", action="append", required=True, help="a target skill; repeatable")
    mine.add_argument("--keywords", action="append", default=[], metavar="SKILL=word,word",
                      help="also offer your messages that mention these words")
    mine.add_argument("--negatives", type=int, default=40, help="random messages of yours to offer as likely negatives")
    mine.add_argument("--skills-dir", action="append", default=[], help="another folder of skills for the menu")
    mine.add_argument("--no-decoys", action="store_true", help="evaluate without your other skills on the menu")
    mine.add_argument("--generate", type=int, default=0, metavar="N",
                      help="also have a model write N should-load and N near-miss prompts per skill from its instructions")
    mine.add_argument("--generator", default="claude", choices=("claude", "codex", "fake"))
    mine.add_argument("--generator-model")
    mine.add_argument("--judge", default="none", choices=("none", "codex", "claude", "fake"),
                      help="a model labels prompts the history doesn't (codex is another model family than Claude)")
    mine.add_argument("--judge-model")
    mine.add_argument("--audit", type=float, default=0.1, help="share of pre-filled prompts flagged for a random check")
    mine.add_argument("--max-calls", type=int, default=30, help="refuse more model calls than this without --yes")
    mine.add_argument("--yes", action="store_true")
    label = optimize_sub.add_parser("label", help="LABEL.md -> train/val/test splits")
    label.add_argument("--test", type=float, default=0.4)
    label.add_argument("--val", type=float, default=0.2)
    review_cmd = optimize_sub.add_parser("review", help="confirm pre-filled labels one prompt at a time from the keyboard")
    review_cmd.add_argument("--all", action="store_true", help="also show prompts you already checked")
    arm = optimize_sub.add_parser("arm", help="set an arm's descriptions, e.g. the control")
    arm.add_argument("arm")
    arm.add_argument("--skill", action="append", default=[], metavar="NAME=text", help="a skill description (trigger runs)")
    arm.add_argument("--text", help="this arm's section text (rule runs) or skill (gskill runs)")
    evaluate = optimize_sub.add_parser("eval", help="run the agent on a split, for each arm")
    evaluate.add_argument("--split", default="test", choices=("train", "val", "test"))
    evaluate.add_argument("--arms", help="comma-separated (default: original,control; for a rule run present,removed,control)")
    evaluate.add_argument("--agent", default="claude-code", choices=("claude-code", "codex", "fake"))
    evaluate.add_argument("--model", help="the agent's model (default: sonnet for Claude Code)")
    evaluate.add_argument("--runs", type=int, default=3, help="runs per prompt per arm")
    evaluate.add_argument("--workers", type=int, default=4)
    evaluate.add_argument("--max-runs", type=int, default=200, help="refuse larger evaluations without --yes")
    evaluate.add_argument("--yes", action="store_true")
    report = optimize_sub.add_parser("report", help="accuracy per arm and paired tests -> REPORT.md")
    report.add_argument("--split", default="test", choices=("train", "val", "test"))
    train = optimize_sub.add_parser("train", help="GEPA rewrites the descriptions (needs the [optimize] extra)")
    train.add_argument("--agent", default="claude-code", choices=("claude-code", "codex", "fake"))
    train.add_argument("--model", help="the agent's model (default: sonnet for Claude Code)")
    train.add_argument("--budget", type=int, default=100, help="GEPA metric calls (one prompt each)")
    train.add_argument("--runs", type=int, default=2, help="agent runs per metric call")
    train.add_argument("--reflection-model", default="sonnet", help="the claude -p model that writes rewrites")
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--max-runs", type=int, default=200, help="refuse larger trainings without --yes")
    train.add_argument("--yes", action="store_true")
    rule = optimize_sub.add_parser("rule", help="test a section of your CLAUDE.md or AGENTS.md on scenarios")
    rule_sub = rule.add_subparsers(dest="rule_command", required=True)
    rule_sub.add_parser("example", help="print an example scenarios file")
    init = rule_sub.add_parser("init", help="start a rule run")
    init.add_argument("--run", required=True)
    init.add_argument("--file", required=True, help="your CLAUDE.md or AGENTS.md")
    init.add_argument("--section", required=True, help="the heading of the section under test")
    init.add_argument("--scenarios", required=True, help="a scenarios JSON file (see `optimize rule example`)")
    skill = optimize_sub.add_parser("gskill", help="make bug-fix tasks from a Python repository, to learn a repo skill")
    skill.add_argument("--run", required=True)
    skill.add_argument("--repo", required=True, help="the repository to copy (it is never changed)")
    skill.add_argument("--test-cmd", required=True, help="e.g. 'python -m pytest -q -rA' or 'python -m unittest -v'")
    skill.add_argument("--venv", help="a virtualenv to link into each copy as .venv (for a test command using .venv/bin/python)")
    skill.add_argument("--tests", action="append", default=[], help=f"test file globs (default: {', '.join(['test_*.py', 'tests/*', '...'])})")
    skill.add_argument("--source", action="append", default=[], help="only mutate files matching these globs")
    skill.add_argument("--k", type=int, default=3, help="bugs per task (single bugs are often too easy)")
    skill.add_argument("--candidates", type=int, default=200, help="mutations to try")
    skill.add_argument("--max-f2p", type=int, default=20, help="most tests a kept bug may break")
    skill.add_argument("--workers", type=int, default=4)
    for command in (mine, review_cmd, label, arm, evaluate, report, train):
        command.add_argument("--run", required=True, help="the run's name (a folder under <out>/optimize)")
    args = parser.parse_args(argv)
    handlers = {"run": cmd_run, "scan": cmd_scan, "backup": cmd_backup, "normalize": cmd_normalize, "index": cmd_index,
                "search": cmd_search, "show": cmd_show, "memories": cmd_memories, "agent-me": cmd_agent_me, "verify": cmd_verify, "optimize": cmd_optimize, None: cmd_run}
    try:
        return handlers[args.command](args)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"aihistory: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
