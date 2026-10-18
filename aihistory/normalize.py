"""Turn sources into a redacted corpus: one JSONL file per source under corpus/."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import SCHEMA
from .adapters import agent_trace, claude_code, codex, exports, hstry
from .redact import Redactor, residual_findings
from .sources import Source
from .util import ZstdUnavailable, private_jsonl, sha256_text, short_id

DERIVED = ("hstry", "agent-trace")  # copies of other apps' sessions: read after the native readers


def _records(source: Source, warnings: list[str]):
    if source.kind == "claude-code":
        for path in source.files:
            record = claude_code.read_session(path, source.root)
            if record:
                yield record
    elif source.kind == "codex":
        for path in source.files:
            try:
                record = codex.read_rollout(path, source.root)
            except ZstdUnavailable as error:
                warnings.append(f"{path.name}: {error}")
                continue
            if record:
                yield record
    elif source.kind == "chatgpt-export":
        for path in source.files:
            yield from exports.read_chatgpt(path)
    elif source.kind == "claude-export":
        for path in source.files:
            yield from exports.read_claude(path)
    elif source.kind == "hstry":
        for path in source.files:
            yield from hstry.read_db(path)
    elif source.kind == "agent-trace":
        for path in source.files:
            record = agent_trace.read_session(path, source.root)
            if record:
                yield record


def normalize(sources: list[Source], out_dir: Path, home: Path | None = None) -> dict:
    redactor = Redactor(home)
    summary, warnings = {}, []
    by_kind: dict[str, list[Source]] = {}
    for source in sources:
        if source.kind != "codex-db":  # kept as a raw snapshot; its threads duplicate the rollouts
            by_kind.setdefault(source.kind, []).append(source)  # e.g. Codex sessions + archived_sessions
    # "<kind>:<session id>" for every native conversation, plus the id's first 16 characters,
    # which is all an agent-trace hook keeps. hstry and agent-trace copies of them are skipped.
    native_keys = set()
    for kind in sorted(by_kind, key=lambda k: k in DERIVED):
        group = by_kind[kind]
        roles, seen, broken = Counter(), set(), []
        records = events = duplicates = residual = native_duplicates = 0
        target = out_dir / "corpus" / f"{kind}.jsonl"
        with private_jsonl(target) as write:
            for record in (r for source in group for r in _records(source, warnings)):
                native, identity = record.pop("native", None), record["identity"]
                if native and f"{native}:{identity}" in native_keys:
                    native_duplicates += 1
                    continue
                for event in record["events"]:
                    event["text"] = redactor.text(event["text"])
                record["title"] = redactor.text(record["title"]) if record.get("title") else None
                record["origin"] = redactor.text(record["origin"])
                record["workspace"] = redactor.text(record["workspace"]) if record.get("workspace") else None
                digest = sha256_text(json.dumps(record["events"], sort_keys=True))
                key = (record["identity"], digest)
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                if ((record.get("stats") or {}).get("chain") or {}).get("status") == "broken":
                    broken.append(record["origin"])
                residual += bool(residual_findings(json.dumps(record["events"])))
                record = {"schema": SCHEMA, "conversation_id": short_id("conv", f"{record['source']}:{record['identity']}"),
                          **{k: v for k, v in record.items() if k != "identity"}}
                write(record)
                if kind in ("claude-code", "codex"):
                    native_keys.update({f"{kind}:{identity}", f"{kind}:{identity[:agent_trace.NATIVE_ID_PREFIX]}"})
                records += 1
                events += len(record["events"])
                roles.update(e["role"] for e in record["events"])
        summary[kind] = {"file": str(target), "conversations": records, "events": events,
                                "exact_duplicates_skipped": duplicates, "roles": dict(roles),
                                "already_read_natively": native_duplicates,
                                "records_with_residual_secret_patterns": residual}
        if kind == "agent-trace":
            summary[kind]["broken_hash_chains"] = broken
    return {"sources": summary, "redactions": dict(redactor.counts), "warnings": warnings[:20],
            "warning_count": len(warnings)}


def read_corpus_offsets(out_dir: Path):
    """(corpus file name, byte offset of the record's line, record) for every record."""
    for path in sorted((out_dir / "corpus").glob("*.jsonl")):
        with open(path, "rb") as handle:
            while line := handle.readline():
                if line.strip():
                    yield path.name, handle.tell() - len(line), json.loads(line)


def read_corpus(out_dir: Path):
    for _, _, record in read_corpus_offsets(out_dir):
        yield record
