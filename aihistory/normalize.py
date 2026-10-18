"""Turn sources into a redacted corpus: one JSONL file per source under corpus/."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import SCHEMA
from .adapters import claude_code, codex, exports
from .redact import Redactor, residual_findings
from .sources import Source
from .util import ZstdUnavailable, private_jsonl, sha256_text, short_id


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


def normalize(sources: list[Source], out_dir: Path, home: Path | None = None) -> dict:
    redactor = Redactor(home)
    summary, warnings = {}, []
    by_kind: dict[str, list[Source]] = {}
    for source in sources:
        if source.kind != "codex-db":  # kept as a raw snapshot; its threads duplicate the rollouts
            by_kind.setdefault(source.kind, []).append(source)  # e.g. Codex sessions + archived_sessions
    for kind, group in by_kind.items():
        roles, seen = Counter(), set()
        records = events = duplicates = residual = 0
        target = out_dir / "corpus" / f"{kind}.jsonl"
        with private_jsonl(target) as write:
            for record in (r for source in group for r in _records(source, warnings)):
                for event in record["events"]:
                    event["text"] = redactor.text(event["text"])
                record["title"] = redactor.text(record["title"]) if record.get("title") else None
                record["origin"] = redactor.text(record["origin"])
                digest = sha256_text(json.dumps(record["events"], sort_keys=True))
                key = (record["identity"], digest)
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                residual += bool(residual_findings(json.dumps(record["events"])))
                record = {"schema": SCHEMA, "conversation_id": short_id("conv", f"{record['source']}:{record['identity']}"),
                          **{k: v for k, v in record.items() if k != "identity"}}
                write(record)
                records += 1
                events += len(record["events"])
                roles.update(e["role"] for e in record["events"])
        summary[kind] = {"file": str(target), "conversations": records, "events": events,
                                "exact_duplicates_skipped": duplicates, "roles": dict(roles),
                                "records_with_residual_secret_patterns": residual}
    return {"sources": summary, "redactions": dict(redactor.counts), "warnings": warnings[:20],
            "warning_count": len(warnings)}


def read_corpus(out_dir: Path):
    for path in sorted((out_dir / "corpus").glob("*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
