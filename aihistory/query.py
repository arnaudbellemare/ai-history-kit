"""Search syntax: words, "exact phrases", -excluded words, prefix*, OR, and filters.

    postgres migration            both words (stemmed: "migrations" matches too)
    "write tests"                 the exact phrase
    kite OR sailboat              either word
    deploy -staging               deploy, but not staging
    migrat*                       any word starting with "migrat"
    source:codex role:user workspace:myproject after:2026-09-01 before:2026-10-01

Every term is quoted before it reaches SQLite FTS5, so no user text is read as FTS5 syntax.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FILTERS = ("source", "role", "workspace", "after", "before")
TOKEN = re.compile(r'(-?)"([^"]*)"|(\S+)')
WORD = re.compile(r"\w", re.UNICODE)


@dataclass
class Query:
    groups: list[list[tuple[str, bool]]] = field(default_factory=list)  # AND of OR-groups of (term, is_prefix)
    excluded: list[str] = field(default_factory=list)
    filters: dict[str, str] = field(default_factory=dict)

    @property
    def terms(self) -> list[str]:
        return [term for group in self.groups for term, _ in group]


def _clean(term: str) -> str:
    term = term.replace('"', " ").strip()
    return term if WORD.search(term) else ""


def parse(text: str) -> Query:
    query, pending_or = Query(), False
    for match in TOKEN.finditer(text):
        negated, phrase, bare = match.groups()
        if phrase is not None:
            term, prefix, negated = _clean(phrase), False, bool(negated)
        else:
            if bare == "OR":
                pending_or = bool(query.groups)
                continue
            key, colon, value = bare.partition(":")
            if colon and key.lower() in FILTERS:
                if value:
                    query.filters[key.lower()] = value
                continue
            negated = bare.startswith("-")
            bare = bare.lstrip("-")
            prefix = bare.endswith("*")
            term = _clean(bare.rstrip("*"))
        if not term:
            continue
        if negated:
            query.excluded.append(term)
        elif pending_or:
            query.groups[-1].append((term, prefix))
        else:
            query.groups.append([(term, prefix)])
        pending_or = False
    return query


def quote(term: str, prefix: bool = False) -> str:
    return '"' + term.replace('"', '""') + '"' + ("*" if prefix else "")


def to_match(query: Query, mode: str = "all") -> str | None:
    """An FTS5 MATCH expression: `all` ANDs the groups, `any` ORs every term. None if nothing to find."""
    if not query.groups:
        return None
    if mode == "any":
        positive = "(" + " OR ".join(quote(t, p) for group in query.groups for t, p in group) + ")"
    else:
        positive = " AND ".join("(" + " OR ".join(quote(t, p) for t, p in group) + ")" for group in query.groups)
    return positive + "".join(f" NOT {quote(term)}" for term in query.excluded)
