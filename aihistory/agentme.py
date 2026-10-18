"""Send the memories you ticked in REVIEW.md to a local Agent-Me as pending entries.

Agent-Me (https://github.com/jzjzzzzzzz/agent-me) is an open-source personal AI twin.
Only loopback URLs are accepted, dry run is the default, and entries arrive as
`pending`: you still confirm each one inside Agent-Me.
"""

from __future__ import annotations

import ipaddress
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .memories import accepted_ids
from .util import write_private_json


def loopback(api: str) -> str:
    parsed = urlparse(api)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        raise ValueError("the Agent-Me URL must be http(s)://localhost:PORT")
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError
        except ValueError:
            raise ValueError("refusing a non-local Agent-Me URL: memories never leave this machine") from None
    if parsed.username or parsed.password or parsed.query or parsed.path not in ("", "/"):
        raise ValueError("the Agent-Me URL must be a bare http(s)://localhost:PORT")
    return api.rstrip("/")


def read_token(token_file: Path | None) -> str:
    if token_file is None:
        token = os.environ.get("AGENT_ME_TOKEN") or os.environ.get("PERSONAL_TOKEN")
        if not token:
            raise ValueError("set AGENT_ME_TOKEN or pass --token-file (Agent-Me's private/personal.env)")
        return token
    for line in token_file.read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() in ("PERSONAL_TOKEN", "AGENT_ME_TOKEN"):
            return value.strip().strip("'\"")
    raise ValueError(f"no PERSONAL_TOKEN in {token_file}")


def _http(method: str, url: str, token: str, payload=None, timeout: float = 10):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    try:
        with urlopen(Request(url, data=data, method=method, headers=headers), timeout=timeout) as response:
            return json.loads(response.read().decode())
    except HTTPError as error:
        raise RuntimeError(f"Agent-Me returned HTTP {error.code}: {error.read(300).decode(errors='replace')}") from error
    except URLError as error:
        raise RuntimeError(f"Agent-Me is not reachable at {url}: {error.reason} (start it in private mode first)") from error


def push(out_dir: Path, api: str, token: str, apply: bool = False) -> dict:
    review = out_dir / "memories" / "REVIEW.md"
    candidates = {c["id"]: c for c in json.loads((out_dir / "memories" / "candidates.json").read_text())["candidates"]}
    ticked = accepted_ids(review)
    unknown = [i for i in ticked if i not in candidates]
    if unknown:
        raise ValueError(f"REVIEW.md ticks ids that are not in candidates.json: {unknown} (re-run `aihistory memories`?)")
    endpoint = loopback(api) + "/api/v1/personal/entries"
    existing = _http("GET", endpoint, token)
    have = {(e.get("kind"), e.get("key"), e.get("content")) for e in existing}
    by_key = {}
    for e in existing:
        by_key.setdefault((e.get("kind"), e.get("key")), set()).add(e.get("content"))
    plan = {"would_submit": [], "already_in_agent_me": [], "conflicts_with_agent_me": []}
    for cid in ticked:
        c = candidates[cid]
        if (c["kind"], c["key"], c["content"]) in have:
            plan["already_in_agent_me"].append(cid)
        elif by_key.get((c["kind"], c["key"])) and c["kind"] == "fact":
            plan["conflicts_with_agent_me"].append(cid)  # e.g. a different identity.name; settle it in Agent-Me
        else:
            plan["would_submit"].append(cid)
    result = {"mode": "apply" if apply else "dry_run", "ticked": len(ticked), **plan, "submitted": []}
    if apply:
        for cid in plan["would_submit"]:
            c = candidates[cid]
            created = _http("POST", endpoint, token, {"kind": c["kind"], "key": c["key"], "content": c["content"]})
            result["submitted"].append({"candidate": cid, "agent_me_entry": created.get("id"), "status": created.get("status")})
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        write_private_json(out_dir / "memories" / f"agent-me-receipt-{stamp}.json", result)
    return result
