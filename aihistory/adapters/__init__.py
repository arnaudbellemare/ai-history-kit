"""Source adapters. Each yields conversation records with roles that separate the
owner (`user`) from text an agent or the harness wrote (`agent`, `harness`)."""

TOOL_CHARS = 2000  # tool inputs/outputs are truncated; messages are kept whole


def clip(text: str, limit: int = TOOL_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f" …[{len(text) - limit} more chars]"
