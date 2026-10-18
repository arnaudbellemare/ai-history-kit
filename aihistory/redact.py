"""Secret and home-path redaction applied before anything is written to the corpus."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

REDACTED = "[redacted]"
SENSITIVE_KEYS = {
    "accesstoken", "apikey", "authorization", "bearertoken", "clientsecret", "cookie",
    "credential", "databasepassword", "idtoken", "oauthtoken", "password", "passwordhash",
    "passwd", "privatekey", "refreshtoken", "secret", "secretaccesskey", "sessionkey",
    "signingsecret", "webhooksecret",
}
PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----", re.DOTALL), "[redacted:private_key]"),
    ("url_credentials", re.compile(r"\b(https?|postgres|postgresql|mysql|mongodb|redis)://([^\s/:@]+):([^\s/@]+)@"), r"\1://[redacted]@"),
    # Names that end in a credential word, with any prefix (GITHUB_TOKEN, DATABASE_PASSWORD,
    # STRIPE_API_KEY, AWS_SECRET_ACCESS_KEY): `\b` alone fails after "_". Names that only
    # contain the word, like max_tokens, token_count, and tokenizer, are left alone.
    ("credential_assignment", re.compile(r"(?i)\b(\w*(?:api[_-]?key|secret[_-]?(?:access[_-]?)?key|token|secret|password|passwd))([\"']?\s*[:=]\s*)([\"']?)(?!\[redacted)([^\s\"',}\]]{8,})"), r"\1\2\3[redacted]"),
    ("bearer_token", re.compile(r"(?i)(Bearer\s+)(?!\[redacted)[A-Za-z0-9._~+/-]{16,}={0,2}"), r"\1[redacted]"),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"), "[redacted:anthropic_key]"),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "[redacted:openai_key]"),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "[redacted:github_token]"),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), "[redacted:aws_access_key]"),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "[redacted:slack_token]"),
    ("google_api_key", re.compile(r"\bAIza[A-Za-z0-9_-]{30,}\b"), "[redacted:google_api_key]"),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"), "[redacted:gitlab_token]"),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), "[redacted:jwt]"),
)
MARKER = re.compile(r"\[redacted(?::[a-z_]+)?\]", re.IGNORECASE)


def _compact(key) -> str:
    return "".join(c.lower() for c in str(key) if c.isalnum())


def key_is_sensitive(key) -> bool:
    compact = _compact(key)
    return (
        compact in SENSITIVE_KEYS
        or compact.endswith(("accesstoken", "refreshtoken", "bearertoken", "clientsecret", "privatekey", "secretkey"))
        or compact.startswith("authorization")
    )


class Redactor:
    """Redacts credential-shaped text and replaces the home directory with $HOME."""

    def __init__(self, home: Path | str | None = None):
        self.home = str(home or Path.home()).rstrip("/")
        self.counts = Counter()

    def text(self, value: str) -> str:
        out = value
        if self.home and self.home in out:
            self.counts["home_path"] += out.count(self.home)
            out = out.replace(self.home, "$HOME")
        for _ in range(3):
            before = out
            for label, pattern, replacement in PATTERNS:
                out, count = pattern.subn(replacement, out)
                if count:
                    self.counts[label] += count
            if out == before:
                break
        return out

    def value(self, value):
        if isinstance(value, dict):
            return {k: (REDACTED if key_is_sensitive(k) and v not in (None, "", [], {}) else self.value(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [self.value(v) for v in value]
        if isinstance(value, str):
            return self.text(value)
        return value


def residual_findings(text: str) -> list[str]:
    """Labels of any secret pattern still present (empty after redaction)."""
    return [label for label, pattern, _ in PATTERNS if pattern.search(text)]
