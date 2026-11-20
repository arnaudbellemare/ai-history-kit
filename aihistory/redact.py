"""Secret, personal-info and home-path redaction applied before anything is written to the corpus."""

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
# A name ends in one of these words when its value is a credential. The text rule and the
# dictionary-key rule share it so they cannot drift apart.
CREDENTIAL_NAME = r"(?:api[_-]?key|secret[_-]?(?:access[_-]?)?key|private[_-]?key|token|secret|password|passwd)"
CREDENTIAL_KEY_NAME = re.compile(r"(?i)" + CREDENTIAL_NAME + r"$")
PATTERNS = (
    # Tool output is often cut before the END line. Without one, take the header and the
    # base64 lines after it (newlines may be JSON-escaped).
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----(?:.*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----|(?:(?:\\[nr]|\s)+[A-Za-z0-9+/=]{16,})*)", re.DOTALL), "[redacted:private_key]"),
    ("url_credentials", re.compile(r"\b(https?|postgres|postgresql|mysql|mongodb|redis)://([^\s/:@]+):([^\s/@]+)@"), r"\1://[redacted]@"),
    # Names that end in a credential word, with any prefix (GITHUB_TOKEN, DATABASE_PASSWORD,
    # STRIPE_API_KEY, AWS_SECRET_ACCESS_KEY): `\b` alone fails after "_". Names that only
    # contain the word, like max_tokens, token_count, and tokenizer, are left alone. Quotes
    # may be backslash-escaped when the text is JSON inside a string; the value stops before
    # an escaped closing quote.
    ("credential_assignment", re.compile(r"(?i)\b(\w*" + CREDENTIAL_NAME + r")((?:\\*[\"'])?\s*[:=]\s*)((?:\\*[\"'])?)(?!\[redacted)((?:[^\s\"',}\]\\]|\\(?![\"'])){8,})"), r"\1\2\3[redacted]"),
    ("bearer_token", re.compile(r"(?i)(Bearer\s+)(?!\[redacted)[A-Za-z0-9._~+/-]{16,}={0,2}"), r"\1[redacted]"),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"), "[redacted:anthropic_key]"),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "[redacted:openai_key]"),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "[redacted:github_token]"),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), "[redacted:aws_access_key]"),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "[redacted:slack_token]"),
    ("google_api_key", re.compile(r"\bAIza[A-Za-z0-9_-]{30,}\b"), "[redacted:google_api_key]"),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"), "[redacted:gitlab_token]"),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), "[redacted:jwt]"),
    ("stripe_key", re.compile(r"\b(?:[sr]k_(?:live|test)_[A-Za-z0-9]{16,}|whsec_[A-Za-z0-9]{24,})\b"), "[redacted:stripe_key]"),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), "[redacted:huggingface_token]"),
    ("replicate_token", re.compile(r"\br8_[A-Za-z0-9]{30,}\b"), "[redacted:replicate_token]"),
    ("groq_key", re.compile(r"\bgsk_[A-Za-z0-9]{40,}\b"), "[redacted:groq_key]"),
    ("xai_key", re.compile(r"\bxai-[A-Za-z0-9]{40,}\b"), "[redacted:xai_key]"),
    ("perplexity_key", re.compile(r"\bpplx-[A-Za-z0-9]{40,}\b"), "[redacted:perplexity_key]"),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"), "[redacted:npm_token]"),
    ("sendgrid_key", re.compile(r"\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}\b"), "[redacted:sendgrid_key]"),
    ("supabase_token", re.compile(r"\b(?:sbp_[a-f0-9]{40}|sb_secret_[A-Za-z0-9_-]{20,})\b"), "[redacted:supabase_token]"),
    ("google_oauth_token", re.compile(r"\bya29\.[A-Za-z0-9_-]{30,}"), "[redacted:google_oauth_token]"),
    ("telegram_bot_token", re.compile(r"\b[0-9]{8,10}:AA[A-Za-z0-9_-]{33}\b"), "[redacted:telegram_bot_token]"),
)
# A literal every match of the rule must contain. A rule whose literals are all absent is
# skipped, which keeps the many prefix-shaped rules cheap on long text.
PATTERN_HINTS = {
    "private_key": ("PRIVATE KEY",),
    "url_credentials": ("://",),
    "anthropic_key": ("sk-ant-",),
    "openai_key": ("sk-",),
    "github_token": ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_"),
    "aws_access_key": ("AKIA", "ASIA"),
    "slack_token": ("xox",),
    "google_api_key": ("AIza",),
    "gitlab_token": ("glpat-",),
    "jwt": ("eyJ",),
    "stripe_key": ("k_live_", "k_test_", "whsec_"),
    "huggingface_token": ("hf_",),
    "replicate_token": ("r8_",),
    "groq_key": ("gsk_",),
    "xai_key": ("xai-",),
    "perplexity_key": ("pplx-",),
    "npm_token": ("npm_",),
    "sendgrid_key": ("SG.",),
    "supabase_token": ("sbp_", "sb_secret_"),
    "google_oauth_token": ("ya29.",),
    "telegram_bot_token": (":AA",),
}

# Addresses that name a machine, a placeholder or a file, not a person.
EMAIL_KEPT_LOCAL_PARTS = {"git", "noreply", "no-reply"}
EMAIL_KEPT_DOMAINS = {"example.com", "example.org", "example.net"}
EMAIL_KEPT_DOMAIN_SUFFIXES = (".local", ".test", ".example", ".invalid", ".localhost")
EMAIL_FILE_SUFFIXES = {"png", "jpg", "jpeg", "gif", "svg", "webp", "pdf", "js", "ts", "py", "json", "md"}
HAS_DIGIT = re.compile(r"[0-9]")


def email_is_personal(address: str) -> bool:
    local, _, domain = address.lower().rpartition("@")
    if local in EMAIL_KEPT_LOCAL_PARTS or domain in EMAIL_KEPT_DOMAINS:
        return False
    if domain.endswith(EMAIL_KEPT_DOMAIN_SUFFIXES):
        return False
    return domain.rsplit(".", 1)[-1] not in EMAIL_FILE_SUFFIXES


def luhn_valid(text: str) -> bool:
    total = 0
    for index, character in enumerate(reversed([c for c in text if c.isdigit()])):
        digit = int(character)
        if index % 2 == 1:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


# A number or code may follow a JSON-escaped newline or tab, where the "n" or "t" would
# otherwise read as the preceding word character.
AFTER_ESCAPED_SPACE = r"(?<=\\[nrt])"
# (label, pattern, accept). A match is redacted only when accept is None or returns true.
# Card and SIN shapes need a valid check digit, so most plain numbers are left alone. Names,
# street addresses and free-text personal details are not detected.
PERSONAL_PATTERNS = (
    ("email", re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,10}\b"), email_is_personal),
    ("card_number", re.compile(r"(?:(?<![0-9A-Za-z])|" + AFTER_ESCAPED_SPACE + r")(?:[3-6][0-9]{3}[ -][0-9]{4}[ -][0-9]{4}[ -][0-9]{4}|4[0-9]{15}|5[1-5][0-9]{14})(?![0-9A-Za-z])"), luhn_valid),
    ("phone", re.compile(r"(?:(?<![\w-])|" + AFTER_ESCAPED_SPACE + r")(?:\+?1[ .-]?)?\(?[2-9][0-9]{2}\)?[ .-][0-9]{3}[ .-][0-9]{4}(?![\w-])"), None),
    ("sin", re.compile(r"(?:(?<![\w-])|" + AFTER_ESCAPED_SPACE + r")[1-79][0-9]{2}[ -][0-9]{3}[ -][0-9]{3}(?![\w-])"), luhn_valid),
    # The separator is required: without it, hex colours such as E0E0E0 match.
    ("postal_code", re.compile(r"(?i)(?:\b|" + AFTER_ESCAPED_SPACE + r")[ABCEGHJ-NPRSTVXY][0-9][ABCEGHJ-NPRSTV-Z][ -][0-9][ABCEGHJ-NPRSTV-Z][0-9]\b"), None),
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
        or bool(CREDENTIAL_KEY_NAME.search(str(key)))
    )


def _redact_secrets(text: str, counts: Counter) -> str:
    for _ in range(3):
        before = text
        for label, pattern, replacement in PATTERNS:
            hints = PATTERN_HINTS.get(label)
            if hints and not any(hint in text for hint in hints):
                continue
            text, count = pattern.subn(replacement, text)
            if count:
                counts[label] += count
        if text == before:
            break
    return text


def _redact_personal(text: str, counts: Counter) -> str:
    if "@" not in text and not HAS_DIGIT.search(text):
        return text
    for label, pattern, accept in PERSONAL_PATTERNS:
        if label == "email" and "@" not in text:
            continue
        hits = 0

        def replace(match):
            nonlocal hits
            if accept is not None and not accept(match.group(0)):
                return match.group(0)
            hits += 1
            return f"[redacted:{label}]"

        text = pattern.sub(replace, text)
        if hits:
            counts[label] += hits
    return text


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
        return _redact_personal(_redact_secrets(out, self.counts), self.counts)

    def value(self, value):
        if isinstance(value, dict):
            return {k: (REDACTED if key_is_sensitive(k) and v not in (None, "", [], {}) else self.value(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [self.value(v) for v in value]
        if isinstance(value, str):
            return self.text(value)
        return value


def residual_findings(text: str) -> list[str]:
    """Labels of anything another redaction pass would still change (empty after redaction)."""
    counts = Counter()
    _redact_personal(_redact_secrets(text, counts), counts)
    order = [label for label, *_ in PATTERNS + PERSONAL_PATTERNS]
    return [label for label in order if counts[label]]
