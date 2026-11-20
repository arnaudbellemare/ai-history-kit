"""Redaction rules: vendor tokens, cut-off key blocks, credential names, and personal info.

Every secret-shaped sample is assembled at runtime so no token-shaped literal is committed.
Addresses use the unassigned .zz domain and phone numbers the fictional 555-01xx range, so
no sample can belong to a real person.
"""

import base64
import json
import unittest

from aihistory.redact import Redactor, residual_findings

BODY = "aB3dE6gH9jK2mN5pQ8sT1vW4xZ7cF0hL" * 3
HEXES = "0123456789abcdef" * 4
FAKE_SECRET = "abcdefgh" + "12345678"


def unsigned_jwt(claims) -> str:
    def segment(value) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    return ".".join((segment({"alg": "HS256", "typ": "JWT"}), segment(claims), "s1gNaTuRe" * 5))


def with_check_digit(prefix: str) -> str:
    """Append the digit that makes the number pass the Luhn check."""
    for digit in "0123456789":
        candidate = prefix + digit
        total = 0
        for index, character in enumerate(reversed(candidate)):
            number = int(character)
            if index % 2 == 1:
                number = number * 2 - 9 if number > 4 else number * 2
            total += number
        if total % 10 == 0:
            return candidate
    raise AssertionError(f"no check digit for {prefix}")


def redactor() -> Redactor:
    return Redactor("/nonexistent")


class RedactionTests(unittest.TestCase):
    def test_vendor_token_shapes_are_redacted(self):
        samples = {
            "stripe_key": ["sk_" + "live_" + BODY[:28], "rk_" + "live_" + BODY[:28],
                           "sk_" + "test_" + BODY[:28], "whsec" + "_" + BODY[:32]],
            "huggingface_token": ["hf" + "_" + BODY[:34]],
            "replicate_token": ["r8" + "_" + BODY[:37]],
            "groq_key": ["gsk" + "_" + BODY[:52]],
            "xai_key": ["xai" + "-" + BODY[:60]],
            "perplexity_key": ["pplx" + "-" + BODY[:48]],
            "npm_token": ["npm" + "_" + BODY[:36]],
            "sendgrid_key": ["SG" + "." + BODY[:22] + "." + BODY[:43]],
            "supabase_token": ["sbp" + "_" + HEXES[:40]],
            "google_oauth_token": ["ya29" + "." + BODY[:60]],
            "telegram_bot_token": ["1234567890" + ":AA" + BODY[:33]],
            # Older rules, listed so a wrong skip literal cannot silently disable one.
            "jwt": [unsigned_jwt({"role": "service_role", "ref": BODY[:20]})],
            "anthropic_key": ["sk" + "-ant-" + BODY[:40]],
            "openai_key": ["sk" + "-" + BODY[:40]],
            "github_token": ["ghp" + "_" + BODY[:36], "github" + "_pat_" + BODY[:40]],
            "aws_access_key": ["AKIA" + "Q7ZK2M9XB4T6WN8P", "ASIA" + "Q7ZK2M9XB4T6WN8P"],
            "slack_token": ["xoxb" + "-" + "123456789012-" + BODY[:24]],
            "google_api_key": ["AIza" + BODY[:35]],
            "gitlab_token": ["glpat" + "-" + BODY[:24]],
        }
        for label, values in samples.items():
            for value in values:
                with self.subTest(label=label, prefix=value[:6]):
                    cleaner = redactor()
                    cleaned = cleaner.text(f"output was {value} here")
                    self.assertEqual(cleaned, f"output was [redacted:{label}] here",
                                     "vendor token survived redaction")
                    self.assertEqual(dict(cleaner.counts), {label: 1})
                    self.assertEqual(residual_findings(cleaned), [])

    def test_private_key_block_cut_before_its_end_marker_is_redacted(self):
        header = "-----BEGIN OPENSSH " + "PRIVATE KEY-----"
        footer = "-----END OPENSSH " + "PRIVATE KEY-----"
        body = ["b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAE" + str(index) * 12 for index in range(3)]
        cases = {
            "\n".join([header] + body + [footer]): "[redacted:private_key]",
            "\n".join([header] + body) + "\n[output truncated]": "[redacted:private_key]\n[output truncated]",
            "\\n".join([header] + body) + '\\n" }': '[redacted:private_key]\\n" }',
            header + " is the first line of a key file": "[redacted:private_key] is the first line of a key file",
        }
        for text, expected in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(redactor().text(text), expected, "private key material survived redaction")

    def test_private_key_assignments_and_escaped_json_credentials_are_redacted(self):
        hex_key = "0x" + "9f3c1e7a" * 8
        cases = {
            f"PRIVATE_KEY={hex_key}": "PRIVATE_KEY=[redacted]",
            f'wallet_private_key: "{hex_key}"': 'wallet_private_key: "[redacted]"',
            f'{{\\"client_secret\\": \\"{FAKE_SECRET}\\"}}': '{\\"client_secret\\": \\"[redacted]\\"}',
            f'\\"GITHUB_TOKEN\\":\\"{FAKE_SECRET}\\",\\"n\\":1': '\\"GITHUB_TOKEN\\":\\"[redacted]\\",\\"n\\":1',
        }
        for text, expected in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(redactor().text(text), expected, "credential value survived redaction")

    def test_values_under_credential_named_keys_are_redacted(self):
        value = redactor().value({
            "token": FAKE_SECRET,
            "OPENROUTER_API_KEY": FAKE_SECRET,
            "env": {"HF_TOKEN": FAKE_SECRET, "walletPrivateKey": FAKE_SECRET},
            "max_tokens": 4096,
            "token_count": 42,
            "tokens": [1, 2],
            "tokenizer": "cl100k_base",
            "next_token": None,
        })
        self.assertEqual(value, {
            "token": "[redacted]",
            "OPENROUTER_API_KEY": "[redacted]",
            "env": {"HF_TOKEN": "[redacted]", "walletPrivateKey": "[redacted]"},
            "max_tokens": 4096,
            "token_count": 42,
            "tokens": [1, 2],
            "tokenizer": "cl100k_base",
            "next_token": None,
        }, "a value under a credential-named key was kept, or a plain setting was redacted")
        self.assertEqual(residual_findings(json.dumps(value)), [])

    def test_personal_contact_details_and_identity_numbers_are_redacted(self):
        sin = with_check_digit("27346158")
        card = with_check_digit("424242424242424")
        cases = {
            "write to jane.tenant@mailbox.zz today": ("write to [redacted:email] today", "email"),
            "<m.tremblay+bail@courriel.zz>": ("<[redacted:email]>", "email"),
            "Message-ID: <{}@mx.mailbox.zz>".format("CAF7x" * 16): ("Message-ID: <[redacted:email]>", "email"),
            "ssh ops@build.corp.internal": ("ssh [redacted:email]", "email"),
            "call 514-555-0142 now": ("call [redacted:phone] now", "phone"),
            "cell (438) 555-0177.": ("cell [redacted:phone].", "phone"),
            "tel: +1 450.555.0123": ("tel: [redacted:phone]", "phone"),
            "T.514-555-0142 / C.+438 555 0177": ("T.[redacted:phone] / C.+[redacted:phone]", "phone"),
            "line one\\n514-555-0142\\nline three": ("line one\\n[redacted:phone]\\nline three", "phone"),
            "unit 4, H2X 1Y4, Montreal": ("unit 4, [redacted:postal_code], Montreal", "postal_code"),
            "code postal j4b-5e9": ("code postal [redacted:postal_code]", "postal_code"),
            "addr\\nH2X 1Y4": ("addr\\n[redacted:postal_code]", "postal_code"),
            f"NAS {sin[:3]} {sin[3:6]} {sin[6:]}": ("NAS [redacted:sin]", "sin"),
            f"card {card[:4]} {card[4:8]} {card[8:12]} {card[12:]}": ("card [redacted:card_number]", "card_number"),
            f"pan={card}": ("pan=[redacted:card_number]", "card_number"),
        }
        for text, (expected, label) in cases.items():
            with self.subTest(label=label, expected=expected):
                cleaner = redactor()
                cleaned = cleaner.text(text)
                self.assertEqual(cleaned, expected, "personal info survived redaction")
                self.assertEqual(dict(cleaner.counts), {label: cleaned.count("[redacted:")})
                self.assertEqual(residual_findings(cleaned), [])

    def test_text_that_only_resembles_personal_info_is_kept(self):
        sin = with_check_digit("27346158")
        wrong_sin = sin[:8] + str((int(sin[8]) + 1) % 10)
        card = with_check_digit("424242424242424")
        wrong_card = card[:15] + str((int(card[15]) + 1) % 10)
        for text in (
            "git clone git@github.com:example/repo.git",
            "Co-authored-by: Bot <noreply@example-vendor.com>",
            "docs use user@example.com",
            "ssh admin@printer.local",
            'srcset="icon@2x.png 2x"',
            "npm install react@18.3.1 && pip install a@b",
            "background: #E0E0E0; color: #A1B2C3",
            "at 2026-09-11T10:22:33.123Z took 1757586153123 ms",
            "range 100-200-3000x and id 7-514-555-0142",
            f"NAS {wrong_sin[:3]} {wrong_sin[3:6]} {wrong_sin[6:]}",
            f"card {wrong_card[:4]} {wrong_card[4:8]} {wrong_card[8:12]} {wrong_card[12:]}",
            "sha 4e3b1c9a7f2d4b1a8c6e0f3a5d7b9c1e2f4a6b8c",
        ):
            with self.subTest(text=text):
                cleaner = redactor()
                self.assertEqual(cleaner.text(text), text, "ordinary text was redacted")
                self.assertEqual(dict(cleaner.counts), {})
                self.assertEqual(residual_findings(text), [], "kept text was reported as a residual finding")

    def test_residual_findings_names_what_a_second_pass_would_still_change(self):
        self.assertEqual(residual_findings("mail jane.tenant@mailbox.zz or call 514-555-0142"), ["email", "phone"])
        self.assertEqual(residual_findings("key " + "sk_" + "live_" + BODY[:28]), ["stripe_key"])


if __name__ == "__main__":
    unittest.main()
