"""Central safety projections for command output and persisted facts."""

from __future__ import annotations

import os
import re
from typing import Any


_CREDENTIAL_PATTERN = re.compile(
    r"\$(?:krb5asrep|krb5tgs)\$[^\s]+|"
    r"(password|passwd|pass|secret|plaintext|ntlmhash|token|hash)\s*[:=]\s*[^\s,;]+",
    flags=re.IGNORECASE,
)
_COMMAND_SECRET_PATTERN = re.compile(
    r"(?i)(--?(?:password|pass|secret)|-[pPwW]|/password)\s+[^\s]+"
)


def redact_output(value: Any, limit: int = 120_000) -> str:
    """Return a bounded, terminal/model-safe projection of command output.

    This is intentionally applied at the tool boundary, before evidence,
    events, durable observations, reports, or prompts can receive the data.
    Credential material remains useful as a category/status, never as a value.
    """

    text = str(value or "")
    secrets = {
        secret for name in (
            "CYBERQA_AD_PASSWORD", "AD_PASSWORD", "CYBERQA_AD_TOKEN", "AD_TOKEN",
        ) if (secret := os.getenv(name))
    }
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = _CREDENTIAL_PATTERN.sub(
        lambda match: (
            "[credential material redacted]"
            if match.group(0).startswith("$")
            else f"{match.group(1)}=[REDACTED]"
        ),
        text,
    )
    text = _COMMAND_SECRET_PATTERN.sub(
        lambda match: f"{match.group(0).split()[0]} [REDACTED]", text
    )
    if len(text) > limit:
        return text[:limit] + "\n[command output truncated]"
    return text


def redact_facts(value: Any, key: str = "") -> Any:
    """Recursively redact sensitive fact values while preserving safe metadata."""

    normalized = key.lower()
    sensitive_keys = {
        "password", "passwd", "pass", "secret", "plaintext", "token", "hash",
        "hash_value", "password_hash", "ntlmhash", "ticket", "ticket_material",
        "credential_material",
    }
    if normalized in sensitive_keys:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(item_key): redact_facts(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        if normalized in {"argv", "command", "command_argv"}:
            result: list[Any] = []
            redact_next = False
            secret_flags = {"-p", "-P", "-w", "-W", "--password", "--pass", "--secret", "/password"}
            for item in value[:500]:
                token = str(item)
                if redact_next:
                    result.append("[REDACTED]")
                    redact_next = False
                elif token in secret_flags:
                    result.append(token)
                    redact_next = True
                else:
                    result.append(redact_output(item, limit=16_000))
            return result
        return [redact_facts(item, key) for item in value[:500]]
    if isinstance(value, tuple):
        return [redact_facts(item, key) for item in value[:500]]
    if isinstance(value, str):
        return redact_output(value, limit=16_000)
    return value
