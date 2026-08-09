"""
Nimbus Support Agent — Guardrails
Code-level safety layer that runs *before* and *after* every LLM call.

Two stages
----------
1. input_guardrails(message)  — called before the message reaches the graph.
   Blocks / sanitises dangerous or malformed input.

2. output_guardrails(response) — called on the final LLM reply before it is
   returned to the user.  Catches leaks, runaway outputs, etc.

Each public function returns a GuardrailResult.  If blocked=True, the caller
should return `sanitized_text` directly to the user instead of invoking the LLM.

All violations are appended to logs/guardrail_violations.jsonl and traced as
a named Langfuse span via @observe.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langfuse import observe

# ── Logging ────────────────────────────────────────────────────────────────────
_LOG_FILE = Path(__file__).parent.parent / "logs" / "guardrail_violations.jsonl"
_LOG_FILE.parent.mkdir(exist_ok=True)


def _log_violation(stage: str, rule: str, original: str, sanitized: str) -> None:
    """Append one violation record to logs/guardrail_violations.jsonl."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "rule": rule,
        "original_preview": original[:200] + ("…" if len(original) > 200 else ""),
        "sanitized_preview": sanitized[:200] + ("…" if len(sanitized) > 200 else ""),
    }
    with open(_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class GuardrailResult:
    """
    Return type for both input_guardrails and output_guardrails.

    Attributes
    ----------
    blocked       : True  → caller should return `sanitized_text` immediately
                    False → continue normal processing
    rule_triggered: Name of the guardrail rule that fired (empty if none)
    sanitized_text: The cleaned text to use (may equal the original if no change)
    """
    blocked: bool
    rule_triggered: str
    sanitized_text: str


# ── Configuration ──────────────────────────────────────────────────────────────
_MAX_INPUT_CHARS  = 2_000   # hard cap on user message length
_MAX_OUTPUT_CHARS = 3_000   # hard cap on LLM response length
_MAX_REPEAT_TURNS = 5       # block if the same message is sent this many times

# Tracks last N messages for repetition detection (session-scoped, in-process)
_recent_messages: list[str] = []


# ── Input Guardrail Patterns ───────────────────────────────────────────────────

# Prompt-injection keywords (case-insensitive)
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"you\s+are\s+now\s+(a|an|the)\s+",
        r"act\s+as\s+(a|an|the|if)\s+",
        r"pretend\s+(you\s+are|to\s+be)\s+",
        r"\bjailbreak\b",
        r"\bDAN\b",                          # Do Anything Now jailbreak
        r"new\s+persona\b",
        r"override\s+(your\s+)?(instructions?|rules?|guidelines?)",
        r"forget\s+(your\s+)?(instructions?|rules?|guidelines?|training)",
        r"system\s*prompt\s*:",               # trying to inject a new system prompt
        r"<\s*system\s*>",                    # XML-style injection
        r"\[INST\]",                          # Llama-style injection token
        r"###\s*instruction",                 # Alpaca-style injection
    ]
]

# PII patterns to scrub before reaching the LLM
_PII_RULES: list[tuple[re.Pattern, str]] = [
    # Credit card numbers (Visa, MC, Amex, etc.)
    (re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|"
                r"6(?:011|5[0-9]{2})[0-9]{12})\b"), "[CARD_REDACTED]"),
    # US Social Security Numbers  (XXX-XX-XXXX or XXXXXXXXX)
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b"), "[SSN_REDACTED]"),
    # Passwords in plain text  (password=xxx, pwd: xxx, etc.)
    (re.compile(r"(?i)(password|passwd|pwd)\s*[=:]\s*\S+"), "[PASSWORD_REDACTED]"),
    # API key / secret patterns
    (re.compile(r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[=:]\s*\S+"),
     "[SECRET_REDACTED]"),
]


# ── Output Guardrail Patterns ──────────────────────────────────────────────────

# Patterns that should never appear in the agent's output
_OUTPUT_LEAK_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Raw JSON blobs with keys that look like database records leaking through
    (re.compile(r'\{["\'](?:password|hashed_password|secret|token|api_key)["\']'),
     "internal_data_leak"),
    # API / secret key patterns in output
    (re.compile(r"(?:sk-|pk-lf-|lsv2_)[A-Za-z0-9_\-]{10,}"),
     "secret_key_in_output"),
    # OpenRouter key pattern
    (re.compile(r"sk-or-v1-[A-Za-z0-9]{30,}"),
     "api_key_in_output"),
]

# Competitor brand names — agent should not mention these
_COMPETITOR_NAMES: list[str] = [
    "Amazon", "Wayfair", "IKEA", "Overstock", "Target", "Walmart", "Costco",
    "HomeDepot", "Home Depot", "Bed Bath", "West Elm", "Pottery Barn",
]
_COMPETITOR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in _COMPETITOR_NAMES) + r")\b",
    re.IGNORECASE,
)

_COMPETITOR_REPLACEMENT = (
    "\n\n---\n*Note: I can only assist with Nimbus-related questions.*"
)


# ── Input Guardrails ───────────────────────────────────────────────────────────
@observe(name="guardrails.input")
def input_guardrails(message: str) -> GuardrailResult:
    """
    Run all input-side guardrail rules on a raw user message.

    Rules (in order of priority)
    -----------------------------
    1. Empty message
    2. Message too long
    3. Repetition flood (same message repeated many times)
    4. Prompt-injection attempt
    5. PII scrubbing (non-blocking — sanitises and continues)

    Returns a GuardrailResult.  If blocked=True, return sanitized_text
    to the user immediately without calling the LLM.
    """

    # ── Rule 1: Empty ──────────────────────────────────────────────────────────
    stripped = message.strip()
    if not stripped:
        return GuardrailResult(
            blocked=True,
            rule_triggered="empty_message",
            sanitized_text="Please type a message before sending.",
        )

    # ── Rule 2: Length ────────────────────────────────────────────────────────
    if len(stripped) > _MAX_INPUT_CHARS:
        _log_violation("input", "message_too_long", stripped, "")
        return GuardrailResult(
            blocked=True,
            rule_triggered="message_too_long",
            sanitized_text=(
                f"Your message is too long ({len(stripped)} characters). "
                f"Please keep messages under {_MAX_INPUT_CHARS} characters and try again."
            ),
        )

    # ── Rule 3: Repetition flood ───────────────────────────────────────────────
    _recent_messages.append(stripped.lower())
    if len(_recent_messages) > _MAX_REPEAT_TURNS:
        _recent_messages.pop(0)
    if len(_recent_messages) >= _MAX_REPEAT_TURNS and len(set(_recent_messages)) == 1:
        _log_violation("input", "repetition_flood", stripped, "")
        return GuardrailResult(
            blocked=True,
            rule_triggered="repetition_flood",
            sanitized_text=(
                "It looks like you've sent the same message several times. "
                "If you're experiencing an issue, please contact our support team at "
                "support@nimbus.com or call 1-800-NIMBUS-1."
            ),
        )

    # ── Rule 4: Prompt injection ───────────────────────────────────────────────
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(stripped):
            _log_violation("input", "prompt_injection", stripped, "")
            return GuardrailResult(
                blocked=True,
                rule_triggered="prompt_injection",
                sanitized_text=(
                    "I'm sorry, I'm not able to process that request. "
                    "I'm here to help with Nimbus product and account questions. "
                    "How can I assist you today?"
                ),
            )

    # ── Rule 5: PII scrubbing (sanitise, do not block) ────────────────────────
    sanitized = stripped
    pii_found: list[str] = []
    for pattern, replacement in _PII_RULES:
        new_text, n = pattern.subn(replacement, sanitized)
        if n > 0:
            pii_found.append(replacement)
            sanitized = new_text

    if pii_found:
        _log_violation("input", "pii_scrubbed", stripped, sanitized)
        # Continue — but with the scrubbed text

    return GuardrailResult(
        blocked=False,
        rule_triggered="pii_scrubbed" if pii_found else "",
        sanitized_text=sanitized,
    )


# ── Output Guardrails ──────────────────────────────────────────────────────────
@observe(name="guardrails.output")
def output_guardrails(response: str) -> GuardrailResult:
    """
    Run all output-side guardrail rules on the LLM's raw response.

    Rules (in order of priority)
    -----------------------------
    1. Internal data / secret key leak → block entirely
    2. Competitor mentions → append scope note (non-blocking)
    3. Response too long → truncate gracefully

    Returns a GuardrailResult.  If blocked=True, return sanitized_text
    instead of the original response.
    """

    # ── Rule 1: Leak / secret detection ───────────────────────────────────────
    for pattern, rule_name in _OUTPUT_LEAK_PATTERNS:
        if pattern.search(response):
            _log_violation("output", rule_name, response, "")
            return GuardrailResult(
                blocked=True,
                rule_triggered=rule_name,
                sanitized_text=(
                    "I encountered an issue preparing your response. "
                    "Please contact our support team at support@nimbus.com "
                    "or call 1-800-NIMBUS-1 for assistance."
                ),
            )

    sanitized = response

    # ── Rule 2: Competitor mentions ────────────────────────────────────────────
    if _COMPETITOR_PATTERN.search(sanitized):
        _log_violation("output", "competitor_mention", sanitized, sanitized)
        # Strip competitor names and append a scope note
        sanitized = _COMPETITOR_PATTERN.sub("[another retailer]", sanitized)
        if _COMPETITOR_REPLACEMENT not in sanitized:
            sanitized += _COMPETITOR_REPLACEMENT

    # ── Rule 3: Length truncation ──────────────────────────────────────────────
    if len(sanitized) > _MAX_OUTPUT_CHARS:
        _log_violation("output", "response_too_long", sanitized, "")
        sanitized = (
            sanitized[:_MAX_OUTPUT_CHARS].rsplit(" ", 1)[0]
            + "\n\n… *(Response truncated — for full details, contact support@nimbus.com)*"
        )

    return GuardrailResult(
        blocked=False,
        rule_triggered="",
        sanitized_text=sanitized,
    )


# ── Convenience helper ─────────────────────────────────────────────────────────
def reset_repetition_tracker() -> None:
    """Clear the in-memory repetition buffer (call on /reset)."""
    _recent_messages.clear()
