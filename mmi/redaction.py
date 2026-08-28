# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Redaction applied before any trace is serialized.

Traces are the most useful thing the harness persists and the most dangerous.
They contain whatever the provider SDK put in the request and response, which
routinely includes authorization headers, signed-URL credentials, and the
absolute paths of the machine that produced them.

This module is an **allowlist**, not a blocklist. A field is dropped unless it
is known to be safe, because the set of keys a future SDK might introduce is
open-ended and a blocklist silently fails open on every one of them.
"""

from __future__ import annotations

import re
from typing import Any

from .fetch import redact_url

REDACTED = "[REDACTED]"

#: Keys that may appear verbatim in a persisted trace.
SAFE_REQUEST_KEYS = frozenset(
    {
        "api",
        "api_version",
        "aspect_ratio",
        "betas",
        "config",
        "content",
        "contents",
        "duration_seconds",
        "filename",
        "input_files",
        "input_modalities",
        "input_support",
        "max_output_tokens",
        "max_tokens",
        "messages",
        "mime_type",
        "modalities",
        "model",
        "output_modalities",
        "provider",
        "response_modalities",
        "role",
        "source_type",
        "status",
        "system_prompt",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_p",
        "type",
        "user_prompt",
    }
)

#: Keys whose values are dropped outright wherever they appear, even if they
#: are also allowlisted. This is defence in depth against a dangerous key being
#: added to the allowlist by mistake later.
SENSITIVE_KEY_PATTERN = re.compile(
    r"(authorization|api[-_]?key|x[-_]api[-_]key|secret|token|password|passwd"
    r"|credential|cookie|set[-_]cookie|session|bearer|signature|private)",
    re.IGNORECASE,
)

#: Keys that trip :data:`SENSITIVE_KEY_PATTERN` but are generation budgets, not
#: credentials. Kept explicit so the exemption is auditable.
SENSITIVE_PATTERN_EXEMPTIONS = frozenset(
    {"max_tokens", "max_output_tokens", "token_count", "total_tokens"}
)

#: Absolute filesystem paths. Persisting these leaks usernames and layout.
_ABSOLUTE_PATH = re.compile(
    r"(?:/(?:home|Users|root|var|tmp|mnt|data)/[^\s\"']*|[A-Za-z]:\\\\[^\s\"']*)"
)

_URL = re.compile(r"https?://[^\s\"'<>]+")

_MAX_STRING = 4000


def _redact_text(value: str) -> str:
    value = _URL.sub(lambda m: redact_url(m.group(0)), value)
    value = _ABSOLUTE_PATH.sub(REDACTED, value)
    if len(value) > _MAX_STRING:
        value = value[:_MAX_STRING] + "…[truncated]"
    return value


def redact(value: Any, *, allowlist: frozenset[str] | None = None) -> Any:
    """Recursively redact a structure for persistence.

    Args:
        value: Any JSON-shaped structure.
        allowlist: Keys permitted at mapping level. Defaults to
            :data:`SAFE_REQUEST_KEYS`. Pass ``frozenset()`` with care — an empty
            allowlist drops every mapping key.
    """
    keys = SAFE_REQUEST_KEYS if allowlist is None else allowlist

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if (
                name.lower() not in SENSITIVE_PATTERN_EXEMPTIONS
                and SENSITIVE_KEY_PATTERN.search(name)
            ):
                out[name] = REDACTED
                continue
            if name not in keys:
                out[name] = REDACTED
                continue
            out[name] = redact(item, allowlist=keys)
        return out

    if isinstance(value, (list, tuple)):
        return [redact(item, allowlist=keys) for item in value]

    if isinstance(value, str):
        return _redact_text(value)

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    return _redact_text(str(value))


def redact_request_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Redact a :class:`~mmi.models.RequestRecord` rendered as a dict."""
    if record is None:
        return None
    return redact(record)
