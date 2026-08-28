# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Deciding whether a provider failure is worth retrying.

This lived in three places as three copies of a substring match, which is how
a permanent ``404 model not available`` came to be retried fifteen times: the
error text contained the word "mig**rate**", and the list contained ``"rate"``.

Model IDs get retired constantly, so a 404 is the single most likely failure a
user of this harness will hit. Retrying it for minutes per prompt, across 893
prompts, turns a clear error into an apparent hang.

Two rules:

1. **Permanent failures are never retried**, and they are checked first.
2. Transient markers are matched on **word boundaries**, so no substring of an
   unrelated English word can trigger a retry.
"""

from __future__ import annotations

import json
import re


def _word(term: str) -> str:
    """Match *term* as a whole word, tolerating ``_``-joined identifiers."""
    return rf"(?<![a-z]){term}(?![a-z])"


#: If any of these appear, the call will never succeed on retry. Checked first.
PERMANENT_MARKERS = (
    r"\b404\b",
    r"\b400\b",
    r"\b401\b",
    r"\b403\b",
    r"not_found",
    r"not found",
    r"invalid_request",
    r"invalid_argument",
    r"unauthorized",
    r"permission_denied",
    r"authentication",
    r"no longer available",
    r"does not exist",
    r"unsupported",
)

#: Transient conditions worth waiting out.
#:
#: Word markers use letter-boundaries rather than ``\b`` because providers wrap
#: them in identifiers: Anthropic returns ``overloaded_error``, and ``\b`` does
#: not match before an underscore.
TRANSIENT_MARKERS = (
    r"\b429\b",
    r"\b500\b",
    r"\b502\b",
    r"\b503\b",
    r"\b504\b",
    r"\b529\b",
    r"rate[ _-]?limit",
    _word("overloaded"),
    _word("capacity"),
    _word("temporarily"),
    _word("unavailable"),
    _word("timeout"),
    r"timed out",
    r"resource[_ ]exhausted",
    r"no[_ ]host",
)

_PERMANENT = re.compile("|".join(PERMANENT_MARKERS), re.IGNORECASE)
_TRANSIENT = re.compile("|".join(TRANSIENT_MARKERS), re.IGNORECASE)


def is_permanent(exc: Exception) -> bool:
    """Whether the failure will recur identically on every retry."""
    return bool(_PERMANENT.search(str(exc)))


def is_retryable(exc: Exception, *, retry_on_bad_json: bool = False) -> bool:
    """Whether to retry after *exc*.

    Args:
        exc: The provider exception.
        retry_on_bad_json: Treat a malformed JSON reply as transient. Judges
            occasionally return unparseable output and succeed on a retry;
            providers should leave this off.
    """
    if retry_on_bad_json and isinstance(exc, json.JSONDecodeError):
        return True
    message = str(exc)
    if _PERMANENT.search(message):
        return False
    return bool(_TRANSIENT.search(message))
