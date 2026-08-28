# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Retry classification.

Model IDs get retired constantly, so a permanent 404 is the most likely failure
a user will hit. Retrying it is not a cosmetic problem: at fifteen attempts with
a thirty-second cap, across 893 prompts, it turns a clear error into a hang.
"""

import json

import pytest

from mmi.retry import is_permanent, is_retryable


class TestPermanentFailures:
    @pytest.mark.parametrize(
        "message",
        [
            "404 NOT_FOUND",
            "Error code: 404 - {'type': 'not_found_error', 'message': 'model: x'}",
            "This model models/gemini-2.0-flash is no longer available.",
            "401 unauthorized",
            "403 permission_denied",
            "invalid_request_error",
            "INVALID_ARGUMENT",
            "unsupported mime type",
        ],
    )
    def test_never_retried(self, message):
        assert is_retryable(Exception(message)) is False
        assert is_permanent(Exception(message)) is True


class TestTransientFailures:
    @pytest.mark.parametrize(
        "message",
        [
            "429 Too Many Requests",
            "rate limit exceeded",
            "rate_limit_error",
            "500 Internal Server Error",
            "503 Service Unavailable",
            "529",
            "{'type': 'overloaded_error'}",
            "Server temporarily unable to service your request",
            "Request timed out",
            "timeout_error",
            "RESOURCE_EXHAUSTED",
        ],
    )
    def test_retried(self, message):
        assert is_retryable(Exception(message)) is True


class TestTheMigrateBug:
    """The specific false positive that motivated this module.

    A permanent 404 was retried fifteen times because the provider's own
    remediation link contained "mig*rate*-to-interactions", and the transient
    list matched the bare substring "rate".
    """

    def test_migrate_in_a_404_is_not_a_rate_limit(self):
        message = (
            "404 NOT_FOUND. {'error': {'code': 404, 'message': 'This model "
            "models/gemini-2.0-flash is no longer available. We recommend you "
            "to use the Interactions API "
            "(https://ai.google.dev/gemini-api/docs/migrate-to-interactions).'}}"
        )
        assert is_retryable(Exception(message)) is False

    @pytest.mark.parametrize(
        "message",
        [
            "please generate a separate response",
            "the accurate answer is 42",
            "corporate policy forbids this",
            "could not enumerate models",
        ],
    )
    def test_english_words_containing_markers_do_not_trigger_retries(self, message):
        assert is_retryable(Exception(message)) is False


class TestPrecedence:
    def test_permanent_wins_over_transient(self):
        """A 404 mentioning a rate limit is still permanent."""
        assert (
            is_retryable(Exception("404 not_found (you also hit a rate limit)"))
            is False
        )


class TestBadJson:
    def test_judges_may_retry_unparseable_output(self):
        exc = json.JSONDecodeError("bad", "{", 0)
        assert is_retryable(exc, retry_on_bad_json=True) is True

    def test_providers_do_not(self):
        exc = json.JSONDecodeError("bad", "{", 0)
        assert is_retryable(exc) is False


class TestSharedByEveryCaller:
    def test_no_module_keeps_its_own_copy(self):
        """Three divergent copies is how the bug survived."""
        import pathlib

        import mmi

        root = pathlib.Path(mmi.__file__).parent
        offenders = [
            p.relative_to(root).as_posix()
            for p in root.rglob("*.py")
            if p.name != "retry.py" and "_RETRYABLE_KEYWORDS" in p.read_text()
        ]
        assert not offenders, f"retry logic duplicated in: {offenders}"
