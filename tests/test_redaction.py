# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Redaction fixtures.

These assert that credentials, signed-URL parameters and local paths *cannot*
be persisted, rather than that a particular known-bad key happens to be caught.
"""

import json

import pytest

from mmi.fetch import redact_url
from mmi.redaction import REDACTED, redact


class TestSensitiveKeys:
    @pytest.mark.parametrize(
        "key",
        [
            "Authorization",
            "authorization",
            "api_key",
            "API-KEY",
            "x-api-key",
            "OPENAI_API_KEY",
            "secret",
            "access_token",
            "Cookie",
            "set-cookie",
            "session_id",
            "private_key",
            "signature",
        ],
    )
    def test_sensitive_keys_are_dropped(self, key):
        assert redact({key: "hunter2"})[key] == REDACTED

    def test_unknown_keys_fail_closed(self):
        """An allowlist, not a blocklist: unknown keys are dropped."""
        assert (
            redact({"some_future_sdk_field": "value"})["some_future_sdk_field"]
            == REDACTED
        )

    def test_allowlisted_keys_survive(self):
        assert redact({"model": "gpt-x", "max_tokens": 32768}) == {
            "model": "gpt-x",
            "max_tokens": 32768,
        }


class TestValues:
    def test_absolute_paths_are_removed(self):
        out = redact({"user_prompt": "see /home/someone/secret/data.csv"})
        assert "/home/someone" not in out["user_prompt"]
        assert REDACTED in out["user_prompt"]

    def test_windows_paths_are_removed(self):
        out = redact({"user_prompt": r"see C:\\Users\\someone\\data.csv"})
        assert "someone" not in out["user_prompt"]

    def test_signed_url_parameters_are_stripped(self):
        signed = (
            "https://example.invalid/a.png?X-Amz-Signature=deadbeef"
            "&X-Amz-Credential=AKIA123&width=64"
        )
        out = redact({"user_prompt": f"asset at {signed}"})["user_prompt"]
        assert "deadbeef" not in out
        assert "AKIA123" not in out
        assert "width=64" in out

    def test_userinfo_is_stripped_from_urls(self):
        assert "hunter2" not in redact_url("https://user:hunter2@example.invalid/a")

    def test_nested_structures_are_redacted(self):
        payload = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "/home/me/x"}]}
            ],
            "headers": {"Authorization": "Bearer abc"},
        }
        out = redact(payload)
        assert out["headers"] == REDACTED
        assert "/home/me" not in json.dumps(out)

    def test_long_strings_are_truncated(self):
        out = redact({"text": "x" * 10000})["text"]
        assert len(out) < 5000


class TestNothingLeaks:
    def test_a_realistic_trace_persists_nothing_sensitive(self):
        trace = {
            "provider": "openai",
            "model": "gpt-x",
            "api_key": "sk-realkeyvalue",
            "headers": {"authorization": "Bearer sk-realkeyvalue"},
            "proxies": {"https": "http://user:pw@proxy.internal:8080"},
            "user_prompt": "read /home/someuser/Documents/secret.txt",
            "messages": [{"role": "user", "content": "hello"}],
        }
        dumped = json.dumps(redact(trace))

        for forbidden in ("sk-realkeyvalue", "pw@proxy", "/home/someuser"):
            assert forbidden not in dumped
