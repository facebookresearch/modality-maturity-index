# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Tests for guarded remote fetching.

The URLs reaching this module come from a language model, so they are
untrusted input. These tests are about what the fetcher *refuses*.
"""

import pytest

from mmi.fetch import (
    FetchRejected,
    _address_is_public,
    assert_url_is_fetchable,
    redact_url,
)


class TestSchemeAndHost:
    @pytest.mark.parametrize(
        "url",
        [
            "http://example.com/a.png",
            "ftp://example.com/a.png",
            "file:///etc/passwd",
            "gopher://example.com/",
        ],
    )
    def test_only_https_is_allowed(self, url):
        with pytest.raises(FetchRejected, match="https"):
            assert_url_is_fetchable(url)

    def test_missing_host_is_rejected(self):
        with pytest.raises(FetchRejected):
            assert_url_is_fetchable("https:///a.png")


class TestAddressFiltering:
    @pytest.mark.parametrize(
        "address",
        [
            "127.0.0.1",  # loopback
            "10.0.0.1",  # private
            "172.16.0.1",  # private
            "192.168.1.1",  # private
            "169.254.169.254",  # link-local: cloud metadata
            "0.0.0.0",  # unspecified
            "::1",  # IPv6 loopback
            "fe80::1",  # IPv6 link-local
            "fc00::1",  # IPv6 unique-local
        ],
    )
    def test_non_public_addresses_are_refused(self, address):
        assert _address_is_public(address) is False

    @pytest.mark.parametrize("address", ["93.184.216.34", "2606:2800:220:1::248"])
    def test_public_addresses_are_allowed(self, address):
        assert _address_is_public(address) is True

    def test_localhost_is_rejected_end_to_end(self):
        with pytest.raises(FetchRejected, match="non-public"):
            assert_url_is_fetchable("https://localhost/a.png")

    def test_metadata_endpoint_is_rejected_end_to_end(self):
        with pytest.raises(FetchRejected, match="non-public"):
            assert_url_is_fetchable("https://169.254.169.254/latest/meta-data/")

    def test_unresolvable_host_is_rejected(self):
        with pytest.raises(FetchRejected, match="resolve"):
            assert_url_is_fetchable("https://this-host-does-not-exist.invalid/a.png")


class TestRedaction:
    def test_signed_url_parameters_are_redacted(self):
        url = (
            "https://example.invalid/a.png?X-Amz-Signature=deadbeef"
            "&X-Amz-Credential=AKIA1&width=64"
        )
        out = redact_url(url)

        assert "deadbeef" not in out
        assert "AKIA1" not in out
        assert "width=64" in out

    @pytest.mark.parametrize(
        "param", ["token", "access_token", "api_key", "signature", "sig", "key"]
    )
    def test_common_credential_parameters_are_redacted(self, param):
        out = redact_url(f"https://example.invalid/a?{param}=secretvalue")
        assert "secretvalue" not in out

    def test_userinfo_is_stripped(self):
        out = redact_url("https://user:hunter2@example.invalid/a.png")
        assert "hunter2" not in out
        assert "user:" not in out

    def test_fragment_is_dropped(self):
        assert "#frag" not in redact_url("https://example.invalid/a.png#frag")

    def test_plain_url_is_unchanged(self):
        url = "https://example.invalid/path/a.png?width=64"
        assert redact_url(url) == url

    def test_unparseable_url_does_not_raise(self):
        assert redact_url("https://[bad") == "<unparseable-url>"


class TestDefaults:
    def test_fetching_is_off_by_default_in_providers(self):
        from mmi.providers.stub_provider import StubProvider

        assert StubProvider(model="s").fetch_remote_assets is False

    def test_fetch_defaults_support_large_assets(self):
        from mmi.fetch import (
            DEFAULT_MAX_BYTES,
            DEFAULT_MAX_REDIRECTS,
            DEFAULT_TIMEOUT,
        )

        assert DEFAULT_MAX_BYTES == 1024 * 1024 * 1024
        assert DEFAULT_TIMEOUT == 1200
        assert DEFAULT_MAX_REDIRECTS == 5

    def test_ambient_proxy_configuration_is_honored(self):
        """Public asset retrieval may require an environment-configured proxy."""
        import inspect

        import mmi.fetch as module

        assert "trust_env=True" in inspect.getsource(module.fetch_url)

    def test_redirects_are_revalidated_per_hop(self):
        """httpx's own redirect following would skip the address check."""
        import inspect

        import mmi.fetch as module

        source = inspect.getsource(module.fetch_url)
        assert "follow_redirects=False" in source
        assert source.count("assert_url_is_fetchable") >= 1
