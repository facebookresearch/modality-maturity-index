# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Tests for output asset capture.

Capture separates three things that used to be conflated: how an artifact
arrived, whether we hold its bytes, and what the metric makes of it. Only the
first two live on the asset.
"""

from __future__ import annotations

import base64

import pytest

from mmi.detection import (
    CAPTURED,
    EXTERNAL_URL,
    PROVIDER_INLINE,
    REFERENCE_ONLY,
    SCORING_NATIVE,
    SCORING_URL,
    SKIPPED,
    scoring_class,
)
from mmi.output_assets import OutputAssetManager, decode_base64_data


def test_capture_bytes_writes_output_asset(tmp_path):
    manager = OutputAssetManager(tmp_path).for_prompt("p1")

    asset = manager.capture_bytes(
        data=b"fakepng",
        modality="Image",
        source_type="provider_inline",
        mime_type="image/png",
    )

    assert asset.capture_status == CAPTURED
    assert asset.delivery == PROVIDER_INLINE
    assert asset.modality == "Image"
    assert asset.size_bytes == 7
    assert asset.sha256
    assert asset.local_path
    assert asset.local_path.endswith(".png")
    assert open(asset.local_path, "rb").read() == b"fakepng"
    assert scoring_class(asset) == SCORING_NATIVE


def test_capture_bytes_without_data_is_skipped_not_captured(tmp_path):
    manager = OutputAssetManager(tmp_path).for_prompt("p1")

    asset = manager.capture_bytes(
        data=None,
        modality="Image",
        source_type="provider_inline",
        mime_type="image/png",
    )

    assert asset.capture_status == SKIPPED
    assert not asset.local_path


def test_capture_reference_is_never_captured(tmp_path):
    """A pointer is not a production.

    This used to be recorded as ``captured``, which handed native credit to
    responses that contained no bytes at all.
    """
    manager = OutputAssetManager(tmp_path).for_prompt("p1")

    asset = manager.capture_reference(
        url="https://example.invalid/generated.png",
        modality="Image",
        mime_type="image/png",
    )

    assert asset.capture_status == REFERENCE_ONLY
    assert scoring_class(asset) != SCORING_NATIVE


def test_asset_ids_are_unique_within_a_prompt(tmp_path):
    manager = OutputAssetManager(tmp_path).for_prompt("p1")

    ids = [
        manager.capture_bytes(
            data=b"x", modality="Image", source_type="t", mime_type="image/png"
        ).asset_id
        for _ in range(5)
    ]
    ids.append(
        manager.capture_reference(url="https://a.invalid/b", modality="Image").asset_id
    )

    assert len(ids) == len(set(ids))


def test_decode_base64_data_url():
    payload = base64.b64encode(b"hello").decode()
    assert decode_base64_data(f"data:text/plain;base64,{payload}") == b"hello"


@pytest.mark.asyncio
async def test_capture_urls_does_not_fetch_by_default(tmp_path, monkeypatch):
    """Remote fetching is opt-in. The default run makes no outbound request."""

    async def _boom(*args, **kwargs):
        raise AssertionError("no fetch should happen with fetching disabled")

    monkeypatch.setattr("mmi.output_assets.fetch_url", _boom)

    assets = (
        await OutputAssetManager(tmp_path)
        .for_prompt("p2")
        .capture_urls("watch https://www.youtube.com/watch?v=abc")
    )

    assert len(assets) == 1
    assert assets[0].delivery == EXTERNAL_URL
    assert assets[0].capture_status == REFERENCE_ONLY
    assert scoring_class(assets[0]) == SCORING_URL


@pytest.mark.asyncio
async def test_fetching_a_url_does_not_make_it_native(tmp_path, monkeypatch):
    """Downloading is a convenience; it must never change a score."""
    from mmi.fetch import FetchResult

    async def _fake_fetch(url, **kwargs):
        return FetchResult(
            url=url,
            final_url=url,
            status_code=200,
            mime_type="image/png",
            data=b"pngbytes",
        )

    monkeypatch.setattr("mmi.output_assets.fetch_url", _fake_fetch)

    asset = (
        await OutputAssetManager(tmp_path)
        .for_prompt("p2")
        .capture_url(
            url="https://flickr.com/photos/example/1.png",
            modality="Image",
            enabled=True,
        )
    )

    assert asset.capture_status == CAPTURED
    assert asset.delivery == EXTERNAL_URL
    assert scoring_class(asset) == SCORING_URL


@pytest.mark.asyncio
async def test_capture_urls_ignores_non_https_urls(tmp_path):
    assets = (
        await OutputAssetManager(tmp_path)
        .for_prompt("p3")
        .capture_urls("local file file:///mnt/data/out.wav")
    )

    assert assets == []


@pytest.mark.asyncio
async def test_capture_urls_ignores_unclassifiable_urls(tmp_path):
    assets = (
        await OutputAssetManager(tmp_path)
        .for_prompt("p4")
        .capture_urls("see https://example.invalid/some/page")
    )

    assert assets == []
