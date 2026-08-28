# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Tests for provider behavior when input-file upload fails.

These tests ensure upload failures are surfaced as prompt-level errors and
requests are not sent with silently dropped modalities.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmi.models import EvalPrompt
from mmi.output_assets import OutputAssetManager
from mmi.providers.anthropic_provider import AnthropicProvider
from mmi.providers.gemini_provider import GeminiProvider
from mmi.providers.openai_provider import OpenAIProvider


def _prompt_with_file() -> EvalPrompt:
    return EvalPrompt(
        prompt_id="p-upload",
        prompt_text="describe",
        input_modalities=["Text"],
        output_modalities=["Text"],
        input_files=["input.png"],
    )


@pytest.mark.asyncio
async def test_openai_upload_failure_returns_error_and_skips_request(
    tmp_path, monkeypatch
):
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.provider_name = "openai"
    provider.model_name = "test-model"
    provider.run_name = "test-run"
    provider._api = "responses"
    provider._max_retries = 1
    provider._retry_backoff = 0
    provider._asset_manager = OutputAssetManager(None)
    provider._base_url = ""
    provider.fetch_remote_assets = False
    provider._inline_input_files = False

    provider._client = MagicMock()
    provider._client.responses = MagicMock()
    provider._client.responses.create = AsyncMock()

    provider._upload_file = AsyncMock(return_value=None)
    provider._delete_file = AsyncMock(return_value=None)

    import mmi.providers.openai_provider as openai_mod

    monkeypatch.setattr(openai_mod, "INPUT_FILES_DIR", tmp_path)
    (tmp_path / "input.png").write_bytes(b"fake")

    result = await provider.send(_prompt_with_file())

    assert result.detection.is_error is True
    assert "upload failed" in (result.error or "")
    provider._client.responses.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_gemini_upload_failure_returns_error_and_skips_request(
    tmp_path, monkeypatch
):
    provider = GeminiProvider.__new__(GeminiProvider)
    provider.provider_name = "gemini"
    provider.model_name = "test-model"
    provider.run_name = "test-run"
    provider._max_retries = 1
    provider._retry_backoff = 0
    provider._asset_manager = OutputAssetManager(None)
    provider._base_url = ""
    provider.fetch_remote_assets = False
    provider._inline_input_files = False

    provider._client = MagicMock()
    provider._client.aio = SimpleNamespace(
        models=SimpleNamespace(generate_content=AsyncMock())
    )

    provider._upload_file = AsyncMock(return_value=None)

    import mmi.providers.gemini_provider as gemini_mod

    monkeypatch.setattr(gemini_mod, "INPUT_FILES_DIR", tmp_path)
    (tmp_path / "input.png").write_bytes(b"fake")

    result = await provider.send(_prompt_with_file())

    assert result.detection.is_error is True
    assert "upload failed" in (result.error or "")
    provider._client.aio.models.generate_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_anthropic_upload_failure_returns_error_and_skips_request(
    tmp_path, monkeypatch
):
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.provider_name = "anthropic"
    provider.model_name = "test-model"
    provider.run_name = "test-run"
    provider._max_retries = 1
    provider._retry_backoff = 0
    provider._asset_manager = OutputAssetManager(None)
    provider._base_url = ""
    provider.fetch_remote_assets = False
    provider._inline_input_files = False

    provider._client = MagicMock()
    provider._client.beta = MagicMock()
    provider._client.beta.messages = MagicMock()
    provider._client.beta.messages.create = AsyncMock()

    provider._upload_file = AsyncMock(return_value=None)

    import mmi.providers.anthropic_provider as anthropic_mod

    monkeypatch.setattr(anthropic_mod, "INPUT_FILES_DIR", tmp_path)
    (tmp_path / "input.png").write_bytes(b"fake")

    result = await provider.send(_prompt_with_file())

    assert result.detection.is_error is True
    assert "upload failed" in (result.error or "")
    provider._client.beta.messages.create.assert_not_awaited()
