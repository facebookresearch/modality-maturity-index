# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Tests for Gemini main-model request construction."""

from types import SimpleNamespace

import pytest

from mmi.models import EvalPrompt
from mmi.output_assets import OutputAssetManager
from mmi.providers.gemini_provider import GeminiProvider


class _FakeGenerateContentConfig(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _FakeModels:
    def __init__(self):
        self.calls = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                text="A neutral answer.",
                                inline_data=None,
                                file_data=None,
                            )
                        ]
                    )
                )
            ]
        )


class _FakeAio:
    def __init__(self):
        self.models = _FakeModels()


class _FakeClient:
    def __init__(self):
        self.aio = _FakeAio()
        self._api_client = SimpleNamespace(
            _http_options=SimpleNamespace(base_url="http://llama.example")
        )


@pytest.mark.asyncio
async def test_gemini_provider_does_not_send_response_modalities_to_main_model(
    monkeypatch,
):
    monkeypatch.setattr(
        "mmi.providers.gemini_provider.types.GenerateContentConfig",
        _FakeGenerateContentConfig,
    )
    monkeypatch.setattr(
        "mmi.providers.gemini_provider.evaluate_input_support",
        lambda **kwargs: SimpleNamespace(to_dict=lambda: {}),
    )
    monkeypatch.setattr(
        "mmi.providers.gemini_provider.mark_observed_success",
        lambda support: support,
    )

    provider = GeminiProvider.__new__(GeminiProvider)
    provider.provider_name = "gemini"
    provider.model_name = "gemini-test"
    provider.run_name = "gemini-test"
    provider._response_modalities = ["TEXT", "IMAGE", "AUDIO"]
    provider._tools = []
    provider._tool_defs = []
    provider._inline_input_files = True
    provider._tool_loop_limit = 3
    provider._media_tool_backends = {}
    provider.fetch_remote_assets = False
    provider._client = _FakeClient()
    provider._asset_manager = OutputAssetManager(None)

    prompt = EvalPrompt(
        prompt_id="p1",
        prompt_text="Answer naturally.",
        input_modalities=["Text"],
        output_modalities=["Image"],
    )

    response = await provider._send_impl(prompt)

    sent_config = provider._client.aio.models.calls[0]["config"]
    assert "response_modalities" not in sent_config
    assert "response_modalities" not in response.request.provider_request["config"]
    assert response.request.response_modalities == ["TEXT", "IMAGE", "AUDIO"]
