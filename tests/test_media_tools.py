# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Tests for neutral MMI media tool definitions and dispatch."""

import pytest

from mmi.config import MediaToolBackend
from mmi.media_tools import (
    AUDIO_GEN,
    IMAGE_GEN,
    VIDEO_GEN,
    build_anthropic_tools,
    build_gemini_tools,
    build_openai_tools,
    dispatch_media_tool,
    validate_neutral_tools,
)
from mmi.output_assets import PromptAssetManager


def test_schema_builders_use_same_neutral_tool_names():
    tools = [IMAGE_GEN, AUDIO_GEN, VIDEO_GEN]

    openai_names = [tool["name"] for tool in build_openai_tools(tools)]
    anthropic_names = [tool["name"] for tool in build_anthropic_tools(tools)]
    gemini_tool = build_gemini_tools(tools)[0]
    gemini_names = [decl.name for decl in gemini_tool.function_declarations]

    assert openai_names == tools
    assert anthropic_names == tools
    assert gemini_names == tools


def test_schema_descriptions_do_not_expose_backend_or_modality_names():
    forbidden = {
        "audio",
        "gemini",
        "google",
        "image",
        "tts",
        "veo",
        "video",
    }
    schemas = build_openai_tools([IMAGE_GEN, AUDIO_GEN, VIDEO_GEN])

    descriptions = " ".join(schema["description"].lower() for schema in schemas)

    assert forbidden.isdisjoint(descriptions.split())


def test_tool_descriptions_are_byte_identical():
    """experiments.tex:66 — the only signal about a tool's modality is its name."""
    schemas = build_openai_tools([IMAGE_GEN, AUDIO_GEN, VIDEO_GEN])
    descriptions = {schema["description"] for schema in schemas}

    assert len(descriptions) == 1


def test_rejects_provider_native_tools():
    with pytest.raises(ValueError):
        validate_neutral_tools(["image_gen", "code_interpreter"])


@pytest.mark.asyncio
async def test_dispatch_returns_captured_asset(monkeypatch):
    async def fake_generate(prompt, prompt_asset_manager, backend):
        return [
            prompt_asset_manager.capture_bytes(
                data=b"fake-png",
                modality="Image",
                source_type="tool_backend",
                mime_type="image/png",
            )
        ]

    monkeypatch.setattr("mmi.media_tools._generate_image", fake_generate)
    manager = PromptAssetManager(None, "p1")

    backends = {IMAGE_GEN: MediaToolBackend(provider="google", model="test-model")}
    result = await dispatch_media_tool(
        IMAGE_GEN, {"prompt": "draw a fox"}, manager, backends
    )

    assert result.assets[0].asset_id == "p1_output_0"
    assert result.assets[0].capture_status == "captured"
    assert result.assets[0].modality == "Image"
    assert result.as_model_payload() == (
        '{"status": "completed", "message": "Generated 1 artifact(s).", '
        '"asset_ids": ["p1_output_0"]}'
    )
