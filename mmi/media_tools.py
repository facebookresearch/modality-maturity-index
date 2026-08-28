# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Neutral media generation tools for MMI tool-calling evals."""

from __future__ import annotations

import asyncio
import io
import json
import os
import time
import wave
from dataclasses import dataclass
from typing import Any

import httpx
from google.genai import Client, types

from .config import MediaToolBackend
from .models import CapturedAsset, ToolCallRecord
from .output_assets import PromptAssetManager, decode_base64_data, modality_from_mime

IMAGE_GEN = "image_gen"
AUDIO_GEN = "audio_gen"
VIDEO_GEN = "video_gen"
NEUTRAL_MEDIA_TOOLS = (IMAGE_GEN, AUDIO_GEN, VIDEO_GEN)

# All three descriptions are byte-identical and deliberately uninformative, so
# that the only signal about a tool's modality is its name (experiments.tex:66).
# Changing any of these strings changes the method. Do not "improve" them.
_TOOL_DESCRIPTIONS = {
    IMAGE_GEN: "Create an external artifact from a text prompt.",
    AUDIO_GEN: "Create an external artifact from a text prompt.",
    VIDEO_GEN: "Create an external artifact from a text prompt.",
}
_TOOL_MIME_DEFAULTS = {
    IMAGE_GEN: "image/png",
    AUDIO_GEN: "audio/wav",
    VIDEO_GEN: "video/mp4",
}


@dataclass(frozen=True)
class ToolExecutionResult:
    """Normalized result returned to a driver model after a tool call."""

    message: str
    assets: list[CapturedAsset]

    def as_model_payload(self) -> str:
        return json.dumps(
            {
                "status": "completed",
                "message": self.message,
                "asset_ids": [asset.asset_id for asset in self.assets],
            },
            ensure_ascii=False,
        )


def validate_neutral_tools(tools: list[str] | None) -> list[str]:
    """Return only supported neutral tool names, rejecting native/provider tools."""

    selected = tools or []
    invalid = [tool for tool in selected if tool not in NEUTRAL_MEDIA_TOOLS]
    if invalid:
        raise ValueError(
            "MMI uniform media-tool configs only support neutral tools "
            f"{list(NEUTRAL_MEDIA_TOOLS)}; got unsupported tools {invalid}"
        )
    return list(selected)


def _parameters_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Text prompt describing the media to generate.",
            }
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }


def build_openai_tools(tools: list[str] | None) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool,
            "description": _TOOL_DESCRIPTIONS[tool],
            "parameters": _parameters_schema(),
        }
        for tool in validate_neutral_tools(tools)
    ]


def build_anthropic_tools(tools: list[str] | None) -> list[dict[str, Any]]:
    return [
        {
            "name": tool,
            "description": _TOOL_DESCRIPTIONS[tool],
            "input_schema": _parameters_schema(),
        }
        for tool in validate_neutral_tools(tools)
    ]


def build_gemini_tools(tools: list[str] | None) -> list[Any]:
    declarations = [
        types.FunctionDeclaration(
            name=tool,
            description=_TOOL_DESCRIPTIONS[tool],
            parameters=_parameters_schema(),
        )
        for tool in validate_neutral_tools(tools)
    ]
    return [types.Tool(function_declarations=declarations)] if declarations else []


def parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"prompt": arguments}
        return parsed if isinstance(parsed, dict) else {}
    if isinstance(arguments, dict):
        return arguments
    if hasattr(arguments, "model_dump"):
        dumped = arguments.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return dict(arguments) if hasattr(arguments, "items") else {}


async def dispatch_media_tool(
    tool_name: str,
    arguments: dict[str, Any],
    prompt_asset_manager: PromptAssetManager,
    backends: dict[str, MediaToolBackend],
) -> ToolExecutionResult:
    """Execute a neutral media tool through its configured backend."""

    if tool_name not in NEUTRAL_MEDIA_TOOLS:
        raise ValueError(f"Unsupported neutral media tool: {tool_name}")
    prompt = str(arguments.get("prompt") or "").strip()
    if not prompt:
        raise ValueError(f"{tool_name} requires a non-empty prompt argument")

    backend = backends.get(tool_name)
    if backend is None:
        raise ValueError(
            f"No backend configured for {tool_name}. Add a [media_tools.{tool_name}] "
            "block with 'provider' and 'model' to the config."
        )

    if tool_name == IMAGE_GEN:
        assets = await _generate_image(prompt, prompt_asset_manager, backend)
    elif tool_name == AUDIO_GEN:
        assets = await _generate_audio(prompt, prompt_asset_manager, backend)
    else:
        assets = await _generate_video(prompt, prompt_asset_manager, backend)

    return ToolExecutionResult(
        message=f"Generated {len(assets)} artifact(s).",
        assets=assets,
    )


def make_tool_call_record(
    *,
    tool_name: str,
    provider_call_id: str,
    arguments: dict[str, Any],
    status: str = "pending",
    error: str = "",
    assets: list[CapturedAsset] | None = None,
) -> ToolCallRecord:
    return ToolCallRecord(
        tool_name=tool_name,
        provider_call_id=provider_call_id,
        arguments=arguments,
        status=status,
        error=error,
        produced_asset_ids=[asset.asset_id for asset in assets or []],
    )


def _pcm_to_wav_bytes(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)
    return buffer.getvalue()


def _backend_client(backend: MediaToolBackend) -> Client:
    """Build a client for a configured media-tool backend.

    Only the ``google`` backend provider is currently implemented. A non-empty
    ``base_url`` is opt-in custom routing; no vendor-specific headers are ever
    injected on the caller's behalf.
    """
    if backend.provider != "google":
        raise ValueError(
            f"Unsupported media-tool backend provider {backend.provider!r}. "
            "Only 'google' is implemented."
        )
    api_key = os.environ.get(backend.api_key_env or "GOOGLE_API_KEY", "")
    if not api_key:
        raise OSError(
            f"{backend.api_key_env or 'GOOGLE_API_KEY'} environment variable is not set"
        )
    http_opts: dict[str, Any] = {
        "httpx_client": httpx.Client(trust_env=True, follow_redirects=True)
    }
    if backend.base_url:
        http_opts["base_url"] = backend.base_url
        http_opts["api_version"] = "v1"
    return Client(api_key=api_key, http_options=types.HttpOptions(**http_opts))


async def _generate_image(
    prompt: str,
    prompt_asset_manager: PromptAssetManager,
    backend: MediaToolBackend,
) -> list[CapturedAsset]:
    def _call():
        client = _backend_client(backend)
        return client.models.generate_content(
            model=backend.model,
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )

    response = await asyncio.to_thread(_call)
    return _capture_genai_response_assets(
        response=response,
        prompt_asset_manager=prompt_asset_manager,
        fallback_tool=IMAGE_GEN,
    )


async def _generate_audio(
    prompt: str,
    prompt_asset_manager: PromptAssetManager,
    backend: MediaToolBackend,
) -> list[CapturedAsset]:
    def _call():
        client = _backend_client(backend)
        return client.models.generate_content(
            model=backend.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Kore"
                        )
                    )
                ),
            ),
        )

    response = await asyncio.to_thread(_call)
    assets = _capture_genai_response_assets(
        response=response,
        prompt_asset_manager=prompt_asset_manager,
        fallback_tool=AUDIO_GEN,
    )
    return assets


async def _generate_video(
    prompt: str,
    prompt_asset_manager: PromptAssetManager,
    backend: MediaToolBackend,
) -> list[CapturedAsset]:
    def _call():
        client = _backend_client(backend)
        op = client.models.generate_videos(
            model=backend.model,
            prompt=prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9",
                duration_seconds=4,
                number_of_videos=1,
                generate_audio=True,
                person_generation="allow_adult",
            ),
        )
        while not op.done:
            time.sleep(10)
            op = client.operations.get(op)
        if op.error:
            raise RuntimeError(f"Video generation failed: {op.error}")
        return getattr(op, "result", None)

    response = await asyncio.to_thread(_call)
    assets: list[CapturedAsset] = []
    operations = getattr(response, "generated_videos", None) or getattr(
        response, "videos", []
    )
    for video in operations or []:
        video_obj = getattr(video, "video", video)
        raw_video_bytes = getattr(video_obj, "video_bytes", None)
        data = (
            raw_video_bytes
            if isinstance(raw_video_bytes, bytes)
            else decode_base64_data(raw_video_bytes)
        )
        mime = getattr(video_obj, "mime_type", "") or _TOOL_MIME_DEFAULTS[VIDEO_GEN]
        if data:
            assets.append(
                prompt_asset_manager.capture_bytes(
                    data=data,
                    modality="Video",
                    source_type="tool_backend",
                    mime_type=mime,
                    metadata={"tool_name": VIDEO_GEN},
                )
            )
            continue
        uri = getattr(video_obj, "uri", "") or getattr(video_obj, "file_uri", "")
        if uri:
            assets.append(
                prompt_asset_manager.capture_reference(
                    url=uri,
                    modality="Video",
                    mime_type=mime,
                    metadata={"tool_name": VIDEO_GEN},
                )
            )
    return assets


def _capture_genai_response_assets(
    *,
    response: Any,
    prompt_asset_manager: PromptAssetManager,
    fallback_tool: str,
) -> list[CapturedAsset]:
    assets: list[CapturedAsset] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            inline_data = getattr(part, "inline_data", None)
            if inline_data:
                mime = (
                    getattr(inline_data, "mime_type", "")
                    or _TOOL_MIME_DEFAULTS[fallback_tool]
                )
                raw_data = getattr(inline_data, "data", None)
                data = (
                    raw_data
                    if isinstance(raw_data, bytes)
                    else decode_base64_data(raw_data)
                )
                if fallback_tool == AUDIO_GEN and data and mime.startswith("audio/L16"):
                    data = _pcm_to_wav_bytes(data)
                    mime = "audio/wav"
                assets.append(
                    prompt_asset_manager.capture_bytes(
                        data=data,
                        modality=modality_from_mime(mime),
                        source_type="tool_backend",
                        mime_type=mime,
                        metadata={"tool_name": fallback_tool},
                    )
                )
            file_data = getattr(part, "file_data", None)
            if file_data:
                mime = (
                    getattr(file_data, "mime_type", "")
                    or _TOOL_MIME_DEFAULTS[fallback_tool]
                )
                uri = getattr(file_data, "file_uri", "") or ""
                assets.append(
                    prompt_asset_manager.capture_reference(
                        url=uri,
                        modality=modality_from_mime(mime),
                        mime_type=mime,
                        metadata={"tool_name": fallback_tool},
                    )
                )
    return assets
