# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Reference provider. Copy this file to add a system under test.

This adapter has no SDK dependency: it replays a canned response so the
contract can be exercised, and so the conformance suite has something to run
against that is known-correct. Everything a real adapter must do is here, in
the order a real adapter must do it.

The contract in one sentence: **extract, do not adjudicate.**

What your adapter is responsible for:

- calling your system and getting a response back
- pulling out user-visible prose as ``response_text``
- turning every artifact into a :class:`~mmi.models.CapturedAsset`, tagged with
  how it arrived (``delivery``) and whether you hold the bytes
- recording tool calls observationally
- handing back ``raw_response`` as plain JSON

What your adapter must **not** do, because doing it would make your scores
incomparable with everyone else's:

- decide which modality an artifact counts as (the shared table does that)
- set pass/fail, or score anything
- put tool traces or request metadata into ``response_text``
- fetch remote content directly (pass ``self.fetch_remote_assets`` to
  ``capture_url`` and let ``mmi.fetch`` do it)
- classify by hostname

See ``docs/ADDING_A_PROVIDER.md`` for the long form.
"""

from __future__ import annotations

from typing import Any

from ..detection import HARNESS_TOOL, PROVIDER_INLINE, PROVIDER_TOOL
from ..media_tools import build_openai_tools
from ..models import EvalPrompt, ProviderResponse, RequestRecord
from ..output_assets import decode_base64_data, modality_from_mime
from .base import BaseProvider

#: A 1x1 PNG. Enough to be real bytes without shipping anyone's generated media.
TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


class StubProvider(BaseProvider):
    """A provider that returns a scripted response instead of calling anything.

    Set ``scripted_response`` to drive it. The shape mirrors what a real
    adapter would have already parsed out of its SDK's response object::

        {
            "text": "here is the chart you asked for",
            "artifacts": [
                {
                    "mime_type": "image/png",
                    "data_b64": TINY_PNG_B64,
                    "delivery": "provider_inline",
                },
                {
                    "mime_type": "video/mp4",
                    "url": "https://example.invalid/clip.mp4",
                    "delivery": "external_url",
                },
            ],
            "tool_calls": [{"name": "image_gen", "arguments": {"prompt": "a fox"}}],
        }
    """

    def __init__(
        self,
        *,
        model: str = "stub",
        run_name: str = "",
        api_key_env: str = "",
        base_url: str = "",
        request_timeout: int = 300,
        max_retries: int = 5,
        retry_backoff: int = 4,
        tools: list[str] | None = None,
        provider_tools: list[dict] | None = None,
        tool_loop_limit: int = 3,
        media_tool_backends: dict | None = None,
        scripted_response: dict[str, Any] | None = None,
        fetch_remote_assets: bool = False,
    ):
        super().__init__(
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            fetch_remote_assets=fetch_remote_assets,
        )
        self.provider_name = "stub"
        self.model_name = model
        self.run_name = run_name or model
        self._tools = tools or []
        self._tool_defs = build_openai_tools(self._tools)
        self._provider_tools = list(provider_tools or [])
        self._base_url = base_url
        self._tool_loop_limit = tool_loop_limit
        self._media_tool_backends = media_tool_backends or {}
        self.scripted_response: dict[str, Any] = scripted_response or {
            "text": "",
            "artifacts": [],
            "tool_calls": [],
        }

    async def _call_system(self, prompt: EvalPrompt) -> dict[str, Any]:
        """Replace this with the call to your system."""
        return self.scripted_response

    async def _send_impl(self, prompt: EvalPrompt) -> ProviderResponse:
        # One asset manager per prompt, shared by everything below, so asset
        # IDs cannot collide across parsing / tools / URL capture.
        prompt_assets = self.new_prompt_assets(prompt)

        raw = await self._call_system(prompt)

        # 1. User-visible prose ONLY. Never tool traces.
        response_text = str(raw.get("text") or "")

        # 2. Artifacts. Say how each one arrived; say nothing about what it
        #    "is" beyond its MIME type.
        output_assets = []
        for artifact in raw.get("artifacts") or []:
            delivery = artifact.get("delivery", PROVIDER_INLINE)
            mime = artifact.get("mime_type", "")
            data = decode_base64_data(artifact.get("data_b64"))

            if data:
                output_assets.append(
                    prompt_assets.capture_bytes(
                        data=data,
                        modality=modality_from_mime(mime),
                        source_type=artifact.get("source_type", "scripted"),
                        delivery=delivery,
                        mime_type=mime,
                        # Only for genuinely opaque MIME. The shared table wins
                        # wherever it can classify, so this cannot mint credit.
                        modality_hint=artifact.get("modality_hint", ""),
                        source_url=artifact.get("url", ""),
                    )
                )
            else:
                # No bytes: a reference, not a production. Never "captured".
                output_assets.append(
                    prompt_assets.capture_reference(
                        url=artifact.get("url", ""),
                        modality=modality_from_mime(mime),
                        delivery=delivery,
                        mime_type=mime,
                        modality_hint=artifact.get("modality_hint", ""),
                    )
                )

        # 3. Tool calls are observational. They never score.
        tool_calls = []
        for index, call in enumerate(raw.get("tool_calls") or []):
            from ..media_tools import make_tool_call_record

            tool_calls.append(
                make_tool_call_record(
                    tool_name=call.get("name", ""),
                    provider_call_id=call.get("id", f"stub-{index}"),
                    arguments=call.get("arguments", {}),
                    status=call.get("status", "completed"),
                )
            )

        request = RequestRecord(
            provider=self.provider_name,
            api="scripted",
            model=self.model_name,
            user_prompt=prompt.prompt_text,
            input_files=list(prompt.input_files),
            input_modalities=list(prompt.input_modalities),
            output_modalities=list(prompt.output_modalities),
            tools=list(self._tools),
            max_output_tokens=32768,
        )

        # 4. finalize() runs URL capture over the prose and assembles the
        #    contract. raw_response must be plain JSON — never str().
        return await self.finalize(
            prompt=prompt,
            prompt_assets=prompt_assets,
            response_text=response_text,
            output_assets=output_assets,
            tool_calls=tool_calls,
            raw_response=self.json_raw_response(raw),
            request=request,
        )


__all__ = ["HARNESS_TOOL", "PROVIDER_TOOL", "StubProvider", "TINY_PNG_B64"]
