# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Anthropic (Claude) provider implementation using the Beta Files API."""

from __future__ import annotations

import base64
import logging
import mimetypes
import shutil
import tempfile
from pathlib import Path

import anthropic

from ..config import (
    DEFAULT_TOOL_LOOP_LIMIT,
    INPUT_FILES_DIR,
    MediaToolBackend,
    get_api_key,
)
from ..detection import PROVIDER_INLINE, PROVIDER_TOOL
from ..input_files import resolve_input_file
from ..input_support import (
    evaluate_input_support,
    mark_observed_success,
    set_current_input_support,
)
from ..media_tools import (
    build_anthropic_tools,
    dispatch_media_tool,
    make_tool_call_record,
    parse_tool_arguments,
)
from ..models import EvalPrompt, ProviderResponse, RequestRecord, ToolCallRecord
from ..output_assets import decode_base64_data
from .base import BaseProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic models (Claude) using the Beta Files API.

    File inputs are uploaded via client.beta.files.upload(), referenced by
    file_id in image/document source blocks, and cleaned up after each request.

    Anthropic supports text and image output. Other modalities detected via URL fallback.
    """

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-20250514",
        run_name: str = "",
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "",
        request_timeout: int = 300,
        max_retries: int = 5,
        retry_backoff: int = 4,
        tools: list[str] | None = None,
        provider_tools: list[dict] | None = None,
        tool_loop_limit: int = DEFAULT_TOOL_LOOP_LIMIT,
        media_tool_backends: dict[str, MediaToolBackend] | None = None,
        fetch_remote_assets: bool = False,
    ):
        super().__init__(
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            fetch_remote_assets=fetch_remote_assets,
        )
        self.provider_name = "anthropic"
        self.model_name = model
        self.run_name = run_name or model
        self._tools = tools or []
        self._tool_defs = build_anthropic_tools(self._tools)
        # Verbatim provider-native tool specs (Decision 4b). The harness does
        # not interpret these; they exist so capable systems are expressible.
        self._provider_tools = list(provider_tools or [])
        if self._provider_tools:
            self._tool_defs = list(self._tool_defs) + self._provider_tools
        self._tool_loop_limit = tool_loop_limit
        self._media_tool_backends = media_tool_backends or {}
        self._base_url = base_url
        self._inline_input_files = bool(base_url)
        client_kwargs = {
            "api_key": get_api_key(api_key_env),
            "timeout": request_timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**client_kwargs)

    async def _upload_file(self, fpath: Path) -> str | None:
        try:
            with open(fpath, "rb") as f:
                uploaded = await self._client.beta.files.upload(file=f)
            return uploaded.id
        except Exception as exc:
            logger.warning("Failed to upload %s: %s", fpath, exc)
            return None

    async def _delete_file(self, file_id: str) -> None:
        try:
            await self._client.beta.files.delete(file_id=file_id)
        except Exception:
            pass

    async def _build_content_blocks(
        self, prompt: EvalPrompt
    ) -> tuple[list[dict], list[str]]:
        blocks: list[dict] = []
        uploaded_ids: list[str] = []

        temp_files: list[Path] = []
        for fname in prompt.input_files:
            fpath = resolve_input_file(fname, root=INPUT_FILES_DIR)
            if not fpath.exists():
                raise FileNotFoundError(
                    f"Missing input media file for prompt {prompt.prompt_id}: {fpath}"
                )

            suffix = fpath.suffix.lower()
            mime_type = (
                mimetypes.guess_type(str(fpath))[0] or "application/octet-stream"
            )
            if self._inline_input_files:
                if suffix in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": base64.b64encode(fpath.read_bytes()).decode(),
                            },
                        }
                    )
                    continue
                if mime_type.startswith("text/") or suffix == ".csv":
                    blocks.append(
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": (
                                    "text/plain" if suffix == ".csv" else mime_type
                                ),
                                "data": base64.b64encode(fpath.read_bytes()).decode(),
                            },
                        }
                    )
                    continue
                raise ValueError(
                    f"Anthropic does not support inline input file type {mime_type} for {fname}"
                )

            upload_path = fpath
            if suffix == ".csv":
                tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
                tmp.close()
                tmp_path = Path(tmp.name)
                shutil.copy2(fpath, tmp_path)
                temp_files.append(tmp_path)
                upload_path = tmp_path

            file_id = await self._upload_file(upload_path)
            if not file_id:
                for t in temp_files:
                    t.unlink(missing_ok=True)
                raise RuntimeError(
                    f"input file upload failed for prompt {prompt.prompt_id}: {fname}"
                )
            uploaded_ids.append(file_id)

            if suffix in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "file", "file_id": file_id},
                    }
                )
            else:
                blocks.append(
                    {
                        "type": "document",
                        "source": {"type": "file", "file_id": file_id},
                    }
                )

        if "Text" in prompt.input_modalities:
            blocks.append({"type": "text", "text": prompt.prompt_text})

        for t in temp_files:
            t.unlink(missing_ok=True)

        return blocks, uploaded_ids

    def _extract_response_text(self, response) -> str:
        """Collect user-visible prose only.

        Tool-call arguments and tool traces are deliberately excluded: they are
        recorded separately and must never be able to mint URL credit.
        """
        text_content = ""
        for block in response.content:
            block_type = block.type
            if block_type == "text" and block.text.strip():
                text_content += block.text
            elif block_type == "tool_result":
                tool_content = getattr(block, "content", [])
                if isinstance(tool_content, list):
                    for sub_block in tool_content:
                        sub_type = (
                            getattr(sub_block, "type", "")
                            if not isinstance(sub_block, dict)
                            else sub_block.get("type", "")
                        )
                        if sub_type == "text":
                            sub_text = (
                                getattr(sub_block, "text", "")
                                if not isinstance(sub_block, dict)
                                else sub_block.get("text", "")
                            )
                            if sub_text:
                                text_content += sub_text
        return text_content

    async def _capture_output_assets(
        self, prompt: EvalPrompt, response, prompt_assets=None
    ) -> list:
        """Extract artifacts. Classification is the harness's job, not ours."""
        assets = []
        prompt_assets = prompt_assets or self._asset_manager.for_prompt(
            prompt.prompt_id
        )
        for block in response.content:
            block_type = getattr(block, "type", "")
            if block_type not in ("image", "document"):
                continue
            source = getattr(block, "source", None)
            media_type = getattr(source, "media_type", "") if source is not None else ""
            data = decode_base64_data(
                getattr(source, "data", None) if source is not None else None
            )
            modality = "Image" if block_type == "image" else "Document"
            assets.append(
                prompt_assets.capture_bytes(
                    data=data,
                    modality=modality,
                    source_type="provider_inline",
                    delivery=PROVIDER_INLINE,
                    mime_type=media_type
                    or ("image/png" if modality == "Image" else "application/pdf"),
                    metadata={"block_type": block_type},
                )
            )
        for block in response.content:
            if getattr(block, "type", "") != "tool_result":
                continue
            tool_content = getattr(block, "content", [])
            if not isinstance(tool_content, list):
                continue
            for sub_block in tool_content:
                sub_type = (
                    getattr(sub_block, "type", "")
                    if not isinstance(sub_block, dict)
                    else sub_block.get("type", "")
                )
                if sub_type != "image":
                    continue
                source = (
                    sub_block.get("source")
                    if isinstance(sub_block, dict)
                    else getattr(sub_block, "source", None)
                )
                media_type = ""
                raw = None
                if isinstance(source, dict):
                    media_type = source.get("media_type", "")
                    raw = source.get("data")
                elif source is not None:
                    media_type = getattr(source, "media_type", "")
                    raw = getattr(source, "data", None)
                assets.append(
                    prompt_assets.capture_bytes(
                        data=decode_base64_data(raw),
                        modality="Image",
                        source_type="tool_result",
                        delivery=PROVIDER_TOOL,
                        mime_type=media_type or "image/png",
                        metadata={"block_type": "tool_result"},
                    )
                )
        return assets

    def _extract_tool_uses(self, response) -> list[dict]:
        calls = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", "") != "tool_use":
                continue
            calls.append(
                {
                    "name": getattr(block, "name", ""),
                    "id": getattr(block, "id", ""),
                    "arguments": parse_tool_arguments(getattr(block, "input", None)),
                }
            )
        return calls

    async def _run_tool_loop(
        self, *, prompt: EvalPrompt, messages: list[dict], prompt_assets
    ) -> tuple[object, list, list[ToolCallRecord]]:
        create_kwargs = {
            "model": self.model_name,
            "max_tokens": 32768,
            "betas": ["files-api-2025-04-14"],
            "messages": messages,
        }
        if self._tool_defs:
            create_kwargs["tools"] = self._tool_defs
        response = await self._client.beta.messages.create(**create_kwargs)
        output_assets = await self._capture_output_assets(
            prompt, response, prompt_assets
        )
        tool_calls: list[ToolCallRecord] = []
        if not self._tool_defs:
            return response, output_assets, tool_calls

        for _ in range(self._tool_loop_limit):
            tool_uses = self._extract_tool_uses(response)
            if not tool_uses:
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        block.model_dump() if hasattr(block, "model_dump") else block
                        for block in response.content
                    ],
                }
            )
            result_blocks = []
            for call in tool_uses:
                try:
                    result = await dispatch_media_tool(
                        call["name"],
                        call["arguments"],
                        prompt_assets,
                        self._media_tool_backends,
                    )
                    output_assets.extend(result.assets)
                    tool_calls.append(
                        make_tool_call_record(
                            tool_name=call["name"],
                            provider_call_id=call["id"],
                            arguments=call["arguments"],
                            status="completed",
                            assets=result.assets,
                        )
                    )
                    result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call["id"],
                            "content": result.as_model_payload(),
                        }
                    )
                except Exception as exc:
                    tool_calls.append(
                        make_tool_call_record(
                            tool_name=call["name"],
                            provider_call_id=call["id"],
                            arguments=call["arguments"],
                            status="error",
                            error=str(exc),
                        )
                    )
                    result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call["id"],
                            "is_error": True,
                            "content": f"Tool execution failed: {exc}",
                        }
                    )
            messages.append({"role": "user", "content": result_blocks})
            response = await self._client.beta.messages.create(**create_kwargs)
            output_assets.extend(
                await self._capture_output_assets(prompt, response, prompt_assets)
            )
        return response, output_assets, tool_calls

    async def _send_impl(self, prompt: EvalPrompt) -> ProviderResponse:
        input_support = evaluate_input_support(
            provider=self.provider_name, base_url=self._base_url, prompt=prompt
        )
        set_current_input_support(input_support)
        content, uploaded_ids = await self._build_content_blocks(prompt)
        prompt_assets = self.new_prompt_assets(prompt)

        try:
            messages = [{"role": "user", "content": content}]
            response, output_assets, tool_calls = await self._run_tool_loop(
                prompt=prompt, messages=messages, prompt_assets=prompt_assets
            )

            request = RequestRecord(
                provider=self.provider_name,
                api="messages",
                model=self.model_name,
                user_prompt=prompt.prompt_text,
                input_files=list(prompt.input_files),
                input_modalities=list(prompt.input_modalities),
                output_modalities=list(prompt.output_modalities),
                tools=list(self._tools),
                max_output_tokens=32768,
                provider_request={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": block.get("type"),
                                    "source_type": block.get("source", {}).get("type"),
                                }
                                for block in content
                            ],
                        }
                    ],
                    "betas": ["files-api-2025-04-14"],
                    "tools": self._tool_defs,
                    "input_support": mark_observed_success(input_support).to_dict(),
                },
            )

            return await self.finalize(
                prompt=prompt,
                prompt_assets=prompt_assets,
                response_text=self._extract_response_text(response),
                output_assets=output_assets,
                tool_calls=tool_calls,
                raw_response=self.json_raw_response(response),
                request=request,
            )
        finally:
            for fid in uploaded_ids:
                await self._delete_file(fid)
