# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""OpenAI provider — supports both Responses API and Chat Completions API."""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path

from openai import AsyncOpenAI

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
    build_openai_tools,
    dispatch_media_tool,
    make_tool_call_record,
    parse_tool_arguments,
)
from ..models import EvalPrompt, ProviderResponse, RequestRecord, ToolCallRecord
from ..output_assets import decode_base64_data, modality_from_mime
from .base import BaseProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI models.

    Supports two API surfaces controlled by the ``api`` parameter:

    - ``"responses"`` (default): Uses the Responses API + Files API.
      File inputs are uploaded via client.files.create(), referenced by
      file_id, and cleaned up after each request.

    - ``"chat_completions"``: Uses the Chat Completions API with
      base64-inline file encoding.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        run_name: str = "",
        api: str = "responses",
        api_key_env: str = "OPENAI_API_KEY",
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
        self.provider_name = "openai"
        self.model_name = model
        self.run_name = run_name or model
        self._api = api
        self._tools = tools or []
        self._tool_defs = build_openai_tools(self._tools)
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
            "max_retries": 0,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**client_kwargs)

    # ------------------------------------------------------------------
    # Responses API helpers
    # ------------------------------------------------------------------

    async def _upload_file(self, fpath: Path) -> str | None:
        try:
            with open(fpath, "rb") as f:
                uploaded = await self._client.files.create(file=f, purpose="user_data")
            return uploaded.id
        except Exception as exc:
            logger.warning("Failed to upload %s: %s", fpath, exc)
            return None

    async def _delete_file(self, file_id: str) -> None:
        try:
            await self._client.files.delete(file_id)
        except Exception:
            pass

    def _extract_response_text_responses(self, response) -> str:
        """Collect user-visible prose only from a Responses-API response."""
        text_content = ""
        for item in response.output:
            if hasattr(item, "content") and item.content:
                for c in item.content:
                    if c.type == "output_text" and c.text and c.text.strip():
                        text_content += c.text
        if not text_content and response.output_text and response.output_text.strip():
            text_content = response.output_text
        return text_content

    async def _capture_output_assets_responses(
        self, prompt: EvalPrompt, response, prompt_assets=None
    ) -> list:
        """Extract artifacts. Classification is the harness's job, not ours."""
        assets = []
        prompt_assets = prompt_assets or self.new_prompt_assets(prompt)
        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", "")
            if item_type == "image_generation_call":
                result = getattr(item, "result", None)
                data = decode_base64_data(result)
                if data:
                    assets.append(
                        prompt_assets.capture_bytes(
                            data=data,
                            modality="Image",
                            source_type="image_generation_call",
                            delivery=PROVIDER_TOOL,
                            mime_type=getattr(item, "output_format", "")
                            and f"image/{item.output_format}"
                            or "image/png",
                            metadata={"response_item_type": item_type},
                        )
                    )
                else:
                    # A "completed" status with no bytes is a claim, not an
                    # artifact. Decision 4a requires actual bytes for native.
                    assets.append(
                        prompt_assets.capture_reference(
                            url="",
                            modality="Image",
                            delivery=PROVIDER_TOOL,
                            mime_type="image/png",
                            metadata={
                                "response_item_type": item_type,
                                "status": getattr(item, "status", ""),
                                "reason": "image_generation_call returned no bytes",
                            },
                        )
                    )
            if item_type == "code_interpreter_call":
                results = getattr(item, "results", [])
                for result in results:
                    result_type = getattr(result, "type", "")
                    if result_type == "image":
                        data = decode_base64_data(getattr(result, "content", None))
                        assets.append(
                            prompt_assets.capture_bytes(
                                data=data,
                                modality="Image",
                                source_type="code_interpreter_call",
                                delivery=PROVIDER_TOOL,
                                mime_type="image/png",
                                metadata={"response_item_type": item_type},
                            )
                        )
                    if result_type == "files":
                        files = getattr(result, "files", [])
                        for f in files:
                            mime = (
                                f.get("mime_type", "")
                                if isinstance(f, dict)
                                else getattr(f, "mime_type", "")
                            )
                            filename = (
                                f.get("filename", "")
                                if isinstance(f, dict)
                                else getattr(f, "filename", "")
                            )
                            data = decode_base64_data(
                                f.get("content", "")
                                if isinstance(f, dict)
                                else getattr(f, "content", "")
                            )
                            if not mime and filename:
                                mime = mimetypes.guess_type(filename)[0] or ""
                            assets.append(
                                prompt_assets.capture_bytes(
                                    data=data,
                                    modality=modality_from_mime(mime),
                                    source_type="code_interpreter_call",
                                    delivery=PROVIDER_TOOL,
                                    mime_type=mime,
                                    filename=filename or "",
                                    metadata={"response_item_type": item_type},
                                )
                            )
            for c in getattr(item, "content", []) or []:
                ctype = getattr(c, "type", "")
                if ctype in ("output_image", "image"):
                    data = decode_base64_data(
                        getattr(c, "data", None) or getattr(c, "image", None)
                    )
                    assets.append(
                        prompt_assets.capture_bytes(
                            data=data,
                            modality="Image",
                            source_type="provider_inline",
                            delivery=PROVIDER_INLINE,
                            mime_type=getattr(c, "mime_type", "") or "image/png",
                            metadata={"content_type": ctype},
                        )
                    )
                if ctype in ("output_audio", "audio"):
                    audio = getattr(c, "audio", None)
                    data = decode_base64_data(
                        getattr(audio, "data", None)
                        if audio is not None
                        else getattr(c, "data", None)
                    )
                    mime = getattr(c, "mime_type", "") or "audio/mpeg"
                    assets.append(
                        prompt_assets.capture_bytes(
                            data=data,
                            modality="Audio",
                            source_type="provider_inline",
                            delivery=PROVIDER_INLINE,
                            mime_type=mime,
                            metadata={"content_type": ctype},
                        )
                    )
                if ctype == "output_file":
                    file_obj = getattr(c, "file", None)
                    data = decode_base64_data(
                        getattr(c, "data", None) or getattr(file_obj, "data", None)
                    )
                    mime = getattr(c, "mime_type", "") or getattr(
                        file_obj, "mime_type", ""
                    )
                    filename = (
                        getattr(c, "filename", "")
                        or getattr(file_obj, "filename", "")
                        or ""
                    )
                    assets.append(
                        prompt_assets.capture_bytes(
                            data=data,
                            modality=modality_from_mime(mime),
                            source_type="provider_inline",
                            delivery=PROVIDER_INLINE,
                            mime_type=mime,
                            filename=filename,
                            metadata={"content_type": ctype},
                        )
                    )
        return assets

    def _extract_function_calls(self, response) -> list[dict]:
        calls = []
        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", "")
            if item_type not in ("function_call", "tool_call"):
                continue
            name = getattr(item, "name", "") or getattr(item, "function", {}).get(
                "name", ""
            )
            if not name:
                continue
            call_id = (
                getattr(item, "call_id", "")
                or getattr(item, "id", "")
                or getattr(item, "tool_call_id", "")
            )
            arguments = getattr(item, "arguments", None)
            calls.append(
                {
                    "name": name,
                    "call_id": call_id,
                    "arguments": parse_tool_arguments(arguments),
                }
            )
        return calls

    async def _run_response_tool_loop(
        self, *, prompt: EvalPrompt, create_kwargs: dict, prompt_assets
    ) -> tuple[object, list, list[ToolCallRecord]]:
        response = await self._client.responses.create(**create_kwargs)
        output_assets = await self._capture_output_assets_responses(
            prompt, response, prompt_assets
        )
        tool_calls: list[ToolCallRecord] = []
        if not self._tool_defs:
            return response, output_assets, tool_calls

        prompt_assets = self._asset_manager.for_prompt(prompt.prompt_id)
        for _ in range(self._tool_loop_limit):
            function_calls = self._extract_function_calls(response)
            if not function_calls:
                break
            function_outputs = []
            for call in function_calls:
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
                            provider_call_id=call["call_id"],
                            arguments=call["arguments"],
                            status="completed",
                            assets=result.assets,
                        )
                    )
                    function_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call["call_id"],
                            "output": result.as_model_payload(),
                        }
                    )
                except Exception as exc:
                    tool_calls.append(
                        make_tool_call_record(
                            tool_name=call["name"],
                            provider_call_id=call["call_id"],
                            arguments=call["arguments"],
                            status="error",
                            error=str(exc),
                        )
                    )
                    function_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call["call_id"],
                            "output": f"Tool execution failed: {exc}",
                        }
                    )
            followup_input = (
                list(getattr(response, "output", []) or []) + function_outputs
            )
            try:
                response = await self._client.responses.create(
                    model=self.model_name,
                    input=function_outputs,
                    previous_response_id=getattr(response, "id", None),
                    tools=self._tool_defs,
                    max_output_tokens=32768,
                )
            except Exception as exc:
                if "previous_response" not in str(exc):
                    raise
                response = await self._client.responses.create(
                    model=self.model_name,
                    input=followup_input,
                    tools=self._tool_defs,
                    max_output_tokens=32768,
                )
            output_assets.extend(
                await self._capture_output_assets_responses(
                    prompt, response, prompt_assets
                )
            )
        return response, output_assets, tool_calls

    async def _send_responses(self, prompt: EvalPrompt) -> ProviderResponse:
        uploaded_file_ids: list[str] = []
        prompt_assets = self.new_prompt_assets(prompt)
        input_support = evaluate_input_support(
            provider=self.provider_name, base_url=self._base_url, prompt=prompt
        )
        set_current_input_support(input_support)
        try:
            content_parts: list[dict] = []

            for fname in prompt.input_files:
                fpath = resolve_input_file(fname, root=INPUT_FILES_DIR)
                if not fpath.exists():
                    raise FileNotFoundError(
                        f"Missing input media file for prompt {prompt.prompt_id}: {fpath}"
                    )
                mime_type = (
                    mimetypes.guess_type(str(fpath))[0] or "application/octet-stream"
                )
                if self._inline_input_files:
                    data_b64 = base64.standard_b64encode(fpath.read_bytes()).decode()
                    if mime_type.startswith("image/"):
                        content_parts.append(
                            {
                                "type": "input_image",
                                "image_url": f"data:{mime_type};base64,{data_b64}",
                            }
                        )
                    elif mime_type.startswith("audio/"):
                        content_parts.append(
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": data_b64,
                                    "format": fpath.suffix.lstrip(".") or "mp3",
                                },
                            }
                        )
                    else:
                        content_parts.append(
                            {
                                "type": "input_file",
                                "filename": fname,
                                "file_data": f"data:{mime_type};base64,{data_b64}",
                            }
                        )
                    continue

                file_id = await self._upload_file(fpath)
                if not file_id:
                    raise RuntimeError(
                        f"input file upload failed for prompt {prompt.prompt_id}: {fname}"
                    )
                uploaded_file_ids.append(file_id)

                if mime_type.startswith("image/"):
                    content_parts.append({"type": "input_image", "file_id": file_id})
                else:
                    content_parts.append({"type": "input_file", "file_id": file_id})

            if "Text" in prompt.input_modalities:
                content_parts.append({"type": "input_text", "text": prompt.prompt_text})

            create_kwargs: dict = {
                "model": self.model_name,
                "input": [{"role": "user", "content": content_parts}],
                "max_output_tokens": 32768,
            }
            if self._tool_defs:
                create_kwargs["tools"] = self._tool_defs
            response, output_assets, tool_calls = await self._run_response_tool_loop(
                prompt=prompt, create_kwargs=create_kwargs, prompt_assets=prompt_assets
            )
            request = RequestRecord(
                provider=self.provider_name,
                api="responses",
                model=self.model_name,
                user_prompt=prompt.prompt_text,
                input_files=list(prompt.input_files),
                input_modalities=list(prompt.input_modalities),
                output_modalities=list(prompt.output_modalities),
                tools=list(self._tools),
                max_output_tokens=32768,
                provider_request={
                    **create_kwargs,
                    "input_support": mark_observed_success(input_support).to_dict(),
                },
            )

            return await self.finalize(
                prompt=prompt,
                prompt_assets=prompt_assets,
                response_text=self._extract_response_text_responses(response),
                output_assets=output_assets,
                tool_calls=tool_calls,
                raw_response=self.json_raw_response(response),
                request=request,
            )
        finally:
            for fid in uploaded_file_ids:
                await self._delete_file(fid)

    # ------------------------------------------------------------------
    # Chat Completions API helpers
    # ------------------------------------------------------------------

    def _build_chat_content_parts(self, prompt: EvalPrompt) -> list[dict]:
        parts: list[dict] = []

        if "Text" in prompt.input_modalities:
            parts.append({"type": "text", "text": prompt.prompt_text})

        for fname in prompt.input_files:
            fpath = resolve_input_file(fname, root=INPUT_FILES_DIR)
            if not fpath.exists():
                raise FileNotFoundError(
                    f"Missing input media file for prompt {prompt.prompt_id}: {fpath}"
                )

            mime_type = (
                mimetypes.guess_type(str(fpath))[0] or "application/octet-stream"
            )
            data_b64 = base64.standard_b64encode(fpath.read_bytes()).decode()

            if mime_type.startswith("image/"):
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{data_b64}"},
                    }
                )
            elif mime_type.startswith("audio/"):
                fmt = fpath.suffix.lstrip(".") or "mp3"
                audio_format = fmt if fmt in ("mp3", "wav") else "mp3"
                parts.append(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": data_b64, "format": audio_format},
                    }
                )
            else:
                parts.append(
                    {
                        "type": "file",
                        "file": {
                            "filename": fname,
                            "file_data": f"data:{mime_type};base64,{data_b64}",
                        },
                    }
                )

        return parts

    @staticmethod
    def _get_val(obj, key: str, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _extract_response_text_chat(self, response) -> str:
        """Collect user-visible prose only from a Chat Completions response."""
        if not response.choices:
            return ""
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content if content.strip() else ""
        text_content = ""
        if isinstance(content, list):
            for part in content:
                if self._get_val(part, "type", "") in ("text", "output_text"):
                    text_val = self._get_val(part, "text", "") or ""
                    if text_val.strip():
                        text_content += text_val
        return text_content

    def _capture_output_assets_chat(self, prompt: EvalPrompt, response, prompt_assets):
        """Extract artifacts from a Chat Completions response."""
        assets = []
        if not response.choices:
            return assets
        message = response.choices[0].message
        content = message.content
        if isinstance(content, list):
            for part in content:
                part_type = self._get_val(part, "type", "")
                if part_type in ("text", "output_text"):
                    continue

                if part_type in ("image_url", "output_image", "image"):
                    raw = self._get_val(part, "image_url", None) or self._get_val(
                        part, "data", None
                    )
                    if isinstance(raw, dict):
                        raw = raw.get("url")
                    assets.append(
                        prompt_assets.capture_bytes(
                            data=decode_base64_data(raw),
                            modality="Image",
                            source_type="provider_inline",
                            delivery=PROVIDER_INLINE,
                            mime_type=self._get_val(part, "mime_type", "")
                            or "image/png",
                            metadata={"content_type": part_type},
                        )
                    )
                    continue

                if part_type in ("output_audio", "audio"):
                    audio = self._get_val(part, "audio", None)
                    raw = (
                        self._get_val(audio, "data", None)
                        if audio is not None
                        else self._get_val(part, "data", None)
                    )
                    assets.append(
                        prompt_assets.capture_bytes(
                            data=decode_base64_data(raw),
                            modality="Audio",
                            source_type="provider_inline",
                            delivery=PROVIDER_INLINE,
                            mime_type=self._get_val(part, "mime_type", "")
                            or "audio/mpeg",
                            metadata={"content_type": part_type},
                        )
                    )
                    continue

                if part_type in ("file", "output_file"):
                    file_obj = self._get_val(part, "file", None)
                    mime = self._get_val(part, "mime_type", "") or ""
                    filename = self._get_val(part, "filename", "") or ""
                    if file_obj is not None:
                        mime = mime or self._get_val(file_obj, "mime_type", "") or ""
                        filename = (
                            filename
                            or self._get_val(file_obj, "filename", "")
                            or self._get_val(file_obj, "name", "")
                            or ""
                        )
                    file_data = self._get_val(part, "file_data", "") or ""
                    if file_obj is not None and not file_data:
                        file_data = self._get_val(file_obj, "file_data", "") or ""
                    if not mime and file_data.startswith("data:"):
                        mime = file_data.split(";", 1)[0].replace("data:", "", 1)
                    if not mime and filename:
                        mime = mimetypes.guess_type(filename)[0] or ""
                    assets.append(
                        prompt_assets.capture_bytes(
                            data=decode_base64_data(file_data),
                            modality=modality_from_mime(mime),
                            source_type="provider_inline",
                            delivery=PROVIDER_INLINE,
                            mime_type=mime,
                            filename=filename,
                            metadata={"content_type": part_type},
                        )
                    )

        audio = getattr(message, "audio", None)
        if audio:
            assets.append(
                prompt_assets.capture_bytes(
                    data=decode_base64_data(getattr(audio, "data", None)),
                    modality="Audio",
                    source_type="provider_inline",
                    delivery=PROVIDER_INLINE,
                    mime_type="audio/mpeg",
                    metadata={"content_type": "message.audio"},
                )
            )
        image = getattr(message, "image", None)
        if image:
            assets.append(
                prompt_assets.capture_bytes(
                    data=decode_base64_data(
                        getattr(image, "data", None)
                        if not isinstance(image, str)
                        else image
                    ),
                    modality="Image",
                    source_type="provider_inline",
                    delivery=PROVIDER_INLINE,
                    mime_type="image/png",
                    metadata={"content_type": "message.image"},
                )
            )
        return assets

    async def _send_chat_completions(self, prompt: EvalPrompt) -> ProviderResponse:
        content_parts = self._build_chat_content_parts(prompt)
        messages = [{"role": "user", "content": content_parts}]

        create_kwargs: dict = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 32768,
        }
        if self._tool_defs:
            create_kwargs["tools"] = self._tool_defs
        response = await self._client.chat.completions.create(**create_kwargs)
        prompt_assets = self.new_prompt_assets(prompt)
        output_assets = self._capture_output_assets_chat(
            prompt, response, prompt_assets
        )
        request = RequestRecord(
            provider=self.provider_name,
            api="chat_completions",
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
                            (
                                {
                                    "type": p.get("type"),
                                    "filename": p.get("file", {}).get("filename"),
                                }
                                if isinstance(p, dict) and p.get("type") == "file"
                                else {"type": p.get("type")}
                            )
                            for p in content_parts
                        ],
                    }
                ],
                "max_tokens": 32768,
            },
        )

        return await self.finalize(
            prompt=prompt,
            prompt_assets=prompt_assets,
            response_text=self._extract_response_text_chat(response),
            output_assets=output_assets,
            tool_calls=[],
            raw_response=self.json_raw_response(response),
            request=request,
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def _send_impl(self, prompt: EvalPrompt) -> ProviderResponse:
        if self._api == "chat_completions":
            return await self._send_chat_completions(prompt)
        return await self._send_responses(prompt)
