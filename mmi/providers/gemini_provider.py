# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Google Gemini provider implementation using the Files API + google-genai SDK."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
from pathlib import Path

import httpx
from google.genai import Client, types

from ..config import (
    DEFAULT_TOOL_LOOP_LIMIT,
    INPUT_FILES_DIR,
    MediaToolBackend,
    get_api_key,
)
from ..detection import PROVIDER_INLINE
from ..input_files import resolve_input_file
from ..input_support import (
    evaluate_input_support,
    mark_observed_success,
    set_current_input_support,
)
from ..media_tools import (
    build_gemini_tools,
    dispatch_media_tool,
    make_tool_call_record,
    parse_tool_arguments,
)
from ..models import EvalPrompt, ProviderResponse, RequestRecord, ToolCallRecord
from ..output_assets import decode_base64_data, modality_from_mime
from .base import BaseProvider

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """Provider for Google Gemini models using the Files API.

    File inputs are uploaded via client.files.upload(), passed as file
    references to generate_content, and cleaned up after each request.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-2.0-flash",
        run_name: str = "",
        api_key_env: str = "GOOGLE_API_KEY",
        base_url: str = "",
        request_timeout: int = 300,
        max_retries: int = 5,
        retry_backoff: int = 4,
        response_modalities: list[str] | None = None,
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
        self.provider_name = "gemini"
        self.model_name = model
        self.run_name = run_name or model
        self._response_modalities = response_modalities or []
        self._tools = tools or []
        self._tool_defs = build_gemini_tools(self._tools)
        # Verbatim provider-native tool specs (Decision 4b). The harness does
        # not interpret these; they exist so capable systems are expressible.
        self._provider_tools = list(provider_tools or [])
        if self._provider_tools:
            self._tool_defs = list(self._tool_defs) + self._provider_tools
        self._tool_loop_limit = tool_loop_limit
        self._media_tool_backends = media_tool_backends or {}
        self._inline_input_files = bool(base_url)

        httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout, connect=60.0),
            trust_env=True,
            follow_redirects=True,
        )
        http_options_kwargs = {
            "timeout": request_timeout * 1000,
            "httpx_async_client": httpx_client,
        }
        if base_url:
            http_options_kwargs["base_url"] = base_url
            http_options_kwargs["api_version"] = "v1"
        self._client = Client(
            api_key=get_api_key(api_key_env),
            http_options=types.HttpOptions(**http_options_kwargs),
        )

    _UPLOAD_POLL_INTERVAL = 2  # seconds between PROCESSING polls
    _UPLOAD_POLL_MAX_WAIT = 60  # give up after this many seconds

    def _upload_file_raw(self, fpath: Path) -> object:
        """Blocking upload — run via asyncio.to_thread()."""
        mime_type = mimetypes.guess_type(str(fpath))[0] or "application/octet-stream"
        if mime_type == "text/csv":
            mime_type = "text/plain"
        return self._client.files.upload(
            file=str(fpath),
            config=types.UploadFileConfig(mime_type=mime_type),
        )

    async def _upload_file(self, fpath: Path) -> object | None:
        """Upload a file and poll until processing completes (async)."""
        try:
            uploaded = await asyncio.to_thread(self._upload_file_raw, fpath)
            waited = 0
            while uploaded.state and uploaded.state.name == "PROCESSING":
                if waited >= self._UPLOAD_POLL_MAX_WAIT:
                    logger.warning(
                        "Upload %s stuck in PROCESSING after %ds, giving up",
                        fpath.name,
                        waited,
                    )
                    return None
                await asyncio.sleep(self._UPLOAD_POLL_INTERVAL)
                waited += self._UPLOAD_POLL_INTERVAL
                uploaded = await asyncio.to_thread(
                    self._client.files.get, name=uploaded.name
                )
            return uploaded
        except Exception as exc:
            logger.warning("Failed to upload %s: %s", fpath, exc)
            return None

    def _delete_file_sync(self, name: str) -> None:
        try:
            self._client.files.delete(name=name)
        except Exception:
            pass

    def _download_file_raw(self, file_name: str) -> bytes:
        """Blocking download — run via asyncio.to_thread()."""
        return self._client.files.download(file=file_name)

    async def _download_file(self, file_name: str) -> bytes | None:
        """Download a file from Gemini Files API."""
        try:
            return await asyncio.to_thread(self._download_file_raw, file_name)
        except Exception as exc:
            logger.warning("Failed to download %s: %s", file_name, exc)
            return None

    async def _capture_output_assets(
        self, prompt: EvalPrompt, response, prompt_assets=None
    ) -> list:
        """Extract artifacts. Classification is the harness's job, not ours."""
        assets = []
        prompt_assets = prompt_assets or self.new_prompt_assets(prompt)
        if not response.candidates:
            return assets
        for candidate in response.candidates:
            if not candidate.content or not candidate.content.parts:
                continue
            for part in candidate.content.parts:
                if part.inline_data:
                    mime = part.inline_data.mime_type or "application/octet-stream"
                    data = decode_base64_data(part.inline_data.data)
                    assets.append(
                        prompt_assets.capture_bytes(
                            data=data,
                            modality=modality_from_mime(mime),
                            source_type="provider_inline",
                            delivery=PROVIDER_INLINE,
                            mime_type=mime,
                        )
                    )
                if hasattr(part, "file_data") and part.file_data:
                    assets.append(
                        await self._capture_file_data(part.file_data, prompt_assets)
                    )
        return assets

    async def _capture_file_data(self, file_data, prompt_assets):
        """Resolve a ``file_data`` part.

        A Files API reference is a structured provider artifact, so it stays
        ``provider_inline`` provenance and earns native credit *if* the bytes
        come back. An unresolvable reference is recorded as reference-only.
        """
        file_uri = getattr(file_data, "file_uri", "") or ""
        mime = getattr(file_data, "mime_type", "") or ""
        modality = modality_from_mime(mime)

        file_name = None
        if "files/" in file_uri:
            tail = file_uri.split("files/", 1)[1]
            file_id = tail.split("/")[0].split("?")[0].split("#")[0]
            if file_id:
                file_name = f"files/{file_id}"

        if file_name:
            data = await self._download_file(file_name)
            if data:
                return prompt_assets.capture_bytes(
                    data=data,
                    modality=modality,
                    source_type="file_data",
                    delivery=PROVIDER_INLINE,
                    mime_type=mime,
                    source_url=file_uri,
                    metadata={"file_name": file_name},
                )
            return prompt_assets.capture_reference(
                url=file_uri,
                modality=modality,
                delivery=PROVIDER_INLINE,
                mime_type=mime,
                metadata={"file_name": file_name, "reason": "download failed"},
            )

        if file_uri.startswith("https://"):
            return await prompt_assets.capture_url(
                url=file_uri,
                modality=modality,
                enabled=self.fetch_remote_assets,
            )

        return prompt_assets.capture_reference(
            url=file_uri,
            modality=modality,
            delivery=PROVIDER_INLINE,
            mime_type=mime,
            metadata={"reason": "file_data URI is not retrievable"},
        )

    def _extract_response_text(self, response) -> str:
        """Collect user-visible prose only."""
        text_content = ""
        if not response.candidates:
            return text_content
        for candidate in response.candidates:
            if not candidate.content or not candidate.content.parts:
                continue
            for part in candidate.content.parts:
                if part.text and part.text.strip():
                    text_content += part.text
                if (
                    hasattr(part, "code_execution_result")
                    and part.code_execution_result
                ):
                    output_text = getattr(part.code_execution_result, "output", "")
                    if output_text and output_text.strip():
                        text_content += "\n" + output_text
        return text_content

    def _extract_function_calls(self, response) -> list[dict]:
        calls = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                function_call = getattr(part, "function_call", None)
                if not function_call:
                    continue
                calls.append(
                    {
                        "name": getattr(function_call, "name", ""),
                        "id": getattr(function_call, "id", "")
                        or getattr(function_call, "name", ""),
                        "arguments": parse_tool_arguments(
                            getattr(function_call, "args", None)
                        ),
                    }
                )
        return calls

    async def _run_tool_loop(
        self, *, prompt: EvalPrompt, contents_list: list, config: object, prompt_assets
    ) -> tuple[object, list, list[ToolCallRecord]]:
        response = await self._client.aio.models.generate_content(
            model=self.model_name,
            contents=contents_list,
            config=config,
        )
        output_assets = await self._capture_output_assets(
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
            function_response_parts = []
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
                            provider_call_id=call["id"],
                            arguments=call["arguments"],
                            status="completed",
                            assets=result.assets,
                        )
                    )
                    function_response_parts.append(
                        types.Part.from_function_response(
                            name=call["name"],
                            response={"result": result.as_model_payload()},
                        )
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
                    function_response_parts.append(
                        types.Part.from_function_response(
                            name=call["name"],
                            response={"error": str(exc)},
                        )
                    )
            contents_list.append(
                types.Content(role="tool", parts=function_response_parts)
            )
            response = await self._client.aio.models.generate_content(
                model=self.model_name,
                contents=contents_list,
                config=config,
            )
            output_assets.extend(
                await self._capture_output_assets(prompt, response, prompt_assets)
            )
        return response, output_assets, tool_calls

    async def _send_impl(self, prompt: EvalPrompt) -> ProviderResponse:
        uploaded_files: list[object] = []
        prompt_assets = self.new_prompt_assets(prompt)

        input_support = evaluate_input_support(
            provider=self.provider_name,
            base_url=self._client._api_client._http_options.base_url or "",
            prompt=prompt,
        )
        set_current_input_support(input_support)

        try:
            contents_list: list = []

            for fname in prompt.input_files:
                fpath = resolve_input_file(fname, root=INPUT_FILES_DIR)
                if not fpath.exists():
                    raise FileNotFoundError(
                        f"Missing input media file for prompt {prompt.prompt_id}: {fpath}"
                    )

                if self._inline_input_files:
                    mime_type = (
                        mimetypes.guess_type(str(fpath))[0]
                        or "application/octet-stream"
                    )
                    if mime_type == "text/csv":
                        mime_type = "text/plain"
                    contents_list.append(
                        types.Part.from_bytes(
                            data=fpath.read_bytes(), mime_type=mime_type
                        )
                    )
                else:
                    uploaded = await self._upload_file(fpath)
                    if not uploaded:
                        raise RuntimeError(
                            f"input file upload failed for prompt {prompt.prompt_id}: {fname}"
                        )
                    uploaded_files.append(uploaded)
                    contents_list.append(uploaded)

            if "Text" in prompt.input_modalities:
                contents_list.append(prompt.prompt_text)

            gen_config_kwargs: dict = {
                "max_output_tokens": 32768,
            }
            if self._tool_defs:
                gen_config_kwargs["tools"] = self._tool_defs
            config = types.GenerateContentConfig(**gen_config_kwargs)
            request_config = dict(gen_config_kwargs)
            if self._tool_defs:
                request_config["tools"] = list(self._tools)

            response, output_assets, tool_calls = await self._run_tool_loop(
                prompt=prompt,
                contents_list=contents_list,
                config=config,
                prompt_assets=prompt_assets,
            )

            request = RequestRecord(
                provider=self.provider_name,
                api="generate_content",
                model=self.model_name,
                user_prompt=prompt.prompt_text,
                input_files=list(prompt.input_files),
                input_modalities=list(prompt.input_modalities),
                output_modalities=list(prompt.output_modalities),
                tools=list(self._tools),
                response_modalities=list(self._response_modalities),
                max_output_tokens=32768,
                provider_request={
                    "contents": [
                        {"type": "input_file", "filename": fname}
                        for fname in prompt.input_files
                    ]
                    + (
                        [{"type": "text", "text": prompt.prompt_text}]
                        if "Text" in prompt.input_modalities
                        else []
                    ),
                    "config": request_config,
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
            for uf in uploaded_files:
                name = getattr(uf, "name", None)
                if name:
                    try:
                        await asyncio.wait_for(
                            asyncio.to_thread(self._delete_file_sync, name),
                            timeout=10,
                        )
                    except TimeoutError:
                        logger.warning("File delete timed out for %s", name)
