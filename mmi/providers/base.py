# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Abstract base class for all LLM providers."""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod

from ..input_support import get_current_input_support, mark_observed_error
from ..models import EvalPrompt, ProviderResponse, RequestRecord
from ..output_assets import OutputAssetManager
from ..retry import is_retryable

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Provider failure with optional raw response/request context."""

    def __init__(self, message: str, *, raw_response=None, request=None):
        super().__init__(message)
        self.raw_response = raw_response
        self.request = request


_MAX_BACKOFF = 30


class BaseProvider(ABC):
    """Abstract provider that sends an EvalPrompt and returns a ProviderResponse."""

    provider_name: str
    model_name: str
    run_name: str

    def __init__(
        self,
        *,
        max_retries: int = 5,
        retry_backoff: int = 4,
        fetch_remote_assets: bool = False,
    ):
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._asset_manager = OutputAssetManager(None)
        # Driven by rubric_enabled: a rubric grades an artifact's content, and
        # there is nothing to grade without the bytes. Retrieval never changes
        # provenance — delivery stays external_url, so a fetched URL cannot
        # become native credit.
        self.fetch_remote_assets = fetch_remote_assets

    def set_output_asset_dir(self, root_dir) -> None:
        self._asset_manager = OutputAssetManager(root_dir)

    def new_prompt_assets(self, prompt: EvalPrompt):
        """One asset manager per prompt, shared by parsing, tools and URL capture.

        Creating a second manager mid-prompt would restart the ID counter and
        collide asset IDs across tool-loop rounds.
        """
        return self._asset_manager.for_prompt(prompt.prompt_id)

    @staticmethod
    def json_raw_response(response) -> object:
        """Serialize a provider response to plain JSON types.

        ``model_dump()`` alone leaves bytes and datetimes in place, which later
        fail ``json.dumps`` and fall back to ``str()`` — that fallback is what
        produced the Python-repr ``raw_response`` values in the historical
        corpus. ``mode="json"`` is the fix.
        """
        if hasattr(response, "model_dump"):
            try:
                return response.model_dump(mode="json")
            except TypeError:
                return response.model_dump()
        if hasattr(response, "to_json_dict"):
            return response.to_json_dict()
        return response

    async def finalize(
        self,
        *,
        prompt: EvalPrompt,
        prompt_assets,
        response_text: str,
        output_assets: list,
        tool_calls: list,
        raw_response,
        request: RequestRecord | None,
    ) -> ProviderResponse:
        """Assemble the provider contract.

        URL capture runs over user-visible prose only. Whether a URL is
        downloaded is irrelevant to the score — the asset stays URL-delivered
        either way.
        """
        output_assets = list(output_assets)
        output_assets.extend(
            await prompt_assets.capture_urls(
                response_text, fetch_enabled=self.fetch_remote_assets
            )
        )
        return ProviderResponse(
            prompt_id=prompt.prompt_id,
            run_name=self.run_name,
            provider=self.provider_name,
            model=self.model_name,
            response_text=response_text,
            raw_response=raw_response,
            request=request,
            output_assets=output_assets,
            tool_calls=tool_calls,
        )

    @abstractmethod
    async def _send_impl(self, prompt: EvalPrompt) -> ProviderResponse:
        """Provider-specific implementation. Subclasses override this."""
        ...

    async def send(self, prompt: EvalPrompt) -> ProviderResponse:
        """Send a prompt with retry + exponential backoff + jitter on transient errors."""
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return await self._send_impl(prompt)
            except Exception as exc:
                last_error = exc
                retryable = is_retryable(exc)
                if not retryable or attempt == self._max_retries:
                    logger.error(
                        "%s prompt %s failed (attempt %d/%d): %s",
                        self.provider_name,
                        prompt.prompt_id,
                        attempt,
                        self._max_retries,
                        exc,
                    )
                    raw_response = getattr(exc, "raw_response", None)
                    request = getattr(exc, "request", None)
                    input_support = get_current_input_support()
                    if input_support is not None:
                        marked_support = mark_observed_error(input_support, exc)
                        if isinstance(request, RequestRecord):
                            request.provider_request.setdefault(
                                "input_support", marked_support.to_dict()
                            )
                        else:
                            request = RequestRecord(
                                provider=self.provider_name,
                                api="unknown",
                                model=self.model_name,
                                user_prompt=prompt.prompt_text,
                                input_files=list(prompt.input_files),
                                input_modalities=list(prompt.input_modalities),
                                output_modalities=list(prompt.output_modalities),
                                provider_request={
                                    "input_support": marked_support.to_dict()
                                },
                            )
                    return ProviderResponse(
                        prompt_id=prompt.prompt_id,
                        run_name=self.run_name,
                        provider=self.provider_name,
                        model=self.model_name,
                        is_error=True,
                        error_type=getattr(exc, "error_type", "") or "provider_error",
                        raw_response=raw_response,
                        error=str(exc),
                        request=request,
                    )
                base_wait = self._retry_backoff * (2 ** (attempt - 1))
                jitter = random.uniform(0, base_wait * 0.5)
                wait = min(base_wait + jitter, _MAX_BACKOFF)
                logger.warning(
                    "%s prompt %s: retryable error (attempt %d/%d), waiting %.1fs: %s",
                    self.provider_name,
                    prompt.prompt_id,
                    attempt,
                    self._max_retries,
                    wait,
                    exc,
                )
                await asyncio.sleep(wait)

        return ProviderResponse(
            prompt_id=prompt.prompt_id,
            run_name=self.run_name,
            provider=self.provider_name,
            model=self.model_name,
            is_error=True,
            error_type="provider_error",
            raw_response=getattr(last_error, "raw_response", None),
            error=str(last_error),
            request=getattr(last_error, "request", None),
        )
