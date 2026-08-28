# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Output asset capture utilities for generated model artifacts."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .detection import (
    CAPTURED,
    EXTERNAL_URL,
    FAILED,
    PROVIDER_INLINE,
    REFERENCE_ONLY,
    SKIPPED,
    classify_mime,
)
from .fetch import fetch_url, redact_url
from .models import CapturedAsset
from .url_modality_detector import classify_url, extract_urls

_MAX_URL_BYTES = 500 * 1024 * 1024
_URL_TIMEOUT = 180.0
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(value: str) -> str:
    cleaned = _SAFE_NAME.sub("_", value.strip())
    return cleaned.strip("._") or "asset"


def modality_from_mime(mime_type: str) -> str:
    """Best-effort modality for filesystem naming only.

    This is **not** the scoring table. Scoring goes through
    :func:`mmi.detection.classify_mime`, which refuses to classify what it does
    not recognise instead of defaulting to Document.
    """
    return classify_mime(mime_type) or "Document"


def extension_for_mime(mime_type: str, default: str = ".bin") -> str:
    ext = mimetypes.guess_extension(mime_type.split(";", 1)[0].strip())
    if ext == ".jpe":
        return ".jpg"
    return ext or default


def decode_base64_data(data: Any) -> bytes | None:
    if data is None:
        return None
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        payload = data
        if payload.startswith("data:") and "," in payload:
            payload = payload.split(",", 1)[1]
        try:
            return base64.b64decode(payload, validate=False)
        except Exception:
            return None
    return None


class OutputAssetManager:
    """Writes generated output assets under a run-scoped directory."""

    def __init__(self, root_dir: Path | None):
        self.root_dir = root_dir

    def for_prompt(self, prompt_id: str) -> "PromptAssetManager":
        return PromptAssetManager(self.root_dir, prompt_id)


class PromptAssetManager:
    """One asset manager per prompt.

    A single instance must be shared across response parsing, tool execution
    and URL capture for that prompt, so asset IDs cannot collide.
    """

    def __init__(self, root_dir: Path | None, prompt_id: str):
        self.root_dir = root_dir
        self.prompt_id = prompt_id
        self._counter = 0

    def _next_path(
        self, modality: str, mime_type: str, filename: str = ""
    ) -> tuple[str, Path | None]:
        asset_id = f"{safe_name(self.prompt_id)}_output_{self._counter}"
        self._counter += 1
        if self.root_dir is None:
            return asset_id, None
        ext = Path(filename).suffix if filename else extension_for_mime(mime_type)
        path = (
            self.root_dir
            / safe_name(self.prompt_id)
            / f"{asset_id}_{modality.lower()}{ext}"
        )
        return asset_id, path

    def capture_bytes(
        self,
        *,
        data: bytes | None,
        modality: str,
        source_type: str,
        delivery: str = PROVIDER_INLINE,
        mime_type: str = "",
        modality_hint: str = "",
        filename: str = "",
        source_url: str = "",
        metadata: dict[str, Any] | None = None,
        error: str = "",
    ) -> CapturedAsset:
        asset_id, path = self._next_path(modality, mime_type, filename)
        if not data:
            return CapturedAsset(
                asset_id=asset_id,
                prompt_id=self.prompt_id,
                modality=modality,
                source_type=source_type,
                delivery=delivery,
                mime_type=mime_type,
                modality_hint=modality_hint,
                source_url=source_url,
                capture_status=FAILED if error else SKIPPED,
                error=error or "no asset bytes available",
                metadata=metadata or {},
            )
        sha = hashlib.sha256(data).hexdigest()
        local_path = ""
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            local_path = str(path)
        return CapturedAsset(
            asset_id=asset_id,
            prompt_id=self.prompt_id,
            modality=modality,
            source_type=source_type,
            delivery=delivery,
            mime_type=mime_type,
            modality_hint=modality_hint,
            local_path=local_path,
            source_url=source_url,
            sha256=sha,
            size_bytes=len(data),
            capture_status=CAPTURED,
            metadata=metadata or {},
        )

    async def capture_urls(
        self, text: str, *, fetch_enabled: bool = False
    ) -> list[CapturedAsset]:
        assets: list[CapturedAsset] = []
        for url in extract_urls(text):
            if not url.startswith("https://"):
                continue
            modality = classify_url(url)
            if modality is None:
                continue
            assets.append(
                await self.capture_url(
                    url=url, modality=modality, enabled=fetch_enabled
                )
            )
        return assets

    def capture_reference(
        self,
        *,
        url: str,
        modality: str,
        delivery: str = PROVIDER_INLINE,
        mime_type: str = "",
        modality_hint: str = "",
        filename: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CapturedAsset:
        """Record an artifact we can point at but hold no bytes for.

        Status is ``reference_only``, never ``captured``: pointing at media is
        not the same as producing it, and the metric must be able to tell them
        apart.
        """
        asset_id, _ = self._next_path(modality, mime_type, filename)
        return CapturedAsset(
            asset_id=asset_id,
            prompt_id=self.prompt_id,
            modality=modality,
            source_type="reference_url",
            delivery=delivery,
            mime_type=mime_type,
            modality_hint=modality_hint,
            source_url=url,
            capture_status=REFERENCE_ONLY,
            metadata=metadata or {},
        )

    async def capture_url(
        self, *, url: str, modality: str, enabled: bool = False
    ) -> CapturedAsset:
        """Optionally download a URL-delivered artifact.

        The delivery provenance stays ``external_url`` whether or not the bytes
        arrive, so downloading can never promote URL evidence to native.
        """
        filename = Path(urlparse(url).path).name
        if not enabled:
            return self.capture_reference(
                url=redact_url(url),
                modality=modality,
                delivery=EXTERNAL_URL,
                filename=filename,
                metadata={"reason": "remote fetching disabled"},
            )
        asset_id, _ = self._next_path(modality, "", filename)
        try:
            fetched = await fetch_url(url)
        except Exception as exc:
            return CapturedAsset(
                asset_id=asset_id,
                prompt_id=self.prompt_id,
                modality=modality,
                source_type="url",
                delivery=EXTERNAL_URL,
                source_url=redact_url(url),
                capture_status=FAILED,
                error=str(exc),
            )
        self._counter -= 1
        return self.capture_bytes(
            data=fetched.data,
            modality=classify_mime(fetched.mime_type) or modality,
            source_type="url",
            delivery=EXTERNAL_URL,
            mime_type=fetched.mime_type,
            filename=filename,
            source_url=fetched.url,
            metadata={
                "final_url": fetched.final_url,
                "status_code": fetched.status_code,
            },
        )
