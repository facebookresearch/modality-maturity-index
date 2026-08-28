# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Route-level input support metadata for rerun planning."""

from __future__ import annotations

import mimetypes
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import INPUT_FILES_DIR
from .input_files import resolve_input_file
from .models import EvalPrompt

# Advisory public-provider input capability metadata. Providers absent from this
# table get no preflight opinion at all.
PROVIDER_INPUT_MODALITIES: dict[str, set[str]] = {
    "openai": {"Text", "Image", "Document"},
    "anthropic": {"Text", "Image", "Document"},
    "gemini": {"Text", "Image", "Audio", "Video", "Document"},
}

UNSUPPORTED_INPUT_MIMES: dict[str, set[str]] = {
    "gemini": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    },
}


@dataclass
class UnsupportedInputFile:
    filename: str
    mime_type: str
    modality: str
    reason: str


@dataclass
class InputSupport:
    """Advisory preflight support plus observed provider outcome."""

    route: str = ""
    preflight_status: str = "supported"
    observed_status: str = "not_observed"
    unsupported_modalities: list[str] = field(default_factory=list)
    unsupported_files: list[dict[str, Any]] = field(default_factory=list)
    rerun_recommended: bool = False
    rerun_reason: str = ""
    provider_error_type: str = ""
    provider_error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def route_name(provider: str, base_url: str) -> str:
    return f"native_{provider}" if not base_url else f"custom_{provider}"


# Request-local so concurrent prompts sharing one provider instance cannot
# observe each other's input-support state.
_current_input_support: ContextVar[InputSupport | None] = ContextVar(
    "mmi_current_input_support", default=None
)


def set_current_input_support(support: InputSupport | None) -> None:
    _current_input_support.set(support)


def get_current_input_support() -> InputSupport | None:
    return _current_input_support.get()


def _file_modality(mime_type: str) -> str:
    if mime_type.startswith("image/"):
        return "Image"
    if mime_type.startswith("audio/"):
        return "Audio"
    if mime_type.startswith("video/"):
        return "Video"
    return "Document"


def evaluate_input_support(
    *,
    provider: str,
    base_url: str,
    prompt: EvalPrompt,
) -> InputSupport:
    """Classify expected route support without preventing the request.

    Purely advisory. Nothing here gates a request or removes a prompt from any
    denominator — unsupported inputs still produce a scored (error) result.
    """

    route = route_name(provider, base_url)
    unsupported_files: list[UnsupportedInputFile] = []
    unsupported_modalities: set[str] = set()

    supported_modalities = PROVIDER_INPUT_MODALITIES.get(provider)
    if supported_modalities is None:
        return InputSupport(route=route)
    for modality in prompt.input_modalities:
        if modality not in supported_modalities:
            unsupported_modalities.add(modality)

    for fname in prompt.input_files:
        path = resolve_input_file(fname, root=INPUT_FILES_DIR, must_exist=False)
        mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        modality = _file_modality(mime_type)
        reason = ""
        if modality not in supported_modalities:
            reason = "unsupported_input_modality"
            unsupported_modalities.add(modality)
        elif mime_type in UNSUPPORTED_INPUT_MIMES.get(provider, set()):
            reason = "unsupported_file_mime"
        elif provider in {"openai", "anthropic"} and modality == "Document":
            allowed = mime_type.startswith("text/") or mime_type in {
                "application/pdf",
                "text/csv",
            }
            if not allowed:
                reason = "unsupported_file_mime"
        if reason:
            unsupported_files.append(
                UnsupportedInputFile(
                    filename=fname,
                    mime_type=mime_type,
                    modality=modality,
                    reason=reason,
                )
            )

    if not unsupported_modalities and not unsupported_files:
        return InputSupport(route=route)

    reason = (
        "unsupported_input_modality"
        if unsupported_modalities
        else "unsupported_file_mime"
    )
    return InputSupport(
        route=route,
        preflight_status="unsupported",
        unsupported_modalities=sorted(unsupported_modalities),
        unsupported_files=[asdict(item) for item in unsupported_files],
        rerun_recommended=True,
        rerun_reason=reason,
    )


def mark_observed_success(input_support: InputSupport) -> InputSupport:
    input_support.observed_status = "supported"
    input_support.rerun_recommended = False
    input_support.rerun_reason = ""
    input_support.provider_error_type = ""
    input_support.provider_error_message = ""
    return input_support


def mark_observed_error(
    input_support: InputSupport,
    exc: Exception,
) -> InputSupport:
    """Classify deterministic unsupported-input provider errors for reruns."""

    message = str(exc)
    lower = message.lower()
    input_support.provider_error_message = message[:2000]

    unsupported_markers = [
        "unsupported mime type",
        "unsupported input file type",
        "does not support inline input file type",
        "input_audio",
        "input_video",
        "input tag 'audio'",
        "input tag 'video'",
        "does not match any of the expected tags",
        "mime_type parameter",
        "mimeType parameter",
        "not supported",
    ]
    transport_markers = [
        "/files' was not found",
        '/files" was not found',
        "upload/v1beta/files",
        "input file upload failed",
    ]

    if any(marker.lower() in lower for marker in unsupported_markers):
        input_support.observed_status = "unsupported"
        input_support.rerun_recommended = True
        input_support.rerun_reason = "provider_rejected_input"
        input_support.provider_error_type = "unsupported_input"
        return input_support

    if any(marker.lower() in lower for marker in transport_markers):
        input_support.observed_status = "transport_error"
        input_support.rerun_recommended = True
        input_support.rerun_reason = "input_transport_error"
        input_support.provider_error_type = "input_transport_error"
        return input_support

    input_support.observed_status = "error"
    input_support.provider_error_type = "provider_error"
    return input_support
