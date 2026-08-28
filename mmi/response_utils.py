# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Utilities for extracting judge context from saved eval results."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

from google.genai import types

from .config import INPUT_FILES_DIR, RESULTS_DIR
from .detection import CAPTURED
from .input_files import InvalidInputFilePath, resolve_input_file
from .models import EvalPrompt, RubricCriterion

#: Inline request payloads are bounded by the provider, so bytes above this are
#: refused before any grading happens.
_MAX_MEDIA_BYTES = 20 * 1024 * 1024

#: What an input artifact may be attached as. Deliberately narrower than what an
#: answer artifact is attached as: an input is context, so a type the judge
#: rejects would fail a criterion that never needed it. Skips are recorded.
_INPUT_MEDIA_PREFIXES = ("image/", "audio/", "video/", "text/")
_INPUT_MEDIA_MIMES = frozenset({"application/pdf"})


def _sendable_as_input(mime: str | None) -> bool:
    if not mime:
        return False
    return mime.startswith(_INPUT_MEDIA_PREFIXES) or mime in _INPUT_MEDIA_MIMES


def _rebase_under_results(path: Path) -> Path | None:
    """Reinterpret an absolute asset path as one under the local ``RESULTS_DIR``.

    ``local_path`` is recorded absolute, so results produced on one machine
    cannot be re-scored on another: every artifact would silently miss and the
    judge would grade blind. The run layout is
    ``RESULTS_DIR/<config>/<timestamp>_<model>_assets/<prompt>/<file>``, so the
    ``*_assets`` directory locates the config-relative tail of a foreign path.
    """
    for i, part in enumerate(path.parts):
        if part.endswith("_assets") and i > 0:
            candidate = RESULTS_DIR.joinpath(*path.parts[i - 1 :])
            return candidate if candidate.exists() else None
    return None


def resolve_asset_path(path_value: str) -> Path | None:
    """The readable file for a recorded ``local_path``, or None if there is none."""
    if not path_value:
        return None
    path = Path(path_value)
    if path.exists():
        return path
    if path.is_absolute():
        return _rebase_under_results(path)
    candidate = RESULTS_DIR / path
    return candidate if candidate.exists() else None


def unsendable_reason(asset: dict[str, Any]) -> str | None:
    """Why a captured artifact's bytes cannot reach the judge, else None.

    Only local, structural obstacles count here. Whether the judge *understands*
    a type is not decided in advance: the bytes are attached and a provider that
    refuses them fails the call, which is recorded as an evaluator error. That
    keeps the harness judge-agnostic, so a judge that accepts more types needs no
    change here.
    """
    path_value = asset.get("local_path") or ""
    if not path_value:
        return "no file path was recorded for the artifact"
    path = resolve_asset_path(path_value)
    if path is None:
        return f"the artifact file is not readable at {path_value}"
    size = path.stat().st_size
    if size > _MAX_MEDIA_BYTES:
        return (
            f"the artifact is {size} bytes, above the {_MAX_MEDIA_BYTES}-byte "
            "inline request limit"
        )
    if not (asset.get("mime_type") or mimetypes.guess_type(str(path))[0]):
        return "the artifact has no MIME type to send it under"
    return None


def _extract_text_from_openai_output(output: Any) -> str:
    chunks: list[str] = []
    if not isinstance(output, list):
        return ""

    for item in output:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text)
        if item.get("type") == "message" and isinstance(item.get("text"), str):
            chunks.append(item["text"])
    return "\n".join(chunks).strip()


def extract_response_text(result_dict: dict) -> str:
    """Extract response text from normalized or provider-specific result rows."""
    response_text = result_dict.get("response_text")
    if isinstance(response_text, str) and response_text.strip():
        return response_text

    raw_response = result_dict.get("raw_response")
    if raw_response is None:
        return ""

    if isinstance(raw_response, str):
        try:
            parsed = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError):
            return raw_response.strip()
        raw_response = parsed

    if not isinstance(raw_response, dict):
        return str(raw_response).strip()

    for key in ("text", "output_text", "content"):
        value = raw_response.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    text = _extract_text_from_openai_output(raw_response.get("output"))
    if text:
        return text

    choices = raw_response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()

    try:
        parts = (
            raw_response.get("candidates", [{}])[0].get("content", {}).get("parts")
            or []
        )
        text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if text.strip():
            return text.strip()
    except (AttributeError, IndexError):
        pass

    return ""


def _normalize_modality(value: Any) -> str:
    modality = str(value or "").strip().lower()
    return {
        "text": "Text",
        "image": "Image",
        "audio": "Audio",
        "video": "Video",
        "document": "Document",
    }.get(modality, "")


def parse_rubric_criteria(value: Any) -> list[RubricCriterion]:
    """Parse the dataset's structured rubric JSON with explicit modalities."""
    if value is None:
        return []
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value:
        return []

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("rubrics must be structured JSON") from exc
    criteria_raw = parsed.get("criteria") if isinstance(parsed, dict) else None
    if not isinstance(criteria_raw, list):
        raise ValueError("rubrics JSON must contain a criteria list")

    criteria: list[RubricCriterion] = []
    for index, item in enumerate(criteria_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"rubric criterion {index} must be an object")
        criterion = str(item.get("criterion", "")).strip()
        if not criterion:
            raise ValueError(f"rubric criterion {index} is missing criterion text")
        modality = _normalize_modality(item.get("modality"))
        if not modality:
            raise ValueError(
                f"rubric criterion {index} has invalid modality: {item.get('modality')!r}"
            )
        criteria.append(
            RubricCriterion(
                id=str(item.get("id") or index),
                criterion=criterion,
                modality=modality,
            )
        )
    return criteria


def build_output_media_parts(
    output_assets: list[dict[str, Any]] | None,
) -> list[types.Part]:
    """Media parts for captured model output assets.

    Every artifact we hold bytes for is attached under its own MIME type. A type
    the judge cannot read is not filtered out here; the call fails and is
    recorded, rather than the artifact vanishing from a grade that still claims
    to have judged it.
    """
    parts: list[types.Part] = []
    for asset in output_assets or []:
        if asset.get("capture_status") != CAPTURED:
            continue
        if unsendable_reason(asset):
            continue
        path = resolve_asset_path(asset.get("local_path") or "")
        if path is None:
            continue
        mime_type = asset.get("mime_type") or mimetypes.guess_type(str(path))[0] or ""
        parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type))
    return parts


def output_asset_summary(output_assets: list[dict[str, Any]] | None) -> str:
    rows = []
    for asset in output_assets or []:
        rows.append(
            "- "
            f"{asset.get('asset_id', 'unknown')}: modality={asset.get('modality', 'unknown')}, "
            f"mime={asset.get('mime_type', 'unknown')}, status={asset.get('capture_status', 'unknown')}, "
            f"bytes={asset.get('size_bytes', 0)}, path={asset.get('local_path', '')}, "
            f"url={asset.get('source_url', '')}, error={asset.get('error', '')}"
        )
    return "\n".join(rows) if rows else "(none)"


def build_input_media_parts(
    prompt: EvalPrompt,
) -> tuple[list[types.Part], list[str]]:
    """Media parts for the artifacts the model was given, and what was skipped.

    Returns ``(parts, skipped_filenames)``. An input is context for grading, not
    the thing being graded, so anything that cannot be attached is dropped and
    named rather than failing the criterion. The names travel with the grade so a
    rubric that compares against an input can be audited afterwards.
    """
    parts: list[types.Part] = []
    skipped: list[str] = []
    for fname in prompt.input_files:
        try:
            path = resolve_input_file(fname, root=INPUT_FILES_DIR)
        except (InvalidInputFilePath, FileNotFoundError):
            skipped.append(fname)
            continue
        if path.stat().st_size > _MAX_MEDIA_BYTES:
            skipped.append(fname)
            continue
        mime_type = mimetypes.guess_type(str(path))[0]
        if not _sendable_as_input(mime_type):
            skipped.append(fname)
            continue
        parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type))
    return parts, skipped


def media_file_summary(prompt: EvalPrompt) -> str:
    rows = []
    for fname in prompt.input_files:
        try:
            path = resolve_input_file(fname, root=INPUT_FILES_DIR)
        except (InvalidInputFilePath, FileNotFoundError):
            path = None
        mime_type = mimetypes.guess_type(fname)[0] or "unknown"
        exists = path is not None
        size = path.stat().st_size if path is not None else 0
        rows.append(f"- {fname}: mime={mime_type}, exists={exists}, bytes={size}")
    return "\n".join(rows) if rows else "(none)"
