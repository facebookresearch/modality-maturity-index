# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""LLM-as-judge for catching false negatives in structural detection (Layer 2).

The judge is a safety net: when Layer 1 (structural parsing) fails to detect
an expected modality, the judge examines the raw response to determine whether
the model *actually produced* that modality and the parser simply missed it.

Uses Gemini via the Google GenAI SDK, talking directly to the Google API with
``GOOGLE_API_KEY``.  Setting ``judge_base_url`` in TOML is opt-in custom routing.

Judge calls are text-only JSON out.  Do not request media generation from the
judge and keep ``response_modalities`` unset for judge calls.  Judge evidence
contributes to lenient scoring only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

import httpx
from google.genai import Client, types

from .config import DEFAULT_JUDGE_MODEL, get_api_key
from .models import DetectionResult, JudgeResult
from .retry import is_retryable

logger = logging.getLogger(__name__)

VALID_MODALITIES = {"Text", "Image", "Audio", "Video", "Document"}

_JUDGE_MAX_RETRIES = 15
_JUDGE_RETRY_BACKOFF = 2  # seconds
_JUDGE_MAX_BACKOFF = 30


def _is_retryable(exc: Exception) -> bool:
    """Transient failures only. A retired model ID must fail fast."""
    return is_retryable(exc, retry_on_bad_json=True)


# The modality definitions below must agree with ``mmi.detection.classify_mime``.
# Scoring takes the union of structural and judge evidence, so a modality the
# judge is told to credit but the table refuses becomes lenient-only credit that
# strict scoring cannot see — which is what HTML did until it was removed here.
JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for a multimodal AI benchmark. Your job is to examine a model's API response and determine which output modalities it genuinely produced.

The five modalities are: Text, Image, Audio, Video, Document.

For each modality, look for concrete evidence the model actually produced it:
- Text: Any substantive text output (not just a refusal or error message).
- Image: Native image data, base64-encoded image content, complete SVG markup, or a URL pointing to a generated image.
- Audio: Native audio data, base64-encoded audio content, or a URL pointing to generated audio.
- Video: Native video data, base64-encoded video content, or a URL pointing to generated video.
- Document: A generated document file such as PDF, DOCX, XLSX, PPTX, CSV, TXT, RTF, ODT or EPUB. HTML, Markdown and JSON are NOT Documents, and neither is prose laid out to look like one.

Do NOT count a modality if the model only wrote ABOUT it. The model must have produced actual media data or file content. Describing what an image would look like, writing a screenplay for a video, or providing lyrics instead of audio are NOT productions of those modalities.

A refusal ("I can't generate images") is NOT a production.

Respond with ONLY valid JSON:
{"detected_modalities": [], "reasoning": "brief explanation"}

The detected_modalities array should list every modality the model genuinely produced. An empty array means the model produced nothing usable. Valid modality names: Text, Image, Audio, Video, Document."""

JUDGE_USER_PROMPT_TEMPLATE = """The model was asked:
"{prompt_text}"

The model's response text was:
---
{text_content}
---

The model's structured/raw API response (JSON, truncated) was:
---
{raw_response_json}
---

Which of the five modalities (Text, Image, Audio, Video, Document) did the model genuinely produce in this response?"""


def _is_modality_detected(detection: DetectionResult, modality: str) -> bool:
    """Check if a modality was detected natively or via URL."""
    mod = detection.modalities.get(modality)
    return bool(mod and (mod.detected_native or mod.detected_via_url))


def get_detected_modalities(
    detection: DetectionResult, expected: list[str]
) -> list[str]:
    """Return expected modalities that were detected (native or URL)."""
    return [m for m in expected if _is_modality_detected(detection, m)]


def get_missing_modalities(
    detection: DetectionResult, expected: list[str]
) -> list[str]:
    """Return expected non-text modalities that were NOT detected."""
    return [
        m for m in expected if m != "Text" and not _is_modality_detected(detection, m)
    ]


def get_unexpected_modalities(
    detection: DetectionResult, expected: list[str]
) -> list[str]:
    """Return non-text modalities detected (native or URL) that are NOT expected."""
    return [
        m
        for m in detection.modalities
        if m != "Text" and m not in expected and _is_modality_detected(detection, m)
    ]


def should_call_judge(
    detection: DetectionResult | None,
    expected_modalities: list[str],
    judge_enabled: bool,
) -> bool:
    """Determine if the judge should be called for this response.

    The judge is called when:
    1. Judge is enabled in config
    2. The response is not an error
    3. At least one expected non-text modality was NOT detected by native
       or URL-based detection, OR at least one non-text modality was
       detected that was NOT expected (unexpected modality).
    """
    if not judge_enabled or detection is None:
        return False
    if detection.is_error:
        return False
    return (
        len(get_missing_modalities(detection, expected_modalities)) > 0
        or len(get_unexpected_modalities(detection, expected_modalities)) > 0
    )


def _format_raw_response_for_judge(raw_response: Any, max_chars: int = 12000) -> str:
    """Convert raw provider response to bounded JSON text for judge context."""
    if raw_response is None:
        return ""

    try:
        if isinstance(raw_response, str):
            normalized = raw_response
        else:
            normalized = json.dumps(raw_response, ensure_ascii=False, sort_keys=True)
    except Exception:
        normalized = str(raw_response)

    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "\n... [truncated]"


class ModalityJudge:
    """LLM-as-judge using Gemini via the Google GenAI SDK.

    Acts as a safety net for false negatives in structural detection.
    When the parser misses an expected modality, the judge examines the
    response to determine if the model actually produced it.

    Talks directly to the Google API.  Pass ``base_url`` only for opt-in
    custom routing.
    """

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        api_key_env: str = "GOOGLE_API_KEY",
        base_url: str = "",
        timeout: int = 120,
    ):
        self._model = model
        httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=60.0),
            trust_env=True,
            follow_redirects=True,
        )
        http_opts: dict = {
            "timeout": timeout * 1000,
            "httpx_async_client": httpx_client,
        }
        if base_url:
            http_opts["base_url"] = base_url
            http_opts["api_version"] = "v1"
        self._client = Client(
            api_key=get_api_key(api_key_env),
            http_options=types.HttpOptions(**http_opts),
        )

    async def evaluate(
        self,
        prompt_text: str,
        text_content: str,
        raw_response: Any = None,
    ) -> JudgeResult:
        """Ask the judge which modalities the model genuinely produced.

        The judge is modality-neutral: it simply examines the response and
        reports every modality it finds.  The caller decides what to do
        with the list (e.g. reconcile with expected vs. detected).

        Args:
            prompt_text: The original user prompt.
            text_content: The model's text response (extracted by Layer 1).
            raw_response: Full provider response payload (structured/JSON).

        Returns:
            JudgeResult whose ``detected_modalities`` lists every modality
            the judge believes the model genuinely produced.
        """
        truncated_text = (
            text_content[:4000] if len(text_content) > 4000 else text_content
        )
        truncated_prompt = prompt_text[:500] if len(prompt_text) > 500 else prompt_text

        raw_response_json = _format_raw_response_for_judge(raw_response)

        user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            prompt_text=truncated_prompt,
            text_content=truncated_text,
            raw_response_json=raw_response_json or "(none)",
        )

        try:
            config = types.GenerateContentConfig(
                system_instruction=JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=32768,
            )

            for attempt in range(1, _JUDGE_MAX_RETRIES + 1):
                try:
                    response = await self._client.aio.models.generate_content(
                        model=self._model,
                        contents=user_prompt,
                        config=config,
                    )
                    raw = response.text or ""
                    break  # success
                except Exception as exc:
                    if not _is_retryable(exc) or attempt == _JUDGE_MAX_RETRIES:
                        raise
                    base_wait = _JUDGE_RETRY_BACKOFF * (2 ** (attempt - 1))
                    jitter = random.uniform(0, base_wait * 0.5)
                    wait = min(base_wait + jitter, _JUDGE_MAX_BACKOFF)
                    logger.warning(
                        "Judge retryable error (attempt %d/%d), waiting %.1fs: %s",
                        attempt,
                        _JUDGE_MAX_RETRIES,
                        wait,
                        exc,
                    )
                    await asyncio.sleep(wait)

            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)

            # Validate modality names — only accept known modalities
            raw_detected = parsed.get("detected_modalities", [])
            validated_detected = [m for m in raw_detected if m in VALID_MODALITIES]

            return JudgeResult(
                detected_modalities=validated_detected,
                reasoning=parsed.get("reasoning", ""),
                judge_model=self._model,
                judge_raw_response=raw,
            )
        except json.JSONDecodeError as exc:
            logger.warning("Judge returned invalid JSON: %s (raw: %s)", exc, raw)
            return JudgeResult(judge_model=self._model, judge_raw_response=raw)
        except Exception as exc:
            logger.warning("Judge call failed: %s", exc)
            return JudgeResult(judge_model=self._model, judge_raw_response=str(exc))
