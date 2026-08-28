# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Rubric-based sample scoring for MMI results."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from typing import Any

import httpx
from google.genai import Client, types

from .config import get_api_key
from .detection import (
    CAPTURED,
    PAYLOAD_ABSENT,
    PAYLOAD_CAPTURE_FAILED,
    PAYLOAD_GRADEABLE,
    PAYLOAD_REFERENCE_ONLY,
    PAYLOAD_SKIPPED,
    PAYLOAD_UNSENDABLE,
    payload_status,
)
from .models import EvalPrompt, RubricCriterion
from .response_utils import (
    build_input_media_parts,
    build_output_media_parts,
    media_file_summary,
    output_asset_summary,
    unsendable_reason,
)
from .retry import is_retryable

logger = logging.getLogger(__name__)

_MAX_RETRIES = 8
_RETRY_BACKOFF = 2
_MAX_BACKOFF = 30

RUBRIC_SYSTEM_PROMPT = """You are a strict grader for one rubric criterion in a multimodal model evaluation.

You will receive the original prompt, expected output modalities, input-file metadata, and exactly one model-produced modality payload for the rubric being graded. Grade only the supplied rubric against only the supplied model-produced modality payload.

Do not infer credit from missing modalities, descriptions of absent media, or any model response content that was not supplied to you. If the supplied payload is insufficient to satisfy the rubric, score it 0.0.

Return only valid JSON with this schema:
{"score": 0.0, "explanation": "brief reason"}

Scoring:
- 1.0 means the criterion is fully satisfied.
- 0.0 means the criterion is not satisfied.
- Use partial credit between 0.0 and 1.0 when the criterion is partly satisfied.
"""

RUBRIC_USER_PROMPT_TEMPLATE = """# Original prompt
{prompt_text}

# Expected output modalities
{expected_modalities}

# Input files
{input_files}

# Rubric modality being graded
{modality}

# Rubric criterion
{rubric}

# Supplied model-produced payload
{payload_summary}

Evaluate only this rubric criterion using only the supplied {modality} payload. Return only JSON."""

#: Labels for the attached media. Without them the judge receives several
#: same-modality artifacts as an unordered heap and cannot tell the artifact it
#: is grading from the artifact the model was given, which for an image-to-image
#: prompt means it may grade the input as if the model had produced it.
_ANSWER_MEDIA_LABEL = (
    "# Attached below: the model-produced {modality} artifact. This is the "
    "artifact to grade."
)
_INPUT_MEDIA_LABEL = (
    "# Attached below: the artifact(s) the model was given as prompt input. "
    "Context only. Never grade these as model output."
)


@dataclass
class RubricScoreResult:
    rubric_used: bool
    rubric_score: float | None
    rubric_grades: list[dict[str, Any]]
    rubric_judge_model: str
    evaluator_errors: int = 0

    @property
    def rubric_score_by_modality(self) -> dict[str, float]:
        """The within-modality means that ``rubric_score`` averages."""
        return score_by_modality(self.rubric_grades)

    def as_result_fields(self) -> dict[str, Any]:
        return {
            "rubric_used": self.rubric_used,
            "rubric_score": self.rubric_score,
            "rubric_grades": self.rubric_grades,
            "rubric_score_by_modality": self.rubric_score_by_modality,
            "rubric_binary_by_modality": collapse_to_binary(self.rubric_grades),
            "rubric_evaluator_errors": self.evaluator_errors,
            "rubric_judge_model": self.rubric_judge_model,
        }


#: A criterion counts as satisfied only at a full score. Partial credit is
#: recorded but does not survive the collapse.
CORRECT_THRESHOLD = 1.0


def _group_by_modality(grades: list[dict[str, Any]]) -> dict[str, list[float]]:
    by_modality: dict[str, list[float]] = {}
    for grade in grades:
        modality = grade.get("modality", "")
        by_modality.setdefault(modality, []).append(float(grade.get("score") or 0.0))
    return by_modality


def collapse_to_binary(grades: list[dict[str, Any]]) -> dict[str, int]:
    """Collapse per-criterion scores to one binary value per modality.

    ``results.tex``: "we collapse the judge scores to binary values by
    assigning a score of 1 if the judge marked correct on **all** rubrics and
    to 0 otherwise." This is what the published judge--human agreement was
    computed over, because the human scores it is compared against are binary.

    The per-criterion grades are kept alongside; this is an additional view,
    not a replacement.
    """
    return {
        modality: int(all(score >= CORRECT_THRESHOLD for score in scores))
        for modality, scores in _group_by_modality(grades).items()
        if modality
    }


def score_by_modality(grades: list[dict[str, Any]]) -> dict[str, float]:
    """Mean criterion score within each modality."""
    return {
        modality: sum(scores) / len(scores)
        for modality, scores in _group_by_modality(grades).items()
    }


def macro_score(grades: list[dict[str, Any]]) -> float | None:
    """The MMI Value for one prompt: mean over modalities of the within-modality mean.

    Averaging within a modality before averaging across them keeps every
    requested modality equally weighted, so a modality does not gain influence
    by having more criteria written for it. The flat mean over all criteria
    would instead weight each modality by its criterion count, which breaks the
    bound relied on in ``benchmark.tex``: a model that returns only the
    criterion-rich modalities would score above its own Modality Presence
    Score.
    """
    per_modality = score_by_modality(grades)
    if not per_modality:
        return None
    return sum(per_modality.values()) / len(per_modality)


def is_evaluator_error(grade: dict[str, Any]) -> bool:
    """Whether a zero came from the evaluator failing rather than the model.

    Ledger 5: infrastructure failures must never be silently collapsed into
    ordinary zeros. They score as zero *and* stay identifiable.

    Carried as a structured flag rather than inferred from the explanation text,
    so aggregation cannot depend on prose a future prompt change might reword.
    """
    return bool(grade.get("evaluator_error"))


def _is_retryable(exc: Exception) -> bool:
    """Transient failures only. A retired model ID must fail fast."""
    return is_retryable(exc, retry_on_bad_json=True)


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _clean_json_text(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


def _validate_grade(
    *,
    grade: dict[str, Any],
    index: int,
    criterion: RubricCriterion,
) -> dict[str, Any]:
    if grade.get("index") != index:
        raise ValueError(
            f"Rubric judge returned grade for index {grade.get('index')} while grading {index}"
        )
    if grade.get("id") != criterion.id:
        raise ValueError(
            f"Rubric judge returned grade id {grade.get('id')!r} while grading {criterion.id!r}"
        )
    if grade.get("rubric") != criterion.criterion:
        raise ValueError("Rubric judge returned a grade for a different criterion")
    if grade.get("modality") != criterion.modality:
        raise ValueError(
            f"Rubric judge returned modality {grade.get('modality')!r} while grading {criterion.modality!r}"
        )
    return grade


#: How a missing payload is explained to a reader of the grades. Keyed by the
#: status ``mmi.detection.payload_status`` returns, so the wording cannot drift
#: from the classification it describes.
_PAYLOAD_EXPLANATIONS = {
    PAYLOAD_ABSENT: (
        "No model-produced {modality} payload was available for this rubric."
    ),
    PAYLOAD_REFERENCE_ONLY: (
        "The model pointed at a {modality} artifact but its bytes were never "
        "retrieved, so its content could not be graded."
    ),
    PAYLOAD_CAPTURE_FAILED: (
        "Retrieving the {modality} artifact failed, so its content could not be "
        "graded. This is a harness failure, not a model failure."
    ),
    PAYLOAD_SKIPPED: (
        "A {modality} artifact was recorded without bytes, so its content could "
        "not be graded."
    ),
}


def _assets_for_modality(
    output_assets: list[dict[str, Any]] | None,
    modality: str,
) -> tuple[list[dict[str, Any]], str]:
    """The gradeable assets of one modality, plus why there are none if so.

    A rubric grades content, so an artifact whose bytes we do not hold cannot be
    graded. It still scores zero — excluding it would inflate the result — but
    the reason travels with the grade instead of being flattened into "the model
    produced nothing". Selecting on ``CAPTURED`` is data selection; what the
    other statuses *mean* is decided by ``mmi.detection.payload_status``.
    """
    of_modality = [
        asset for asset in output_assets or [] if asset.get("modality") == modality
    ]
    gradeable = [
        asset for asset in of_modality if asset.get("capture_status") == CAPTURED
    ]
    status = payload_status(
        [asset.get("capture_status") or "" for asset in of_modality]
    )
    return gradeable, status


def _zero_grade(
    *,
    index: int,
    criterion: RubricCriterion,
    explanation: str,
    evaluator_error: bool = False,
    payload: str = PAYLOAD_GRADEABLE,
) -> dict[str, Any]:
    return {
        "index": index,
        "id": criterion.id,
        "rubric": criterion.criterion,
        "modality": criterion.modality,
        "score": 0.0,
        "explanation": explanation,
        "evaluator_error": evaluator_error,
        "payload_status": payload,
    }


def _note_input_skips(grade: dict[str, Any], skipped: list[str]) -> dict[str, Any]:
    """Record prompt inputs that could not be attached to the grading call.

    A criterion that compares the output against an input is graded differently
    when the input is absent, so the absence is recorded rather than leaving the
    two cases indistinguishable in the results.
    """
    if skipped:
        grade["input_media_skipped"] = skipped
    return grade


class RubricJudge:
    """Gemini-backed rubric judge using the same routing style as ModalityJudge."""

    def __init__(
        self,
        model: str,
        api_key_env: str,
        base_url: str = "",
        timeout: int = 180,
    ):
        self.model = model
        httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=60.0),
            trust_env=True,
            follow_redirects=True,
        )
        http_opts: dict[str, Any] = {
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

    async def judge_rubric(
        self,
        *,
        prompt: EvalPrompt,
        index: int,
        criterion: RubricCriterion,
        response_text: str,
        output_assets: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        modality = criterion.modality
        if modality == "Text":
            payload_summary = response_text.strip()[:12000]
            contents: list[Any] = [
                RUBRIC_USER_PROMPT_TEMPLATE.format(
                    prompt_text=prompt.prompt_text[:4000],
                    expected_modalities=", ".join(prompt.output_modalities),
                    input_files=media_file_summary(prompt),
                    modality=modality,
                    rubric=criterion.criterion,
                    payload_summary=payload_summary,
                )
            ]
        else:
            payload_summary = output_asset_summary(output_assets)
            contents = [
                RUBRIC_USER_PROMPT_TEMPLATE.format(
                    prompt_text=prompt.prompt_text[:4000],
                    expected_modalities=", ".join(prompt.output_modalities),
                    input_files=media_file_summary(prompt),
                    modality=modality,
                    rubric=criterion.criterion,
                    payload_summary=payload_summary,
                )
            ]
            answer_parts = build_output_media_parts(output_assets)
            if answer_parts:
                contents.append(_ANSWER_MEDIA_LABEL.format(modality=modality))
                contents.extend(answer_parts)

        input_parts, input_skipped = build_input_media_parts(prompt)
        if input_parts:
            contents.append(_INPUT_MEDIA_LABEL)
            contents.extend(input_parts)

        config = types.GenerateContentConfig(
            system_instruction=RUBRIC_SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=2048,
        )

        raw = ""
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                raw = response.text or ""
                parsed = json.loads(_clean_json_text(raw))
                return _note_input_skips(
                    {
                        "index": index,
                        "id": criterion.id,
                        "rubric": criterion.criterion,
                        "modality": modality,
                        "score": _clamp_score(parsed.get("score")),
                        "explanation": str(parsed.get("explanation", "")).strip(),
                        "evaluator_error": False,
                        "payload_status": PAYLOAD_GRADEABLE,
                    },
                    input_skipped,
                )
            except Exception as exc:
                if not _is_retryable(exc) or attempt == _MAX_RETRIES:
                    logger.warning(
                        "Rubric judge failed for %s rubric %d: %s",
                        prompt.prompt_id,
                        index,
                        exc,
                    )
                    return _note_input_skips(
                        _zero_grade(
                            index=index,
                            criterion=criterion,
                            explanation=f"Judge error: {exc}",
                            evaluator_error=True,
                        ),
                        input_skipped,
                    )
                base_wait = _RETRY_BACKOFF * (2 ** (attempt - 1))
                jitter = random.uniform(0, base_wait * 0.5)
                await asyncio.sleep(min(base_wait + jitter, _MAX_BACKOFF))

        return _note_input_skips(
            _zero_grade(
                index=index,
                criterion=criterion,
                explanation="Judge failed without returning a response.",
                evaluator_error=True,
            ),
            input_skipped,
        )


async def score_prompt_rubrics(
    *,
    judge: RubricJudge,
    prompt: EvalPrompt,
    response_text: str,
    raw_response: Any,
    output_assets: list[dict[str, Any]] | None = None,
) -> RubricScoreResult:
    """Score each rubric independently against its explicit output modality.

    The prompt's MMI Value is the mean over its output modalities of the mean
    criterion score within each modality — see :func:`macro_score`.
    """
    criteria = prompt.rubric_criteria
    if not criteria:
        return RubricScoreResult(False, None, [], judge.model)

    invalid_modalities = [
        criterion.modality
        for criterion in criteria
        if criterion.modality not in prompt.output_modalities
    ]
    if invalid_modalities:
        raise ValueError(
            f"Prompt {prompt.prompt_id} has rubric modalities that do not match "
            f"output_modalities: invalid={invalid_modalities}, "
            f"output_modalities={prompt.output_modalities}"
        )

    grades: list[dict[str, Any]] = []
    for index, criterion in enumerate(criteria, start=1):
        if criterion.modality == "Text":
            if not response_text.strip():
                grades.append(
                    _zero_grade(
                        index=index,
                        criterion=criterion,
                        explanation="No model-produced Text payload was available for this rubric.",
                        payload=PAYLOAD_ABSENT,
                    )
                )
                continue
            grade = await judge.judge_rubric(
                prompt=prompt,
                index=index,
                criterion=criterion,
                response_text=response_text,
            )
            grades.append(
                _validate_grade(grade=grade, index=index, criterion=criterion)
            )
            continue

        modality_assets, payload = _assets_for_modality(
            output_assets, criterion.modality
        )
        if not modality_assets:
            grades.append(
                _zero_grade(
                    index=index,
                    criterion=criterion,
                    explanation=_PAYLOAD_EXPLANATIONS[payload].format(
                        modality=criterion.modality
                    ),
                    payload=payload,
                )
            )
            continue

        # Ledger 4: only the first asset of a modality is graded. Recorded in
        # the grade so the caveat is visible in the data, not just the docs.
        selected_assets = modality_assets[:1]

        # The capture status says we hold the bytes; it does not say they reached
        # the judge. Without this the judge is called with no artifact attached
        # and grades from the text summary alone, or worse, from the prompt's own
        # input artifact. A zero here is the harness failing, not the model.
        blocked = unsendable_reason(selected_assets[0])
        if blocked:
            blocked_grade = _zero_grade(
                index=index,
                criterion=criterion,
                explanation=(
                    f"The model-produced {criterion.modality} artifact could "
                    f"not be sent to the judge: {blocked}. Its content was "
                    "never graded. This is a harness failure, not a model "
                    "failure."
                ),
                evaluator_error=True,
                payload=PAYLOAD_UNSENDABLE,
            )
            blocked_grade["selected_asset_id"] = selected_assets[0].get("asset_id", "")
            blocked_grade["candidate_asset_ids"] = [
                asset.get("asset_id", "") for asset in modality_assets
            ]
            blocked_grade["extra_asset_count"] = max(0, len(modality_assets) - 1)
            grades.append(blocked_grade)
            continue

        grade = await judge.judge_rubric(
            prompt=prompt,
            index=index,
            criterion=criterion,
            response_text="",
            output_assets=selected_assets,
        )
        validated = _validate_grade(grade=grade, index=index, criterion=criterion)
        validated["selected_asset_id"] = selected_assets[0].get("asset_id", "")
        validated["candidate_asset_ids"] = [
            asset.get("asset_id", "") for asset in modality_assets
        ]
        validated["extra_asset_count"] = max(0, len(modality_assets) - 1)
        grades.append(validated)

    if len(grades) != len(criteria):
        raise ValueError(
            f"Prompt {prompt.prompt_id} produced {len(grades)} rubric grades for "
            f"{len(criteria)} rubric criteria"
        )

    score = macro_score(grades)
    return RubricScoreResult(
        True,
        score,
        grades,
        judge.model,
        evaluator_errors=sum(1 for grade in grades if is_evaluator_error(grade)),
    )
