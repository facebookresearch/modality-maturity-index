# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Scorer: compare expected vs produced modalities."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence

from .detection import ALL_MODALITIES
from .models import EvalPrompt, EvalResult, JudgeResult, ModalityScore, ProviderResponse
from .redaction import redact_request_record


def dataclass_to_dict(value):
    if value is None:
        return None
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return value


def compute_prf(
    expected_modalities: Sequence[str],
    detected: set[str],
    detected_strict: set[str],
) -> tuple[float, float, float, float, float, float]:
    """Compute precision, recall, and F1 for lenient and strict detection.

    Args:
        expected_modalities: The modalities the prompt expects the model to produce.
        detected: Modalities detected via any method (native, URL, or judge).
        detected_strict: Modalities detected via native detection only.

    Returns:
        A 6-tuple of (precision, recall, f1, precision_strict, recall_strict, f1_strict).
    """
    expected = set(expected_modalities)

    # Ignore Text in detected if it is not an expected modality, so that
    # a model returning text alongside the correct modality is not penalised.
    if "Text" not in expected:
        detected = detected - {"Text"}
        detected_strict = detected_strict - {"Text"}

    true_pos = len(expected & detected)
    precision = true_pos / len(detected) if detected else 1.0
    recall = true_pos / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    true_pos_strict = len(expected & detected_strict)
    precision_strict = (
        true_pos_strict / len(detected_strict) if detected_strict else 1.0
    )
    recall_strict = true_pos_strict / len(expected) if expected else 0.0
    f1_strict = (
        2 * precision_strict * recall_strict / (precision_strict + recall_strict)
        if (precision_strict + recall_strict)
        else 0.0
    )

    return precision, recall, f1, precision_strict, recall_strict, f1_strict


# ---------------------------------------------------------------------------
# Single entry-point for all scoring / rescoring
# ---------------------------------------------------------------------------


def rescore_dict(
    result_dict: dict,
    judge_result: JudgeResult | None = None,
) -> dict:
    """Score (or re-score) a result dict in-place.

    This is the **single place** where scoring math happens.  Every code
    path — fresh scoring, rejudging, and rescoring — calls this function.

    It:
    1. Normalises ``per_modality`` to contain ALL modalities.
    2. Applies *judge_result* detection flags (if provided).
    3. Recomputes ``pass_strict`` / ``pass_lenient`` for every modality.
    4. Recomputes ``all_pass_*``, ``produced_modalities``, and P/R/F1.

    Returns the mutated *result_dict*.
    """
    # 1. Normalise per_modality to ALL_MODALITIES
    per_modality = result_dict.setdefault("per_modality", {})
    for modality in ALL_MODALITIES:
        if modality not in per_modality:
            per_modality[modality] = {
                "detected_native": False,
                "detected_via_url": False,
                "detected_via_judge": False,
                "pass_strict": False,
                "pass_lenient": False,
            }
        else:
            per_modality[modality].setdefault("detected_via_judge", False)

    # 2. Apply judge flags
    judge_detected = set(judge_result.detected_modalities if judge_result else [])
    for modality, ms in per_modality.items():
        if judge_result is not None:
            ms["detected_via_judge"] = modality in judge_detected
        # 3. Recompute pass flags from detection flags
        ms["pass_strict"] = ms["detected_native"]
        ms["pass_lenient"] = (
            ms["detected_native"] or ms["detected_via_url"] or ms["detected_via_judge"]
        )

    # 4. Derive aggregates
    expected = result_dict.get("expected_modalities", [])

    detected = {
        m
        for m, ms in per_modality.items()
        if ms.get("detected_native")
        or ms.get("detected_via_url")
        or ms.get("detected_via_judge")
    }
    detected_strict = {m for m, ms in per_modality.items() if ms.get("detected_native")}

    result_dict["all_pass_lenient"] = (
        all(per_modality[m]["pass_lenient"] for m in expected) if expected else False
    )
    result_dict["all_pass_strict"] = (
        all(per_modality[m]["pass_strict"] for m in expected) if expected else False
    )
    result_dict["produced_modalities"] = sorted(detected)

    precision, recall, f1, precision_strict, recall_strict, f1_strict = compute_prf(
        expected, detected, detected_strict
    )
    result_dict["precision"] = precision
    result_dict["recall"] = recall
    result_dict["f1"] = f1
    result_dict["precision_strict"] = precision_strict
    result_dict["recall_strict"] = recall_strict
    result_dict["f1_strict"] = f1_strict

    # Store judge metadata
    if judge_result is not None:
        result_dict["judge_used"] = True
        result_dict["judge_reasoning"] = judge_result.reasoning
        result_dict["judge_model"] = judge_result.judge_model

    return result_dict


# ---------------------------------------------------------------------------
# Fresh scoring from dataclass objects
# ---------------------------------------------------------------------------


def score(
    prompt: EvalPrompt,
    response: ProviderResponse,
    judge_result: JudgeResult | None = None,
) -> EvalResult:
    """Score a single prompt/response pair.

    Builds a result dict from the dataclass inputs and delegates all
    scoring math to :func:`rescore_dict`.
    """
    detection = response.detection

    # Build per_modality from the single shared detector
    per_modality: dict[str, dict] = {}
    for modality in ALL_MODALITIES:
        mod_det = detection.modalities.get(modality)
        per_modality[modality] = {
            "detected_native": mod_det.detected_native if mod_det else False,
            "detected_via_url": mod_det.detected_via_url if mod_det else False,
            "detected_via_judge": False,
            "pass_strict": False,
            "pass_lenient": False,
        }

    raw = response.raw_response
    if raw is not None:
        # Must round-trip as JSON. A repr string here would make the persisted
        # form unparseable and break fresh/rescore parity.
        json.dumps(raw, ensure_ascii=False)

    result_dict: dict = {
        "prompt_id": prompt.prompt_id,
        "run_name": response.run_name,
        "provider": response.provider,
        "model": response.model,
        "expected_modalities": prompt.output_modalities,
        "per_modality": per_modality,
        "judge_used": False,
        "is_error": detection.is_error,
        "error_message": response.error or detection.error_message,
        "error_type": detection.error_type,
        "raw_response": raw,
        "response_text": detection.text_content,
        "request": redact_request_record(dataclass_to_dict(response.request)),
        "output_assets": [dataclass_to_dict(asset) for asset in response.output_assets],
        "tool_calls": [dataclass_to_dict(call) for call in response.tool_calls],
    }

    # Single place for all scoring math
    rescore_dict(result_dict, judge_result)

    # Convert to EvalResult
    return EvalResult(
        prompt_id=result_dict["prompt_id"],
        run_name=result_dict["run_name"],
        provider=result_dict["provider"],
        model=result_dict["model"],
        expected_modalities=result_dict["expected_modalities"],
        per_modality={
            m: ModalityScore(**ms) for m, ms in result_dict["per_modality"].items()
        },
        all_pass_lenient=result_dict["all_pass_lenient"],
        all_pass_strict=result_dict["all_pass_strict"],
        precision=result_dict["precision"],
        recall=result_dict["recall"],
        f1=result_dict["f1"],
        precision_strict=result_dict["precision_strict"],
        recall_strict=result_dict["recall_strict"],
        f1_strict=result_dict["f1_strict"],
        judge_used=result_dict.get("judge_used", False),
        judge_model=result_dict.get("judge_model", ""),
        judge_reasoning=result_dict.get("judge_reasoning", ""),
        is_error=result_dict["is_error"],
        error_message=result_dict["error_message"],
        error_type=result_dict["error_type"],
        raw_response=result_dict["raw_response"],
        response_text=result_dict.get("response_text", ""),
        request=result_dict.get("request"),
        output_assets=result_dict.get("output_assets", []),
        tool_calls=result_dict.get("tool_calls", []),
        rubric_used=result_dict.get("rubric_used", False),
        rubric_score=result_dict.get("rubric_score"),
        rubric_grades=result_dict.get("rubric_grades", []),
        rubric_score_by_modality=result_dict.get("rubric_score_by_modality", {}),
        rubric_judge_model=result_dict.get("rubric_judge_model", ""),
    )
