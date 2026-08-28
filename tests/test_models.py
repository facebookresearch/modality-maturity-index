# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Tests for the data models."""

import pytest

from mmi.detection import make_empty_modalities
from mmi.models import (
    CapturedAsset,
    DetectionResult,
    EvalPrompt,
    EvalResult,
    ModalityDetection,
    ModalityScore,
    ProviderResponse,
    RequestRecord,
    RubricCriterion,
    ToolCallRecord,
)
from mmi.rubric_scorer import score_prompt_rubrics


def test_eval_prompt_defaults():
    p = EvalPrompt(
        prompt_id="p001",
        prompt_text="hello",
        input_modalities=["Text"],
        output_modalities=["Text"],
    )
    assert p.input_files == []
    assert p.prompt_id == "p001"


def test_eval_prompt_with_files():
    p = EvalPrompt(
        prompt_id="p002",
        prompt_text="describe this",
        input_modalities=["Image", "Text"],
        output_modalities=["Text"],
        input_files=["p002.jpg"],
    )
    assert p.input_files == ["p002.jpg"]
    assert p.input_modalities == ["Image", "Text"]


def test_request_record_and_captured_asset_defaults():
    request = RequestRecord(provider="openai", api="responses", model="gpt-test")
    asset = CapturedAsset(
        asset_id="a1",
        prompt_id="p001",
        modality="Image",
        source_type="provider_inline",
    )
    assert request.input_files == []
    assert request.provider_request == {}
    assert asset.capture_status == "captured"
    assert asset.metadata == {}


def test_tool_call_record_defaults():
    record = ToolCallRecord(tool_name="image_gen", provider_call_id="call_1")

    assert record.arguments == {}
    assert record.status == "pending"
    assert record.produced_asset_ids == []


def test_provider_response():
    r = ProviderResponse(
        prompt_id="p001",
        run_name="gpt-4o",
        provider="openai",
        model="gpt-4o",
    )
    assert r.error is None
    assert r.raw_response is None
    assert r.run_name == "gpt-4o"
    assert r.output_assets == []
    assert r.tool_calls == []


def test_provider_response_with_error():
    r = ProviderResponse(
        prompt_id="p001",
        run_name="gpt-4o",
        provider="openai",
        model="gpt-4o",
        error="rate limited",
    )
    assert r.error == "rate limited"


def test_eval_result():
    r = EvalResult(
        prompt_id="p001",
        run_name="gpt-4o",
        provider="openai",
        model="gpt-4o",
        expected_modalities=["Text", "Image"],
        per_modality={
            "Text": ModalityScore(
                detected_native=True, pass_strict=True, pass_lenient=True
            ),
            "Image": ModalityScore(
                detected_native=False, pass_strict=False, pass_lenient=False
            ),
        },
        all_pass_lenient=False,
        all_pass_strict=False,
    )
    assert not r.all_pass_lenient
    assert r.per_modality["Text"].pass_lenient is True
    assert r.per_modality["Image"].pass_lenient is False
    assert r.run_name == "gpt-4o"
    assert r.produced_modalities == ["Text"]


def test_eval_result_produced_includes_judge():
    """produced_modalities includes judge-detected modalities."""
    r = EvalResult(
        prompt_id="p001",
        run_name="gpt-4o",
        provider="openai",
        model="gpt-4o",
        expected_modalities=["Text", "Image"],
        per_modality={
            "Text": ModalityScore(
                detected_native=True, pass_strict=True, pass_lenient=True
            ),
            "Image": ModalityScore(
                detected_via_judge=True, pass_strict=False, pass_lenient=True
            ),
        },
        all_pass_lenient=True,
        all_pass_strict=False,
    )
    assert r.produced_modalities == ["Image", "Text"]


def test_modality_detection():
    d = ModalityDetection()
    assert d.detected_native is False
    assert d.detected_via_url is False
    assert d.native_evidence == ""


def test_detection_result():
    mods = make_empty_modalities()
    mods["Text"].detected_native = True
    d = DetectionResult(modalities=mods, text_content="hello")
    assert d.modalities["Text"].detected_native is True
    assert d.text_content == "hello"
    assert d.is_error is False


def test_modality_score():
    ms = ModalityScore(
        detected_native=True,
        pass_strict=True,
        pass_lenient=True,
    )
    assert ms.detected_native is True
    assert ms.detected_via_judge is False


@pytest.mark.asyncio
async def test_rubric_scoring_uses_first_asset_for_modality():
    class FakeJudge:
        model = "fake-judge"

        async def judge_rubric(
            self, *, prompt, index, criterion, response_text, output_assets=None
        ):
            assert [asset["asset_id"] for asset in output_assets] == ["asset_first"]
            return {
                "index": index,
                "id": criterion.id,
                "rubric": criterion.criterion,
                "modality": criterion.modality,
                "score": 1.0,
                "explanation": "graded first asset only",
            }

    prompt = EvalPrompt(
        prompt_id="p-rubric",
        prompt_text="make one image",
        input_modalities=["Text"],
        output_modalities=["Image"],
        rubric_criteria=[
            RubricCriterion(id="img", criterion="is relevant", modality="Image")
        ],
    )
    output_assets = [
        {"asset_id": "asset_first", "modality": "Image", "capture_status": "captured"},
        {"asset_id": "asset_second", "modality": "Image", "capture_status": "captured"},
    ]

    result = await score_prompt_rubrics(
        judge=FakeJudge(),
        prompt=prompt,
        response_text="",
        raw_response=None,
        output_assets=output_assets,
    )

    grade = result.rubric_grades[0]
    assert grade["selected_asset_id"] == "asset_first"
    assert grade["candidate_asset_ids"] == ["asset_first", "asset_second"]
    assert grade["extra_asset_count"] == 1
