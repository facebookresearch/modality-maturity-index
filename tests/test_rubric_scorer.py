# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Tests for rubric grading: what reaches the judge, and what happens when it cannot."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mmi.detection import CAPTURED, PAYLOAD_UNSENDABLE
from mmi.models import EvalPrompt, RubricCriterion
from mmi.rubric_scorer import (
    _ANSWER_MEDIA_LABEL,
    _INPUT_MEDIA_LABEL,
    RubricJudge,
    macro_score,
    score_by_modality,
    score_prompt_rubrics,
)


def _prompt(*, input_files=None, output_modalities=None, criteria=None) -> EvalPrompt:
    modalities = output_modalities or ["Image"]
    return EvalPrompt(
        prompt_id="p1",
        prompt_text="edit this",
        input_modalities=["Image"],
        output_modalities=modalities,
        input_files=input_files or [],
        rubric_criteria=criteria
        or [
            RubricCriterion(
                id="1", criterion="Delivers the artifact.", modality=modalities[0]
            )
        ],
    )


def _asset(path, *, mime="image/png", asset_id="a1", modality="Image"):
    return {
        "asset_id": asset_id,
        "modality": modality,
        "mime_type": mime,
        "capture_status": CAPTURED,
        "local_path": str(path),
        "size_bytes": 10,
    }


def _judge(monkeypatch, *, reply='{"score": 1.0, "explanation": "ok"}'):
    """A RubricJudge whose transport is captured rather than called."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    with patch("mmi.rubric_scorer.Client") as client_cls:
        client = MagicMock()
        client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(text=reply)
        )
        client_cls.return_value = client
        judge = RubricJudge(model="gemini-test", api_key_env="GOOGLE_API_KEY")
    return judge, client.aio.models.generate_content


def _sent_contents(call):
    return call.call_args.kwargs["contents"]


# ---------------------------------------------------------------------------
# Labelling: the judge must be able to tell the two artifacts apart
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_and_input_media_are_labelled_in_order(monkeypatch, tmp_path):
    answer = tmp_path / "answer.png"
    answer.write_bytes(b"answer-bytes")
    source = tmp_path / "p1.jpg"
    source.write_bytes(b"input-bytes")
    monkeypatch.setattr("mmi.response_utils.INPUT_FILES_DIR", tmp_path)

    judge, call = _judge(monkeypatch)
    await score_prompt_rubrics(
        judge=judge,
        prompt=_prompt(input_files=["p1.jpg"]),
        response_text="",
        raw_response={},
        output_assets=[_asset(answer)],
    )

    contents = _sent_contents(call)
    labels = [c for c in contents if isinstance(c, str) and c.startswith("# Attached")]
    assert labels == [_ANSWER_MEDIA_LABEL.format(modality="Image"), _INPUT_MEDIA_LABEL]
    # Each label must precede the bytes it describes.
    assert contents.index(labels[0]) < contents.index(labels[1])
    assert len(contents) == 5  # prompt, answer label, answer, input label, input


@pytest.mark.asyncio
async def test_input_label_omitted_when_prompt_has_no_input_files(
    monkeypatch, tmp_path
):
    answer = tmp_path / "answer.png"
    answer.write_bytes(b"answer-bytes")

    judge, call = _judge(monkeypatch)
    await score_prompt_rubrics(
        judge=judge,
        prompt=_prompt(input_files=[]),
        response_text="",
        raw_response={},
        output_assets=[_asset(answer)],
    )

    contents = _sent_contents(call)
    assert _INPUT_MEDIA_LABEL not in contents
    assert _ANSWER_MEDIA_LABEL.format(modality="Image") in contents


# ---------------------------------------------------------------------------
# The answer artifact must actually reach the judge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unreadable_answer_artifact_is_not_graded_blind(monkeypatch, tmp_path):
    """A captured artifact whose bytes we cannot read must not be judged."""
    judge, call = _judge(monkeypatch)
    result = await score_prompt_rubrics(
        judge=judge,
        prompt=_prompt(),
        response_text="",
        raw_response={},
        output_assets=[_asset(tmp_path / "gone.png")],
    )

    call.assert_not_awaited()
    grade = result.rubric_grades[0]
    assert grade["score"] == 0.0
    assert grade["payload_status"] == PAYLOAD_UNSENDABLE
    assert grade["evaluator_error"] is True
    assert grade["selected_asset_id"] == "a1"
    assert result.evaluator_errors == 1


@pytest.mark.asyncio
async def test_oversize_answer_artifact_is_not_graded_blind(monkeypatch, tmp_path):
    answer = tmp_path / "big.mp4"
    answer.write_bytes(b"x")
    monkeypatch.setattr("mmi.response_utils._MAX_MEDIA_BYTES", 0)

    judge, call = _judge(monkeypatch)
    result = await score_prompt_rubrics(
        judge=judge,
        prompt=_prompt(output_modalities=["Video"]),
        response_text="",
        raw_response={},
        output_assets=[_asset(answer, mime="video/mp4", modality="Video")],
    )

    call.assert_not_awaited()
    assert result.rubric_grades[0]["payload_status"] == PAYLOAD_UNSENDABLE


@pytest.mark.asyncio
async def test_office_document_is_sent_rather_than_silently_dropped(
    monkeypatch, tmp_path
):
    """A type this judge may reject is still attached, so the failure is the judge's.

    Filtering by assumed provider support would make the artifact disappear from a
    grade that still reported itself as judged, and would freeze the harness to one
    judge's capabilities.
    """
    answer = tmp_path / "report.docx"
    answer.write_bytes(b"docx-bytes")
    ooxml = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    judge, call = _judge(monkeypatch)
    result = await score_prompt_rubrics(
        judge=judge,
        prompt=_prompt(output_modalities=["Document"]),
        response_text="",
        raw_response={},
        output_assets=[_asset(answer, mime=ooxml, modality="Document")],
    )

    call.assert_awaited_once()
    sent = [c for c in _sent_contents(call) if not isinstance(c, str)]
    assert any(getattr(p, "inline_data", None) for p in sent)
    assert result.rubric_grades[0]["payload_status"] != PAYLOAD_UNSENDABLE


@pytest.mark.asyncio
async def test_judge_rejection_is_recorded_as_evaluator_error(monkeypatch, tmp_path):
    answer = tmp_path / "report.docx"
    answer.write_bytes(b"docx-bytes")

    judge, call = _judge(monkeypatch)
    call.side_effect = Exception("400 INVALID_ARGUMENT: unsupported mime type")

    result = await score_prompt_rubrics(
        judge=judge,
        prompt=_prompt(output_modalities=["Document"]),
        response_text="",
        raw_response={},
        output_assets=[_asset(answer, mime="application/msword", modality="Document")],
    )

    grade = result.rubric_grades[0]
    assert grade["score"] == 0.0
    assert grade["evaluator_error"] is True
    assert "INVALID_ARGUMENT" in grade["explanation"]
    # Fails fast: a rejected type is not a transient error.
    assert call.await_count == 1


# ---------------------------------------------------------------------------
# Input artifacts are context: absence is recorded, never fatal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_input_artifact_is_recorded_but_still_graded(
    monkeypatch, tmp_path
):
    answer = tmp_path / "answer.png"
    answer.write_bytes(b"answer-bytes")
    monkeypatch.setattr("mmi.response_utils.INPUT_FILES_DIR", tmp_path / "empty")

    judge, call = _judge(monkeypatch)
    result = await score_prompt_rubrics(
        judge=judge,
        prompt=_prompt(input_files=["p1.jpg"]),
        response_text="",
        raw_response={},
        output_assets=[_asset(answer)],
    )

    call.assert_awaited_once()
    grade = result.rubric_grades[0]
    assert grade["score"] == 1.0
    assert grade["input_media_skipped"] == ["p1.jpg"]


@pytest.mark.asyncio
async def test_gradeable_call_records_no_input_skips(monkeypatch, tmp_path):
    answer = tmp_path / "answer.png"
    answer.write_bytes(b"answer-bytes")
    (tmp_path / "p1.jpg").write_bytes(b"input-bytes")
    monkeypatch.setattr("mmi.response_utils.INPUT_FILES_DIR", tmp_path)

    judge, _ = _judge(monkeypatch)
    result = await score_prompt_rubrics(
        judge=judge,
        prompt=_prompt(input_files=["p1.jpg"]),
        response_text="",
        raw_response={},
        output_assets=[_asset(answer)],
    )

    assert "input_media_skipped" not in result.rubric_grades[0]


@pytest.mark.asyncio
async def test_text_criterion_still_carries_input_media(monkeypatch, tmp_path):
    (tmp_path / "p1.jpg").write_bytes(b"input-bytes")
    monkeypatch.setattr("mmi.response_utils.INPUT_FILES_DIR", tmp_path)

    judge, call = _judge(monkeypatch)
    await score_prompt_rubrics(
        judge=judge,
        prompt=_prompt(
            input_files=["p1.jpg"],
            output_modalities=["Text"],
            criteria=[
                RubricCriterion(id="1", criterion="Describes it.", modality="Text")
            ],
        ),
        response_text="a description",
        raw_response={},
        output_assets=[],
    )

    contents = _sent_contents(call)
    assert _INPUT_MEDIA_LABEL in contents
    assert not any(
        isinstance(c, str) and c.startswith("# Attached below: the model-produced")
        for c in contents
    )


@pytest.mark.asyncio
async def test_grade_json_is_parsed_from_the_reply(monkeypatch, tmp_path):
    answer = tmp_path / "answer.png"
    answer.write_bytes(b"answer-bytes")

    judge, _ = _judge(
        monkeypatch, reply=json.dumps({"score": 0.5, "explanation": "partly"})
    )
    result = await score_prompt_rubrics(
        judge=judge,
        prompt=_prompt(),
        response_text="",
        raw_response={},
        output_assets=[_asset(answer)],
    )

    grade = result.rubric_grades[0]
    assert grade["score"] == 0.5
    assert grade["explanation"] == "partly"
    assert grade["evaluator_error"] is False


# ---------------------------------------------------------------------------
# Aggregation: the MMI Value weights modalities, not criteria
# ---------------------------------------------------------------------------


def _grades(*pairs):
    return [{"modality": modality, "score": score} for modality, score in pairs]


def test_score_by_modality_means_within_each_modality():
    grades = _grades(("Text", 1.0), ("Text", 1.0), ("Text", 0.0), ("Image", 0.0))
    assert score_by_modality(grades) == {"Text": pytest.approx(2 / 3), "Image": 0.0}


def test_macro_score_weights_modalities_equally_not_criteria():
    """Three Text criteria must not outvote the single Image one."""
    grades = _grades(("Text", 1.0), ("Text", 1.0), ("Text", 0.0), ("Image", 0.0))

    # A flat mean over criteria would give 2/4 = 0.5, letting the
    # criterion-rich modality dominate.
    assert macro_score(grades) == pytest.approx(1 / 3)


def test_macro_score_cannot_exceed_modality_recall():
    """The bound benchmark.tex relies on: omitting a modality forfeits its share.

    A model that returns only the criterion-rich modality, perfectly, must not
    score above the fraction of requested modalities it produced. Under a flat
    mean over criteria this case scored 9/12 = 0.75 against a recall of 0.5.
    """
    grades = _grades(*([("Text", 0.0)] * 3 + [("Document", 1.0)] * 9))

    assert macro_score(grades) == pytest.approx(0.5)


def test_macro_and_flat_mean_agree_for_a_single_modality():
    grades = _grades(("Audio", 1.0), ("Audio", 0.0), ("Audio", 0.5))
    assert macro_score(grades) == pytest.approx(0.5)


def test_macro_score_of_no_grades_is_none():
    assert macro_score([]) is None


@pytest.mark.asyncio
async def test_prompt_score_is_macro_and_breakdown_is_recorded(monkeypatch, tmp_path):
    """End-to-end: two modalities, one produced, one absent."""
    answer = tmp_path / "answer.png"
    answer.write_bytes(b"answer-bytes")

    criteria = [
        RubricCriterion(id="1", criterion="Image is right.", modality="Image"),
        RubricCriterion(id="2", criterion="Image is sharp.", modality="Image"),
        RubricCriterion(id="3", criterion="Image is captioned.", modality="Image"),
        RubricCriterion(id="4", criterion="Audio is right.", modality="Audio"),
    ]
    judge, _ = _judge(monkeypatch)
    result = await score_prompt_rubrics(
        judge=judge,
        prompt=_prompt(output_modalities=["Image", "Audio"], criteria=criteria),
        response_text="",
        raw_response={},
        output_assets=[_asset(answer)],
    )

    # Image: 3/3 satisfied. Audio: never produced, so zero.
    assert result.rubric_score_by_modality == {"Image": 1.0, "Audio": 0.0}
    # Macro over the two modalities, not 3/4 over the criteria.
    assert result.rubric_score == pytest.approx(0.5)
    assert result.as_result_fields()["rubric_score_by_modality"] == {
        "Image": 1.0,
        "Audio": 0.0,
    }
