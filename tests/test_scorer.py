# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Tests for the scorer module.

Responses here are built from real artifacts and real prose and adjudicated by
the real detector, so these tests pin the whole persisted-form path rather than
hand-asserting detection flags the harness would never actually see.
"""

import pytest

from mmi.detection import CAPTURED, PROVIDER_INLINE
from mmi.models import (
    CapturedAsset,
    EvalPrompt,
    JudgeResult,
    ProviderResponse,
)
from mmi.scorer import score

#: One canonical MIME per modality, for building native artifacts.
_NATIVE_MIME = {
    "Image": "image/png",
    "Audio": "audio/mpeg",
    "Video": "video/mp4",
    "Document": "application/pdf",
}

#: One canonical public URL per modality, for building URL evidence.
_URL_FOR = {
    "Image": "https://flickr.com/photos/example/1",
    "Audio": "https://soundcloud.com/example/track",
    "Video": "https://www.youtube.com/watch?v=example",
    "Document": "https://example.invalid/report.pdf",
}


def _make_prompt(output_modalities: list[str]) -> EvalPrompt:
    return EvalPrompt(
        prompt_id="p001",
        prompt_text="test",
        input_modalities=["Text"],
        output_modalities=output_modalities,
    )


def _make_response(native=None, via_url=None, error=None) -> ProviderResponse:
    """Build a response that genuinely produces the requested evidence."""
    native = list(native or [])
    via_url = list(via_url or [])

    text_parts = []
    if "Text" in native:
        text_parts.append("a substantive textual answer")
    for modality in via_url:
        if modality == "Text":
            text_parts.append("a substantive textual answer")
        else:
            text_parts.append(_URL_FOR[modality])

    assets = []
    for index, modality in enumerate(m for m in native if m != "Text"):
        assets.append(
            CapturedAsset(
                asset_id=f"p001_output_{index}",
                prompt_id="p001",
                modality=modality,
                source_type="test",
                delivery=PROVIDER_INLINE,
                mime_type=_NATIVE_MIME[modality],
                capture_status=CAPTURED,
                sha256="0" * 64,
                size_bytes=8,
            )
        )

    return ProviderResponse(
        prompt_id="p001",
        run_name="test-run",
        provider="test",
        model="test-model",
        response_text=" ".join(text_parts),
        output_assets=assets,
        error=error,
        is_error=bool(error),
    )


class TestScorer:
    def test_all_pass(self):
        prompt = _make_prompt(["Text", "Image"])
        response = _make_response(native=["Text", "Image"])
        result = score(prompt, response)
        assert result.all_pass_lenient is True
        assert result.all_pass_strict is True
        assert result.per_modality["Text"].pass_lenient is True
        assert result.per_modality["Image"].pass_lenient is True
        assert result.run_name == "test-run"

    def test_partial_pass(self):
        prompt = _make_prompt(["Text", "Video"])
        response = _make_response(native=["Text"])
        result = score(prompt, response)
        assert result.all_pass_lenient is False
        assert result.per_modality["Text"].pass_lenient is True
        assert result.per_modality["Video"].pass_lenient is False

    def test_all_fail(self):
        prompt = _make_prompt(["Image", "Audio"])
        response = _make_response(native=["Text"])
        result = score(prompt, response)
        assert result.all_pass_lenient is False
        assert result.per_modality["Image"].pass_lenient is False
        assert result.per_modality["Audio"].pass_lenient is False

    def test_empty_response(self):
        prompt = _make_prompt(["Text"])
        response = _make_response()
        result = score(prompt, response)
        assert result.all_pass_lenient is False
        assert result.per_modality["Text"].pass_lenient is False

    def test_extra_modalities_in_response(self):
        """Extra modalities in response are tracked in per_modality."""
        prompt = _make_prompt(["Text"])
        response = _make_response(native=["Text", "Image", "Audio"])
        result = score(prompt, response)
        assert result.all_pass_lenient is True
        assert "Text" in result.per_modality
        assert "Image" in result.per_modality
        assert result.per_modality["Image"].detected_native is True
        assert "Audio" in result.per_modality
        assert result.per_modality["Audio"].detected_native is True

    def test_single_modality_pass(self):
        prompt = _make_prompt(["Video"])
        response = _make_response(native=["Video"])
        result = score(prompt, response)
        assert result.all_pass_lenient is True

    def test_single_modality_fail(self):
        prompt = _make_prompt(["Video"])
        response = _make_response(native=["Text"])
        result = score(prompt, response)
        assert result.all_pass_lenient is False

    def test_result_metadata(self):
        prompt = _make_prompt(["Text"])
        response = _make_response(native=["Text"])
        result = score(prompt, response)
        assert result.prompt_id == "p001"
        assert result.provider == "test"
        assert result.model == "test-model"
        assert result.run_name == "test-run"
        assert result.expected_modalities == ["Text"]
        assert result.produced_modalities == ["Text"]


class TestScorerWithDetection:
    def test_native_detection_passes_strict_and_lenient(self):
        """Native detection → pass_strict=True, pass_lenient=True."""
        prompt = _make_prompt(["Text", "Image"])
        response = _make_response(native=["Text", "Image"])
        result = score(prompt, response)
        assert result.per_modality["Image"].pass_strict is True
        assert result.per_modality["Image"].pass_lenient is True
        assert result.all_pass_strict is True
        assert result.all_pass_lenient is True

    def test_url_only_passes_lenient_not_strict(self):
        """URL-only detection → pass_lenient=True, pass_strict=False."""
        prompt = _make_prompt(["Text", "Video"])
        response = _make_response(native=["Text"], via_url=["Video"])
        result = score(prompt, response)
        assert result.per_modality["Video"].pass_strict is False
        assert result.per_modality["Video"].pass_lenient is True
        assert result.all_pass_lenient is True
        assert result.all_pass_strict is False

    def test_no_judge_defaults(self):
        """score(prompt, response) without judge_result → no crash, clean defaults."""
        prompt = _make_prompt(["Text"])
        response = _make_response(native=["Text"])
        result = score(prompt, response)
        assert result.judge_used is False

    def test_produced_modalities_property(self):
        """produced_modalities derived property returns detected modalities."""
        prompt = _make_prompt(["Text", "Image", "Video"])
        response = _make_response(native=["Text", "Image"], via_url=["Video"])
        result = score(prompt, response)
        assert result.produced_modalities == ["Image", "Text", "Video"]


class TestScorerJudgeDetection:
    """Tests for the judge-as-safety-net scoring semantics.

    Judge-detected modalities (parser false negatives) count as lenient
    passes.
    """

    def test_judge_detected_passes_lenient(self):
        """Judge-detected modality → pass_lenient=True, pass_strict=False."""
        prompt = _make_prompt(["Text", "Image"])
        response = _make_response(native=["Text"])
        judge_result = JudgeResult(detected_modalities=["Image"])
        result = score(prompt, response, judge_result)

        assert result.per_modality["Image"].detected_via_judge is True
        assert result.per_modality["Image"].pass_lenient is True
        assert result.per_modality["Image"].pass_strict is False
        assert result.all_pass_lenient is True
        assert result.all_pass_strict is False
        assert result.judge_used is True

    def test_judge_detected_in_produced_modalities(self):
        """Judge-detected modalities appear in produced_modalities."""
        prompt = _make_prompt(["Text", "Audio"])
        response = _make_response(native=["Text"])
        judge_result = JudgeResult(detected_modalities=["Audio"])
        result = score(prompt, response, judge_result)
        assert "Audio" in result.produced_modalities

    def test_empty_judge_result(self):
        """JudgeResult with no detections → no change in scoring."""
        prompt = _make_prompt(["Text", "Image"])
        response = _make_response(native=["Text"])
        judge_result = JudgeResult()
        result = score(prompt, response, judge_result)

        assert result.per_modality["Image"].pass_lenient is False
        assert result.per_modality["Image"].detected_via_judge is False
        assert result.judge_used is True

    def test_multiple_judge_detections(self):
        """Judge can detect multiple modalities at once."""
        prompt = _make_prompt(["Text", "Image", "Audio"])
        response = _make_response(native=["Text"])
        judge_result = JudgeResult(detected_modalities=["Image", "Audio"])
        result = score(prompt, response, judge_result)

        assert result.per_modality["Image"].detected_via_judge is True
        assert result.per_modality["Image"].pass_lenient is True
        assert result.per_modality["Audio"].detected_via_judge is True
        assert result.per_modality["Audio"].pass_lenient is True
        assert result.all_pass_lenient is True


class TestPrecisionRecallF1:
    """Tests for precision, recall, and F1 computation."""

    def test_perfect_match(self):
        """All expected modalities detected, nothing extra → P=R=F1=1."""
        prompt = _make_prompt(["Text", "Image"])
        response = _make_response(native=["Text", "Image"])
        result = score(prompt, response)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_nothing_detected(self):
        """No modalities detected → precision=1 (no false positives), R=F1=0."""
        prompt = _make_prompt(["Image"])
        response = _make_response()
        result = score(prompt, response)
        assert result.precision == 1.0
        assert result.recall == 0.0
        assert result.f1 == 0.0

    def test_partial_recall(self):
        """Only some expected modalities detected."""
        prompt = _make_prompt(["Text", "Image", "Audio"])
        response = _make_response(native=["Text"])
        result = score(prompt, response)
        assert result.precision == 1.0
        assert result.recall == pytest.approx(1 / 3)
        assert result.f1 == pytest.approx(2 * 1.0 * (1 / 3) / (1.0 + 1 / 3))

    def test_extra_non_text_modality_lowers_precision(self):
        """Extra non-Text modality counts as false positive."""
        prompt = _make_prompt(["Text"])
        response = _make_response(native=["Text", "Image"])
        result = score(prompt, response)
        assert result.precision == pytest.approx(1 / 2)
        assert result.recall == 1.0
        assert result.f1 == pytest.approx(2 / 3)

    def test_extra_text_ignored_when_not_expected(self):
        """Text detected but not expected is ignored → precision unaffected."""
        prompt = _make_prompt(["Image"])
        response = _make_response(native=["Image", "Text"])
        result = score(prompt, response)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_text_counts_when_expected(self):
        """Text detected and expected is counted normally."""
        prompt = _make_prompt(["Text", "Image"])
        response = _make_response(native=["Text"])
        result = score(prompt, response)
        assert result.precision == 1.0
        assert result.recall == pytest.approx(1 / 2)

    def test_url_detection_counts_for_metrics(self):
        """Modalities detected via URL count as detected."""
        prompt = _make_prompt(["Video"])
        response = _make_response(via_url=["Video"])
        result = score(prompt, response)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_judge_detection_counts_for_metrics(self):
        """Modalities detected via judge count as detected."""
        prompt = _make_prompt(["Image"])
        response = _make_response()
        judge_result = JudgeResult(detected_modalities=["Image"])
        result = score(prompt, response, judge_result)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_extra_text_with_multiple_expected(self):
        """Text not expected, model returns Image + Text → Text ignored."""
        prompt = _make_prompt(["Image", "Audio"])
        response = _make_response(native=["Image", "Text"])
        result = score(prompt, response)
        assert result.precision == 1.0
        assert result.recall == pytest.approx(1 / 2)


class TestPrecisionRecallF1Strict:
    """Tests for strict precision, recall, and F1 (native detection only)."""

    def test_perfect_match_strict(self):
        prompt = _make_prompt(["Text", "Image"])
        response = _make_response(native=["Text", "Image"])
        result = score(prompt, response)
        assert result.precision_strict == 1.0
        assert result.recall_strict == 1.0
        assert result.f1_strict == 1.0

    def test_url_only_not_counted_strict(self):
        """URL-detected modality does not count for strict metrics."""
        prompt = _make_prompt(["Video"])
        response = _make_response(via_url=["Video"])
        result = score(prompt, response)
        assert result.precision_strict == 1.0
        assert result.recall_strict == 0.0
        assert result.f1_strict == 0.0
        # lenient should still pass
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_judge_only_not_counted_strict(self):
        """Judge-detected modality does not count for strict metrics."""
        prompt = _make_prompt(["Image"])
        response = _make_response()
        judge_result = JudgeResult(detected_modalities=["Image"])
        result = score(prompt, response, judge_result)
        assert result.precision_strict == 1.0
        assert result.recall_strict == 0.0
        assert result.f1_strict == 0.0

    def test_mixed_native_and_url(self):
        """Native modality counts strict, URL modality does not."""
        prompt = _make_prompt(["Text", "Video"])
        response = _make_response(native=["Text"], via_url=["Video"])
        result = score(prompt, response)
        assert result.precision_strict == 1.0
        assert result.recall_strict == pytest.approx(1 / 2)
        # lenient sees both
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_extra_text_ignored_strict(self):
        """Text not expected, natively detected → ignored in strict too."""
        prompt = _make_prompt(["Image"])
        response = _make_response(native=["Image", "Text"])
        result = score(prompt, response)
        assert result.precision_strict == 1.0
        assert result.recall_strict == 1.0
        assert result.f1_strict == 1.0

    def test_extra_non_text_lowers_precision_strict(self):
        """Extra non-Text native modality lowers strict precision."""
        prompt = _make_prompt(["Text"])
        response = _make_response(native=["Text", "Image"])
        result = score(prompt, response)
        assert result.precision_strict == pytest.approx(1 / 2)
        assert result.recall_strict == 1.0


class TestErrorSemantics:
    """Errors are scored outcomes, not missing rows.

    The paper counts an errored prompt as a failure over the full 893, so an
    error row must carry valid metrics rather than be excluded.
    """

    @staticmethod
    def _error_response(message="provider failure", error_type="provider_error"):
        return ProviderResponse(
            prompt_id="p001",
            run_name="test-run",
            provider="test",
            model="test-model",
            is_error=True,
            error=message,
            error_type=error_type,
        )

    def test_error_rows_have_valid_precision_recall_f1(self):
        prompt = _make_prompt(["Text", "Image"])
        result = score(prompt, self._error_response())

        assert result.is_error is True
        assert result.precision == 1.0
        assert result.recall == 0.0
        assert result.f1 == 0.0
        assert result.precision_strict == 1.0
        assert result.recall_strict == 0.0
        assert result.f1_strict == 0.0

    def test_error_rows_metrics_match_non_error_empty_response(self):
        """Errors must not be special-cased in the math."""
        prompt = _make_prompt(["Image"])
        result_ok = score(prompt, _make_response())
        result_err = score(prompt, self._error_response("timeout", "timeout"))

        assert result_err.precision == result_ok.precision
        assert result_err.recall == result_ok.recall
        assert result_err.f1 == result_ok.f1
        assert result_err.precision_strict == result_ok.precision_strict
        assert result_err.recall_strict == result_ok.recall_strict
        assert result_err.f1_strict == result_ok.f1_strict

    def test_error_fields_propagate(self):
        result = score(_make_prompt(["Text"]), self._error_response())

        assert result.is_error is True
        assert result.error_message == "provider failure"
        assert result.error_type == "provider_error"

    def test_error_row_detects_nothing(self):
        """An error row cannot accidentally carry detection evidence."""
        result = score(_make_prompt(["Text"]), self._error_response())

        assert result.produced_modalities == []
        assert all(not s.pass_lenient for s in result.per_modality.values())


class TestNothingIsSilentlyDropped:
    """Every field the scorer computes must have a home on ``EvalResult``.

    ``_result_to_dict`` serializes via ``dataclasses.asdict``, which walks only
    *declared* fields. Anything the scorer writes into the result dict but that
    EvalResult does not declare is computed, looks correct in memory, and then
    vanishes on write. That silently lost the judge model, the judge reasoning,
    and the paper's binary rubric metric before this test existed.
    """

    #: Bookkeeping the result dict carries but EvalResult exposes differently.
    KNOWN_NON_FIELDS = {"produced_modalities"}

    def test_rescore_dict_writes_only_declared_fields(self):
        import dataclasses

        from mmi.models import EvalResult, JudgeResult
        from mmi.scorer import rescore_dict

        record = {
            "per_modality": {},
            "expected_modalities": ["Image"],
        }
        rescore_dict(
            record,
            JudgeResult(
                detected_modalities=["Image"],
                reasoning="because",
                judge_model="test-judge",
            ),
        )

        declared = {f.name for f in dataclasses.fields(EvalResult)}
        undeclared = set(record) - declared - self.KNOWN_NON_FIELDS

        assert not undeclared, (
            f"these would be dropped by asdict(): {sorted(undeclared)}"
        )

    def test_judge_metadata_survives_scoring(self):
        from mmi.models import JudgeResult

        prompt = _make_prompt(["Image"])
        response = _make_response()
        judge = JudgeResult(
            detected_modalities=["Image"],
            reasoning="the model returned an inline image",
            judge_model="gemini-test",
        )

        result = score(prompt, response, judge)

        assert result.judge_used is True
        assert result.judge_model == "gemini-test"
        assert result.judge_reasoning == "the model returned an inline image"
        assert result.per_modality["Image"].detected_via_judge is True
        assert result.per_modality["Image"].pass_lenient is True
        assert result.per_modality["Image"].pass_strict is False

    def test_judge_metadata_survives_the_jsonl_round_trip(self):
        from mmi.models import JudgeResult
        from mmi.runner import _dict_to_eval_result, _result_to_dict

        result = score(
            _make_prompt(["Image"]),
            _make_response(),
            JudgeResult(
                detected_modalities=["Image"],
                reasoning="why",
                judge_model="gemini-test",
            ),
        )

        restored = _dict_to_eval_result(_result_to_dict(result))

        assert restored.judge_model == "gemini-test"
        assert restored.judge_reasoning == "why"
