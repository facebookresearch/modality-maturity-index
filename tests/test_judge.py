# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Tests for the LLM judge module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mmi.detection import classify_mime
from mmi.judge import JUDGE_SYSTEM_PROMPT, ModalityJudge, should_call_judge
from mmi.models import DetectionResult, ModalityDetection


class TestShouldCallJudge:
    def _det(self, text=False, image=False, audio=False, is_error=False):
        mods = {
            m: ModalityDetection()
            for m in ["Text", "Image", "Audio", "Video", "Document"]
        }
        mods["Text"].detected_native = text
        mods["Image"].detected_native = image
        mods["Audio"].detected_native = audio
        d = DetectionResult(modalities=mods)
        d.is_error = is_error
        return d

    def test_disabled(self):
        assert (
            should_call_judge(self._det(text=True), ["Text", "Image"], False) is False
        )

    def test_missing_non_text_without_text_still_calls_judge(self):
        assert should_call_judge(self._det(text=False), ["Text", "Image"], True) is True

    def test_all_found(self):
        assert (
            should_call_judge(self._det(text=True, image=True), ["Text", "Image"], True)
            is False
        )

    def test_missing_non_text(self):
        assert (
            should_call_judge(
                self._det(text=True, image=False), ["Text", "Image"], True
            )
            is True
        )

    def test_text_only_expected(self):
        assert should_call_judge(self._det(text=True), ["Text"], True) is False

    def test_error(self):
        assert (
            should_call_judge(
                self._det(text=True, is_error=True), ["Text", "Image"], True
            )
            is False
        )

    def test_none(self):
        assert should_call_judge(None, ["Text"], True) is False


class TestModalityJudge:
    def test_init_defaults_to_official_google_endpoint(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        with patch("mmi.judge.Client") as mock_client:
            ModalityJudge(model="gemini-test", api_key_env="GOOGLE_API_KEY")

        http_options = mock_client.call_args.kwargs["http_options"]
        assert not getattr(http_options, "base_url", None)
        assert not getattr(http_options, "headers", None)

    def test_init_honours_explicit_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        with patch("mmi.judge.Client") as mock_client:
            ModalityJudge(
                model="gemini-test",
                api_key_env="GOOGLE_API_KEY",
                base_url="https://example.invalid/v1",
            )

        http_options = mock_client.call_args.kwargs["http_options"]
        assert http_options.base_url == "https://example.invalid/v1"
        assert http_options.api_version == "v1"
        assert not getattr(http_options, "headers", None)

    def _make_judge(self, response_text: str = "", side_effect=None):
        """Create a ModalityJudge with a mocked Google GenAI client."""
        judge = ModalityJudge.__new__(ModalityJudge)
        judge._model = "test-model"

        mock_response = MagicMock()
        mock_response.text = response_text

        mock_aio = MagicMock()
        if side_effect:
            mock_aio.models.generate_content = AsyncMock(side_effect=side_effect)
        else:
            mock_aio.models.generate_content = AsyncMock(return_value=mock_response)

        judge._client = MagicMock()
        judge._client.aio = mock_aio
        return judge

    @pytest.mark.asyncio
    async def test_no_modalities_detected_by_default(self):
        """When model merely describes media, detected_modalities is empty."""
        judge = self._make_judge(
            '{"detected_modalities":[],"reasoning":"model described the image in text, did not produce it"}'
        )
        result = await judge.evaluate(
            "prompt",
            "Here is what the image would look like...",
            {"expected": ["Image"]},
        )
        assert result.detected_modalities == []

    @pytest.mark.asyncio
    async def test_parser_miss_detected(self):
        """Judge catches a base64 image the parser missed."""
        judge = self._make_judge(
            '{"detected_modalities":["Image"],"reasoning":"base64 PNG found"}'
        )
        result = await judge.evaluate(
            "prompt", "data:image/png;base64,iVBOR...", {"expected": ["Image"]}
        )
        assert result.detected_modalities == ["Image"]

    @pytest.mark.asyncio
    async def test_refusal_returns_empty(self):
        """Refusals yield no detected modalities."""
        judge = self._make_judge(
            '{"detected_modalities":[],"reasoning":"refusal, nothing produced"}'
        )
        result = await judge.evaluate(
            "prompt", "I can't generate images", {"expected": ["Image"]}
        )
        assert result.detected_modalities == []

    @pytest.mark.asyncio
    async def test_invalid_json_fails_open(self):
        """Invalid JSON → default JudgeResult (fail open, don't block eval)."""
        judge = self._make_judge("not json at all")
        result = await judge.evaluate("p", "t", {"expected": ["Image"]})
        assert result.detected_modalities == []

    @pytest.mark.asyncio
    async def test_api_error_fails_open(self):
        """API error → default JudgeResult (fail open)."""
        judge = self._make_judge(side_effect=Exception("permission denied"))
        result = await judge.evaluate("p", "t", {"expected": ["Image"]})
        assert result.detected_modalities == []

    @pytest.mark.asyncio
    async def test_markdown_fences_stripped(self):
        """JSON wrapped in markdown code fences is still parsed correctly."""
        judge = self._make_judge(
            '```json\n{"detected_modalities":["Audio"],"reasoning":"wav data found"}\n```'
        )
        result = await judge.evaluate("p", "t", {"expected": ["Audio"]})
        assert result.detected_modalities == ["Audio"]
        assert result.reasoning == "wav data found"

    @pytest.mark.asyncio
    async def test_judge_model_recorded(self):
        """judge_model field records which model was used."""
        judge = self._make_judge('{"detected_modalities":[],"reasoning":"ok"}')
        result = await judge.evaluate("p", "t", {"expected": ["Image"]})
        assert result.judge_model == "test-model"

    @pytest.mark.asyncio
    async def test_invalid_modality_names_filtered(self):
        """Modality names not in VALID_MODALITIES are silently dropped."""
        judge = self._make_judge(
            '{"detected_modalities":["Image","InvalidMod",""],"reasoning":"test"}'
        )
        result = await judge.evaluate("p", "t", {"expected": ["Image"]})
        assert result.detected_modalities == ["Image"]

    @pytest.mark.asyncio
    async def test_multiple_modalities_detected(self):
        """Judge can detect multiple missed modalities at once."""
        judge = self._make_judge(
            '{"detected_modalities":["Image","Audio"],"reasoning":"both found"}'
        )
        result = await judge.evaluate(
            "prompt",
            "base64 data...",
            {"expected": ["Image", "Audio"], "detected": ["Text"]},
        )
        assert set(result.detected_modalities) == {"Image", "Audio"}


class TestJudgeDocumentDefinition:
    """The judge's Document definition must not outrun the scoring table.

    Scoring unions structural and judge evidence, so a format the judge is told
    to credit but ``classify_mime`` refuses becomes lenient-only credit that
    strict scoring can never see.
    """

    @pytest.mark.parametrize(
        "mime",
        [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/rtf",
            "application/epub+zip",
            "application/vnd.oasis.opendocument.text",
            "text/csv",
            "text/plain",
        ],
    )
    def test_every_credited_format_classifies_as_document(self, mime):
        assert classify_mime(mime) == "Document"

    @pytest.mark.parametrize("mime", ["text/html", "text/markdown", "application/json"])
    def test_excluded_formats_are_refused_by_the_table(self, mime):
        assert classify_mime(mime) is None

    def test_prompt_names_the_exclusions(self):
        """Pinned verbatim: reintroducing HTML here silently creates the gap."""
        assert "HTML, Markdown and JSON are NOT Documents" in JUDGE_SYSTEM_PROMPT
