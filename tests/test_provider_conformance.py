# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Provider conformance suite.

Run this against your adapter to find out whether you got the contract right.
A contract without a conformance suite is a docstring, so this file is the
actual specification — ``docs/ADDING_A_PROVIDER.md`` is the prose version.

To include your provider, add it to :data:`PROVIDER_CASES`. Each entry supplies
a factory that returns an adapter primed to produce a given scenario, so the
suite can exercise every adapter identically without any of them needing a
network.
"""

import json

import pytest

from mmi.detection import (
    CAPTURED,
    EXTERNAL_URL,
    PROVIDER_INLINE,
    PROVIDER_TOOL,
    REFERENCE_ONLY,
    SCORING_NATIVE,
    SCORING_NONE,
    SCORING_URL,
    scoring_class,
)
from mmi.models import EvalPrompt
from mmi.providers.base import ProviderError
from mmi.providers.stub_provider import TINY_PNG_B64, StubProvider

PROMPT = EvalPrompt(
    prompt_id="p1",
    prompt_text="make me a picture",
    input_modalities=["Text"],
    output_modalities=["Image"],
)


def stub(scripted) -> StubProvider:
    return StubProvider(model="stub-1", scripted_response=scripted)


#: Adapters under test. Add yours here.
#:
#: A factory takes a scenario name and returns either a primed provider or
#: ``None`` to skip that scenario (for capabilities your system genuinely
#: cannot express).
PROVIDER_CASES = {
    "stub": lambda scenario: _stub_for(scenario),
}


def _stub_for(scenario: str):
    scripts = {
        "artifact_with_bytes": {
            "text": "here it is",
            "artifacts": [
                {
                    "mime_type": "image/png",
                    "data_b64": TINY_PNG_B64,
                    "delivery": PROVIDER_INLINE,
                }
            ],
        },
        "url_in_prose": {
            "text": "I cannot make one, but see https://www.youtube.com/watch?v=abc",
            "artifacts": [],
        },
        "tool_produced_file": {
            "text": "done",
            "artifacts": [
                {
                    "mime_type": "image/png",
                    "data_b64": TINY_PNG_B64,
                    "delivery": PROVIDER_TOOL,
                }
            ],
        },
        "opaque_mime": {
            "text": "attached",
            "artifacts": [
                {
                    "mime_type": "",
                    "data_b64": TINY_PNG_B64,
                    "delivery": PROVIDER_INLINE,
                    "modality_hint": "Audio",
                }
            ],
        },
        "bytesless_reference": {
            "text": "generated",
            "artifacts": [
                {
                    "mime_type": "image/png",
                    "url": "https://example.invalid/generated.png",
                    "delivery": PROVIDER_TOOL,
                }
            ],
        },
        "hidden_tool_trace": {
            "text": "Sorry, I cannot generate video.",
            "artifacts": [],
            "tool_calls": [
                {
                    "name": "video_gen",
                    "arguments": {"prompt": "see https://vimeo.com/999 for style"},
                }
            ],
        },
        "empty": {"text": "", "artifacts": []},
    }
    if scenario not in scripts:
        return None
    return stub(scripts[scenario])


ADAPTERS = sorted(PROVIDER_CASES)


async def run(name: str, scenario: str):
    provider = PROVIDER_CASES[name](scenario)
    if provider is None:
        pytest.skip(f"{name} cannot express scenario {scenario!r}")
    return await provider.send(PROMPT)


@pytest.mark.parametrize("name", ADAPTERS)
class TestProviderConformance:
    async def test_artifact_with_bytes_is_native(self, name):
        response = await run(name, "artifact_with_bytes")
        captured = [a for a in response.output_assets if a.capture_status == CAPTURED]
        assert captured, "an artifact with bytes must be captured"
        assert any(scoring_class(a) == SCORING_NATIVE for a in captured)
        assert response.detection.modalities["Image"].detected_native is True

    async def test_url_in_prose_is_url_class_not_native(self, name):
        response = await run(name, "url_in_prose")
        modalities = response.detection.modalities
        assert modalities["Video"].detected_via_url is True
        assert modalities["Video"].detected_native is False

    async def test_tool_produced_file_records_tool_provenance(self, name):
        response = await run(name, "tool_produced_file")
        tool_assets = [a for a in response.output_assets if a.delivery == PROVIDER_TOOL]
        assert tool_assets, "platform-tool artifacts must be recorded distinctly"
        assert response.detection.modalities["Image"].detected_native is True

    async def test_opaque_mime_uses_hint_and_records_it(self, name):
        response = await run(name, "opaque_mime")
        hinted = [a for a in response.output_assets if a.modality_hint]
        assert hinted, "the hint must survive into the trace"
        assert response.detection.modalities["Audio"].detected_native is True

    async def test_bytesless_reference_earns_no_native_credit(self, name):
        response = await run(name, "bytesless_reference")
        assert all(
            a.capture_status != CAPTURED
            for a in response.output_assets
            if a.delivery == PROVIDER_TOOL
        )
        assert response.detection.modalities["Image"].detected_native is False

    async def test_hidden_tool_trace_gives_no_url_credit(self, name):
        response = await run(name, "hidden_tool_trace")
        assert response.detection.modalities["Video"].detected_via_url is False
        assert response.detection.modalities["Video"].detected_native is False
        assert response.tool_calls, "tool calls are still recorded, just not scored"

    async def test_raw_response_is_json_serialisable(self, name):
        response = await run(name, "artifact_with_bytes")
        # A repr string here is the specific historical failure this guards.
        dumped = json.dumps(response.raw_response, ensure_ascii=False)
        assert not dumped.strip().startswith('"<')

    async def test_asset_ids_are_unique(self, name):
        response = await run(name, "artifact_with_bytes")
        ids = [a.asset_id for a in response.output_assets]
        assert len(ids) == len(set(ids))

    async def test_provider_does_not_score(self, name):
        response = await run(name, "artifact_with_bytes")
        for asset in response.output_assets:
            assert not hasattr(asset, "detected_native")
            assert not hasattr(asset, "pass_strict")

    async def test_empty_response_detects_nothing(self, name):
        response = await run(name, "empty")
        assert all(
            not d.detected_native and not d.detected_via_url
            for d in response.detection.modalities.values()
        )


class TestProviderErrorsStillCount:
    """A provider error is a scored outcome, not a missing row."""

    async def test_error_produces_an_error_result(self):
        class Failing(StubProvider):
            async def _call_system(self, prompt):
                raise ProviderError("upstream exploded")

        response = await Failing(model="stub-err", max_retries=1).send(PROMPT)

        assert response.is_error is True
        assert "upstream exploded" in (response.error or "")
        assert response.detection.is_error is True

    async def test_error_result_scores_as_a_failure(self):
        from mmi.scorer import score

        class Failing(StubProvider):
            async def _call_system(self, prompt):
                raise ProviderError("upstream exploded")

        response = await Failing(model="stub-err", max_retries=1).send(PROMPT)
        result = score(PROMPT, response)

        assert result.is_error is True
        assert result.recall == 0.0
        assert result.f1 == 0.0
        assert result.all_pass_lenient is False


class TestNoModalityLogicInProviders:
    """Adjudication must not leak back into adapters."""

    def test_provider_modules_do_not_classify(self):
        import pathlib

        import mmi.providers as providers

        banned = ("detected_native", "detected_via_url", "pass_strict", "pass_lenient")
        root = pathlib.Path(providers.__file__).parent
        offenders = []
        for path in root.glob("*.py"):
            text = path.read_text()
            for token in banned:
                if token in text:
                    offenders.append(f"{path.name}: {token}")
        assert not offenders, f"providers must not decide modalities: {offenders}"


class TestScoringClassTable:
    @pytest.mark.parametrize(
        "delivery,status,expected",
        [
            (PROVIDER_INLINE, CAPTURED, SCORING_NATIVE),
            (PROVIDER_TOOL, CAPTURED, SCORING_NATIVE),
            (PROVIDER_INLINE, REFERENCE_ONLY, SCORING_NONE),
            (EXTERNAL_URL, CAPTURED, SCORING_URL),
            (EXTERNAL_URL, REFERENCE_ONLY, SCORING_URL),
        ],
    )
    def test_table(self, delivery, status, expected):
        from mmi.models import CapturedAsset

        asset = CapturedAsset(
            asset_id="a",
            prompt_id="p",
            modality="Image",
            source_type="t",
            delivery=delivery,
            mime_type="image/png",
            capture_status=status,
        )
        assert scoring_class(asset) == expected
