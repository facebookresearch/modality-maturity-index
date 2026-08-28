# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Tests for route-level input support metadata.

Input support is advisory only: it annotates results, and never gates a
request or removes a prompt from any denominator.
"""

from mmi.input_support import evaluate_input_support, mark_observed_error
from mmi.models import EvalPrompt


def test_native_openai_audio_preflight_is_rerunnable():
    support = evaluate_input_support(
        provider="openai",
        base_url="",
        prompt=EvalPrompt(
            prompt_id="p",
            prompt_text="transcribe",
            input_modalities=["Audio"],
            output_modalities=["Text"],
            input_files=["p697879.mp3"],
        ),
    )

    assert support.route == "native_openai"
    assert support.preflight_status == "unsupported"
    assert support.rerun_recommended is True
    assert support.rerun_reason == "unsupported_input_modality"
    assert support.unsupported_modalities == ["Audio"]


def test_custom_route_is_labelled_custom():
    support = evaluate_input_support(
        provider="openai",
        base_url="https://example.invalid/v1",
        prompt=EvalPrompt(
            prompt_id="p",
            prompt_text="hi",
            input_modalities=["Text"],
            output_modalities=["Text"],
        ),
    )

    assert support.route == "custom_openai"


def test_unknown_provider_gets_no_preflight_opinion():
    support = evaluate_input_support(
        provider="stub",
        base_url="",
        prompt=EvalPrompt(
            prompt_id="p",
            prompt_text="transcribe",
            input_modalities=["Audio"],
            output_modalities=["Text"],
            input_files=["p697879.mp3"],
        ),
    )

    assert support.preflight_status == "supported"
    assert support.unsupported_modalities == []


def test_observed_unsupported_error_overrides_preflight():
    support = evaluate_input_support(
        provider="openai",
        base_url="",
        prompt=EvalPrompt(
            prompt_id="p",
            prompt_text="transcribe",
            input_modalities=["Audio"],
            output_modalities=["Text"],
            input_files=["p697879.mp3"],
        ),
    )

    marked = mark_observed_error(
        support,
        ValueError("Invalid value: 'input_audio'. Supported values are input_text."),
    )

    assert marked.observed_status == "unsupported"
    assert marked.rerun_reason == "provider_rejected_input"
    assert marked.provider_error_type == "unsupported_input"
    assert marked.rerun_recommended is True


def test_observed_success_clears_rerun_recommendation_for_image():
    support = evaluate_input_support(
        provider="anthropic",
        base_url="",
        prompt=EvalPrompt(
            prompt_id="p",
            prompt_text="describe",
            input_modalities=["Image", "Text"],
            output_modalities=["Text"],
            input_files=["p139334.jpg"],
        ),
    )

    assert support.preflight_status == "supported"
    assert support.rerun_recommended is False
