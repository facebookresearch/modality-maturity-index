# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Mocked end-to-end pipeline test for runner execution.

Covers:
- dataset loading
- provider send path
- scoring + JSONL persistence
- summary computation + summary JSON persistence
"""

from __future__ import annotations

import json

import pytest

from mmi.config import HarnessConfig, ModelRunConfig
from mmi.detection import CAPTURED, PROVIDER_INLINE
from mmi.models import CapturedAsset, EvalPrompt, ProviderResponse, RequestRecord
from mmi.runner import run


@pytest.mark.asyncio
async def test_run_end_to_end_mocked_pipeline(tmp_path, monkeypatch):
    prompt = EvalPrompt(
        prompt_id="p-e2e-1",
        prompt_text="Say hello",
        input_modalities=["Text"],
        output_modalities=["Text"],
        input_files=[],
    )

    async def _fake_send(_prompt: EvalPrompt) -> ProviderResponse:
        return ProviderResponse(
            prompt_id=_prompt.prompt_id,
            run_name="mock-model",
            provider="openai",
            model="mock-model-id",
            response_text="hello",
            raw_response={"mock": True, "content": "hello"},
            request=RequestRecord(
                provider="openai",
                api="responses",
                model="mock-model-id",
                user_prompt=_prompt.prompt_text,
                input_files=list(_prompt.input_files),
            ),
            output_assets=[
                CapturedAsset(
                    asset_id="p-e2e-1_output_0",
                    prompt_id="p-e2e-1",
                    modality="Image",
                    source_type="provider_inline",
                    delivery=PROVIDER_INLINE,
                    mime_type="image/png",
                    local_path=str(tmp_path / "asset.png"),
                    capture_status=CAPTURED,
                    size_bytes=3,
                )
            ],
        )

    class _FakeProvider:
        def set_output_asset_dir(self, root_dir):
            self.root_dir = root_dir

        async def send(self, p: EvalPrompt) -> ProviderResponse:
            return await _fake_send(p)

    monkeypatch.setattr("mmi.runner.load_dataset", lambda: [prompt])
    monkeypatch.setattr("mmi.runner.RESULTS_DIR", tmp_path)
    monkeypatch.setattr("mmi.runner._make_provider", lambda *_: _FakeProvider())

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = HarnessConfig(
        sample=False,
        judge_enabled=False,
        models=[
            ModelRunConfig(
                name="mock-model",
                provider="openai",
                model_id="mock-model-id",
                api_key_env="OPENAI_API_KEY",
            )
        ],
    )

    summary = await run(config, "testconfig")

    assert "mock-model" in summary
    assert summary["mock-model"]["_total"] == 1
    assert summary["mock-model"]["_errors"] == 0
    assert summary["mock-model"]["Text"]["lenient"] == 1.0
    assert summary["mock-model"]["Text"]["strict"] == 1.0

    out_dir = tmp_path / "testconfig"
    jsonl_files = sorted(out_dir.glob("*_mock-model.jsonl"))
    summary_files = sorted(out_dir.glob("*_summary.json"))

    assert len(jsonl_files) == 1
    assert len(summary_files) == 1

    lines = jsonl_files[0].read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["prompt_id"] == "p-e2e-1"
    assert row["all_pass_lenient"] is True
    assert row["all_pass_strict"] is True
    assert row["is_error"] is False
    assert isinstance(row.get("raw_response"), dict)
    assert row["request"]["user_prompt"] == "Say hello"
    assert row["output_assets"][0]["asset_id"] == "p-e2e-1_output_0"

    persisted_summary = json.loads(summary_files[0].read_text())
    assert persisted_summary["mock-model"]["_total"] == 1
