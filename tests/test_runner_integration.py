# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Integration tests for runner and config with judge support."""

import json
from unittest.mock import AsyncMock

import pytest

from mmi.config import (
    DEFAULT_JUDGE_MODEL,
    HarnessConfig,
    ModelRunConfig,
    load_config,
)
from mmi.judge import should_call_judge
from mmi.models import (
    DetectionResult,
    EvalPrompt,
    EvalResult,
    ModalityDetection,
    ModalityScore,
)
from mmi.runner import (
    _compute_summary,
    _dict_to_eval_result,
    _effective_runner_deadline,
    _make_provider,
    _result_to_dict,
    _results_config_name,
    _run_model,
)


class TestConfigJudgeFields:
    def test_defaults_when_absent(self, tmp_path):
        """Config without judge fields → judge_enabled=False, defaults for model/key."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[settings]\nconcurrency = 3\n[[models]]\nname="t"\nprovider="openai"\nmodel_id="gpt-4"'
        )
        # Temporarily override CONFIGS_DIR for this test
        import mmi.config as cfg_mod

        orig = cfg_mod.CONFIGS_DIR
        cfg_mod.CONFIGS_DIR = tmp_path
        try:
            config = load_config("test.toml")
            assert config.judge_enabled is False
            assert config.judge_model == DEFAULT_JUDGE_MODEL
            assert config.judge_api_key_env == "GOOGLE_API_KEY"
            assert config.models[0].api_key_env == "OPENAI_API_KEY"
            assert config.models[0].base_url == ""
        finally:
            cfg_mod.CONFIGS_DIR = orig

    def test_explicit_judge_settings(self, tmp_path):
        """Config with explicit judge fields → parsed correctly."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[settings]\njudge_enabled = true\njudge_model = "claude-3"\n'
            'judge_api_key_env = "ANTHROPIC_API_KEY"\n'
            '[[models]]\nname="t"\nprovider="openai"\nmodel_id="gpt-4"'
        )
        import mmi.config as cfg_mod

        orig = cfg_mod.CONFIGS_DIR
        cfg_mod.CONFIGS_DIR = tmp_path
        try:
            config = load_config("test.toml")
            assert config.judge_enabled is True
            assert config.judge_model == "claude-3"
            assert config.judge_api_key_env == "ANTHROPIC_API_KEY"
        finally:
            cfg_mod.CONFIGS_DIR = orig


class TestConfigLookup:
    def test_existing_explicit_path_precedes_checkout_config(
        self, tmp_path, monkeypatch
    ):
        import mmi.config as cfg_mod

        explicit_dir = tmp_path / "explicit"
        checkout_dir = tmp_path / "checkout"
        explicit_dir.mkdir()
        checkout_dir.mkdir()
        explicit_path = explicit_dir / "same.toml"
        explicit_path.write_text(
            '[[models]]\nname="explicit"\nprovider="stub"\nmodel_id="explicit"'
        )
        (checkout_dir / "same.toml").write_text(
            '[[models]]\nname="checkout"\nprovider="stub"\nmodel_id="checkout"'
        )
        monkeypatch.setattr(cfg_mod, "CONFIGS_DIR", checkout_dir)

        config = load_config(str(explicit_path))

        assert config.models[0].name == "explicit"

    @pytest.mark.parametrize(
        "config_name",
        ["../default.toml", "configs/default.toml", r"configs\\default.toml"],
    )
    def test_relative_names_are_not_resolved_against_the_configs_dir(
        self, config_name, tmp_path, monkeypatch
    ):
        import mmi.config as cfg_mod

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cfg_mod, "CONFIGS_DIR", tmp_path / "missing")

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(config_name)

    def test_missing_bare_config_raises_file_not_found(self, tmp_path, monkeypatch):
        import mmi.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "CONFIGS_DIR", tmp_path)

        with pytest.raises(FileNotFoundError, match="does-not-exist.toml"):
            load_config("does-not-exist.toml")


class TestResultsConfigName:
    @pytest.mark.parametrize(
        "config_arg,expected",
        [
            ("default.toml", "default"),
            ("paper_main_eval", "paper_main_eval"),
            ("/tmp/custom/config.toml", "config"),
        ],
    )
    def test_uses_config_stem_only(self, config_arg, expected):
        assert _results_config_name(config_arg) == expected

    def test_absolute_path_does_not_retain_parent_directories(self, tmp_path):
        config_path = tmp_path / "nested" / "release.toml"

        result_name = _results_config_name(str(config_path.resolve()))

        assert result_name == "release"
        assert "/" not in result_name
        assert "\\" not in result_name

    @pytest.mark.parametrize("rejudge_mode", [False, True])
    @pytest.mark.parametrize(
        "config_arg,expected_name",
        [
            ("default.toml", "default"),
            ("/tmp/custom/release.toml", "release"),
        ],
    )
    def test_main_loads_original_path_but_uses_stem_for_results(
        self, monkeypatch, config_arg, expected_name, rejudge_mode
    ):
        import sys

        import mmi.runner as runner_mod

        calls = {}
        config = HarnessConfig()

        def fake_load_config(value):
            calls["loaded"] = value
            return config

        async def fake_run(value, config_name, *, resume=False):
            calls["executed"] = (value, config_name, resume)
            return {}

        async def fake_rejudge(value, config_name):
            calls["executed"] = (value, config_name, None)
            return {}

        monkeypatch.setattr(runner_mod, "load_config", fake_load_config)
        monkeypatch.setattr(runner_mod, "run", fake_run)
        monkeypatch.setattr(runner_mod, "rejudge", fake_rejudge)
        argv = ["mmi", "--config", config_arg]
        if rejudge_mode:
            argv.append("--rejudge")
        monkeypatch.setattr(sys, "argv", argv)

        assert runner_mod.main() == 0
        assert calls["loaded"] == config_arg
        assert calls["executed"] == (
            config,
            expected_name,
            False if not rejudge_mode else None,
        )


class TestRunnerJudgeIntegration:
    @pytest.mark.asyncio
    async def test_judge_called_only_when_should_call_judge_true(self):
        """should_call_judge returns True only for missing non-text modalities."""
        mods_all_found = {
            m: ModalityDetection()
            for m in ["Text", "Image", "Audio", "Video", "Document"]
        }
        mods_all_found["Text"].detected_native = True
        mods_all_found["Image"].detected_native = True
        det_all = DetectionResult(modalities=mods_all_found)

        mods_missing = {
            m: ModalityDetection()
            for m in ["Text", "Image", "Audio", "Video", "Document"]
        }
        mods_missing["Text"].detected_native = True
        det_missing = DetectionResult(modalities=mods_missing)

        # All modalities found natively → judge NOT called
        assert should_call_judge(det_all, ["Text", "Image"], True) is False

        # Missing non-text modality → judge IS called
        assert should_call_judge(det_missing, ["Text", "Image"], True) is True

    @pytest.mark.asyncio
    async def test_judge_not_initialized_when_disabled(self):
        """When judge_enabled=False, should_call_judge returns False."""
        mods = {
            m: ModalityDetection()
            for m in ["Text", "Image", "Audio", "Video", "Document"]
        }
        mods["Text"].detected_native = True
        det = DetectionResult(modalities=mods)
        assert should_call_judge(det, ["Text", "Image"], False) is False

    @pytest.mark.asyncio
    async def test_judge_init_failure_logs_warning(self):
        """When judge API key is missing, ModalityJudge raises EnvironmentError."""
        import os

        from mmi.judge import ModalityJudge

        # Ensure the env var is NOT set
        key = "TEST_NONEXISTENT_KEY_FOR_JUDGE"
        os.environ.pop(key, None)
        with pytest.raises(EnvironmentError):
            ModalityJudge(api_key_env=key)


class TestRunnerDeadline:
    @pytest.mark.parametrize(
        "rubric_enabled,expected",
        [(False, 360), (True, 1560)],
    )
    def test_deadline_includes_fetch_timeout_only_when_fetching(
        self, rubric_enabled, expected
    ):
        config = HarnessConfig(
            request_timeout=300,
            runner_timeout_padding=60,
            rubric_enabled=rubric_enabled,
        )

        assert _effective_runner_deadline(config) == expected

    @pytest.mark.asyncio
    async def test_timeout_uses_and_logs_effective_deadline(
        self, tmp_path, monkeypatch, caplog
    ):
        provider = AsyncMock()
        provider.set_output_asset_dir = lambda _path: None
        monkeypatch.setattr("mmi.runner._make_provider", lambda *_args: provider)

        async def raise_timeout(coro, *, timeout):
            coro.close()
            raise TimeoutError

        wait_for = AsyncMock(side_effect=raise_timeout)
        monkeypatch.setattr("mmi.runner.asyncio.wait_for", wait_for)
        config = HarnessConfig(
            request_timeout=10,
            runner_timeout_padding=5,
            rubric_enabled=True,
        )
        model_run = ModelRunConfig(name="timeout", provider="stub")
        prompt = EvalPrompt(
            prompt_id="p1",
            prompt_text="return text",
            input_modalities=["Text"],
            output_modalities=["Text"],
        )

        with caplog.at_level("ERROR"):
            results = await _run_model(
                model_run, [prompt], config, tmp_path, "20260101_000000"
            )

        deadline = _effective_runner_deadline(config)
        assert wait_for.await_args.kwargs["timeout"] == deadline
        assert f"timed out after {deadline:g}s" in caplog.text
        assert results[0].is_error is True
        assert results[0].error_type == "timeout"


class TestRunnerResumeBehavior:
    @pytest.mark.asyncio
    async def test_run_model_returns_existing_results_when_all_prompts_already_done(
        self, tmp_path, monkeypatch
    ):
        model_run = ModelRunConfig(
            name="resume-test", provider="openai", model_id="gpt-4"
        )
        config = HarnessConfig(concurrency=2, judge_enabled=False)
        timestamp = "20260101_000000"

        results_file = tmp_path / f"{timestamp}_{model_run.name}.jsonl"
        existing_result = {
            "prompt_id": "p1",
            "run_name": model_run.name,
            "provider": model_run.provider,
            "model": model_run.model_id,
            "expected_modalities": ["Text"],
            "per_modality": {
                "Text": {
                    "detected_native": True,
                    "detected_via_url": False,
                    "detected_via_judge": False,
                    "pass_strict": True,
                    "pass_lenient": True,
                }
            },
            "all_pass_lenient": True,
            "all_pass_strict": True,
            "judge_used": False,
            "raw_response": None,
        }
        results_file.write_text(json.dumps(existing_result) + "\n")

        def _fail_make_provider(*_args, **_kwargs):
            raise AssertionError(
                "Provider should not be created when all prompts are done"
            )

        monkeypatch.setattr("mmi.runner._make_provider", _fail_make_provider)

        prompts = [
            EvalPrompt(
                prompt_id="p1",
                prompt_text="return text",
                input_modalities=["Text"],
                output_modalities=["Text"],
            )
        ]

        results = await _run_model(model_run, prompts, config, tmp_path, timestamp)

        assert len(results) == 1
        assert results[0].prompt_id == "p1"
        assert results[0].all_pass_lenient is True


class TestRunnerErrorSemantics:
    def test_eval_result_roundtrip_preserves_error_fields(self):
        result = EvalResult(
            prompt_id="p1",
            run_name="r1",
            provider="openai",
            model="gpt-test",
            expected_modalities=["Text"],
            per_modality={
                "Text": ModalityScore(
                    detected_native=False,
                    detected_via_url=False,
                    detected_via_judge=False,
                    pass_strict=False,
                    pass_lenient=False,
                )
            },
            all_pass_lenient=False,
            all_pass_strict=False,
            judge_used=False,
            is_error=True,
            error_message="upload failed",
            error_type="input_upload_failure",
            raw_response={"detail": "x"},
        )

        as_dict = _result_to_dict(result)
        restored = _dict_to_eval_result(as_dict)

        assert restored.is_error is True
        assert restored.error_message == "upload failed"
        assert restored.error_type == "input_upload_failure"

    def test_eval_result_roundtrip_preserves_tool_calls(self):
        result = EvalResult(
            prompt_id="p1",
            run_name="r1",
            provider="openai",
            model="gpt-test",
            expected_modalities=["Image"],
            per_modality={"Image": ModalityScore(detected_native=True)},
            all_pass_lenient=True,
            all_pass_strict=True,
            tool_calls=[
                {
                    "tool_name": "image_gen",
                    "provider_call_id": "call_1",
                    "arguments": {"prompt": "draw"},
                    "status": "completed",
                    "error": "",
                    "produced_asset_ids": ["asset_1"],
                }
            ],
        )

        as_dict = _result_to_dict(result)
        restored = _dict_to_eval_result(as_dict)

        assert restored.tool_calls[0]["tool_name"] == "image_gen"
        assert restored.tool_calls[0]["produced_asset_ids"] == ["asset_1"]

    def test_summary_counts_errors_as_failures_over_all_prompts(self):
        ok_result = EvalResult(
            prompt_id="ok",
            run_name="r1",
            provider="openai",
            model="gpt-test",
            expected_modalities=["Text"],
            per_modality={
                "Text": ModalityScore(
                    detected_native=True,
                    detected_via_url=False,
                    detected_via_judge=False,
                    pass_strict=True,
                    pass_lenient=True,
                )
            },
            all_pass_lenient=True,
            all_pass_strict=True,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            precision_strict=1.0,
            recall_strict=1.0,
            f1_strict=1.0,
            is_error=False,
        )
        err_result = EvalResult(
            prompt_id="err",
            run_name="r1",
            provider="openai",
            model="gpt-test",
            expected_modalities=["Text"],
            per_modality={
                "Text": ModalityScore(
                    detected_native=False,
                    detected_via_url=False,
                    detected_via_judge=False,
                    pass_strict=False,
                    pass_lenient=False,
                )
            },
            all_pass_lenient=False,
            all_pass_strict=False,
            is_error=True,
            error_message="timed out",
            error_type="timeout",
        )

        summary = _compute_summary({"r1": [ok_result, err_result]})
        model_summary = summary["r1"]

        assert model_summary["_total"] == 2
        assert model_summary["_errors"] == 1
        assert model_summary["_non_error_total"] == 1
        assert model_summary["_overall"] == 0.5
        assert model_summary["_overall_strict"] == 0.5
        assert model_summary["_mean_f1"] == 0.5
        assert model_summary["_mean_f1_strict"] == 0.5
        assert model_summary["Text"]["lenient"] == 0.5
        assert model_summary["Text"]["strict"] == 0.5


class TestRubricImpliesFetching:
    """Rubrics grade content, so enabling them must retrieve the bytes.

    ``fetch_remote_assets`` used to be a ``BaseProvider`` parameter with no path
    from configuration to reach it, so every URL-delivered artifact stayed
    ``reference_only`` and every non-Text rubric over one scored zero by
    construction, in every possible configuration.
    """

    @pytest.mark.parametrize("rubric_enabled", [True, False])
    def test_config_derives_fetching_from_rubrics(self, rubric_enabled):
        assert (
            HarnessConfig(rubric_enabled=rubric_enabled).fetch_remote_assets
            is rubric_enabled
        )

    @pytest.mark.parametrize(
        "provider,model_id",
        [
            ("openai", "gpt-test"),
            ("anthropic", "claude-test"),
            ("gemini", "gemini-test"),
            ("stub", "stub-test"),
        ],
    )
    @pytest.mark.parametrize("rubric_enabled", [True, False])
    def test_every_provider_receives_it(
        self, monkeypatch, provider, model_id, rubric_enabled
    ):
        """A provider that drops the kwarg would silently never fetch."""
        monkeypatch.setenv("TEST_KEY", "x")
        built = _make_provider(
            ModelRunConfig(
                name="m",
                provider=provider,
                model_id=model_id,
                api_key_env="TEST_KEY",
            ),
            HarnessConfig(request_timeout=1, rubric_enabled=rubric_enabled),
        )
        assert built.fetch_remote_assets is rubric_enabled

    def test_manifest_records_the_retrieval_posture(self):
        from mmi.run_manifest import build_manifest

        manifest = build_manifest(
            config=HarnessConfig(rubric_enabled=True),
            config_name="c.toml",
            prompts=[],
            prompt_selection="all",
            timestamp="20260101_000000",
        )
        assert manifest.rubric_profile["fetch_remote_assets"] is True

    def test_make_provider_passes_uniform_tools_but_strips_response_modalities(
        self, monkeypatch
    ):
        monkeypatch.setenv("TEST_KEY", "x")
        config = HarnessConfig(request_timeout=1)

        openai = _make_provider(
            ModelRunConfig(
                name="o",
                provider="openai",
                model_id="gpt-test",
                api_key_env="TEST_KEY",
                tools=["image_gen"],
            ),
            config,
        )
        anthropic = _make_provider(
            ModelRunConfig(
                name="a",
                provider="anthropic",
                model_id="claude-test",
                api_key_env="TEST_KEY",
                tools=["audio_gen"],
            ),
            config,
        )
        gemini = _make_provider(
            ModelRunConfig(
                name="g",
                provider="gemini",
                model_id="gemini-test",
                api_key_env="TEST_KEY",
                tools=["video_gen"],
                response_modalities=["TEXT", "VIDEO"],
            ),
            config,
        )

        assert openai._tools == ["image_gen"]
        assert anthropic._tools == ["audio_gen"]
        assert gemini._tools == ["video_gen"]
        assert gemini._response_modalities == []
        assert "openai.com" in str(openai._client.base_url)
        assert "anthropic.com" in str(anthropic._client.base_url)
        assert (
            "generativelanguage.googleapis.com"
            in gemini._client._api_client._http_options.base_url
        )


class TestRunManifest:
    """Resume is gated on a manifest, not on a filename matching a glob."""

    def _config(self, **overrides):
        base = {
            "models": [ModelRunConfig(name="m1", provider="openai", model_id="gpt-x")],
        }
        base.update(overrides)
        return HarnessConfig(**base)

    def _prompts(self, ids=("p1", "p2")):
        return [
            EvalPrompt(
                prompt_id=pid,
                prompt_text="x",
                input_modalities=["Text"],
                output_modalities=["Text"],
            )
            for pid in ids
        ]

    def _manifest(self, config=None, prompts=None, selection="full"):
        from mmi.run_manifest import build_manifest

        return build_manifest(
            config=config or self._config(),
            config_name="test",
            prompts=prompts or self._prompts(),
            prompt_selection=selection,
        )

    def test_manifest_records_what_produced_the_run(self):
        manifest = self._manifest()

        assert manifest.schema_version >= 1
        assert manifest.prompt_count == 2
        assert manifest.models[0]["provider"] == "openai"
        assert manifest.runtime["python"]
        assert manifest.dataset_prompt_id_hash

    def test_identical_configs_are_compatible(self):
        self._manifest().assert_compatible_with(self._manifest())

    def test_changing_the_model_blocks_resume(self):
        from mmi.run_manifest import IncompatibleManifest

        other = self._config(
            models=[ModelRunConfig(name="m1", provider="openai", model_id="gpt-y")]
        )

        with pytest.raises(IncompatibleManifest, match="config_hash"):
            self._manifest().assert_compatible_with(self._manifest(config=other))

    def test_changing_the_prompt_set_blocks_resume(self):
        from mmi.run_manifest import IncompatibleManifest

        with pytest.raises(IncompatibleManifest, match="dataset_prompt_id_hash"):
            self._manifest().assert_compatible_with(
                self._manifest(prompts=self._prompts(("p1", "p3")))
            )

    def test_changing_the_selection_mode_blocks_resume(self):
        from mmi.run_manifest import IncompatibleManifest

        with pytest.raises(IncompatibleManifest, match="prompt_selection"):
            self._manifest().assert_compatible_with(self._manifest(selection="sample"))

    def test_concurrency_does_not_block_resume(self):
        """Execution details must not invalidate a run."""
        fast = self._config(concurrency=1, verbose=True)
        slow = self._config(concurrency=8, verbose=False)

        self._manifest(config=fast).assert_compatible_with(self._manifest(config=slow))

    def test_manifest_round_trips_through_disk(self, tmp_path):
        from mmi.run_manifest import RunManifest

        original = self._manifest()
        original.write(tmp_path)
        restored = RunManifest.read(tmp_path)

        assert restored.to_dict() == original.to_dict()

    def test_missing_manifest_reads_as_none(self, tmp_path):
        from mmi.run_manifest import RunManifest

        assert RunManifest.read(tmp_path) is None

    def test_tool_profile_records_resolved_backends(self):
        from mmi.config import MediaToolBackend

        config = self._config(
            media_tool_backends={
                "image_gen": MediaToolBackend(provider="google", model="img-v1")
            },
            tool_loop_limit=3,
        )
        manifest = self._manifest(config=config)

        assert manifest.tool_profile["tool_loop_limit"] == 3
        assert manifest.tool_profile["backends"]["image_gen"]["model"] == "img-v1"


class TestRubricPersistence:
    """The paper's rubric metric must survive serialization.

    ``rubric_binary_by_modality`` was computed, set on the result, and then
    silently dropped by ``dataclasses.asdict`` because the field was not
    declared — so the metric ``results.tex`` actually defines never reached
    disk.
    """

    def test_binary_collapse_survives_the_round_trip(self):
        from mmi.models import EvalResult
        from mmi.runner import _dict_to_eval_result, _result_to_dict

        result = EvalResult(
            prompt_id="p1",
            run_name="r",
            provider="openai",
            model="m",
            expected_modalities=["Image"],
            per_modality={},
            all_pass_lenient=False,
            all_pass_strict=False,
            rubric_used=True,
            rubric_score=0.5,
            rubric_binary_by_modality={"Image": 0},
        )

        as_dict = _result_to_dict(result)
        assert as_dict["rubric_binary_by_modality"] == {"Image": 0}
        assert _dict_to_eval_result(as_dict).rubric_binary_by_modality == {"Image": 0}

    def test_as_result_fields_only_sets_declared_fields(self):
        """Anything returned here must exist on EvalResult or it is lost."""
        import dataclasses

        from mmi.models import EvalResult
        from mmi.rubric_scorer import RubricScoreResult

        fields = {f.name for f in dataclasses.fields(EvalResult)}
        produced = RubricScoreResult(True, 1.0, [], "judge").as_result_fields()

        assert set(produced) <= fields, set(produced) - fields

    def test_summary_reports_the_binary_metric_per_modality(self):
        from mmi.runner import _compute_rubric_summary

        rows = [
            {
                "run_name": "m",
                "rubric_used": True,
                "rubric_score": 0.75,
                "response_text": "x",
                "rubric_binary_by_modality": {"Image": 0, "Text": 1},
            },
            {
                "run_name": "m",
                "rubric_used": True,
                "rubric_score": 1.0,
                "response_text": "y",
                "rubric_binary_by_modality": {"Image": 1, "Text": 1},
            },
        ]

        summary = _compute_rubric_summary(rows)["m"]

        assert summary["_rubric_binary_by_modality"] == {"Image": 0.5, "Text": 1.0}
        assert summary["_rubric_mean_score"] == 0.875


class TestEvaluatorErrorVisibility:
    """A degraded judge must not be reportable as a bad model.

    Judge failures score zero — anything else would inflate results — but the
    aggregation has to say how many zeros were the evaluator's fault, or a
    rate-limited run is indistinguishable from a model that produced nothing.
    """

    def _grade(self, modality, score, evaluator_error):
        return {
            "index": 1,
            "id": "c1",
            "rubric": "r",
            "modality": modality,
            "score": score,
            "explanation": "Judge error: boom" if evaluator_error else "fine",
            "evaluator_error": evaluator_error,
        }

    def test_flag_is_structured_not_inferred_from_prose(self):
        """Aggregation must not depend on the explanation wording."""
        from mmi.rubric_scorer import is_evaluator_error

        assert is_evaluator_error(self._grade("Image", 0.0, True))
        assert not is_evaluator_error(self._grade("Image", 0.0, False))
        # Prose that merely looks like a judge error is not one.
        assert not is_evaluator_error(
            {"explanation": "Judge error: as the model described", "score": 1.0}
        )

    def test_evaluator_error_still_scores_zero(self):
        """Excluding it would inflate the metric, so it stays a zero."""
        from mmi.rubric_scorer import collapse_to_binary

        assert collapse_to_binary([self._grade("Image", 0.0, True)]) == {"Image": 0}

    def test_summary_counts_evaluator_errors_separately(self):
        from mmi.runner import _compute_rubric_summary

        rows = [
            {
                "run_name": "m",
                "rubric_used": True,
                "rubric_score": 0.0,
                "response_text": "x",
                "rubric_binary_by_modality": {"Image": 0},
                "rubric_grades": [
                    self._grade("Image", 0.0, True),
                    self._grade("Image", 0.0, True),
                ],
            },
            {
                "run_name": "m",
                "rubric_used": True,
                "rubric_score": 0.0,
                "response_text": "y",
                "rubric_binary_by_modality": {"Image": 0},
                "rubric_grades": [self._grade("Image", 0.0, False)],
            },
        ]

        summary = _compute_rubric_summary(rows)["m"]

        assert summary["_rubric_evaluator_error_grades"] == 2
        assert summary["_rubric_rows_with_evaluator_error"] == 1
        assert summary["_rubric_evaluator_errors_by_modality"] == {"Image": 2}
        # The metric itself is untouched: both rows still count as failures.
        assert summary["_rubric_binary_by_modality"] == {"Image": 0.0}

    def test_clean_run_reports_zero_evaluator_errors(self):
        from mmi.runner import _compute_rubric_summary

        rows = [
            {
                "run_name": "m",
                "rubric_used": True,
                "rubric_score": 1.0,
                "response_text": "x",
                "rubric_binary_by_modality": {"Text": 1},
                "rubric_grades": [self._grade("Text", 1.0, False)],
            }
        ]

        summary = _compute_rubric_summary(rows)["m"]

        assert summary["_rubric_evaluator_error_grades"] == 0
        assert summary["_rubric_rows_with_evaluator_error"] == 0
        assert summary["_rubric_evaluator_errors_by_modality"] == {}

    def test_error_count_survives_the_round_trip(self):
        from mmi.models import EvalResult
        from mmi.runner import _dict_to_eval_result, _result_to_dict

        result = EvalResult(
            prompt_id="p1",
            run_name="r",
            provider="openai",
            model="m",
            expected_modalities=["Image"],
            per_modality={},
            all_pass_lenient=False,
            all_pass_strict=False,
            rubric_used=True,
            rubric_evaluator_errors=3,
        )

        as_dict = _result_to_dict(result)
        assert as_dict["rubric_evaluator_errors"] == 3
        assert _dict_to_eval_result(as_dict).rubric_evaluator_errors == 3

    @pytest.mark.asyncio
    async def test_judge_failure_is_flagged_and_counted_end_to_end(self):
        """The real path: a judge exception becomes a flagged, counted zero."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from mmi.models import EvalPrompt, RubricCriterion
        from mmi.rubric_scorer import RubricJudge, score_prompt_rubrics

        client = MagicMock()
        # "404" is a permanent marker, so this returns without burning retries.
        client.aio.models.generate_content = AsyncMock(
            side_effect=RuntimeError("404 model not found")
        )
        with (
            patch("mmi.rubric_scorer.get_api_key", return_value="k"),
            patch("mmi.rubric_scorer.Client", return_value=client),
        ):
            judge = RubricJudge(model="j", api_key_env="GOOGLE_API_KEY")

        prompt = EvalPrompt(
            prompt_id="p1",
            prompt_text="write something",
            input_modalities=["Text"],
            output_modalities=["Text"],
            rubric_criteria=[
                RubricCriterion(id="c1", criterion="is relevant", modality="Text")
            ],
        )

        result = await score_prompt_rubrics(
            judge=judge, prompt=prompt, response_text="some text", raw_response=None
        )

        grade = result.rubric_grades[0]
        assert grade["evaluator_error"] is True
        assert grade["score"] == 0.0
        assert grade["explanation"].startswith("Judge error:")
        assert result.evaluator_errors == 1
        assert result.as_result_fields()["rubric_evaluator_errors"] == 1


class TestUngradeablePayloadIsExplained:
    """An ungradeable artifact still scores zero, but says why.

    ``_assets_for_modality``'s docstring claimed such an asset was "excluded
    here rather than graded as a failure of the model", while its caller turned
    the resulting empty list into exactly that — a zero indistinguishable from
    the model having produced nothing.
    """

    def _prompt(self):
        from mmi.models import EvalPrompt, RubricCriterion

        return EvalPrompt(
            prompt_id="p1",
            prompt_text="make an image",
            input_modalities=["Text"],
            output_modalities=["Image"],
            rubric_criteria=[
                RubricCriterion(id="c1", criterion="is relevant", modality="Image")
            ],
        )

    class _NeverCalledJudge:
        model = "j"

        async def judge_rubric(self, **kwargs):
            raise AssertionError("the judge must not be called without a payload")

    @pytest.mark.parametrize(
        "capture_status,expected_status,expected_phrase",
        [
            ("reference_only", "reference_only", "bytes were never retrieved"),
            ("failed", "capture_failed", "harness failure, not a model failure"),
            ("skipped", "skipped", "recorded without bytes"),
        ],
    )
    @pytest.mark.asyncio
    async def test_reason_travels_with_the_grade(
        self, capture_status, expected_status, expected_phrase
    ):
        from mmi.rubric_scorer import score_prompt_rubrics

        result = await score_prompt_rubrics(
            judge=self._NeverCalledJudge(),
            prompt=self._prompt(),
            response_text="",
            raw_response=None,
            output_assets=[
                {
                    "asset_id": "a1",
                    "modality": "Image",
                    "capture_status": capture_status,
                }
            ],
        )

        grade = result.rubric_grades[0]
        assert grade["score"] == 0.0
        assert grade["payload_status"] == expected_status
        assert expected_phrase in grade["explanation"]
        # Not the judge's fault, so it must not inflate the evaluator-error count.
        assert grade["evaluator_error"] is False
        assert result.evaluator_errors == 0

    @pytest.mark.asyncio
    async def test_no_asset_at_all_is_absent(self):
        from mmi.rubric_scorer import score_prompt_rubrics

        result = await score_prompt_rubrics(
            judge=self._NeverCalledJudge(),
            prompt=self._prompt(),
            response_text="",
            raw_response=None,
            output_assets=[],
        )

        grade = result.rubric_grades[0]
        assert grade["payload_status"] == "absent"
        assert "No model-produced Image payload" in grade["explanation"]

    def test_summary_counts_payload_statuses(self):
        from mmi.runner import _compute_rubric_summary

        rows = [
            {
                "run_name": "m",
                "rubric_used": True,
                "rubric_score": 0.0,
                "response_text": "x",
                "rubric_binary_by_modality": {"Image": 0},
                "rubric_grades": [
                    {
                        "modality": "Image",
                        "score": 0.0,
                        "payload_status": "capture_failed",
                    },
                    {
                        "modality": "Image",
                        "score": 0.0,
                        "payload_status": "reference_only",
                    },
                    {"modality": "Text", "score": 1.0, "payload_status": "gradeable"},
                ],
            }
        ]

        summary = _compute_rubric_summary(rows)["m"]

        assert summary["_rubric_payload_status_counts"] == {
            "capture_failed": 1,
            "gradeable": 1,
            "reference_only": 1,
        }
