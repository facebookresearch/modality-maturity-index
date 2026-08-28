# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Runner: orchestrate config loading → provider calls → scoring → saving."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import (
    PROVIDER_REGISTRY,
    RESULTS_DIR,
    HarnessConfig,
    ModelRunConfig,
    get_api_key,
    load_config,
)
from .dataset import SAMPLE_PROMPT_IDS, load_dataset
from .detection import PAYLOAD_CAPTURE_FAILED
from .fetch import DEFAULT_TIMEOUT as FETCH_TIMEOUT
from .models import (
    EvalPrompt,
    EvalResult,
    ModalityScore,
    ProviderResponse,
)
from .response_utils import extract_response_text
from .rubric_scorer import RubricJudge, is_evaluator_error, score_prompt_rubrics
from .run_manifest import (
    MANIFEST_FILENAME,
    IncompatibleManifest,
    RunManifest,
    build_manifest,
)
from .scorer import rescore_dict, score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def _make_provider(model_run: ModelRunConfig, config: HarnessConfig):
    """Instantiate the right provider class for a model run config entry."""
    registry = PROVIDER_REGISTRY.get(model_run.provider, {})
    base_url = model_run.base_url
    common = {
        "model": model_run.model_id,
        "run_name": model_run.name,
        "api_key_env": model_run.api_key_env or registry.get("api_key_env", ""),
        "request_timeout": config.request_timeout,
        "max_retries": config.max_retries,
        "retry_backoff": config.retry_backoff,
        "tool_loop_limit": config.tool_loop_limit,
        "media_tool_backends": config.media_tool_backends,
        "provider_tools": model_run.provider_tools,
        "fetch_remote_assets": config.fetch_remote_assets,
    }
    # Tools are offered uniformly so the model can choose an output modality.
    # Provider-level response_modalities are not forwarded because they constrain
    # or suggest the answer modality independently of the prompt.
    model_tools = model_run.tools
    model_response_modalities: list[str] = []

    match model_run.provider:
        case "openai":
            from .providers.openai_provider import OpenAIProvider

            return OpenAIProvider(
                **common,
                api=model_run.api or "responses",
                base_url=base_url,
                tools=model_run.tools,
            )
        case "anthropic":
            from .providers.anthropic_provider import AnthropicProvider

            return AnthropicProvider(**common, base_url=base_url, tools=model_tools)
        case "gemini":
            from .providers.gemini_provider import GeminiProvider

            return GeminiProvider(
                **common,
                base_url=base_url,
                response_modalities=model_response_modalities,
                tools=model_tools,
            )
        case "stub":
            from .providers.stub_provider import StubProvider

            return StubProvider(**common, base_url=base_url, tools=model_tools)
        case _:
            raise ValueError(f"Unknown provider: {model_run.provider}")


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------


def _result_to_dict(result: EvalResult) -> dict:
    d = dataclasses.asdict(result)
    d["produced_modalities"] = result.produced_modalities
    # raw_response is stored as a JSON string on EvalResult; parse it back
    # to a dict so the outer json.dumps() serializes it as a nested object
    # instead of double-encoding it as an escaped string.
    if isinstance(d.get("raw_response"), str):
        try:
            d["raw_response"] = json.loads(d["raw_response"])
        except (json.JSONDecodeError, ValueError):
            pass  # keep as string if not valid JSON
    return d


def _dict_to_eval_result(obj: dict) -> EvalResult:
    per_modality = {
        modality: ModalityScore(**score_data)
        for modality, score_data in obj["per_modality"].items()
    }
    return EvalResult(
        prompt_id=obj["prompt_id"],
        run_name=obj["run_name"],
        provider=obj["provider"],
        model=obj["model"],
        expected_modalities=obj["expected_modalities"],
        per_modality=per_modality,
        all_pass_lenient=obj["all_pass_lenient"],
        all_pass_strict=obj["all_pass_strict"],
        precision=obj.get("precision", 0.0),
        recall=obj.get("recall", 0.0),
        f1=obj.get("f1", 0.0),
        precision_strict=obj.get("precision_strict", 0.0),
        recall_strict=obj.get("recall_strict", 0.0),
        f1_strict=obj.get("f1_strict", 0.0),
        judge_used=obj.get("judge_used", False),
        judge_model=obj.get("judge_model", ""),
        judge_reasoning=obj.get("judge_reasoning", ""),
        is_error=obj.get("is_error", False),
        error_message=obj.get("error_message", ""),
        error_type=obj.get("error_type", ""),
        raw_response=obj.get("raw_response"),
        response_text=obj.get("response_text", ""),
        rubric_used=obj.get("rubric_used", False),
        rubric_score=obj.get("rubric_score"),
        request=obj.get("request"),
        output_assets=obj.get("output_assets", []),
        tool_calls=obj.get("tool_calls", []),
        rubric_grades=obj.get("rubric_grades", []),
        rubric_score_by_modality=obj.get("rubric_score_by_modality", {}),
        rubric_binary_by_modality=obj.get("rubric_binary_by_modality", {}),
        rubric_evaluator_errors=obj.get("rubric_evaluator_errors", 0),
        rubric_judge_model=obj.get("rubric_judge_model", ""),
    )


def _load_existing_results(path: Path) -> list[EvalResult]:
    """Load existing eval results from JSONL for resume support."""
    existing: list[EvalResult] = []
    if not path.exists():
        return existing
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                existing.append(_dict_to_eval_result(obj))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return existing


# ---------------------------------------------------------------------------
# Single model run
# ---------------------------------------------------------------------------


def _effective_runner_deadline(config: HarnessConfig) -> float:
    """Return the outer timeout covering provider work and optional fetching."""
    deadline = config.request_timeout + config.runner_timeout_padding
    if config.fetch_remote_assets:
        deadline += FETCH_TIMEOUT
    return deadline


async def _run_model(
    model_run: ModelRunConfig,
    prompts: list[EvalPrompt],
    config: HarnessConfig,
    results_dir: Path,
    timestamp: str,
) -> list[EvalResult]:
    """Run all prompts through a single model with concurrency control."""
    name_safe = model_run.name.replace("/", "_").replace(" ", "_")
    results_file = results_dir / f"{timestamp}_{name_safe}.jsonl"

    existing_results = _load_existing_results(results_file)
    done_ids = {r.prompt_id for r in existing_results}
    remaining = [p for p in prompts if p.prompt_id not in done_ids]
    if done_ids:
        logger.info(
            "%s: resuming — %d done, %d remaining",
            model_run.name,
            len(done_ids),
            len(remaining),
        )
    if not remaining:
        return existing_results

    provider = _make_provider(model_run, config)
    runner_deadline = _effective_runner_deadline(config)
    assets_dir = results_dir / f"{timestamp}_{name_safe}_assets"
    provider.set_output_asset_dir(assets_dir)

    judge = None
    if config.judge_enabled:
        from .judge import ModalityJudge

        try:
            judge = ModalityJudge(
                model=config.judge_model,
                api_key_env=config.judge_api_key_env,
                base_url=config.judge_base_url,
            )
        except OSError:
            logger.warning("Judge disabled: %s not set", config.judge_api_key_env)

    rubric_judge = None
    if config.rubric_enabled:
        try:
            rubric_judge = RubricJudge(
                model=config.rubric_judge_model,
                api_key_env=config.rubric_judge_api_key_env,
                base_url=config.rubric_judge_base_url,
            )
        except OSError:
            logger.warning(
                "Rubric judge disabled: %s not set", config.rubric_judge_api_key_env
            )

    semaphore = asyncio.Semaphore(config.concurrency)
    write_lock = asyncio.Lock()
    results: list[EvalResult] = list(existing_results)
    completed = 0
    total = len(remaining)

    async def process_prompt(prompt: EvalPrompt) -> EvalResult:
        nonlocal completed
        async with semaphore:
            try:
                response = await asyncio.wait_for(
                    provider.send(prompt), timeout=runner_deadline
                )
            except TimeoutError:
                logger.error(
                    "%s prompt %s timed out after %ds",
                    model_run.name,
                    prompt.prompt_id,
                    runner_deadline,
                )
                response = ProviderResponse(
                    prompt_id=prompt.prompt_id,
                    run_name=model_run.name,
                    provider=model_run.provider,
                    model=model_run.model_id,
                    error="prompt timed out",
                    is_error=True,
                    error_type="timeout",
                )

            judge_result = None
            if judge:
                from .judge import should_call_judge

                if should_call_judge(
                    response.detection, prompt.output_modalities, config.judge_enabled
                ):
                    judge_result = await judge.evaluate(
                        prompt_text=prompt.prompt_text,
                        text_content=response.detection.text_content,
                        raw_response=response.raw_response,
                    )

            result = score(prompt, response, judge_result)
            if rubric_judge and prompt.rubric_criteria:
                rubric_result = await score_prompt_rubrics(
                    judge=rubric_judge,
                    prompt=prompt,
                    response_text=response.detection.text_content,
                    raw_response=response.raw_response,
                    output_assets=[
                        dataclasses.asdict(asset) for asset in response.output_assets
                    ],
                )
                for key, value in rubric_result.as_result_fields().items():
                    setattr(result, key, value)

            async with write_lock:
                with open(results_file, "a") as f:
                    f.write(json.dumps(_result_to_dict(result)) + "\n")

            completed += 1
            status = "✅" if result.all_pass_lenient else "❌"
            logger.info(
                "%s [%d/%d] %s %s — expected=%s produced=%s",
                model_run.name,
                completed,
                total,
                status,
                prompt.prompt_id,
                result.expected_modalities,
                result.produced_modalities,
            )
            return result

    tasks = [process_prompt(p) for p in remaining]
    new_results = await asyncio.gather(*tasks)
    results.extend(new_results)

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _compute_summary(all_results: dict[str, list[EvalResult]]) -> dict:
    """Compute per-model, per-modality pass rates with strict/lenient breakdown."""
    summary: dict = {}

    for run_name, results in all_results.items():
        if not results:
            continue

        modality_counts: dict[str, int] = {}
        modality_passes_lenient: dict[str, int] = {}
        modality_passes_strict: dict[str, int] = {}

        for r in results:
            expected = set(r.expected_modalities)
            for modality in expected:
                ms = r.per_modality.get(modality)
                if ms is None:
                    continue
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
                if ms.pass_lenient:
                    modality_passes_lenient[modality] = (
                        modality_passes_lenient.get(modality, 0) + 1
                    )
                if ms.pass_strict:
                    modality_passes_strict[modality] = (
                        modality_passes_strict.get(modality, 0) + 1
                    )

        total = len(results)
        error_count = sum(1 for r in results if r.is_error)
        all_pass_lenient_count = sum(1 for r in results if r.all_pass_lenient)
        all_pass_strict_count = sum(1 for r in results if r.all_pass_strict)

        # Aggregate P/R/F1 across all results
        sum_p = sum_r = sum_f1 = 0.0
        sum_ps = sum_rs = sum_f1s = 0.0
        for r in results:
            sum_p += r.precision
            sum_r += r.recall
            sum_f1 += r.f1
            sum_ps += r.precision_strict
            sum_rs += r.recall_strict
            sum_f1s += r.f1_strict

        model_summary = {}
        for mod, count in modality_counts.items():
            model_summary[mod] = {
                "lenient": modality_passes_lenient.get(mod, 0) / count,
                "strict": modality_passes_strict.get(mod, 0) / count,
                "count": count,
            }

        model_summary["_overall"] = all_pass_lenient_count / total
        model_summary["_overall_strict"] = all_pass_strict_count / total
        model_summary["_mean_precision"] = sum_p / total
        model_summary["_mean_recall"] = sum_r / total
        model_summary["_mean_f1"] = sum_f1 / total
        model_summary["_mean_precision_strict"] = sum_ps / total
        model_summary["_mean_recall_strict"] = sum_rs / total
        model_summary["_mean_f1_strict"] = sum_f1s / total
        model_summary["_total"] = total
        model_summary["_errors"] = error_count
        model_summary["_non_error_total"] = total - error_count

        summary[run_name] = model_summary

    return summary


def _print_summary(summary: dict, title: str = "MMI Eval Summary") -> None:
    """Pretty-print a summary dict to stdout."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    for run_name, rates in summary.items():
        print(f"\n  {run_name}:")
        for key, value in sorted(rates.items()):
            if key.startswith("_"):
                continue
            lenient = value.get("lenient", 0)
            strict = value.get("strict", 0)
            print(f"    {key}: {lenient:.1%} (strict: {strict:.1%})")
        overall = rates.get("_overall", 0)
        overall_strict = rates.get("_overall_strict", 0)
        print(f"    Overall (lenient): {overall:.1%}")
        print(f"    Overall (strict):  {overall_strict:.1%}")
        mean_f1 = rates.get("_mean_f1", 0)
        mean_f1s = rates.get("_mean_f1_strict", 0)
        mean_p = rates.get("_mean_precision", 0)
        mean_r = rates.get("_mean_recall", 0)
        print(f"    Mean P/R/F1:       {mean_p:.3f} / {mean_r:.3f} / {mean_f1:.3f}")
        print(f"    Mean F1 (strict):  {mean_f1s:.3f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def _find_latest_run(results_dir: Path, model_name: str) -> str | None:
    """Find the timestamp of the most recent results file for a model."""
    name_safe = model_name.replace("/", "_").replace(" ", "_")
    pattern = f"*_{name_safe}.jsonl"
    matches = sorted(results_dir.glob(pattern), reverse=True)
    if matches:
        # Extract timestamp from filename: <timestamp>_<model>.jsonl
        stem = matches[0].stem  # e.g. "20260403_210552_gemini-3.1-pro"
        # timestamp is everything before the last occurrence of _<name_safe>
        suffix = f"_{name_safe}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return None


async def run(config: HarnessConfig, config_name: str, resume: bool = False) -> dict:
    """Run the eval across all models specified in the config.

    Args:
        config: Harness configuration.
        config_name: Name of the config (used as results subdirectory).
        resume: If True, find the most recent incomplete run and resume it
                instead of starting fresh.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    results_dir = RESULTS_DIR / config_name
    results_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_dataset()
    if config.prompt_ids:
        requested_ids = set(config.prompt_ids)
        prompts = [p for p in prompts if p.prompt_id in requested_ids]
        found_ids = {p.prompt_id for p in prompts}
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            raise ValueError(f"Config requested unknown prompt_ids: {missing_ids}")
        logger.info("Prompt ID mode: filtered to %d prompts", len(prompts))
        prompt_selection = "prompt_ids"
    elif config.sample:
        prompts = [p for p in prompts if p.prompt_id in SAMPLE_PROMPT_IDS]
        logger.info("Sample mode: filtered to %d prompts", len(prompts))
        prompt_selection = "sample"
    elif config.max_prompts:
        prompts = prompts[: config.max_prompts]
        prompt_selection = f"max_prompts:{config.max_prompts}"
    else:
        prompt_selection = "full"
    logger.info("Running %d prompts", len(prompts))

    # A manifest, not a filename, decides whether a run can be resumed.
    manifest = build_manifest(
        config=config,
        config_name=config_name,
        prompts=prompts,
        prompt_selection=prompt_selection,
        timestamp=timestamp,
    )
    existing = RunManifest.read(results_dir)
    if resume:
        if existing is None:
            raise IncompatibleManifest(
                f"Cannot resume: no {MANIFEST_FILENAME} in {results_dir}. "
                "Runs from before manifests were recorded cannot be resumed safely."
            )
        existing.assert_compatible_with(manifest)
        logger.info(
            "Resuming run %s (config hash %s)",
            existing.run_timestamp,
            existing.config_hash,
        )
    else:
        manifest.write(results_dir)

    available: list[ModelRunConfig] = []
    for model_run in config.models:
        try:
            if model_run.api_key_env:
                get_api_key(model_run.api_key_env)
            available.append(model_run)
        except OSError:
            logger.warning(
                "Skipping %s: %s not set", model_run.name, model_run.api_key_env
            )

    if not available:
        logger.error("No models available (no API keys set)")
        return {}

    logger.info("Running models: %s", [m.name for m in available])

    if resume:
        # Use the most recent existing timestamp for each model so we
        # append to / resume from the same results file.
        ts_for = {}
        for m in available:
            found = _find_latest_run(results_dir, m.name)
            if found:
                ts_for[m.name] = found
                logger.info("Resuming %s from timestamp %s", m.name, found)
            else:
                ts_for[m.name] = timestamp
    else:
        ts_for = {m.name: timestamp for m in available}

    all_results: dict[str, list[EvalResult]] = {}
    model_tasks = []
    for model_run in available:
        coro = _run_model(
            model_run, prompts, config, results_dir, ts_for[model_run.name]
        )
        model_tasks.append((model_run.name, coro))

    gathered = await asyncio.gather(
        *(coro for _, coro in model_tasks), return_exceptions=True
    )

    for (name, _), result in zip(model_tasks, gathered):
        if isinstance(result, Exception):
            logger.error("Model %s failed: %s", name, result)
            all_results[name] = []
        else:
            all_results[name] = result

    summary = _compute_summary(all_results)

    summary_file = results_dir / f"{timestamp}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_file)

    _print_summary(summary, "MMI Eval Summary")

    return summary


# ---------------------------------------------------------------------------
# Rejudge: run judge on existing results without re-running models
# ---------------------------------------------------------------------------


def _find_latest_jsonl(results_dir: Path, model_name: str) -> Path | None:
    """Find the latest results JSONL for a model by name.

    First tries matching by filename (the config model name). If that fails,
    scans all JSONL files looking for one whose records have a matching
    ``run_name`` — this handles cases where the config model name changed
    since the original run.
    """
    name_safe = model_name.replace("/", "_").replace(" ", "_")

    # Try direct filename match first
    pattern = f"*_{name_safe}.jsonl"
    matches = sorted(results_dir.glob(pattern), reverse=True)
    # Filter out _rejudged files so we always pick the original
    matches = [m for m in matches if "_rejudged" not in m.name]
    if matches:
        return matches[0]

    # Fallback: scan all JSONL files for matching run_name in first record
    for path in sorted(results_dir.glob("*.jsonl"), reverse=True):
        if "_rejudged" in path.name:
            continue
        try:
            with open(path) as f:
                first_line = f.readline().strip()
                if first_line:
                    obj = json.loads(first_line)
                    if obj.get("run_name") == model_name:
                        return path
        except (json.JSONDecodeError, OSError):
            continue

    return None


def _needs_judge(result_dict: dict) -> bool:
    """Determine if a result needs judging.

    Returns True when the result is not an error and at least one expected
    non-text modality was not detected by native or URL detection, OR at
    least one non-text modality was detected that is not in expected.
    """
    if result_dict.get("is_error"):
        return False
    per_modality = result_dict.get("per_modality", {})
    expected = set(result_dict.get("expected_modalities", []))
    for modality, ms in per_modality.items():
        if modality == "Text":
            continue
        detected = ms.get("detected_native") or ms.get("detected_via_url")
        if modality in expected and not detected:
            return True
        if modality not in expected and detected:
            return True
    return False


async def rejudge(config: HarnessConfig, config_name: str) -> dict:
    """Run the LLM judge on existing model results.

    For each model in config, finds the latest results JSONL, runs the judge
    on results that have missing non-text modalities, and writes a new
    ``*_rejudged.jsonl`` file alongside the original.

    Returns:
        Summary dict (same format as ``run()``).
    """
    results_dir = RESULTS_DIR / config_name
    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir)
        return {}

    # Load dataset for prompt texts
    prompts_list = load_dataset()
    prompts_by_id: dict[str, EvalPrompt] = {p.prompt_id: p for p in prompts_list}

    # Instantiate judge
    from .judge import ModalityJudge

    try:
        judge = ModalityJudge(
            model=config.judge_model,
            api_key_env=config.judge_api_key_env,
            base_url=config.judge_base_url,
        )
    except OSError as e:
        logger.error("Cannot create judge: %s", e)
        return {}

    logger.info(
        "Rejudge using model=%s (config judge_model)",
        config.judge_model,
    )

    all_results: dict[str, list[EvalResult]] = {}

    # Only rejudge models listed in the config
    source_files: list[tuple[str, Path]] = []
    for model_run in config.models:
        path = _find_latest_jsonl(results_dir, model_run.name)
        if path:
            source_files.append((model_run.name, path))
        else:
            logger.warning(
                "No results JSONL found for model '%s' in %s — skipping",
                model_run.name,
                results_dir,
            )

    if not source_files:
        logger.error(
            "No result JSONL files found for any config model in %s", results_dir
        )
        return {}

    logger.info(
        "Found %d model result file(s): %s",
        len(source_files),
        [name for name, _ in source_files],
    )

    for model_name, source_path in source_files:
        logger.info("Rejudging %s from %s", model_name, source_path.name)

        # Load all result dicts (keep as dicts to preserve raw_response as-is)
        result_dicts: list[dict] = []
        with open(source_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    result_dicts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not result_dicts:
            logger.warning("No results in %s", source_path.name)
            continue

        # Process each result
        semaphore = asyncio.Semaphore(config.concurrency)
        completed = 0
        total = len(result_dicts)
        judged_count = 0

        async def process_one(rd: dict) -> dict:
            nonlocal completed, judged_count

            judge_result = None
            if _needs_judge(rd):
                prompt = prompts_by_id.get(rd["prompt_id"])
                if prompt is None:
                    logger.warning(
                        "Prompt %s not found in dataset — skipping judge",
                        rd["prompt_id"],
                    )
                else:
                    # Extract text content from raw_response if possible
                    text_content = ""
                    raw_resp = rd.get("raw_response")
                    if isinstance(raw_resp, dict):
                        # Try common locations for text in raw responses
                        text_content = (
                            raw_resp.get("text", "")
                            or raw_resp.get("content", "")
                            or ""
                        )
                        if not text_content:
                            # Gemini-style: candidates[0].content.parts[*].text
                            try:
                                parts = (
                                    raw_resp.get("candidates", [{}])[0]
                                    .get("content", {})
                                    .get("parts")
                                ) or []
                                text_content = " ".join(
                                    p.get("text", "") for p in parts if "text" in p
                                )
                            except (IndexError, AttributeError):
                                pass
                    elif isinstance(raw_resp, str):
                        text_content = raw_resp[:4000]

                    async with semaphore:
                        judge_result = await judge.evaluate(
                            prompt_text=prompt.prompt_text,
                            text_content=text_content,
                            raw_response=raw_resp,
                        )
                        judged_count += 1

            updated = rescore_dict(rd, judge_result)

            completed += 1
            status = "✅" if updated.get("all_pass_lenient") else "❌"
            changed = " 🔄" if judge_result and judge_result.detected_modalities else ""
            logger.info(
                "%s [%d/%d] %s%s %s — expected=%s produced=%s",
                model_name,
                completed,
                total,
                status,
                changed,
                updated["prompt_id"],
                updated.get("expected_modalities", []),
                updated.get("produced_modalities", []),
            )
            return updated

        tasks = [process_one(rd) for rd in result_dicts]
        updated_dicts = await asyncio.gather(*tasks)

        # Write rejudged file alongside original
        rejudged_path = source_path.with_name(source_path.stem + "_rejudged.jsonl")
        with open(rejudged_path, "w") as f:
            for rd in updated_dicts:
                f.write(json.dumps(rd, ensure_ascii=False) + "\n")

        logger.info(
            "%s: wrote %d results to %s (judge called on %d)",
            model_name,
            len(updated_dicts),
            rejudged_path.name,
            judged_count,
        )

        # Convert to EvalResult for summary computation
        all_results[model_name] = [_dict_to_eval_result(rd) for rd in updated_dicts]

    # Compute and save summary
    summary = _compute_summary(all_results)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    summary_file = results_dir / f"{timestamp}_rejudged_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Rejudged summary saved to %s", summary_file)

    _print_summary(summary, "MMI Eval — Rejudged Summary")

    return summary


# ---------------------------------------------------------------------------
# Rescore: recompute P/R/F1 from existing detection flags
# ---------------------------------------------------------------------------

EXPECTED_ROW_COUNT = 893


async def rescore(jsonl_path: str) -> dict:
    """Recompute precision/recall/F1 from existing detection flags in a JSONL file.

    This does NOT re-run models or the judge — it only recalculates the
    aggregate P/R/F1 metrics from the per-modality detection booleans that
    are already stored in each result row.

    Args:
        jsonl_path: Path to the JSONL results file to rescore.

    Returns:
        Summary dict (same format as ``run()``).
    """
    path = Path(jsonl_path)
    if not path.exists():
        logger.error("File not found: %s", path)
        return {}

    # Load all result dicts
    result_dicts: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result_dicts.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not result_dicts:
        logger.error("No results found in %s", path)
        return {}

    if len(result_dicts) != EXPECTED_ROW_COUNT:
        logger.warning(
            "Row count mismatch: expected %d rows but found %d in %s",
            EXPECTED_ROW_COUNT,
            len(result_dicts),
            path.name,
        )

    # Normalize per_modality and recompute all scores for each result
    for rd in result_dicts:
        rescore_dict(rd)

    # Write rescored file alongside original
    rescored_path = path.with_name(path.stem + "_rescored.jsonl")
    with open(rescored_path, "w") as f:
        for rd in result_dicts:
            f.write(json.dumps(rd, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rescored results to %s", len(result_dicts), rescored_path)

    # Convert to EvalResult and group by run_name for summary
    all_results: dict[str, list[EvalResult]] = {}
    for rd in result_dicts:
        er = _dict_to_eval_result(rd)
        all_results.setdefault(er.run_name, []).append(er)

    summary = _compute_summary(all_results)

    # Save summary JSON
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    summary_file = path.parent / f"{timestamp}_rescored_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Rescored summary saved to %s", summary_file)

    _print_summary(summary, "MMI Eval — Rescored Summary")

    return summary


# ---------------------------------------------------------------------------
# Rubric scoring: run rubric judge on existing results
# ---------------------------------------------------------------------------


def _compute_rubric_summary(result_dicts: list[dict]) -> dict:
    """Aggregate rubric outcomes per run.

    Three numbers, deliberately all three: the mean of per-prompt MMI Values
    (each itself a mean over modalities of the within-modality mean criterion
    score, with partial credit, which is what the judge returns), the paper's
    binary per-modality metric — 1 only if every rubric for that modality was
    marked correct — and the count of zeros that came from the judge failing
    rather than the model. The published judge--human agreement is computed
    over the binary form, because the human scores it is compared against are
    binary.

    Evaluator errors are **not** removed from the metric — excluding them would
    inflate it. They are reported alongside so a reader can tell a degraded run
    from a bad model, which is the whole point of tracking them.
    """
    by_run: dict[str, dict[str, Any]] = {}
    for rd in result_dicts:
        run_name = rd.get("run_name", "unknown")
        summary = by_run.setdefault(
            run_name,
            {
                "_total": 0,
                "_rubric_total": 0,
                "_rubric_scored": 0,
                "_rubric_missing_response": 0,
                "_rubric_score_sum": 0.0,
                "_rubric_mean_score": None,
                "_rubric_binary_by_modality": {},
                "_rubric_binary_counts": {},
                "_rubric_evaluator_error_grades": 0,
                "_rubric_rows_with_evaluator_error": 0,
                "_rubric_evaluator_errors_by_modality": {},
                "_rubric_payload_status_counts": {},
            },
        )
        summary["_total"] += 1
        if rd.get("rubric_used"):
            summary["_rubric_total"] += 1
            if rd.get("rubric_score") is not None:
                summary["_rubric_scored"] += 1
                summary["_rubric_score_sum"] += float(rd["rubric_score"])
                if not extract_response_text(rd):
                    summary["_rubric_missing_response"] += 1

            failed = [
                grade
                for grade in rd.get("rubric_grades") or []
                if is_evaluator_error(grade)
            ]
            if failed:
                summary["_rubric_evaluator_error_grades"] += len(failed)
                summary["_rubric_rows_with_evaluator_error"] += 1
                for grade in failed:
                    modality = grade.get("modality", "")
                    if modality:
                        summary["_rubric_evaluator_errors_by_modality"][modality] = (
                            summary["_rubric_evaluator_errors_by_modality"].get(
                                modality, 0
                            )
                            + 1
                        )

            for grade in rd.get("rubric_grades") or []:
                status = grade.get("payload_status") or ""
                if status:
                    counts = summary["_rubric_payload_status_counts"]
                    counts[status] = counts.get(status, 0) + 1

            for modality, value in (rd.get("rubric_binary_by_modality") or {}).items():
                counts = summary["_rubric_binary_counts"].setdefault(
                    modality, {"correct": 0, "graded": 0}
                )
                counts["graded"] += 1
                counts["correct"] += int(value)

    for summary in by_run.values():
        scored = summary["_rubric_scored"]
        summary["_rubric_mean_score"] = (
            summary["_rubric_score_sum"] / scored if scored else None
        )
        del summary["_rubric_score_sum"]
        summary["_rubric_binary_by_modality"] = {
            modality: counts["correct"] / counts["graded"]
            for modality, counts in sorted(summary["_rubric_binary_counts"].items())
            if counts["graded"]
        }
        summary["_rubric_evaluator_errors_by_modality"] = dict(
            sorted(summary["_rubric_evaluator_errors_by_modality"].items())
        )
        summary["_rubric_payload_status_counts"] = dict(
            sorted(summary["_rubric_payload_status_counts"].items())
        )
    return by_run


async def rubric_score(jsonl_path: str, config: HarnessConfig) -> dict:
    """Run rubric judging on existing JSONL rows and append *_rubric_scored.jsonl."""
    path = Path(jsonl_path)
    if not path.exists():
        logger.error("File not found: %s", path)
        return {}

    prompts_by_id = {prompt.prompt_id: prompt for prompt in load_dataset()}
    result_dicts: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result_dicts.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not result_dicts:
        logger.error("No results found in %s", path)
        return {}

    output_path = path.with_name(path.stem + "_rubric_scored.jsonl")
    existing_dicts: list[dict] = []
    scored_prompt_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt_id = row.get("prompt_id")
                if isinstance(prompt_id, str):
                    scored_prompt_ids.add(prompt_id)
                existing_dicts.append(row)

    remaining_dicts = [
        rd for rd in result_dicts if rd.get("prompt_id") not in scored_prompt_ids
    ]
    if scored_prompt_ids:
        logger.info(
            "Rubric scoring resume: %d already written, %d remaining",
            len(scored_prompt_ids),
            len(remaining_dicts),
        )

    rubric_judge = RubricJudge(
        model=config.rubric_judge_model,
        api_key_env=config.rubric_judge_api_key_env,
        base_url=config.rubric_judge_base_url,
    )
    semaphore = asyncio.Semaphore(config.concurrency)
    write_lock = asyncio.Lock()
    completed = 0
    updated_dicts: list[dict] = list(existing_dicts)
    rubric_total = sum(
        1
        for rd in remaining_dicts
        if prompts_by_id.get(rd.get("prompt_id"))
        and prompts_by_id[rd["prompt_id"]].rubric_criteria
    )

    async def process_one(rd: dict) -> dict:
        nonlocal completed
        prompt = prompts_by_id.get(rd.get("prompt_id"))
        if prompt is None or not prompt.rubric_criteria:
            rd.setdefault("rubric_used", False)
            rd.setdefault("rubric_score", None)
            rd.setdefault("rubric_grades", [])
            rd.setdefault("rubric_score_by_modality", {})
            rd.setdefault("rubric_evaluator_errors", 0)
            rd.setdefault("rubric_judge_model", "")
        else:
            async with semaphore:
                response_text = extract_response_text(rd)
                rd["response_text"] = response_text
                rubric_result = await score_prompt_rubrics(
                    judge=rubric_judge,
                    prompt=prompt,
                    response_text=response_text,
                    raw_response=rd.get("raw_response"),
                    output_assets=rd.get("output_assets", []),
                )
                rd.update(rubric_result.as_result_fields())
                completed += 1
                logger.info(
                    "Rubric scored [%d/%d] %s score=%s rubrics=%d",
                    completed,
                    rubric_total,
                    rd.get("prompt_id"),
                    rd.get("rubric_score"),
                    len(prompt.rubric_criteria),
                )

        async with write_lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(rd, ensure_ascii=False) + "\n")
            updated_dicts.append(rd)
        return rd

    await asyncio.gather(*[process_one(rd) for rd in remaining_dicts])
    logger.info("Wrote %d rubric-scored results to %s", len(updated_dicts), output_path)

    summary = _compute_rubric_summary(updated_dicts)
    degraded = sum(s["_rubric_evaluator_error_grades"] for s in summary.values())
    if degraded:
        logger.warning(
            "%d rubric criteria scored zero because the judge failed, not the "
            "model. These are counted in the metric but reported separately per "
            "run as _rubric_evaluator_error_grades; treat affected runs as "
            "degraded rather than comparable.",
            degraded,
        )
    unretrieved = sum(
        s["_rubric_payload_status_counts"].get(PAYLOAD_CAPTURE_FAILED, 0)
        for s in summary.values()
    )
    if unretrieved:
        logger.warning(
            "%d rubric criteria scored zero because retrieving the artifact "
            "failed. That is a harness failure, not a model failure; see "
            "_rubric_payload_status_counts.",
            unretrieved,
        )
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    summary_file = path.parent / f"{timestamp}_rubric_scored_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Rubric summary saved to %s", summary_file)
    print(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _results_config_name(config_arg: str) -> str:
    """Return the config basename used for the results subdirectory."""
    return Path(config_arg).stem


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run MMI eval harness")
    parser.add_argument(
        "--config",
        default="default.toml",
        help="Config filename in configs/ directory (default: default.toml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the most recent incomplete run instead of starting fresh",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--rejudge",
        action="store_true",
        help=(
            "Run the LLM judge on existing results without re-running models. "
            "Picks up the latest JSONL for each model in the config and writes "
            "a new *_rejudged.jsonl alongside it."
        ),
    )
    mode_group.add_argument(
        "--rescore",
        type=str,
        metavar="PATH",
        help=(
            "Recompute precision/recall/F1 from detection flags in the given "
            "JSONL file. Does not re-run models or the judge. Writes a new "
            "*_rescored.jsonl alongside the original."
        ),
    )
    mode_group.add_argument(
        "--rubric-score",
        type=str,
        metavar="PATH",
        help=(
            "Run rubric judging on prompts with rubrics in the given JSONL file. "
            "Writes *_rubric_scored.jsonl alongside the original."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        metavar="PATH",
        help=(
            "Local dataset override (.jsonl or .parquet) for offline work or a "
            "controlled mirror. Equivalent to MMI_DATASET_PATH."
        ),
    )
    args = parser.parse_args()

    if args.dataset_path:
        os.environ["MMI_DATASET_PATH"] = args.dataset_path

    config = load_config(args.config)

    if args.rescore:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        )
        asyncio.run(rescore(args.rescore))
        return 0

    if args.rubric_score:
        logging.basicConfig(
            level=logging.DEBUG if config.verbose else logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        )
        asyncio.run(rubric_score(args.rubric_score, config))
        return 0

    logging.basicConfig(
        level=logging.DEBUG if config.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    config_name = _results_config_name(args.config)

    if args.rejudge:
        # Force judge enabled for rejudge regardless of config
        config.judge_enabled = True
        asyncio.run(rejudge(config, config_name))
    else:
        asyncio.run(run(config, config_name, resume=args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
