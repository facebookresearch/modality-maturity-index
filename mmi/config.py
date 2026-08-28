# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Configuration: TOML config loader, provider registry, paths."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

# ---------------------------------------------------------------------------
# Paths
#
# The repository holds code and configuration. Dataset content and run outputs
# live outside it: loading the dataset must never write into the source tree.
# ---------------------------------------------------------------------------

HARNESS_DIR = Path(__file__).resolve().parent.parent  # project root
CONFIGS_DIR = HARNESS_DIR / "configs"


def _xdg_dir(env_var: str, default_suffix: str) -> Path:
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base).expanduser() / "mmi" / default_suffix


#: Where dataset input media is materialized at runtime. Overridable with
#: ``MMI_INPUT_FILES_DIR``; defaults to the XDG cache, never the repo.
INPUT_FILES_DIR = _xdg_dir("MMI_INPUT_FILES_DIR", "input_files")

#: Where run outputs are written. Overridable with ``MMI_RESULTS_DIR``.
RESULTS_DIR = Path(
    os.environ.get("MMI_RESULTS_DIR", HARNESS_DIR / "results")
).expanduser()

# ---------------------------------------------------------------------------
# API key helper
# ---------------------------------------------------------------------------


def get_api_key(env_var: str) -> str:
    key = os.environ.get(env_var, "")
    if not key:
        raise OSError(f"{env_var} environment variable is not set")
    return key


# ---------------------------------------------------------------------------
# Provider registry — static metadata, no model names
#
# An empty ``base_url`` means "use the SDK's own official endpoint". Setting a
# non-empty ``base_url`` in TOML is opt-in custom routing; the harness never
# injects gateway-specific headers on behalf of a custom route.
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: dict[str, dict[str, str]] = {
    "openai": {"api_key_env": "OPENAI_API_KEY", "base_url": ""},
    "anthropic": {"api_key_env": "ANTHROPIC_API_KEY", "base_url": ""},
    "gemini": {"api_key_env": "GOOGLE_API_KEY", "base_url": ""},
    "stub": {"api_key_env": "", "base_url": ""},
}

# ---------------------------------------------------------------------------
# Judge provider presets — preconfigured judge routing
# ---------------------------------------------------------------------------

JUDGE_PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "google": {
        "api_key_env": "GOOGLE_API_KEY",
        "base_url": "",
    },
}

DEFAULT_JUDGE_PROVIDER = "google"

# ---------------------------------------------------------------------------
# Neutral media-tool backends (Decision 6)
#
# The model-facing tool *schemas* are frozen in ``media_tools``. Which backend
# fulfils each tool is configuration. Neutrality means the backend is constant
# across drivers within an experiment and its resolved identity is recorded in
# the run manifest — not that it must be any particular vendor.
# ---------------------------------------------------------------------------

MEDIA_TOOL_NAMES = ("image_gen", "audio_gen", "video_gen")

DEFAULT_TOOL_LOOP_LIMIT = 3

# "Gemini 3 Flash" in the paper. Override in TOML with ``judge_model``.
DEFAULT_JUDGE_MODEL = "gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelRunConfig:
    """A single model run entry from the [[models]] array in config TOML."""

    name: str
    provider: str
    model_id: str = ""
    api: str = ""
    api_key_env: str = ""
    base_url: str = ""
    tools: list[str] = field(default_factory=list)
    response_modalities: list[str] = field(default_factory=list)
    # Verbatim provider-native tool specifications, passed through to the SDK
    # untouched (Decision 4b). The harness does not interpret these; they exist
    # so provider-native / agentic systems under test are expressible at all.
    provider_tools: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.model_id:
            self.model_id = self.name


@dataclass
class MediaToolBackend:
    """Backend that fulfils one neutral media tool.

    The model-facing schema is frozen; this is only the resolution of *which*
    system generates the artifact.
    """

    provider: str
    model: str
    base_url: str = ""
    api_key_env: str = ""


@dataclass
class HarnessConfig:
    """Parsed harness configuration from a TOML file."""

    concurrency: int = 3
    max_retries: int = 5
    retry_backoff: int = 4
    request_timeout: int = 300
    # Runner-level wait_for() padding on top of request_timeout (ledger 11).
    # The effective runner deadline is request_timeout + runner_timeout_padding.
    runner_timeout_padding: int = 60
    verbose: bool = False
    max_prompts: int | None = None
    # Cost guardrail. True restricts the run to the curated sample set in
    # ``dataset.SAMPLE_PROMPT_IDS``; the full 893-prompt run is opt-in.
    sample: bool = True
    prompt_ids: list[str] = field(default_factory=list)
    models: list[ModelRunConfig] = field(default_factory=list)
    judge_enabled: bool = False
    judge_provider: str = DEFAULT_JUDGE_PROVIDER
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_api_key_env: str = "GOOGLE_API_KEY"
    judge_base_url: str = ""
    rubric_enabled: bool = False
    rubric_judge_provider: str = DEFAULT_JUDGE_PROVIDER
    rubric_judge_model: str = DEFAULT_JUDGE_MODEL
    rubric_judge_api_key_env: str = "GOOGLE_API_KEY"
    rubric_judge_base_url: str = ""
    tool_loop_limit: int = DEFAULT_TOOL_LOOP_LIMIT
    media_tool_backends: dict[str, MediaToolBackend] = field(default_factory=dict)

    @property
    def fetch_remote_assets(self) -> bool:
        """Whether URL-delivered artifacts are retrieved.

        Tied to ``rubric_enabled`` because a rubric grades an artifact's content
        and there is nothing to grade without the bytes. With rubrics off,
        nothing needs the content and URLs are left unretrieved.

        Retrieval never promotes URL evidence to native: ``delivery`` stays
        ``external_url``, so this changes what can be *graded*, not where an
        artifact is deemed to have come from.
        """
        return self.rubric_enabled


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _parse_media_tool_backends(raw: dict) -> dict[str, MediaToolBackend]:
    """Parse the ``[media_tools]`` table into resolved backends.

    Each entry is ``{provider, model, base_url?, api_key_env?}``. Tools with no
    configured backend are simply not offered to the model.
    """
    backends: dict[str, MediaToolBackend] = {}
    for tool_name, entry in raw.items():
        if tool_name not in MEDIA_TOOL_NAMES:
            raise ValueError(
                f"Unknown media tool '{tool_name}'. Available: {list(MEDIA_TOOL_NAMES)}"
            )
        provider = entry.get("provider")
        model = entry.get("model")
        if not provider or not model:
            raise ValueError(
                f"[media_tools.{tool_name}] requires both 'provider' and 'model'"
            )
        backends[tool_name] = MediaToolBackend(
            provider=provider,
            model=model,
            base_url=entry.get("base_url", ""),
            api_key_env=entry.get(
                "api_key_env",
                PROVIDER_REGISTRY.get(provider, {}).get("api_key_env", ""),
            ),
        )
    return backends


def _open_config(config_name: str) -> tuple[BinaryIO, str]:
    """Open an explicit path, or a bare config name from the checkout."""
    explicit_path = Path(config_name).expanduser()
    if explicit_path.is_file():
        return explicit_path.open("rb"), str(explicit_path)

    checkout_path = CONFIGS_DIR / config_name
    if checkout_path.is_file():
        return checkout_path.open("rb"), str(checkout_path)

    raise FileNotFoundError(f"Config file not found: {config_name}")


def load_config(config_name: str = "default.toml") -> HarnessConfig:
    """Load and validate a TOML config from a path or a name in ``configs/``."""
    config_file, config_source = _open_config(config_name)
    with config_file:
        raw = tomllib.load(config_file)

    # Parse [settings]
    settings = raw.get("settings", {})
    # Resolve judge provider preset, then allow explicit overrides
    judge_provider = settings.get("judge_provider", DEFAULT_JUDGE_PROVIDER)
    if judge_provider not in JUDGE_PROVIDER_PRESETS:
        raise ValueError(
            f"Unknown judge_provider '{judge_provider}'. "
            f"Available: {list(JUDGE_PROVIDER_PRESETS.keys())}"
        )
    preset = JUDGE_PROVIDER_PRESETS[judge_provider]

    rubric_judge_provider = settings.get("rubric_judge_provider", judge_provider)
    if rubric_judge_provider not in JUDGE_PROVIDER_PRESETS:
        raise ValueError(
            f"Unknown rubric_judge_provider '{rubric_judge_provider}'. "
            f"Available: {list(JUDGE_PROVIDER_PRESETS.keys())}"
        )
    rubric_preset = JUDGE_PROVIDER_PRESETS[rubric_judge_provider]

    config = HarnessConfig(
        concurrency=settings.get("concurrency", 3),
        max_retries=settings.get("max_retries", 5),
        retry_backoff=settings.get("retry_backoff", 4),
        request_timeout=settings.get("request_timeout", 300),
        runner_timeout_padding=settings.get("runner_timeout_padding", 60),
        verbose=settings.get("verbose", False),
        max_prompts=settings.get("max_prompts"),
        sample=settings.get("sample", True),
        prompt_ids=settings.get("prompt_ids", []),
        judge_enabled=settings.get("judge_enabled", False),
        judge_provider=judge_provider,
        judge_model=settings.get("judge_model", DEFAULT_JUDGE_MODEL),
        judge_api_key_env=settings.get("judge_api_key_env", preset["api_key_env"]),
        judge_base_url=settings.get("judge_base_url", preset["base_url"]),
        rubric_enabled=settings.get("rubric_enabled", False),
        rubric_judge_provider=rubric_judge_provider,
        rubric_judge_model=settings.get(
            "rubric_judge_model",
            settings.get("judge_model", DEFAULT_JUDGE_MODEL),
        ),
        rubric_judge_api_key_env=settings.get(
            "rubric_judge_api_key_env", rubric_preset["api_key_env"]
        ),
        rubric_judge_base_url=settings.get(
            "rubric_judge_base_url", rubric_preset["base_url"]
        ),
        tool_loop_limit=settings.get("tool_loop_limit", DEFAULT_TOOL_LOOP_LIMIT),
        media_tool_backends=_parse_media_tool_backends(raw.get("media_tools", {})),
    )

    # Parse [[models]]
    models_raw = raw.get("models", [])
    if not models_raw:
        raise ValueError(f"No [[models]] entries in {config_source}")

    for entry in models_raw:
        name = entry.get("name")
        provider = entry.get("provider")
        if not name or not provider:
            raise ValueError(
                f"Each [[models]] entry requires 'name' and 'provider': {entry}"
            )
        if provider not in PROVIDER_REGISTRY:
            raise ValueError(
                f"Unknown provider '{provider}' for model '{name}'. "
                f"Available: {list(PROVIDER_REGISTRY.keys())}"
            )

        # Resolve api_key_env: per-model override > provider registry default
        api_key_env = entry.get(
            "api_key_env", PROVIDER_REGISTRY[provider]["api_key_env"]
        )

        config.models.append(
            ModelRunConfig(
                name=name,
                provider=provider,
                model_id=entry.get("model_id", ""),
                api=entry.get("api", ""),
                api_key_env=api_key_env,
                base_url=entry.get(
                    "base_url", PROVIDER_REGISTRY[provider].get("base_url", "")
                ),
                tools=entry.get("tools", []),
                response_modalities=entry.get("response_modalities", []),
                provider_tools=entry.get("provider_tools", []),
            )
        )

    return config
