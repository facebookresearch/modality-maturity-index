# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Run manifests.

A results file on its own does not say what produced it. Filenames are not a
contract: resuming a run because a file happens to match a glob is how a
half-finished run against one dataset revision gets silently completed against
another.

Every run therefore writes a manifest recording what it is, and resume refuses
to continue a run whose manifest is incompatible with the current
configuration.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Bumped whenever the persisted result shape changes incompatibly.
SCHEMA_VERSION = 1

MANIFEST_FILENAME = "run_manifest.json"

#: Fields that must match for a resume to be safe. Anything outside this set
#: may differ without changing what the numbers mean.
COMPATIBILITY_FIELDS = (
    "schema_version",
    "dataset_revision",
    "dataset_prompt_id_hash",
    "prompt_selection",
    "config_hash",
)


class IncompatibleManifest(RuntimeError):
    """The existing run cannot be resumed under the current configuration."""


@dataclass
class ModelRoute:
    """How one model was reached, and what actually answered."""

    name: str
    provider: str
    model_id: str
    api: str = ""
    #: Empty means the SDK's own official endpoint.
    base_url: str = ""
    api_key_env: str = ""
    tools: list[str] = field(default_factory=list)
    provider_tools_count: int = 0
    #: Filled in from the response where the provider reports it. Every model
    #: ID in the paper configs will stop resolving eventually; this records
    #: what answered on the day.
    resolved_model_version: str = ""


@dataclass
class RunManifest:
    schema_version: int
    run_timestamp: str
    config_name: str
    config_hash: str

    dataset_revision: str
    dataset_prompt_id_hash: str
    prompt_count: int
    prompt_selection: str

    harness_version: str
    harness_commit: str

    models: list[dict[str, Any]] = field(default_factory=list)

    judge_profile: dict[str, Any] = field(default_factory=dict)
    rubric_profile: dict[str, Any] = field(default_factory=dict)
    tool_profile: dict[str, Any] = field(default_factory=dict)

    runtime: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, directory: Path) -> Path:
        path = directory / MANIFEST_FILENAME
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def read(cls, directory: Path) -> "RunManifest | None":
        path = directory / MANIFEST_FILENAME
        if not path.exists():
            return None
        return cls(**json.loads(path.read_text()))

    def assert_compatible_with(self, other: "RunManifest") -> None:
        """Refuse to resume across a change that would corrupt the results."""
        differences = [
            f"{field_name}: existing={getattr(self, field_name)!r} "
            f"current={getattr(other, field_name)!r}"
            for field_name in COMPATIBILITY_FIELDS
            if getattr(self, field_name) != getattr(other, field_name)
        ]
        if differences:
            raise IncompatibleManifest(
                "Refusing to resume: the existing run is not compatible with the "
                "current configuration.\n  " + "\n  ".join(differences)
            )


def config_hash(config) -> str:
    """Hash the settings that change what the numbers mean.

    Deliberately excludes concurrency and verbosity: they affect how the run
    executes, not what it measures.
    """
    material = {
        "models": [
            {
                "name": m.name,
                "provider": m.provider,
                "model_id": m.model_id,
                "api": m.api,
                "base_url": m.base_url,
                "tools": sorted(m.tools),
                "provider_tools": m.provider_tools,
            }
            for m in sorted(config.models, key=lambda m: m.name)
        ],
        "judge_enabled": config.judge_enabled,
        "judge_model": config.judge_model,
        "rubric_enabled": config.rubric_enabled,
        "rubric_judge_model": config.rubric_judge_model,
        "tool_loop_limit": config.tool_loop_limit,
        "media_tool_backends": {
            name: {"provider": b.provider, "model": b.model, "base_url": b.base_url}
            for name, b in sorted(config.media_tool_backends.items())
        },
        "max_retries": config.max_retries,
        "request_timeout": config.request_timeout,
    }
    blob = json.dumps(material, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _harness_commit() -> str:
    """Best-effort commit id. Absent in an installed package, which is fine."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _harness_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("mmi")
    except PackageNotFoundError:
        return "unknown"


def build_manifest(
    *,
    config,
    config_name: str,
    prompts,
    prompt_selection: str,
    timestamp: str = "",
) -> RunManifest:
    from .dataset import dataset_revision, prompt_id_set_hash

    return RunManifest(
        schema_version=SCHEMA_VERSION,
        run_timestamp=timestamp or datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
        config_name=config_name,
        config_hash=config_hash(config),
        dataset_revision=dataset_revision(),
        dataset_prompt_id_hash=prompt_id_set_hash(prompts),
        prompt_count=len(prompts),
        prompt_selection=prompt_selection,
        harness_version=_harness_version(),
        harness_commit=_harness_commit(),
        models=[
            asdict(
                ModelRoute(
                    name=m.name,
                    provider=m.provider,
                    model_id=m.model_id,
                    api=m.api,
                    base_url=m.base_url,
                    api_key_env=m.api_key_env,
                    tools=list(m.tools),
                    provider_tools_count=len(m.provider_tools),
                )
            )
            for m in config.models
        ],
        judge_profile={
            "enabled": config.judge_enabled,
            "provider": config.judge_provider,
            "model": config.judge_model,
            "base_url": config.judge_base_url,
        },
        rubric_profile={
            "enabled": config.rubric_enabled,
            "provider": config.rubric_judge_provider,
            "model": config.rubric_judge_model,
            "base_url": config.rubric_judge_base_url,
            # Rubrics need artifact bytes, so enabling them enables retrieval.
            # Recorded because it changes which criteria were gradeable at all.
            "fetch_remote_assets": config.fetch_remote_assets,
        },
        tool_profile={
            "tool_loop_limit": config.tool_loop_limit,
            "backends": {
                name: {
                    "provider": b.provider,
                    "model": b.model,
                    "base_url": b.base_url,
                }
                for name, b in sorted(config.media_tool_backends.items())
            },
        },
        runtime={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    )
