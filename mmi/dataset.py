# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Load the MMI eval dataset into EvalPrompt objects.

The dataset lives on Hugging Face at ``facebook/mmi`` and is loaded into the
standard Hugging Face cache. It is not carried in this repository.

Set ``MMI_DATASET_PATH`` (or pass ``path=``) to load a local JSONL or Parquet
file instead, for offline work or a mirror. Set ``MMI_DATASET_REVISION`` to pin
a specific revision; the default is the dataset's default branch.

Loading writes only into cache directories. It never writes into this
repository or into an installed copy of the package.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path

from .config import INPUT_FILES_DIR
from .input_files import (
    InvalidInputFilePath,
    resolve_input_file,
    write_input_file_if_missing,
)
from .models import EvalPrompt
from .response_utils import parse_rubric_criteria

logger = logging.getLogger(__name__)

_ALLOWED_MODALITIES = {"Text", "Image", "Audio", "Video", "Document"}

HF_REPO_ID = "facebook/mmi"

# 25 hand-picked prompt IDs covering a representative mix of input/output
# modality combinations.  Used when sample = true in config.
#
# Coverage:
#   Text-only input → each single output modality (Text, Image, Audio, Video, Document)
#   Text-only input → multi-output (Audio+Image+Video, Image+Text, Audio+Text)
#   Image input → Text, Image, Audio, Video, Document
#   Image+Text input → Text, Image, Video
#   Audio input → Text, Audio, Image
#   Document input → Text, Document, Image
#   Multi-modal input (Audio+Image, Document+Image+Text)
#   Multi-output with files (Audio → Document+Text+Video)
SAMPLE_PROMPT_IDS: set[str] = {
    "p842100",  # Text → Text
    "p232244",  # Text → Image
    "p123013",  # Text → Audio
    "p990358",  # Text → Video
    "p433625",  # Text → Document
    "p496527",  # Text → Audio, Image, Video
    "p561559",  # Text → Image, Text
    "p304382",  # Text → Audio, Text
    "p765740",  # Image → Text
    "p211938",  # Image → Image
    "p894960",  # Image → Audio
    "p792273",  # Image → Video
    "p990064",  # Image → Document
    "p740684",  # Image, Text → Text
    "p139334",  # Image, Text → Image
    "p728009",  # Image, Text → Video
    "p697879",  # Audio → Text
    "p220626",  # Audio → Audio
    "p210850",  # Audio → Image
    "p235070",  # Document → Text
    "p277655",  # Document → Document
    "p250735",  # Document → Image
    "p928747",  # Audio, Image → Audio
    "p599103",  # Document, Image, Text → Audio
    "p107156",  # Audio → Document, Text, Video
}


class DatasetError(RuntimeError):
    """The dataset could not be loaded, or a record is malformed."""


def dataset_revision() -> str:
    """The revision to load. Recorded in the run manifest."""
    return os.environ.get("MMI_DATASET_REVISION", "main")


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------


def _parse_list(value: str, sep: str = ",") -> list[str]:
    """Split a delimited string into a list, stripping whitespace and filtering empties."""
    if not value or not value.strip():
        return []
    return [item.strip() for item in value.split(sep) if item.strip()]


def _parse_and_validate_modalities(
    raw_value: str,
    *,
    field_name: str,
    line_num: int,
    prompt_id: str,
    required: bool,
) -> list[str]:
    modalities = _parse_list(raw_value)
    if required and not modalities:
        raise DatasetError(
            f"Invalid dataset at line {line_num} (prompt {prompt_id}): "
            f"{field_name} is required and must not be empty"
        )

    invalid = [m for m in modalities if m not in _ALLOWED_MODALITIES]
    if invalid:
        raise DatasetError(
            f"Invalid dataset at line {line_num} (prompt {prompt_id}): "
            f"{field_name} has unsupported modalities: {', '.join(invalid)}. "
            f"Allowed: {', '.join(sorted(_ALLOWED_MODALITIES))}"
        )

    return modalities


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    return False


def _payload_bytes(value) -> bytes | None:
    if _is_missing(value):
        return None
    if isinstance(value, Mapping):
        data = value.get("bytes")
        return bytes(data) if data is not None else None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return None


def _write_asset_if_missing(filename: str, data: bytes | None) -> None:
    """Materialize one input asset into the cache.

    Writes to the cache directory, never the repository. Existing files are
    left alone so a run cannot be perturbed mid-flight.
    """
    if data is None:
        return
    write_input_file_if_missing(filename, data, root=INPUT_FILES_DIR)


def _materialize_embedded_assets(row: Mapping, input_files: list[str]) -> None:
    """Extract media embedded in the dataset row into the cache."""
    images = row.get("input_images")
    image_filenames = [
        name for name in input_files if Path(name).suffix.lower() in (".jpg", ".jpeg")
    ]
    if images is not None and not _is_missing(images):
        for filename, payload in zip(image_filenames, images, strict=False):
            _write_asset_if_missing(filename, _payload_bytes(payload))

    for ext, column in ((".mp3", "input_audio"), (".mp4", "input_video")):
        filename = next(
            (name for name in input_files if Path(name).suffix.lower() == ext),
            None,
        )
        if filename:
            _write_asset_if_missing(filename, _payload_bytes(row.get(column)))


def _validate_input_file_names(
    input_files: list[str], *, line_num: int, prompt_id: str
) -> None:
    for filename in input_files:
        try:
            resolve_input_file(filename, root=INPUT_FILES_DIR, must_exist=False)
        except InvalidInputFilePath as exc:
            raise DatasetError(
                f"Invalid dataset at line {line_num} (prompt {prompt_id}): {exc}"
            ) from exc


def _validate_input_files(
    input_files: list[str], *, line_num: int, prompt_id: str
) -> None:
    missing_files = []
    for filename in input_files:
        try:
            resolve_input_file(filename, root=INPUT_FILES_DIR)
        except (InvalidInputFilePath, FileNotFoundError):
            missing_files.append(f"{filename} ({INPUT_FILES_DIR / filename})")
    if missing_files:
        raise DatasetError(
            f"Invalid dataset at line {line_num} (prompt {prompt_id}): "
            f"missing or invalid input media files: {', '.join(missing_files)}. "
            f"Input media is cached under {INPUT_FILES_DIR}; "
            "set MMI_INPUT_FILES_DIR to point elsewhere."
        )


def _prompt_from_record(
    obj: Mapping, *, line_num: int, require_media: bool = True
) -> EvalPrompt | None:
    missing = [
        f for f in ("prompt_id", "prompt_text", "output_modalities") if f not in obj
    ]
    if missing:
        logger.warning(
            "Skipping line %d: missing required fields: %s",
            line_num,
            ", ".join(missing),
        )
        return None

    prompt_id = str(obj["prompt_id"])
    input_files = _parse_list(str(obj.get("input_files") or ""), sep="\n")
    _validate_input_file_names(input_files, line_num=line_num, prompt_id=prompt_id)
    _materialize_embedded_assets(obj, input_files)
    if require_media:
        _validate_input_files(input_files, line_num=line_num, prompt_id=prompt_id)

    try:
        rubric_criteria = parse_rubric_criteria(obj.get("rubrics"))
    except ValueError as exc:
        raise DatasetError(
            f"Invalid dataset at line {line_num} (prompt {prompt_id}): {exc}"
        ) from exc

    output_modalities = _parse_and_validate_modalities(
        str(obj.get("output_modalities") or ""),
        field_name="output_modalities",
        line_num=line_num,
        prompt_id=prompt_id,
        required=True,
    )

    # Rubric judging is per modality and keyed off the expected modalities, so
    # a rubric for a modality the prompt does not expect can never be graded.
    invalid_rubric_modalities = [
        criterion.modality
        for criterion in rubric_criteria
        if criterion.modality not in output_modalities
    ]
    if invalid_rubric_modalities:
        raise DatasetError(
            f"Invalid dataset at line {line_num} (prompt {prompt_id}): "
            "rubric modalities must match output_modalities; "
            f"invalid={invalid_rubric_modalities}, output_modalities={output_modalities}"
        )

    return EvalPrompt(
        prompt_id=prompt_id,
        prompt_text=str(obj["prompt_text"]),
        input_modalities=_parse_and_validate_modalities(
            str(obj.get("input_modalities") or ""),
            field_name="input_modalities",
            line_num=line_num,
            prompt_id=prompt_id,
            required=False,
        ),
        output_modalities=output_modalities,
        input_files=input_files,
        rubric_criteria=rubric_criteria,
    )


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _read_local(path: Path) -> list[Mapping]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    raise DatasetError(f"Dataset override must be .jsonl or .parquet: {path}")


def _read_hub() -> list[Mapping]:
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise DatasetError(
            "The 'datasets' package is required to load the dataset from the Hub."
        ) from exc

    return list(hf_load_dataset(HF_REPO_ID, revision=dataset_revision(), split="train"))


def _local_override() -> Path | None:
    raw = os.environ.get("MMI_DATASET_PATH")
    return Path(raw).expanduser() if raw else None


def load_dataset(
    path: Path | None = None, *, require_media: bool = True
) -> list[EvalPrompt]:
    """Load the eval dataset and return a list of EvalPrompt objects.

    Resolution order: explicit ``path`` → ``MMI_DATASET_PATH`` → the Hugging
    Face dataset.

    Args:
        path: Explicit local override (``.jsonl`` or ``.parquet``).
        require_media: Fail if a referenced input file is not in the cache.
            Set ``False`` to list prompts without materialized media.
    """
    source = path or _local_override()

    if source is not None:
        if not source.exists():
            raise DatasetError(f"Eval dataset not found: {source}")
        records = _read_local(source)
        origin = str(source)
    else:
        records = _read_hub()
        origin = f"{HF_REPO_ID}@{dataset_revision()}"

    prompts = []
    for line_num, obj in enumerate(records, start=1):
        prompt = _prompt_from_record(
            obj, line_num=line_num, require_media=require_media
        )
        if prompt is not None:
            prompts.append(prompt)

    logger.info("Loaded %d prompts from %s", len(prompts), origin)
    return prompts


def prompt_id_set_hash(prompts: list[EvalPrompt]) -> str:
    """Order-independent hash of the prompt-ID set.

    Recorded in the run manifest so a resume cannot silently continue against a
    different set of prompts.
    """
    return hashlib.sha256(
        "\n".join(sorted(p.prompt_id for p in prompts)).encode()
    ).hexdigest()
