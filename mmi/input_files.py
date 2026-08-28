# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Safe resolution and materialization of dataset input files."""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath

from .config import INPUT_FILES_DIR


class InvalidInputFilePath(ValueError):
    """An input filename escapes, or could escape, the configured cache root."""


def _relative_input_path(filename: str) -> Path:
    if not isinstance(filename, str) or not filename or "\x00" in filename:
        raise InvalidInputFilePath("input filename must be a non-empty string")

    path = Path(filename)
    windows_path = PureWindowsPath(filename)
    if path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise InvalidInputFilePath(f"input filename must be relative: {filename!r}")
    if ".." in path.parts or ".." in windows_path.parts:
        raise InvalidInputFilePath(
            f"input filename must not contain parent traversal: {filename!r}"
        )
    return path


def resolve_input_file(
    filename: str,
    *,
    root: Path | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve an input filename while requiring it to remain under ``root``."""
    relative_path = _relative_input_path(filename)
    root_path = Path(root or INPUT_FILES_DIR).expanduser().resolve(strict=False)
    try:
        candidate = (root_path / relative_path).resolve(strict=must_exist)
        candidate.relative_to(root_path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise InvalidInputFilePath(
            f"input filename escapes the configured cache: {filename!r}"
        ) from exc

    if must_exist and not candidate.is_file():
        raise InvalidInputFilePath(f"input path is not a regular file: {filename!r}")
    return candidate


def write_input_file_if_missing(
    filename: str,
    data: bytes,
    *,
    root: Path | None = None,
) -> Path:
    """Create a contained cache file without following a final symlink."""
    path = resolve_input_file(filename, root=root, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Re-resolve after creating parents so a pre-existing parent symlink cannot
    # redirect the write outside the cache.
    path = resolve_input_file(filename, root=root, must_exist=False)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        return resolve_input_file(filename, root=root, must_exist=True)

    with os.fdopen(fd, "wb") as output:
        output.write(data)
    return path
