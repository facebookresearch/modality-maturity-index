# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Tests for the dataset loader."""

import json
from pathlib import Path

import pandas as pd
import pytest

from mmi.dataset import DatasetError, _parse_list, load_dataset, prompt_id_set_hash
from mmi.response_utils import parse_rubric_criteria


class TestParseList:
    def test_comma_separated(self):
        assert _parse_list("Text, Image, Audio") == ["Text", "Image", "Audio"]

    def test_comma_no_space(self):
        assert _parse_list("Text,Image") == ["Text", "Image"]

    def test_single_item(self):
        assert _parse_list("Text") == ["Text"]

    def test_empty_string(self):
        assert _parse_list("") == []

    def test_whitespace_only(self):
        assert _parse_list("   ") == []

    def test_newline_separated(self):
        assert _parse_list("file1.jpg\nfile2.mp3", sep="\n") == [
            "file1.jpg",
            "file2.mp3",
        ]

    def test_newline_with_empty(self):
        assert _parse_list("file1.jpg\n\nfile2.mp3", sep="\n") == [
            "file1.jpg",
            "file2.mp3",
        ]


class TestParseRubricCriteria:
    def test_json_criteria_object_rubrics(self):
        raw = json.dumps(
            {
                "criteria": [
                    {"id": "1", "criterion": "First criterion.", "modality": "text"},
                    {"id": "2", "criterion": "Second criterion.", "modality": "image"},
                ]
            }
        )

        criteria = parse_rubric_criteria(raw)

        assert [(c.id, c.criterion, c.modality) for c in criteria] == [
            ("1", "First criterion.", "Text"),
            ("2", "Second criterion.", "Image"),
        ]

    def test_fails_without_explicit_modality(self):
        raw = json.dumps({"criteria": [{"id": "1", "criterion": "Missing modality."}]})

        with pytest.raises(ValueError, match="invalid modality"):
            parse_rubric_criteria(raw)

    def test_fails_on_non_structured_rubrics(self):
        with pytest.raises(ValueError, match="structured JSON"):
            parse_rubric_criteria("1. First criterion.\n2. Second criterion.")
        with pytest.raises(ValueError, match="criteria list"):
            parse_rubric_criteria('["A", "B"]')
        assert parse_rubric_criteria("") == []


class TestLoadDataset:
    def _write_parquet(self, tmpdir: Path, records: list[dict]) -> Path:
        path = tmpdir / "test.parquet"
        pd.DataFrame(records).to_parquet(path, index=False)
        return path

    def test_basic_load(self, tmp_path, monkeypatch):
        records = [
            {
                "prompt_id": "p001",
                "prompt_text": "Hello",
                "input_modalities": "Text",
                "output_modalities": "Text",
                "input_files": "",
            },
            {
                "prompt_id": "p002",
                "prompt_text": "Describe image",
                "input_modalities": "Image, Text",
                "output_modalities": "Text, Image",
                "input_files": "p002.jpg",
            },
        ]
        path = self._write_parquet(tmp_path, records)

        import mmi.dataset as dataset_mod

        monkeypatch.setattr(dataset_mod, "INPUT_FILES_DIR", tmp_path)
        (tmp_path / "p002.jpg").write_bytes(b"fake")

        prompts = load_dataset(path)

        assert len(prompts) == 2
        assert prompts[0].prompt_id == "p001"
        assert prompts[0].input_modalities == ["Text"]
        assert prompts[0].output_modalities == ["Text"]
        assert prompts[0].input_files == []

        assert prompts[1].prompt_id == "p002"
        assert prompts[1].input_modalities == ["Image", "Text"]
        assert prompts[1].output_modalities == ["Text", "Image"]
        assert prompts[1].input_files == ["p002.jpg"]
        assert prompts[1].rubric_criteria == []

    def test_parquet_load_materializes_embedded_media(self, tmp_path, monkeypatch):
        import mmi.dataset as dataset_mod

        media_dir = tmp_path / "input_files"
        monkeypatch.setattr(dataset_mod, "INPUT_FILES_DIR", media_dir)
        path = tmp_path / "test.parquet"
        pd.DataFrame(
            [
                {
                    "prompt_id": "p001",
                    "prompt_text": "Describe image and audio",
                    "input_modalities": "Audio, Image, Text",
                    "output_modalities": "Text",
                    "input_files": "p001_1.jpg\np001_2.mp3",
                    "rubrics": json.dumps(
                        {
                            "criteria": [
                                {
                                    "id": "1",
                                    "criterion": "Check image.",
                                    "modality": "text",
                                },
                                {
                                    "id": "2",
                                    "criterion": "Check audio.",
                                    "modality": "text",
                                },
                            ]
                        }
                    ),
                    "input_images": [{"bytes": b"image-bytes"}],
                    "input_audio": {"bytes": b"audio-bytes"},
                    "input_video": None,
                    "input_document": None,
                }
            ]
        ).to_parquet(path, index=False)

        prompts = load_dataset(path)

        assert len(prompts) == 1
        assert prompts[0].prompt_id == "p001"
        assert prompts[0].input_files == ["p001_1.jpg", "p001_2.mp3"]
        assert [(c.criterion, c.modality) for c in prompts[0].rubric_criteria] == [
            ("Check image.", "Text"),
            ("Check audio.", "Text"),
        ]
        assert (media_dir / "p001_1.jpg").read_bytes() == b"image-bytes"
        assert (media_dir / "p001_2.mp3").read_bytes() == b"audio-bytes"

    def test_rubric_modality_must_match_output_modality(self, tmp_path):
        path = self._write_parquet(
            tmp_path,
            [
                {
                    "prompt_id": "p001",
                    "prompt_text": "Hello",
                    "input_modalities": "Text",
                    "output_modalities": "Text",
                    "input_files": "",
                    "rubrics": json.dumps(
                        {
                            "criteria": [
                                {
                                    "id": "1",
                                    "criterion": "Must produce an image.",
                                    "modality": "image",
                                }
                            ]
                        }
                    ),
                }
            ],
        )

        with pytest.raises(DatasetError, match="rubric modalities must match"):
            load_dataset(path)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(DatasetError, match="not found"):
            load_dataset(tmp_path / "nonexistent.parquet")

    def test_jsonl_load(self, tmp_path, monkeypatch):
        records = [
            {
                "prompt_id": "p001",
                "prompt_text": "Hello",
                "input_modalities": "Text",
                "output_modalities": "Text",
                "input_files": "",
                "rubrics": json.dumps(
                    {
                        "criteria": [
                            {
                                "id": "1",
                                "criterion": "Say hello.",
                                "modality": "text",
                            }
                        ]
                    }
                ),
            },
            {
                "prompt_id": "p002",
                "prompt_text": "Describe image",
                "input_modalities": "Image, Text",
                "output_modalities": "Text, Image",
                "input_files": "p002.jpg",
            },
        ]
        path = tmp_path / "dataset.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records))

        import mmi.dataset as dataset_mod

        monkeypatch.setattr(dataset_mod, "INPUT_FILES_DIR", tmp_path)
        (tmp_path / "p002.jpg").write_bytes(b"fake")

        prompts = load_dataset(path)

        assert len(prompts) == 2
        assert prompts[0].prompt_id == "p001"
        assert [(c.criterion, c.modality) for c in prompts[0].rubric_criteria] == [
            ("Say hello.", "Text"),
        ]
        assert prompts[1].input_files == ["p002.jpg"]

    def test_rejects_unsupported_dataset_path(self, tmp_path):
        import pytest

        path = tmp_path / "dataset.csv"
        path.write_text("")
        with pytest.raises(DatasetError, match="must be .jsonl or .parquet"):
            load_dataset(path)

    def test_loading_does_not_write_into_the_repository(self, tmp_path, monkeypatch):
        """A harness that mutates its own source tree is not reproducible."""
        import mmi.dataset as dataset_mod
        from mmi.config import HARNESS_DIR

        before = {
            p: p.stat().st_mtime
            for p in HARNESS_DIR.rglob("*")
            if p.is_file() and ".venv" not in p.parts and ".git" not in p.parts
        }

        path = tmp_path / "dataset.jsonl"
        path.write_text(
            json.dumps(
                {
                    "prompt_id": "p001",
                    "prompt_text": "Hello",
                    "input_modalities": "Text",
                    "output_modalities": "Text",
                }
            )
        )
        monkeypatch.setattr(dataset_mod, "INPUT_FILES_DIR", tmp_path / "cache")
        load_dataset(path)

        after = {
            p: p.stat().st_mtime
            for p in HARNESS_DIR.rglob("*")
            if p.is_file() and ".venv" not in p.parts and ".git" not in p.parts
        }
        assert before == after

    def test_input_media_cache_is_outside_the_repository(self):
        from mmi.config import HARNESS_DIR, INPUT_FILES_DIR

        assert HARNESS_DIR not in INPUT_FILES_DIR.parents
        assert INPUT_FILES_DIR != HARNESS_DIR / "data" / "input_files"


class TestInputFileContainment:
    @pytest.mark.parametrize(
        "filename",
        [
            "../escaped.jpg",
            "nested/../../escaped.jpg",
            "/tmp/escaped.jpg",
            r"C:\\escaped.jpg",
        ],
    )
    def test_rejects_unsafe_input_filenames_before_materializing(
        self, filename, tmp_path, monkeypatch
    ):
        import mmi.dataset as dataset_mod

        media_dir = tmp_path / "input_files"
        monkeypatch.setattr(dataset_mod, "INPUT_FILES_DIR", media_dir)
        dataset_path = tmp_path / "dataset.parquet"
        pd.DataFrame(
            [
                {
                    "prompt_id": "p001",
                    "prompt_text": "Describe image",
                    "input_modalities": "Image, Text",
                    "output_modalities": "Text",
                    "input_files": filename,
                    "input_images": [{"bytes": b"outside-write"}],
                }
            ]
        ).to_parquet(dataset_path, index=False)

        with pytest.raises(DatasetError, match="input filename"):
            load_dataset(dataset_path, require_media=False)

        assert not (tmp_path / "escaped.jpg").exists()

    def test_rejects_symlink_escape_before_materializing(self, tmp_path, monkeypatch):
        import mmi.dataset as dataset_mod

        media_dir = tmp_path / "input_files"
        outside_dir = tmp_path / "outside"
        media_dir.mkdir()
        outside_dir.mkdir()
        (media_dir / "linked").symlink_to(outside_dir, target_is_directory=True)
        monkeypatch.setattr(dataset_mod, "INPUT_FILES_DIR", media_dir)
        dataset_path = tmp_path / "dataset.parquet"
        pd.DataFrame(
            [
                {
                    "prompt_id": "p001",
                    "prompt_text": "Describe image",
                    "input_modalities": "Image, Text",
                    "output_modalities": "Text",
                    "input_files": "linked/escaped.jpg",
                    "input_images": [{"bytes": b"outside-write"}],
                }
            ]
        ).to_parquet(dataset_path, index=False)

        with pytest.raises(DatasetError, match="escapes the configured cache"):
            load_dataset(dataset_path, require_media=False)

        assert not (outside_dir / "escaped.jpg").exists()

    def test_allows_nested_relative_input_filename(self, tmp_path, monkeypatch):
        import mmi.dataset as dataset_mod

        media_dir = tmp_path / "input_files"
        monkeypatch.setattr(dataset_mod, "INPUT_FILES_DIR", media_dir)
        dataset_path = tmp_path / "dataset.parquet"
        pd.DataFrame(
            [
                {
                    "prompt_id": "p001",
                    "prompt_text": "Describe image",
                    "input_modalities": "Image, Text",
                    "output_modalities": "Text",
                    "input_files": "nested/p001.jpg",
                    "input_images": [{"bytes": b"image-bytes"}],
                }
            ]
        ).to_parquet(dataset_path, index=False)

        prompts = load_dataset(dataset_path)

        assert prompts[0].input_files == ["nested/p001.jpg"]
        assert (media_dir / "nested" / "p001.jpg").read_bytes() == b"image-bytes"


class TestPromptIdHash:
    """The hash the run manifest uses to gate resume."""

    @staticmethod
    def _prompt(pid):
        from mmi.models import EvalPrompt

        return EvalPrompt(
            prompt_id=pid,
            prompt_text="x",
            input_modalities=["Text"],
            output_modalities=["Text"],
        )

    def test_is_order_independent(self):
        forward = [self._prompt("p1"), self._prompt("p2"), self._prompt("p3")]
        assert prompt_id_set_hash(forward) == prompt_id_set_hash(forward[::-1])

    def test_differs_when_the_prompt_set_differs(self):
        a = [self._prompt("p1"), self._prompt("p2")]
        b = [self._prompt("p1"), self._prompt("p3")]
        assert prompt_id_set_hash(a) != prompt_id_set_hash(b)


class TestSource:
    def test_local_override_takes_precedence(self, tmp_path, monkeypatch):
        """A local override must not reach the Hub."""
        import mmi.dataset as dataset_mod

        def _boom():
            raise AssertionError("must not hit the Hub when overridden")

        monkeypatch.setattr(dataset_mod, "_read_hub", _boom)
        path = tmp_path / "dataset.jsonl"
        path.write_text(
            json.dumps(
                {
                    "prompt_id": "p001",
                    "prompt_text": "Hello",
                    "input_modalities": "Text",
                    "output_modalities": "Text",
                }
            )
        )

        assert len(load_dataset(path)) == 1

    def test_env_override_is_honoured(self, tmp_path, monkeypatch):
        path = tmp_path / "dataset.jsonl"
        path.write_text(
            json.dumps(
                {
                    "prompt_id": "p001",
                    "prompt_text": "Hello",
                    "input_modalities": "Text",
                    "output_modalities": "Text",
                }
            )
        )
        monkeypatch.setenv("MMI_DATASET_PATH", str(path))

        assert len(load_dataset()) == 1

    def test_revision_defaults_to_main_and_is_overridable(self, monkeypatch):
        from mmi.dataset import dataset_revision

        monkeypatch.delenv("MMI_DATASET_REVISION", raising=False)
        assert dataset_revision() == "main"

        monkeypatch.setenv("MMI_DATASET_REVISION", "abc123")
        assert dataset_revision() == "abc123"
