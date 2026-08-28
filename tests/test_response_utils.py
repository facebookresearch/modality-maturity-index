# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Tests for response text and rubric utilities."""

from mmi.response_utils import (
    extract_response_text,
    resolve_asset_path,
    unsendable_reason,
)


def test_extract_plain_text_field():
    assert extract_response_text({"raw_response": {"text": "hello"}}) == "hello"


def test_extract_openai_responses_text():
    row = {
        "raw_response": {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "first"},
                        {"type": "output_text", "text": "second"},
                    ],
                }
            ]
        }
    }
    assert extract_response_text(row) == "first\nsecond"


def test_response_text_field_wins():
    assert (
        extract_response_text(
            {"response_text": "cached", "raw_response": {"text": "raw"}}
        )
        == "cached"
    )


# ---------------------------------------------------------------------------
# Locating a recorded artifact
# ---------------------------------------------------------------------------


def test_absolute_path_from_another_machine_is_rebased_under_results(
    monkeypatch, tmp_path
):
    """Results written elsewhere must stay re-scorable here.

    local_path is absolute, so without re-basing every artifact of an imported run
    would miss and each rubric would be graded with no artifact attached.
    """
    results = tmp_path / "results"
    local = results / "my_config" / "20260101_120000_some-model_assets" / "p1"
    local.mkdir(parents=True)
    (local / "p1_output_0_image.png").write_bytes(b"bytes")
    monkeypatch.setattr("mmi.response_utils.RESULTS_DIR", results)

    foreign = (
        "/data/users/someone/mmi/results/my_config/"
        "20260101_120000_some-model_assets/p1/p1_output_0_image.png"
    )
    assert resolve_asset_path(foreign) == local / "p1_output_0_image.png"


def test_relative_path_resolves_under_results(monkeypatch, tmp_path):
    results = tmp_path / "results"
    (results / "cfg").mkdir(parents=True)
    (results / "cfg" / "a.png").write_bytes(b"bytes")
    monkeypatch.setattr("mmi.response_utils.RESULTS_DIR", results)

    assert resolve_asset_path("cfg/a.png") == results / "cfg" / "a.png"


def test_unresolvable_path_is_none(monkeypatch, tmp_path):
    monkeypatch.setattr("mmi.response_utils.RESULTS_DIR", tmp_path)
    assert resolve_asset_path("/nowhere/at/all/x.png") is None
    assert resolve_asset_path("") is None


def test_existing_path_is_used_as_recorded(tmp_path):
    asset = tmp_path / "a.png"
    asset.write_bytes(b"bytes")
    assert resolve_asset_path(str(asset)) == asset


# ---------------------------------------------------------------------------
# Whether an artifact can be put in front of the judge
# ---------------------------------------------------------------------------


def test_readable_artifact_is_sendable(tmp_path):
    asset = tmp_path / "a.png"
    asset.write_bytes(b"bytes")
    assert (
        unsendable_reason({"local_path": str(asset), "mime_type": "image/png"}) is None
    )


def test_office_document_is_sendable(tmp_path):
    """Provider capability is not second-guessed here; see test_rubric_scorer."""
    asset = tmp_path / "a.docx"
    asset.write_bytes(b"bytes")
    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert unsendable_reason({"local_path": str(asset), "mime_type": mime}) is None


def test_missing_file_is_unsendable(monkeypatch, tmp_path):
    monkeypatch.setattr("mmi.response_utils.RESULTS_DIR", tmp_path)
    reason = unsendable_reason({"local_path": "/nope/a.png", "mime_type": "image/png"})
    assert reason and "not readable" in reason


def test_absent_path_is_unsendable():
    reason = unsendable_reason({"local_path": "", "mime_type": "image/png"})
    assert reason and "no file path" in reason


def test_oversize_artifact_is_unsendable(monkeypatch, tmp_path):
    asset = tmp_path / "a.mp4"
    asset.write_bytes(b"bytes")
    monkeypatch.setattr("mmi.response_utils._MAX_MEDIA_BYTES", 1)
    reason = unsendable_reason({"local_path": str(asset), "mime_type": "video/mp4"})
    assert reason and "inline request limit" in reason


def test_untyped_artifact_is_unsendable(tmp_path):
    asset = tmp_path / "a.unknownext"
    asset.write_bytes(b"bytes")
    reason = unsendable_reason({"local_path": str(asset), "mime_type": ""})
    assert reason and "no MIME type" in reason
