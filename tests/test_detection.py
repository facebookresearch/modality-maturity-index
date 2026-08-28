# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Tests for the single modality detector.

Detection is one pure function over the persisted form, so these tests need no
providers, no SDKs, and no network. That is the point: if a case cannot be
expressed as ``(response_text, assets)`` then it is not something the metric
can see, and no provider should be able to score it.
"""

import pytest

from mmi.detection import (
    ALL_MODALITIES,
    CAPTURED,
    EXTERNAL_URL,
    FAILED,
    HARNESS_TOOL,
    PAYLOAD_ABSENT,
    PAYLOAD_CAPTURE_FAILED,
    PAYLOAD_GRADEABLE,
    PAYLOAD_REFERENCE_ONLY,
    PAYLOAD_SKIPPED,
    PROVIDER_INLINE,
    PROVIDER_TOOL,
    REFERENCE_ONLY,
    SCORING_NATIVE,
    SCORING_NONE,
    SCORING_URL,
    SKIPPED,
    asset_modality,
    classify_mime,
    detect,
    payload_status,
    scoring_class,
)
from mmi.models import CapturedAsset


def asset(**kwargs) -> CapturedAsset:
    base = {
        "asset_id": "a0",
        "prompt_id": "p1",
        "modality": "Image",
        "source_type": "test",
        "delivery": PROVIDER_INLINE,
        "mime_type": "image/png",
        "capture_status": CAPTURED,
    }
    base.update(kwargs)
    return CapturedAsset(**base)


class TestClassifyMime:
    @pytest.mark.parametrize(
        "mime,expected",
        [
            ("image/png", "Image"),
            ("image/svg+xml", "Image"),
            ("IMAGE/PNG", "Image"),
            ("image/jpeg; charset=binary", "Image"),
            ("audio/mpeg", "Audio"),
            ("video/mp4", "Video"),
            ("application/pdf", "Document"),
            ("application/msword", "Document"),
            ("application/rtf", "Document"),
            ("application/epub+zip", "Document"),
            (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "Document",
            ),
            (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "Document",
            ),
            ("application/vnd.ms-excel", "Document"),
            ("application/vnd.oasis.opendocument.text", "Document"),
            ("text/csv", "Document"),
            ("text/plain", "Document"),
        ],
    )
    def test_recognised(self, mime, expected):
        assert classify_mime(mime) == expected

    @pytest.mark.parametrize(
        "mime",
        [
            "",
            "text/html",
            "text/markdown",
            "weird/thing",
            "application/json",
            "application/zip",
            "application/x-www-form-urlencoded",
        ],
    )
    def test_refuses_rather_than_guessing(self, mime):
        """The table refuses what it does not know instead of defaulting.

        Defaulting to Document is how ``text/html`` used to score Document from
        one code path and nothing from another.
        """
        assert classify_mime(mime) is None

    def test_unknown_binary_is_not_a_document(self):
        """``application/octet-stream`` is "I could not tell", not "document".

        Providers emit it as their fallback when no type is available, so
        classifying it would hand native Document credit to any unidentifiable
        bytes.
        """
        assert classify_mime("application/octet-stream") is None


class TestScoringClass:
    @pytest.mark.parametrize("delivery", [PROVIDER_INLINE, PROVIDER_TOOL, HARNESS_TOOL])
    def test_structured_artifact_with_bytes_is_native(self, delivery):
        assert scoring_class(asset(delivery=delivery)) == SCORING_NATIVE

    @pytest.mark.parametrize("status", [REFERENCE_ONLY, SKIPPED, FAILED])
    def test_structured_artifact_without_bytes_earns_nothing(self, status):
        assert scoring_class(asset(capture_status=status)) == SCORING_NONE

    def test_url_delivery_is_url_class_even_with_bytes(self):
        """Downloading a URL must never promote it to native."""
        downloaded = asset(delivery=EXTERNAL_URL, capture_status=CAPTURED)
        assert scoring_class(downloaded) == SCORING_URL

    def test_provenance_ignores_hostname(self):
        """There is no CDN allowlist, and there never will be one."""
        for host in (
            "https://cdn.example.com/a.png",
            "https://images.example.org/a.png",
            "https://storage.example.net/a.png",
        ):
            assert (
                scoring_class(asset(delivery=EXTERNAL_URL, source_url=host))
                == SCORING_URL
            )
            assert (
                scoring_class(asset(delivery=PROVIDER_INLINE, source_url=host))
                == SCORING_NATIVE
            )


class TestModalityHint:
    def test_shared_table_wins_over_hint(self):
        """An adapter cannot mint credit the table would have refused."""
        hinted = asset(mime_type="image/png", modality_hint="Video")
        assert asset_modality(hinted) == "Image"

    @pytest.mark.parametrize("mime", ["", "application/octet-stream"])
    def test_hint_used_for_opaque_mime(self, mime):
        """Both forms of "no usable type" must reach the hint.

        ``application/octet-stream`` is the fallback providers emit when they
        cannot determine a type, so it has to behave like a missing type rather
        than being absorbed by the Document branch.
        """
        assert asset_modality(asset(mime_type=mime, modality_hint="Audio")) == "Audio"

    def test_hint_cannot_override_a_classifiable_mime(self):
        hinted = asset(mime_type="application/pdf", modality_hint="Audio")
        assert asset_modality(hinted) == "Document"

    def test_unclassifiable_mime_without_hint_earns_nothing(self):
        assert asset_modality(asset(mime_type="application/octet-stream")) is None

    def test_nonsense_hint_is_ignored(self):
        assert asset_modality(asset(mime_type="", modality_hint="Hologram")) is None


class TestDetect:
    def test_returns_every_modality(self):
        result = detect("")
        assert sorted(result) == sorted(ALL_MODALITIES)

    def test_text_from_prose(self):
        result = detect("here you go")
        assert result["Text"].detected_native is True

    def test_whitespace_is_not_text(self):
        assert detect("   \n  ")["Text"].detected_native is False

    @pytest.mark.parametrize(
        "mime,modality",
        [
            ("image/png", "Image"),
            ("audio/mpeg", "Audio"),
            ("video/mp4", "Video"),
            ("application/pdf", "Document"),
        ],
    )
    def test_native_from_captured_artifact(self, mime, modality):
        result = detect("", [asset(mime_type=mime)])
        assert result[modality].detected_native is True
        assert result[modality].detected_via_url is False

    def test_platform_tool_artifact_is_native(self):
        result = detect("", [asset(delivery=PROVIDER_TOOL, mime_type="image/png")])
        assert result["Image"].detected_native is True

    def test_bytesless_artifact_earns_no_credit(self):
        result = detect(
            "", [asset(capture_status=REFERENCE_ONLY, source_url="https://x.invalid/a")]
        )
        assert result["Image"].detected_native is False
        assert result["Image"].detected_via_url is False

    def test_url_delivered_artifact_is_url_class(self):
        result = detect(
            "",
            [
                asset(
                    delivery=EXTERNAL_URL,
                    mime_type="video/mp4",
                    modality="Video",
                    source_url="https://example.invalid/clip.mp4",
                )
            ],
        )
        assert result["Video"].detected_via_url is True
        assert result["Video"].detected_native is False

    def test_url_fallback_from_prose(self):
        result = detect("watch https://www.youtube.com/watch?v=abc")
        assert result["Video"].detected_via_url is True
        assert result["Video"].detected_native is False

    def test_hidden_tool_trace_cannot_mint_url_credit(self):
        """Only user-visible prose reaches ``detect``.

        This is the invariant that keeps tool-capable systems comparable with
        tool-less ones, and it gets more load-bearing as models gain tools.
        """
        visible = "I could not make a video."
        assert detect(visible)["Video"].detected_via_url is False

    def test_unclassifiable_artifact_is_ignored(self):
        result = detect("", [asset(mime_type="text/html")])
        assert all(not d.detected_native for d in result.values())

    def test_evidence_is_recorded(self):
        result = detect("", [asset(asset_id="a7", mime_type="image/png")])
        assert "a7" in result["Image"].native_evidence
        assert "image/png" in result["Image"].native_evidence


class TestPurity:
    def test_same_inputs_give_same_outputs(self):
        """Fresh scoring and --rescore must not be able to disagree."""
        text = "see https://vimeo.com/123"
        assets = [asset(mime_type="image/png")]
        first = detect(text, assets)
        second = detect(text, assets)
        assert {
            m: (d.detected_native, d.detected_via_url) for m, d in first.items()
        } == {m: (d.detected_native, d.detected_via_url) for m, d in second.items()}

    def test_does_not_mutate_assets(self):
        a = asset()
        before = (a.modality, a.capture_status, a.delivery, a.mime_type)
        detect("x", [a])
        assert (a.modality, a.capture_status, a.delivery, a.mime_type) == before


class TestPayloadStatus:
    """Why a modality's artifacts can or cannot be graded for content.

    The rubric scorer used to filter on ``capture_status == CAPTURED`` and hand
    its caller a bare list, so "the model produced nothing", "we never fetched
    it" and "fetching failed" all arrived as an empty list and were reported
    identically. This taxonomy lives here so there is one opinion about it.
    """

    def test_no_artifacts_is_absent(self):
        assert payload_status([]) == PAYLOAD_ABSENT

    @pytest.mark.parametrize(
        "status,expected",
        [
            (CAPTURED, PAYLOAD_GRADEABLE),
            (REFERENCE_ONLY, PAYLOAD_REFERENCE_ONLY),
            (FAILED, PAYLOAD_CAPTURE_FAILED),
            (SKIPPED, PAYLOAD_SKIPPED),
        ],
    )
    def test_single_artifact(self, status, expected):
        assert payload_status([status]) == expected

    @pytest.mark.parametrize("other", [REFERENCE_ONLY, FAILED, SKIPPED])
    def test_any_bytes_make_it_gradeable(self, other):
        assert payload_status([other, CAPTURED]) == PAYLOAD_GRADEABLE

    @pytest.mark.parametrize("passive", [REFERENCE_ONLY, SKIPPED])
    def test_failure_outranks_the_passive_states(self, passive):
        """Our failure must stay visible rather than reading as a passive miss."""
        assert payload_status([passive, FAILED]) == PAYLOAD_CAPTURE_FAILED

    def test_unrecognised_status_does_not_become_gradeable(self):
        assert payload_status(["something-new"]) == PAYLOAD_SKIPPED
