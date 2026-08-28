# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""The single modality detector.

Everything that decides *what a system produced* lives here. Providers extract
artifacts; this module adjudicates them. That split is what makes scores
comparable across systems under test: if each adapter answered "does
``image/svg+xml`` count as Image?" for itself, adding a SUT would silently
redefine the benchmark.

``detect`` is pure and operates only on the persisted representation, so fresh
scoring and ``--rescore`` call the identical function on identical inputs.

Provenance is **structural, never hostname-based**. An artifact is
native-eligible because of *how it arrived* — inline bytes, a file part, an
attachment, a platform-tool file — not because of which host served it. There
is no CDN allowlist and there never will be one.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import CapturedAsset, DetectionResult, ModalityDetection
from .url_modality_detector import classify_urls_by_modality, extract_urls

ALL_MODALITIES = ["Text", "Image", "Audio", "Video", "Document"]

# ---------------------------------------------------------------------------
# Delivery provenance — how the artifact reached us
# ---------------------------------------------------------------------------

#: Bytes carried directly in the model's own response body.
PROVIDER_INLINE = "provider_inline"
#: Produced by a tool belonging to the provider's platform (code interpreter,
#: image-generation call). Recorded distinctly from ``PROVIDER_INLINE`` so
#: "the model emitted it" can later be separated from "the model's platform
#: tool emitted it" without re-running anything.
PROVIDER_TOOL = "provider_tool"
#: Produced by one of MMI's own neutral media tools on the model's request.
HARNESS_TOOL = "harness_tool"
#: The artifact was only ever pointed at by a URL.
EXTERNAL_URL = "external_url"

#: Deliveries that can earn native credit, given actual bytes.
NATIVE_ELIGIBLE_DELIVERIES = frozenset({PROVIDER_INLINE, PROVIDER_TOOL, HARNESS_TOOL})

# ---------------------------------------------------------------------------
# Capture status — whether we actually hold the artifact
# ---------------------------------------------------------------------------

#: We hold the bytes.
CAPTURED = "captured"
#: We hold an identifier or URL but no bytes.
REFERENCE_ONLY = "reference_only"
#: Nothing to capture.
SKIPPED = "skipped"
#: Capture was attempted and failed.
FAILED = "failed"

# ---------------------------------------------------------------------------
# Scoring class — what the metric makes of it
# ---------------------------------------------------------------------------

SCORING_NATIVE = "native"
SCORING_URL = "url"
SCORING_NONE = "none"

# ---------------------------------------------------------------------------
# Payload status — whether a modality's artifacts can be graded for content
# ---------------------------------------------------------------------------

#: We hold the bytes, so a rubric can be graded against the content.
PAYLOAD_GRADEABLE = "gradeable"
#: No artifact of the modality was produced at all.
PAYLOAD_ABSENT = "absent"
#: An artifact was pointed at but never retrieved, so there is no content.
PAYLOAD_REFERENCE_ONLY = "reference_only"
#: An artifact was recorded with no bytes and nothing to retrieve.
PAYLOAD_SKIPPED = "skipped"
#: Retrieval was attempted and failed. A harness failure, not a model failure.
PAYLOAD_CAPTURE_FAILED = "capture_failed"
#: We hold the bytes but could not put them in front of the judge. Derived from
#: the artifact on disk rather than from a capture status, so it is not part of
#: ``_PAYLOAD_PRECEDENCE``. A harness failure, not a model failure.
PAYLOAD_UNSENDABLE = "unsendable"

#: Which capture status wins when a modality has several artifacts. Any bytes at
#: all make it gradeable; otherwise a failed retrieval outranks the passive
#: states, because that one is *our* fault and has to stay visible.
_PAYLOAD_PRECEDENCE = (
    (CAPTURED, PAYLOAD_GRADEABLE),
    (FAILED, PAYLOAD_CAPTURE_FAILED),
    (REFERENCE_ONLY, PAYLOAD_REFERENCE_ONLY),
    (SKIPPED, PAYLOAD_SKIPPED),
)


def payload_status(capture_statuses: Sequence[str]) -> str:
    """Why a modality's artifacts can or cannot be graded for their content.

    Rubrics grade content, which requires the bytes. This is the one place that
    decides what a capture status means for gradeability, so the rubric scorer
    consumes an answer rather than forming a second opinion about it.
    """
    if not capture_statuses:
        return PAYLOAD_ABSENT
    for status, result in _PAYLOAD_PRECEDENCE:
        if status in capture_statuses:
            return result
    return PAYLOAD_SKIPPED


# Document is an allowlist, not an ``application/*`` prefix match.
# ``application/octet-stream`` is what a provider emits when it cannot determine
# a type at all, so a prefix match turned "unknown bytes" into native Document
# credit and, because the table is consulted before ``modality_hint``, also
# suppressed the one mechanism an adapter has for saying what those bytes are.
_DOCUMENT_MIMES = frozenset(
    {
        "application/epub+zip",
        "application/msword",
        "application/pdf",
        "application/rtf",
        "text/csv",
        "text/plain",
        "text/rtf",
    }
)

_DOCUMENT_MIME_PREFIXES = (
    "application/vnd.ms-",
    "application/vnd.oasis.opendocument.",
    "application/vnd.openxmlformats-officedocument.",
)


def classify_mime(mime: str) -> str | None:
    """The one MIME → modality table.

    Deliberately conservative: a MIME family this table does not recognise is
    *not* classified. When a future provider ships something unclassifiable the
    fix is one line here, not an opinion in that provider's adapter.
    """
    mime = (mime or "").split(";", 1)[0].strip().lower()
    if not mime:
        return None
    if mime.startswith("image/"):
        return "Image"
    if mime.startswith("audio/"):
        return "Audio"
    if mime.startswith("video/"):
        return "Video"
    if mime in _DOCUMENT_MIMES or mime.startswith(_DOCUMENT_MIME_PREFIXES):
        return "Document"
    return None


def scoring_class(asset: CapturedAsset) -> str:
    """Map an asset's delivery provenance and capture status onto the metric.

    Native requires *actual bytes*: a reference we could not fetch is evidence
    that something was pointed at, not evidence that it was produced.
    """
    if asset.delivery in NATIVE_ELIGIBLE_DELIVERIES:
        return SCORING_NATIVE if asset.capture_status == CAPTURED else SCORING_NONE
    if asset.delivery == EXTERNAL_URL:
        return SCORING_URL
    return SCORING_NONE


def asset_modality(asset: CapturedAsset) -> str | None:
    """Resolve an asset to a modality.

    The shared table wins wherever it can classify. The provider's optional
    ``modality_hint`` is consulted only for genuinely opaque MIME, so an adapter
    can never mint credit the table would have refused.
    """
    resolved = classify_mime(asset.mime_type)
    if resolved is not None:
        return resolved
    hint = (asset.modality_hint or "").strip()
    return hint if hint in ALL_MODALITIES else None


def make_empty_modalities() -> dict[str, ModalityDetection]:
    return {m: ModalityDetection() for m in ALL_MODALITIES}


def detect(
    response_text: str,
    assets: Sequence[CapturedAsset] = (),
) -> dict[str, ModalityDetection]:
    """Derive per-modality evidence from the persisted response form.

    Args:
        response_text: **User-visible prose only.** Hidden tool traces and
            request metadata must never reach this argument, or a tool-trace URL
            could mint URL credit the user never saw.
        assets: Normalized artifacts captured from the response.

    Returns:
        One :class:`ModalityDetection` per modality in :data:`ALL_MODALITIES`.
    """
    modalities = make_empty_modalities()

    if response_text.strip():
        modalities["Text"].detected_native = True
        modalities["Text"].native_evidence = "response text"

    for asset in assets:
        modality = asset_modality(asset)
        if modality is None:
            continue
        cls = scoring_class(asset)
        evidence = (
            f"{asset.delivery} asset {asset.asset_id} "
            f"mime={asset.mime_type or 'unknown'}"
        )
        if cls == SCORING_NATIVE:
            modalities[modality].detected_native = True
            modalities[modality].native_evidence = evidence
        elif cls == SCORING_URL:
            modalities[modality].detected_via_url = True
            modalities[modality].url_evidence = asset.source_url or evidence

    for modality, urls in classify_urls_by_modality(response_text).items():
        modalities[modality].detected_via_url = True
        modalities[modality].url_evidence = ", ".join(urls)

    return modalities


def build_detection_result(
    response_text: str,
    assets: Sequence[CapturedAsset] = (),
) -> DetectionResult:
    """``detect`` plus the persisted context the result record carries."""
    return DetectionResult(
        modalities=detect(response_text, assets),
        text_content=response_text,
        urls_found=extract_urls(response_text),
    )
