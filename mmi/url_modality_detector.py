# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Detect media modalities from URLs found in text responses.

This module is a fallback: if a model can't produce a media type natively
but returns a URL pointing to that media (e.g. a YouTube link for Video),
the modality is still counted as produced.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Platform patterns by modality
# ---------------------------------------------------------------------------

VIDEO_PLATFORMS = [
    r"(youtube\.com|youtu\.be)",
    r"(instagram\.com/(reel|p)/)",
    r"(tiktok\.com/)",
    r"(vimeo\.com/)",
    r"(dailymotion\.com/)",
    r"(twitch\.tv/)",
]

AUDIO_PLATFORMS = [
    r"(soundcloud\.com/)",
    r"(open\.spotify\.com/)",
    r"(music\.apple\.com/)",
    r"(bandcamp\.com/)",
]

IMAGE_PLATFORMS = [
    r"(flickr\.com/)",
    r"(imgur\.com/)",
    r"(unsplash\.com/photos/)",
]

DOCUMENT_PLATFORMS = [
    r"(docs\.google\.com/)",
    r"(drive\.google\.com/)",
    r"(dropbox\.com/)",
]

# ---------------------------------------------------------------------------
# File extension patterns by modality
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = r"\.(mp4|webm|mov|avi|mkv|flv|wmv)(\?|$)"
AUDIO_EXTENSIONS = r"\.(mp3|wav|ogg|flac|aac|m4a|wma)(\?|$)"
IMAGE_EXTENSIONS = r"\.(jpg|jpeg|png|gif|webp|bmp|svg|tiff)(\?|$)"
DOCUMENT_EXTENSIONS = r"\.(pdf|docx?|xlsx?|pptx?|csv|txt)(\?|$)"

# ---------------------------------------------------------------------------
# URL extraction pattern
# ---------------------------------------------------------------------------

_URL_PATTERN = re.compile(r"https?://[^\s\)\]\"'`]+")

# Trailing characters that are valid in URLs but almost always sentence
# punctuation when they appear at the very end of an extracted URL.
_TRAILING_PUNCT = re.compile(r"[.,;:!]+$")


def extract_urls(text: str) -> list[str]:
    """Extract all URLs from text, stripping trailing punctuation."""
    return [_TRAILING_PUNCT.sub("", url) for url in _URL_PATTERN.findall(text)]


def classify_url(url: str) -> str | None:
    """Classify a single URL into a modality string, or None if unrecognised."""
    url_lower = url.lower()

    if any(re.search(p, url_lower) for p in VIDEO_PLATFORMS) or re.search(
        VIDEO_EXTENSIONS, url_lower
    ):
        return "Video"

    if any(re.search(p, url_lower) for p in AUDIO_PLATFORMS) or re.search(
        AUDIO_EXTENSIONS, url_lower
    ):
        return "Audio"

    if any(re.search(p, url_lower) for p in IMAGE_PLATFORMS) or re.search(
        IMAGE_EXTENSIONS, url_lower
    ):
        return "Image"

    if any(re.search(p, url_lower) for p in DOCUMENT_PLATFORMS) or re.search(
        DOCUMENT_EXTENSIONS, url_lower
    ):
        return "Document"

    return None


def classify_urls_by_modality(text: str) -> dict[str, list[str]]:
    """Scan text for URLs and return a dict mapping each modality to its evidence URLs."""
    urls = extract_urls(text)
    found: dict[str, list[str]] = {}

    for url in urls:
        mod = classify_url(url)
        if mod:
            found.setdefault(mod, []).append(url)

    return found


def detect_modalities_from_urls(text: str) -> set[str]:
    """Scan text for URLs and return the set of modalities they represent."""
    return set(classify_urls_by_modality(text).keys())
