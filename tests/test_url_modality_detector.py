# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Tests for the URL modality detector."""

from mmi.url_modality_detector import detect_modalities_from_urls, extract_urls


class TestVideoDetection:
    def test_youtube_url(self):
        text = "Check out this video: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert "Video" in detect_modalities_from_urls(text)

    def test_youtu_be_short_url(self):
        text = "See https://youtu.be/dQw4w9WgXcQ"
        assert "Video" in detect_modalities_from_urls(text)

    def test_tiktok_url(self):
        text = "TikTok: https://www.tiktok.com/@user/video/12345"
        assert "Video" in detect_modalities_from_urls(text)

    def test_vimeo_url(self):
        text = "https://vimeo.com/123456789"
        assert "Video" in detect_modalities_from_urls(text)

    def test_instagram_reel(self):
        text = "https://instagram.com/reel/ABC123/"
        assert "Video" in detect_modalities_from_urls(text)

    def test_mp4_extension(self):
        text = "Download: https://example.com/clip.mp4"
        assert "Video" in detect_modalities_from_urls(text)

    def test_webm_extension(self):
        text = "https://cdn.example.com/video.webm?token=abc"
        assert "Video" in detect_modalities_from_urls(text)


class TestAudioDetection:
    def test_soundcloud(self):
        text = "Listen: https://soundcloud.com/artist/track"
        assert "Audio" in detect_modalities_from_urls(text)

    def test_spotify(self):
        text = "https://open.spotify.com/track/abc123"
        assert "Audio" in detect_modalities_from_urls(text)

    def test_mp3_extension(self):
        text = "https://example.com/song.mp3"
        assert "Audio" in detect_modalities_from_urls(text)

    def test_wav_extension(self):
        text = "https://example.com/audio.wav"
        assert "Audio" in detect_modalities_from_urls(text)

    def test_ogg_extension(self):
        text = "https://cdn.example.com/file.ogg?v=2"
        assert "Audio" in detect_modalities_from_urls(text)


class TestImageDetection:
    def test_imgur(self):
        text = "https://imgur.com/gallery/abc123"
        assert "Image" in detect_modalities_from_urls(text)

    def test_flickr(self):
        text = "https://www.flickr.com/photos/user/12345"
        assert "Image" in detect_modalities_from_urls(text)

    def test_jpg_extension(self):
        text = "https://example.com/photo.jpg"
        assert "Image" in detect_modalities_from_urls(text)

    def test_png_extension(self):
        text = "https://example.com/image.png"
        assert "Image" in detect_modalities_from_urls(text)

    def test_webp_extension(self):
        text = "https://cdn.example.com/pic.webp"
        assert "Image" in detect_modalities_from_urls(text)


class TestDocumentDetection:
    def test_google_docs(self):
        text = "https://docs.google.com/document/d/abc123"
        assert "Document" in detect_modalities_from_urls(text)

    def test_google_drive(self):
        text = "https://drive.google.com/file/d/abc123/view"
        assert "Document" in detect_modalities_from_urls(text)

    def test_pdf_extension(self):
        text = "https://example.com/report.pdf"
        assert "Document" in detect_modalities_from_urls(text)

    def test_docx_extension(self):
        text = "https://example.com/document.docx"
        assert "Document" in detect_modalities_from_urls(text)

    def test_xlsx_extension(self):
        text = "https://example.com/data.xlsx"
        assert "Document" in detect_modalities_from_urls(text)

    def test_csv_extension(self):
        text = "https://example.com/data.csv"
        assert "Document" in detect_modalities_from_urls(text)


class TestEdgeCases:
    def test_no_urls(self):
        text = "This is just plain text with no links."
        assert detect_modalities_from_urls(text) == set()

    def test_empty_string(self):
        assert detect_modalities_from_urls("") == set()

    def test_multiple_modalities(self):
        text = (
            "Video: https://youtube.com/watch?v=abc "
            "Audio: https://soundcloud.com/track "
            "Image: https://example.com/pic.jpg "
            "Doc: https://docs.google.com/document/d/123"
        )
        result = detect_modalities_from_urls(text)
        assert result == {"Video", "Audio", "Image", "Document"}

    def test_non_media_url(self):
        text = "Visit https://www.google.com for more info"
        assert detect_modalities_from_urls(text) == set()

    def test_url_with_query_params(self):
        text = "https://example.com/video.mp4?token=abc&quality=hd"
        assert "Video" in detect_modalities_from_urls(text)


class TestExtractUrls:
    def test_extracts_multiple(self):
        urls = extract_urls("See https://a.com/x.jpg and https://b.com/y.mp4")
        assert len(urls) == 2

    def test_empty(self):
        assert extract_urls("") == []

    def test_no_urls(self):
        assert extract_urls("just text") == []
