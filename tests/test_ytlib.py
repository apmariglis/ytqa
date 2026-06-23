"""
Tests for ytlib.py shared utilities.

Group H: extract_video_id() — ensures all URL formats we support are parsed
correctly, and unsupported formats raise clearly.

Group I: load_transcript() — ensures the cache-hit / cache-miss branching
works correctly without hitting YouTube's network.

Group K: load_transcript_segments() and format_transcript_with_timestamps() —
ensures the JSON sidecar is read correctly and timestamps are formatted so
Claude can cite them in answers.
"""

import json

import pytest

import ytlib
from ytlib import (
    extract_video_id,
    load_transcript,
    load_transcript_segments,
    format_transcript_with_timestamps,
    TRANSCRIPTS_DIR,
)


# ---------------------------------------------------------------------------
# Group H — extract_video_id()
# ---------------------------------------------------------------------------

STANDARD_WATCH_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
SHORT_URL = "https://youtu.be/dQw4w9WgXcQ"
NO_WWW_URL = "https://youtube.com/watch?v=dQw4w9WgXcQ"
EXPECTED_VIDEO_ID = "dQw4w9WgXcQ"


def test_extracts_video_id_from_standard_youtube_watch_url():
    result = extract_video_id(STANDARD_WATCH_URL)

    assert result == EXPECTED_VIDEO_ID


def test_extracts_video_id_from_youtu_be_short_url():
    result = extract_video_id(SHORT_URL)

    assert result == EXPECTED_VIDEO_ID


def test_raises_value_error_for_unrecognized_url_format():
    unrecognized_url = "https://vimeo.com/123456"

    with pytest.raises(ValueError):
        extract_video_id(unrecognized_url)


def test_handles_youtube_com_url_without_www_prefix():
    result = extract_video_id(NO_WWW_URL)

    assert result == EXPECTED_VIDEO_ID


# ---------------------------------------------------------------------------
# Group I — load_transcript() caching
# ---------------------------------------------------------------------------

CACHED_VIDEO_ID = "abc123cached"
CACHED_TRANSCRIPT_TEXT = "This is the cached transcript."


def test_returns_from_cache_true_when_transcript_file_exists(tmp_path, monkeypatch):
    # Redirect TRANSCRIPTS_DIR to a temp directory so the real fs is untouched.
    monkeypatch.setattr(ytlib, "TRANSCRIPTS_DIR", tmp_path)
    (tmp_path / f"{CACHED_VIDEO_ID}.txt").write_text(CACHED_TRANSCRIPT_TEXT, encoding="utf-8")

    _text, from_cache = load_transcript(CACHED_VIDEO_ID)

    assert from_cache is True


def test_returns_from_cache_false_when_transcript_must_be_fetched(tmp_path, monkeypatch):
    # No cached file exists — fetch_and_save_transcript is called instead.
    monkeypatch.setattr(ytlib, "TRANSCRIPTS_DIR", tmp_path)
    monkeypatch.setattr(
        ytlib, "fetch_and_save_transcript", lambda _video_id: CACHED_TRANSCRIPT_TEXT
    )

    _text, from_cache = load_transcript("uncached_video_id")

    assert from_cache is False


def test_returns_transcript_text_from_cache_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ytlib, "TRANSCRIPTS_DIR", tmp_path)
    (tmp_path / f"{CACHED_VIDEO_ID}.txt").write_text(CACHED_TRANSCRIPT_TEXT, encoding="utf-8")

    text, _from_cache = load_transcript(CACHED_VIDEO_ID)

    assert text == CACHED_TRANSCRIPT_TEXT


# ---------------------------------------------------------------------------
# Group K — load_transcript_segments() and format_transcript_with_timestamps()
# ---------------------------------------------------------------------------

SEGMENTS_VIDEO_ID = "seg_vid_abcde"
SAMPLE_SEGMENTS = [
    {"start": 0.0, "text": "Hello everyone."},
    {"start": 5.5, "text": "Today we discuss the GIL."},
    {"start": 135.0, "text": "The GIL is a mutex."},
]


def test_load_transcript_segments_returns_none_when_json_file_does_not_exist(tmp_path, monkeypatch):
    # No sidecar file means we have no timing data — return None rather than error.
    monkeypatch.setattr(ytlib, "TRANSCRIPTS_DIR", tmp_path)

    result = load_transcript_segments(SEGMENTS_VIDEO_ID)

    assert result is None


def test_load_transcript_segments_returns_segments_from_json_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ytlib, "TRANSCRIPTS_DIR", tmp_path)
    (tmp_path / f"{SEGMENTS_VIDEO_ID}.json").write_text(
        json.dumps(SAMPLE_SEGMENTS), encoding="utf-8"
    )

    result = load_transcript_segments(SEGMENTS_VIDEO_ID)

    assert result == SAMPLE_SEGMENTS


def test_format_transcript_with_timestamps_prefixes_each_segment_with_timestamp():
    # Every segment must appear on its own line so Claude can cite a specific moment.
    segments = [{"start": 0.0, "text": "Hello."}, {"start": 60.0, "text": "Goodbye."}]

    result = format_transcript_with_timestamps(segments)
    lines = result.splitlines()

    assert len(lines) == 2


def test_format_transcript_with_timestamps_converts_seconds_to_m_colon_ss():
    # 135 seconds = 2 minutes 15 seconds → [2:15]
    segments = [{"start": 135.0, "text": "The GIL is a mutex."}]

    result = format_transcript_with_timestamps(segments)

    assert result.startswith("[2:15]")
