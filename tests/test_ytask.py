"""
Tests for ytask.py.

Group A  — VideoResult dataclass
Group B  — search_youtube() normal results
Group C  — search_youtube() error handling
Group E  — fetch_transcripts_for_videos()
Group N  — build_plan_tool_definition()
Group O  — extract_keyword_windows()
Group P  — build_video_excerpts()
Group F  — run_agent_loop() planning phase
Group G  — run_agent_loop() synthesis phase
Group L  — run_agent_loop() session logging integration
Group M  — _create_with_retry() transient error handling
Group J  — main() entry point
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

import ytask
from ytask import (
    VideoResult,
    SearchError,
    SearchSession,
    build_plan_tool_definition,
    extract_keyword_windows,
    build_video_excerpts,
    fetch_transcripts_for_videos,
    run_agent_loop,
    run_followup,
    search_youtube,
)
from ytlog import SessionLog


# ---------------------------------------------------------------------------
# Helpers — lightweight stand-ins for Anthropic response objects
# ---------------------------------------------------------------------------

@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict = field(default_factory=dict)
    type: str = "tool_use"


@dataclass
class FakeResponse:
    stop_reason: str
    content: list


def _make_synthesis_response(answer_text: str) -> FakeResponse:
    return FakeResponse(stop_reason="end_turn", content=[FakeTextBlock(text=answer_text)])


# ---------------------------------------------------------------------------
# Helpers — planning response factory
# ---------------------------------------------------------------------------

PLAN_TOOL_USE_ID = "tu_plan_123"
PLAN_SEARCH_QUERIES = ["Python GIL explained", "python global interpreter lock"]
PLAN_KEYWORDS = ["GIL", "global interpreter lock", "mutex"]


def _plan_response(
    search_queries: list[str] | None = None,
    transcript_keywords: list[str] | None = None,
) -> FakeResponse:
    return FakeResponse(
        stop_reason="tool_use",
        content=[FakeToolUseBlock(
            id=PLAN_TOOL_USE_ID,
            name="plan_search",
            input={
                "search_queries": search_queries if search_queries is not None else PLAN_SEARCH_QUERIES,
                "transcript_keywords": transcript_keywords if transcript_keywords is not None else PLAN_KEYWORDS,
            },
        )],
    )


# ---------------------------------------------------------------------------
# Helpers — yt-dlp entry factory
# ---------------------------------------------------------------------------

def _yt_entry(
    video_id: str = "abc1234defg",
    title: str = "Test Video",
    description: str | None = "A description",
    view_count: int | None = 1000,
    duration: int | None = 300,
) -> dict:
    return {
        "id": video_id,
        "title": title,
        "description": description,
        "view_count": view_count,
        "duration": duration,
    }


def _mock_ydl(mocker, entries: list[dict] | None) -> MagicMock:
    """Patch yt_dlp.YoutubeDL so extract_info returns {entries: <entries>}."""
    info = {"entries": entries}
    mock_instance = MagicMock()
    mock_instance.extract_info.return_value = info
    mock_class = mocker.patch("ytask.yt_dlp.YoutubeDL")
    mock_class.return_value.__enter__.return_value = mock_instance
    mock_class.return_value.__exit__.return_value = False
    return mock_instance


# ===========================================================================
# Group A — VideoResult dataclass
# ===========================================================================

def test_video_result_stores_all_fields_accessible_by_name():
    # VideoResult must expose every field as a named attribute after construction.
    result = VideoResult(
        video_id="abc1234defg",
        title="My Video",
        url="https://www.youtube.com/watch?v=abc1234defg",
        description="A test description",
        view_count=42000,
        duration_seconds=180,
    )

    assert result.video_id == "abc1234defg"
    assert result.title == "My Video"
    assert result.url == "https://www.youtube.com/watch?v=abc1234defg"
    assert result.description == "A test description"
    assert result.view_count == 42000
    assert result.duration_seconds == 180


def test_video_result_url_contains_video_id():
    # The watch URL must embed the video_id so links are directly usable.
    video_id = "xyz9876abcd"
    result = VideoResult(
        video_id=video_id,
        title="Video",
        url=f"https://www.youtube.com/watch?v={video_id}",
        description=None,
        view_count=None,
        duration_seconds=None,
    )

    assert video_id in result.url


# ===========================================================================
# Group B — search_youtube() normal results
# ===========================================================================

def test_search_youtube_returns_a_list_of_video_result_objects(mocker):
    _mock_ydl(mocker, [_yt_entry()])

    results = search_youtube("test query")

    assert isinstance(results, list)
    assert all(isinstance(r, VideoResult) for r in results)


def test_search_youtube_maps_video_id_from_yt_dlp_id_field(mocker):
    _mock_ydl(mocker, [_yt_entry(video_id="abc1234defg")])

    results = search_youtube("test query")

    assert results[0].video_id == "abc1234defg"


def test_search_youtube_maps_title_from_yt_dlp_title_field(mocker):
    _mock_ydl(mocker, [_yt_entry(title="Python Tutorial")])

    results = search_youtube("python")

    assert results[0].title == "Python Tutorial"


def test_search_youtube_constructs_watch_url_from_video_id(mocker):
    _mock_ydl(mocker, [_yt_entry(video_id="abc1234defg")])

    results = search_youtube("test query")

    assert results[0].url == "https://www.youtube.com/watch?v=abc1234defg"


def test_search_youtube_respects_max_results_limit(mocker):
    # Return more entries than max_results to confirm truncation.
    entries = [_yt_entry(video_id=f"vid{i:09d}") for i in range(10)]
    _mock_ydl(mocker, entries)

    results = search_youtube("query", max_results=3)

    assert len(results) == 3


def test_search_youtube_handles_missing_description_field(mocker):
    _mock_ydl(mocker, [_yt_entry(description=None)])

    results = search_youtube("query")

    assert results[0].description is None


def test_search_youtube_handles_missing_view_count_field(mocker):
    _mock_ydl(mocker, [_yt_entry(view_count=None)])

    results = search_youtube("query")

    assert results[0].view_count is None


# ===========================================================================
# Group C — search_youtube() error handling
# ===========================================================================

def test_search_youtube_raises_search_error_when_yt_dlp_raises_download_error(mocker):
    import yt_dlp

    mock_class = mocker.patch("ytask.yt_dlp.YoutubeDL")
    mock_instance = MagicMock()
    mock_instance.extract_info.side_effect = yt_dlp.utils.DownloadError("network error")
    mock_class.return_value.__enter__.return_value = mock_instance
    mock_class.return_value.__exit__.return_value = False

    with pytest.raises(SearchError):
        search_youtube("query")


def test_search_youtube_returns_empty_list_when_yt_dlp_returns_no_entries(mocker):
    _mock_ydl(mocker, [])

    results = search_youtube("query")

    assert results == []


def test_search_youtube_returns_empty_list_when_yt_dlp_entries_key_is_none(mocker):
    mock_instance = MagicMock()
    mock_instance.extract_info.return_value = {"entries": None}
    mock_class = mocker.patch("ytask.yt_dlp.YoutubeDL")
    mock_class.return_value.__enter__.return_value = mock_instance
    mock_class.return_value.__exit__.return_value = False

    results = search_youtube("query")

    assert results == []


# ===========================================================================
# Group E — fetch_transcripts_for_videos()
# ===========================================================================

TRANSCRIPT_VIDEO_ID = "vid1111aaaaa"
TRANSCRIPT_TEXT = "Hello world transcript."


def test_fetch_transcripts_for_videos_returns_dict_keyed_by_video_id(mocker):
    mocker.patch("ytask.load_transcript", return_value=(TRANSCRIPT_TEXT, True, None))

    result = fetch_transcripts_for_videos([TRANSCRIPT_VIDEO_ID])

    assert TRANSCRIPT_VIDEO_ID in result


def test_fetch_transcripts_for_videos_calls_load_transcript_once_per_video_id(mocker):
    mock_load = mocker.patch("ytask.load_transcript", return_value=(TRANSCRIPT_TEXT, True, None))

    fetch_transcripts_for_videos(["vid_a1bbbbbbb", "vid_b2ccccccc"])

    assert mock_load.call_count == 2


def test_fetch_transcripts_for_videos_skips_videos_where_transcript_fetch_fails(mocker):
    # The first video fails; the second succeeds. Only the second should appear.
    def load_side_effect(video_id):
        if video_id == "bad_vid_aaaa":
            raise RuntimeError("transcript unavailable")
        return (TRANSCRIPT_TEXT, False, None)

    mocker.patch("ytask.load_transcript", side_effect=load_side_effect)

    result = fetch_transcripts_for_videos(["bad_vid_aaaa", "good_vid_aaa"])

    assert "bad_vid_aaaa" not in result
    assert "good_vid_aaa" in result


def test_fetch_transcripts_for_videos_returns_empty_dict_for_empty_input(mocker):
    mock_load = mocker.patch("ytask.load_transcript")

    result = fetch_transcripts_for_videos([])

    mock_load.assert_not_called()
    assert result == {}


# ===========================================================================
# Group N — build_plan_tool_definition()
# ===========================================================================

def test_build_plan_tool_definition_returns_dict_with_name_plan_search():
    tool = build_plan_tool_definition()

    assert tool["name"] == "plan_search"


def test_build_plan_tool_definition_includes_search_queries_array_in_schema():
    tool = build_plan_tool_definition()

    prop = tool["input_schema"]["properties"]["search_queries"]
    assert prop["type"] == "array"


def test_build_plan_tool_definition_includes_transcript_keywords_array_in_schema():
    tool = build_plan_tool_definition()

    prop = tool["input_schema"]["properties"]["transcript_keywords"]
    assert prop["type"] == "array"


def test_build_plan_tool_definition_marks_search_queries_as_required():
    tool = build_plan_tool_definition()

    assert "search_queries" in tool["input_schema"]["required"]


def test_build_plan_tool_definition_marks_transcript_keywords_as_required():
    tool = build_plan_tool_definition()

    assert "transcript_keywords" in tool["input_schema"]["required"]


# ===========================================================================
# Group O — extract_keyword_windows()
# ===========================================================================

# Segments used across multiple window tests.
_SEGMENTS = [
    {"start": 0.0,   "text": "Welcome to this video."},
    {"start": 5.0,   "text": "Today we discuss the GIL."},
    {"start": 10.0,  "text": "It is a mutex protecting Python objects."},
    {"start": 15.0,  "text": "This prevents true parallelism."},
    {"start": 20.0,  "text": "Thanks for watching."},
]


def test_extract_keyword_windows_returns_empty_list_when_no_keywords_match():
    # No segment contains "asyncio" so no windows should be returned.
    result = extract_keyword_windows(_SEGMENTS, ["asyncio"])

    assert result == []


def test_extract_keyword_windows_returns_empty_list_for_empty_segments():
    result = extract_keyword_windows([], ["GIL"])

    assert result == []


def test_extract_keyword_windows_returns_empty_list_for_empty_keywords():
    result = extract_keyword_windows(_SEGMENTS, [])

    assert result == []


def test_extract_keyword_windows_returns_window_containing_matched_segment():
    # "GIL" appears in segment index 1; the window should include that text.
    result = extract_keyword_windows(_SEGMENTS, ["GIL"], window_size=1)

    combined = " ".join(w["text"] for w in result)
    assert "GIL" in combined


def test_extract_keyword_windows_clamps_window_at_start_of_transcript():
    # Segment 0 matches; window cannot extend before the beginning.
    segments = [
        {"start": 0.0, "text": "The GIL is important."},
        {"start": 5.0, "text": "It protects objects."},
        {"start": 10.0, "text": "True parallelism is blocked."},
    ]

    result = extract_keyword_windows(segments, ["GIL"], window_size=3)

    # Start index must be 0, not negative.
    assert result[0]["start"] == 0.0


def test_extract_keyword_windows_clamps_window_at_end_of_transcript():
    # Last segment matches; window cannot extend beyond the end.
    segments = [
        {"start": 0.0,  "text": "Introduction."},
        {"start": 5.0,  "text": "More content."},
        {"start": 10.0, "text": "The GIL is a mutex."},
    ]

    result = extract_keyword_windows(segments, ["mutex"], window_size=5)

    # Should not raise and should include the last segment's text.
    combined = " ".join(w["text"] for w in result)
    assert "mutex" in combined


def test_extract_keyword_windows_matches_case_insensitively():
    # Keyword "gil" in lowercase should match "GIL" in the segment text.
    result = extract_keyword_windows(_SEGMENTS, ["gil"])

    assert len(result) > 0


def test_extract_keyword_windows_merges_overlapping_windows():
    # Segments 1 and 2 both match; with window_size=1 their windows overlap
    # (indices 0–2 and 1–3) and should merge into a single excerpt.
    segments = [
        {"start": 0.0,  "text": "Introduction."},
        {"start": 5.0,  "text": "The GIL is here."},
        {"start": 10.0, "text": "It is a mutex."},
        {"start": 15.0, "text": "Conclusion."},
    ]

    result = extract_keyword_windows(segments, ["GIL", "mutex"], window_size=1)

    assert len(result) == 1


def test_extract_keyword_windows_keeps_separate_distant_windows():
    # Matches far apart in the transcript should produce separate windows.
    segments = [
        {"start": 0.0,   "text": "The GIL is introduced."},
        {"start": 10.0,  "text": "Unrelated content."},
        {"start": 20.0,  "text": "More unrelated content."},
        {"start": 30.0,  "text": "Still unrelated."},
        {"start": 40.0,  "text": "The mutex is explained."},
    ]

    result = extract_keyword_windows(segments, ["GIL", "mutex"], window_size=0)

    assert len(result) == 2


def test_extract_keyword_windows_respects_configurable_window_size():
    # window_size=0 means only the matching segment itself, no context.
    result_narrow = extract_keyword_windows(_SEGMENTS, ["GIL"], window_size=0)
    result_wide   = extract_keyword_windows(_SEGMENTS, ["GIL"], window_size=2)

    # Wider window should include more text.
    narrow_text = " ".join(w["text"] for w in result_narrow)
    wide_text   = " ".join(w["text"] for w in result_wide)
    assert len(wide_text) > len(narrow_text)


def test_extract_keyword_windows_returns_start_timestamp_of_first_segment_in_window():
    # The "start" field of each window should be the timestamp of its first segment.
    result = extract_keyword_windows(_SEGMENTS, ["GIL"], window_size=1)

    # Segment 1 ("Today we discuss the GIL.") is at 5.0s.
    # window_size=1 pulls in segment 0 (0.0s) as the leading context.
    assert result[0]["start"] == 0.0


def test_extract_keyword_windows_concatenates_text_of_all_segments_in_window():
    # Text from every segment in the merged window should appear in the output.
    result = extract_keyword_windows(_SEGMENTS, ["GIL"], window_size=1)

    combined = " ".join(w["text"] for w in result)
    assert "Welcome to this video" in combined   # segment 0 (context before match)
    assert "Today we discuss the GIL" in combined  # segment 1 (the match)
    assert "mutex" in combined                    # segment 2 (context after match)


# ===========================================================================
# Group P — build_video_excerpts()
# ===========================================================================

_VIDEO_ID_A = "video_aaaaaa"
_VIDEO_ID_B = "video_bbbbbb"

_VIDEO_A = VideoResult(
    video_id=_VIDEO_ID_A,
    title="Understanding the GIL",
    url=f"https://www.youtube.com/watch?v={_VIDEO_ID_A}",
    description=None, view_count=None, duration_seconds=None,
)
_VIDEO_B = VideoResult(
    video_id=_VIDEO_ID_B,
    title="Python Asyncio Tutorial",
    url=f"https://www.youtube.com/watch?v={_VIDEO_ID_B}",
    description=None, view_count=None, duration_seconds=None,
)

_SEGMENTS_A = [
    {"start": 0.0,  "text": "Welcome."},
    {"start": 5.0,  "text": "The GIL protects Python objects."},
    {"start": 10.0, "text": "It prevents true parallelism."},
]
_SEGMENTS_B = [
    {"start": 0.0,  "text": "Asyncio is an event loop."},
    {"start": 5.0,  "text": "It handles concurrency differently."},
]


def test_build_video_excerpts_returns_empty_dict_when_no_videos_match():
    # No video's segments contain "asyncio" so the result should be empty.
    excerpts = build_video_excerpts(
        {_VIDEO_ID_A: _VIDEO_A},
        {_VIDEO_ID_A: _SEGMENTS_A},
        keywords=["asyncio"],
    )

    assert excerpts == {}


def test_build_video_excerpts_skips_videos_with_no_keyword_matches():
    # Video B has no segments matching "GIL", so it must not appear in the output.
    excerpts = build_video_excerpts(
        {_VIDEO_ID_A: _VIDEO_A, _VIDEO_ID_B: _VIDEO_B},
        {_VIDEO_ID_A: _SEGMENTS_A, _VIDEO_ID_B: _SEGMENTS_B},
        keywords=["GIL"],
    )

    assert _VIDEO_ID_B not in excerpts


def test_build_video_excerpts_includes_videos_with_keyword_matches():
    # Video A contains "GIL" so it must appear in the output.
    excerpts = build_video_excerpts(
        {_VIDEO_ID_A: _VIDEO_A},
        {_VIDEO_ID_A: _SEGMENTS_A},
        keywords=["GIL"],
    )

    assert _VIDEO_ID_A in excerpts


def test_build_video_excerpts_includes_title_and_url_from_video_result():
    excerpts = build_video_excerpts(
        {_VIDEO_ID_A: _VIDEO_A},
        {_VIDEO_ID_A: _SEGMENTS_A},
        keywords=["GIL"],
    )

    assert excerpts[_VIDEO_ID_A]["title"] == _VIDEO_A.title
    assert excerpts[_VIDEO_ID_A]["url"] == _VIDEO_A.url


def test_build_video_excerpts_includes_non_empty_excerpts_list():
    excerpts = build_video_excerpts(
        {_VIDEO_ID_A: _VIDEO_A},
        {_VIDEO_ID_A: _SEGMENTS_A},
        keywords=["GIL"],
    )

    assert len(excerpts[_VIDEO_ID_A]["excerpts"]) > 0


def test_build_video_excerpts_passes_window_size_to_keyword_matching():
    # window_size=0 should produce shorter excerpts than window_size=2.
    excerpts_narrow = build_video_excerpts(
        {_VIDEO_ID_A: _VIDEO_A},
        {_VIDEO_ID_A: _SEGMENTS_A},
        keywords=["GIL"],
        window_size=0,
    )
    excerpts_wide = build_video_excerpts(
        {_VIDEO_ID_A: _VIDEO_A},
        {_VIDEO_ID_A: _SEGMENTS_A},
        keywords=["GIL"],
        window_size=2,
    )

    narrow_text = excerpts_narrow[_VIDEO_ID_A]["excerpts"][0]["text"]
    wide_text   = excerpts_wide[_VIDEO_ID_A]["excerpts"][0]["text"]
    assert len(wide_text) > len(narrow_text)


# ===========================================================================
# Group F — run_agent_loop() planning phase
# ===========================================================================

SEARCH_QUERY = "how does Python GIL work"
DEFAULT_MODEL = "claude-3-opus-20240229"

SELECTED_VIDEO_ID = "abc1234defg"
SELECTED_VIDEO_TITLE = "Understanding the GIL"
SELECTED_VIDEO_URL = f"https://www.youtube.com/watch?v={SELECTED_VIDEO_ID}"
SELECTED_TRANSCRIPT = "The GIL is a mutex that protects Python objects."
SYNTHESIS_ANSWER = f"The GIL protects objects. See **[{SELECTED_VIDEO_TITLE}]({SELECTED_VIDEO_URL})**"

SEGMENTS_WITH_TIMESTAMPS = [
    {"start": 0.0,   "text": "Hello everyone."},
    {"start": 5.5,   "text": "Today we talk about the GIL."},
    {"start": 135.0, "text": "The GIL is a mutex."},
]


def _make_client(*responses) -> MagicMock:
    client = MagicMock()
    client.messages.create.side_effect = list(responses)
    return client


def test_run_agent_loop_makes_planning_call_with_plan_search_tool(mocker):
    # First Claude call must offer the plan_search tool so Claude can specify
    # what to search for and what keywords to look for in transcripts.
    mocker.patch("ytask.search_youtube", return_value=[])
    client = _make_client(
        _plan_response(["python GIL"], ["GIL"]),
        _make_synthesis_response("No info found."),
    )

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    first_call_kwargs = client.messages.create.call_args_list[0].kwargs
    tool_names = [t["name"] for t in first_call_kwargs["tools"]]
    assert "plan_search" in tool_names


def test_run_agent_loop_calls_search_youtube_for_each_query_in_plan(mocker):
    # One search_youtube call must be made per query returned by the planning step.
    mock_search = mocker.patch("ytask.search_youtube", return_value=[])
    client = _make_client(
        _plan_response(["python GIL", "GIL threading python"], ["GIL"]),
        _make_synthesis_response("Answer."),
    )

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    assert mock_search.call_count == 2


def test_run_agent_loop_deduplicates_videos_from_multiple_searches(mocker):
    # If the same video appears in two search results it should only be
    # fetched once — deduplication keyed on video_id.
    shared_video = VideoResult(
        video_id=SELECTED_VIDEO_ID, title=SELECTED_VIDEO_TITLE, url=SELECTED_VIDEO_URL,
        description=None, view_count=None, duration_seconds=None,
    )
    mocker.patch("ytask.search_youtube", return_value=[shared_video])
    mock_load = mocker.patch("ytask.load_transcript", return_value=(SELECTED_TRANSCRIPT, False, "auto-generated EN"))
    mocker.patch("ytask.load_transcript_segments", return_value=None)
    client = _make_client(
        _plan_response(["query one", "query two"], ["GIL"]),
        _make_synthesis_response("Answer."),
    )

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    assert mock_load.call_count == 1


def test_run_agent_loop_reports_each_search_query_via_status_callback(mocker):
    # The TUI should show which queries are being executed as they run.
    mocker.patch("ytask.search_youtube", return_value=[])
    status_events: list[tuple[str, str]] = []
    client = _make_client(
        _plan_response(["python GIL", "global interpreter lock"], ["GIL"]),
        _make_synthesis_response("Answer."),
    )

    run_agent_loop(
        SEARCH_QUERY, client, DEFAULT_MODEL,
        status_callback=lambda msg, kind="info", **kw: status_events.append((msg, kind)),
    )

    search_messages = [m for m, _ in status_events if "search" in m.lower()]
    assert len(search_messages) >= 2


def test_run_agent_loop_reports_found_video_title_and_url_via_status_callback(mocker):
    # After each search, title and URL of every new video should be shown.
    search_result = VideoResult(
        video_id=SELECTED_VIDEO_ID, title=SELECTED_VIDEO_TITLE, url=SELECTED_VIDEO_URL,
        description=None, view_count=None, duration_seconds=None,
    )
    mocker.patch("ytask.search_youtube", return_value=[search_result])
    mocker.patch("ytask.load_transcript", return_value=(SELECTED_TRANSCRIPT, False, "auto-generated EN"))
    mocker.patch("ytask.load_transcript_segments", return_value=None)
    status_events: list[tuple[str, str]] = []
    client = _make_client(
        _plan_response(["Python GIL"], ["GIL"]),
        _make_synthesis_response(SYNTHESIS_ANSWER),
    )

    run_agent_loop(
        SEARCH_QUERY, client, DEFAULT_MODEL,
        status_callback=lambda msg, kind="info", **kw: status_events.append((msg, kind)),
    )

    found_messages = [m for m, _ in status_events if SELECTED_VIDEO_TITLE in m]
    assert len(found_messages) >= 1
    assert any(SELECTED_VIDEO_URL in m for m in found_messages)


# ===========================================================================
# Group G — run_agent_loop() synthesis phase
# ===========================================================================

def _setup_synthesis_scenario(mocker):
    """
    Planning returns one search query + keywords; the video's transcript segments
    contain the keywords so excerpts are extracted and sent to synthesis.
    Returns (client, mock_load_transcript).
    """
    search_result = VideoResult(
        video_id=SELECTED_VIDEO_ID, title=SELECTED_VIDEO_TITLE, url=SELECTED_VIDEO_URL,
        description=None, view_count=None, duration_seconds=None,
    )
    mocker.patch("ytask.search_youtube", return_value=[search_result])
    mock_load = mocker.patch(
        "ytask.load_transcript",
        return_value=(SELECTED_TRANSCRIPT, False, "auto-generated EN"),
    )
    # Segments contain "GIL" and "mutex" — both are in PLAN_KEYWORDS.
    mocker.patch("ytask.load_transcript_segments", return_value=SEGMENTS_WITH_TIMESTAMPS)

    client = _make_client(
        _plan_response(["Python GIL"], ["GIL", "mutex"]),
        _make_synthesis_response(SYNTHESIS_ANSWER),
    )
    return client, mock_load


def test_run_agent_loop_fetches_transcripts_for_all_discovered_videos(mocker):
    # Every video found during the search phase should have its transcript fetched.
    client, mock_load = _setup_synthesis_scenario(mocker)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    mock_load.assert_called_once_with(SELECTED_VIDEO_ID, auto=True)


def test_run_agent_loop_includes_keyword_matched_excerpts_in_synthesis_prompt(mocker):
    # Synthesis prompt must contain text from the keyword-matching segments,
    # not the full transcript verbatim.
    client, _ = _setup_synthesis_scenario(mocker)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    synthesis_messages = client.messages.create.call_args_list[-1].kwargs["messages"]
    combined_content = " ".join(
        m["content"] for m in synthesis_messages if isinstance(m.get("content"), str)
    )
    # "Today we talk about the GIL." is in SEGMENTS_WITH_TIMESTAMPS and matches keyword "GIL".
    assert "Today we talk about the GIL" in combined_content


def test_run_agent_loop_returns_text_from_claude_synthesis_response(mocker):
    client, _ = _setup_synthesis_scenario(mocker)

    session = run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    assert session.answer == SYNTHESIS_ANSWER


def test_run_agent_loop_includes_video_title_and_url_in_synthesis_context(mocker):
    client, _ = _setup_synthesis_scenario(mocker)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    synthesis_messages = client.messages.create.call_args_list[-1].kwargs["messages"]
    combined_content = " ".join(
        m["content"] for m in synthesis_messages if isinstance(m.get("content"), str)
    )
    assert SELECTED_VIDEO_TITLE in combined_content
    assert SELECTED_VIDEO_URL in combined_content


def test_run_agent_loop_includes_timestamps_in_synthesis_excerpts(mocker):
    # Excerpt windows carry [M:SS] timestamps so Claude can produce cited answers.
    # window_size=0 keeps each matching segment isolated so the 135s segment
    # ([2:15]) appears as its own excerpt rather than being merged into [0:00].
    client, _ = _setup_synthesis_scenario(mocker)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL, window_size=0)

    synthesis_messages = client.messages.create.call_args_list[-1].kwargs["messages"]
    combined_content = " ".join(
        m["content"] for m in synthesis_messages if isinstance(m.get("content"), str)
    )
    # 135 seconds → [2:15] from SEGMENTS_WITH_TIMESTAMPS
    assert "[2:15]" in combined_content


def test_run_agent_loop_reports_transcript_fetch_per_video_via_status_callback(mocker):
    # The user should see which video's transcript is being fetched.
    client, _ = _setup_synthesis_scenario(mocker)
    status_messages: list[str] = []

    run_agent_loop(
        SEARCH_QUERY, client, DEFAULT_MODEL,
        status_callback=lambda msg, kind="info", **kw: status_messages.append(msg),
    )

    fetch_messages = [m for m in status_messages if SELECTED_VIDEO_TITLE in m]
    assert len(fetch_messages) >= 1


def test_run_agent_loop_uses_success_kind_when_transcript_loads(mocker):
    # A green "Loaded" signal lets the user see which videos contributed.
    client, _ = _setup_synthesis_scenario(mocker)
    status_events: list[tuple[str, str]] = []

    run_agent_loop(
        SEARCH_QUERY, client, DEFAULT_MODEL,
        status_callback=lambda msg, kind="info", **kw: status_events.append((msg, kind)),
    )

    loaded_events = [(m, k) for m, k in status_events if "Loaded" in m and SELECTED_VIDEO_TITLE in m]
    assert len(loaded_events) >= 1


def test_run_agent_loop_uses_skip_kind_when_no_captions_available(mocker):
    # A yellow "skip" signal distinguishes no-captions from fetch errors.
    from youtube_transcript_api import CouldNotRetrieveTranscript

    search_result = VideoResult(
        video_id=SELECTED_VIDEO_ID, title=SELECTED_VIDEO_TITLE, url=SELECTED_VIDEO_URL,
        description=None, view_count=None, duration_seconds=None,
    )
    mocker.patch("ytask.search_youtube", return_value=[search_result])
    mocker.patch("ytask.load_transcript", side_effect=CouldNotRetrieveTranscript(SELECTED_VIDEO_ID))
    status_events: list[tuple[str, str]] = []
    client = _make_client(
        _plan_response(["Python GIL"], ["GIL"]),
        _make_synthesis_response(SYNTHESIS_ANSWER),
    )

    run_agent_loop(
        SEARCH_QUERY, client, DEFAULT_MODEL,
        status_callback=lambda msg, kind="info", **kw: status_events.append((msg, kind)),
    )

    no_caption_events = [m for m, k in status_events if "no captions" in m.lower()]
    assert len(no_caption_events) >= 1


def test_run_agent_loop_reports_no_captions_message_in_status(mocker):
    # Message text should clearly say "no captions" not a generic error.
    from youtube_transcript_api import CouldNotRetrieveTranscript

    search_result = VideoResult(
        video_id=SELECTED_VIDEO_ID, title=SELECTED_VIDEO_TITLE, url=SELECTED_VIDEO_URL,
        description=None, view_count=None, duration_seconds=None,
    )
    mocker.patch("ytask.search_youtube", return_value=[search_result])
    mocker.patch("ytask.load_transcript", side_effect=CouldNotRetrieveTranscript(SELECTED_VIDEO_ID))
    status_events: list[tuple[str, str]] = []
    client = _make_client(
        _plan_response(["Python GIL"], ["GIL"]),
        _make_synthesis_response(SYNTHESIS_ANSWER),
    )

    run_agent_loop(
        SEARCH_QUERY, client, DEFAULT_MODEL,
        status_callback=lambda msg, kind="info", **kw: status_events.append((msg, kind)),
    )

    skip_messages = [m for m, _ in status_events if "no captions" in m.lower()]
    assert len(skip_messages) >= 1


def test_run_agent_loop_skips_videos_with_no_keyword_matches_in_synthesis(mocker):
    # A video whose transcript contains none of the planned keywords should not
    # appear in the synthesis context — it has no relevant content.
    search_result = VideoResult(
        video_id=SELECTED_VIDEO_ID, title=SELECTED_VIDEO_TITLE, url=SELECTED_VIDEO_URL,
        description=None, view_count=None, duration_seconds=None,
    )
    mocker.patch("ytask.search_youtube", return_value=[search_result])
    mocker.patch("ytask.load_transcript", return_value=(SELECTED_TRANSCRIPT, False, "auto-generated EN"))
    # Segments contain no keyword match for "asyncio".
    mocker.patch("ytask.load_transcript_segments", return_value=SEGMENTS_WITH_TIMESTAMPS)
    client = _make_client(
        _plan_response(["Python asyncio"], ["asyncio", "event loop"]),
        _make_synthesis_response(SYNTHESIS_ANSWER),
    )

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    synthesis_messages = client.messages.create.call_args_list[-1].kwargs["messages"]
    combined_content = " ".join(
        m["content"] for m in synthesis_messages if isinstance(m.get("content"), str)
    )
    assert SELECTED_VIDEO_TITLE not in combined_content


# ===========================================================================
# Group M — _create_with_retry() transient error handling
# ===========================================================================

class _FakeAPIError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _make_retrying_client(failing_status_code: int, fail_times: int, success_response):
    """Client whose .messages.create raises an HTTP error `fail_times` times then succeeds."""
    call_count = 0

    def _create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= fail_times:
            raise _FakeAPIError(failing_status_code)
        return success_response

    client = MagicMock()
    client.messages.create.side_effect = _create
    return client


def test_create_with_retry_returns_response_on_first_success(mocker):
    # When the API succeeds immediately, the response is returned without retrying.
    mocker.patch("ytask.time.sleep")
    client = MagicMock()
    expected = MagicMock()
    client.messages.create.return_value = expected

    result = ytask._create_with_retry(client, model="m", messages=[])

    assert result is expected


def test_create_with_retry_retries_on_529_and_eventually_succeeds(mocker):
    # A single 529 (overloaded) should be retried and succeed on the next attempt.
    mock_sleep = mocker.patch("ytask.time.sleep")
    success = MagicMock()
    client = _make_retrying_client(failing_status_code=529, fail_times=1, success_response=success)

    result = ytask._create_with_retry(client, model="m", messages=[])

    assert result is success
    mock_sleep.assert_called_once()


def test_create_with_retry_retries_on_429_and_eventually_succeeds(mocker):
    # 429 (rate limited) is also retryable and should be handled the same way.
    mock_sleep = mocker.patch("ytask.time.sleep")
    success = MagicMock()
    client = _make_retrying_client(failing_status_code=429, fail_times=1, success_response=success)

    result = ytask._create_with_retry(client, model="m", messages=[])

    assert result is success
    mock_sleep.assert_called_once()


def test_create_with_retry_uses_exponential_backoff(mocker):
    # Each retry waits 2^attempt seconds: 1st retry waits 1s, 2nd waits 2s.
    mock_sleep = mocker.patch("ytask.time.sleep")
    success = MagicMock()
    client = _make_retrying_client(failing_status_code=529, fail_times=2, success_response=success)

    ytask._create_with_retry(client, model="m", messages=[])

    sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
    assert sleep_calls == [1, 2]


def test_create_with_retry_raises_after_max_retries_exhausted(mocker):
    # If the API keeps returning 529 beyond the retry limit, the error propagates.
    mocker.patch("ytask.time.sleep")
    client = MagicMock()
    client.messages.create.side_effect = _FakeAPIError(529)

    with pytest.raises(_FakeAPIError):
        ytask._create_with_retry(client, model="m", messages=[])


def test_create_with_retry_does_not_retry_non_transient_errors(mocker):
    # A 400 bad request should not be retried — it is a permanent client error.
    mock_sleep = mocker.patch("ytask.time.sleep")
    client = MagicMock()
    client.messages.create.side_effect = _FakeAPIError(400)

    with pytest.raises(_FakeAPIError):
        ytask._create_with_retry(client, model="m", messages=[])

    mock_sleep.assert_not_called()


def test_create_with_retry_notifies_status_callback_before_each_retry(mocker):
    # Status callback is called so the TUI can show the user a "retrying" message.
    mocker.patch("ytask.time.sleep")
    success = MagicMock()
    client = _make_retrying_client(failing_status_code=529, fail_times=1, success_response=success)
    status_events: list[tuple[str, str]] = []

    ytask._create_with_retry(
        client,
        lambda msg, kind="info", **kw: status_events.append((msg, kind)),
        model="m",
        messages=[],
    )

    assert len(status_events) == 1
    assert "retry" in status_events[0][0].lower() or "overload" in status_events[0][0].lower()


# ===========================================================================
# Group J — main() entry point
# ===========================================================================

def _patch_app(mocker) -> MagicMock:
    """Replace YtsearchApp so we don't spin up a real TUI."""
    mock_app_class = mocker.patch("ytask.YtsearchApp")
    mock_app_instance = MagicMock()
    mock_app_class.return_value = mock_app_instance
    return mock_app_class


def test_main_reads_query_from_command_line_argument_when_provided(mocker):
    mocker.patch.dict("os.environ", {"YTQA_MODEL": "claude-3-opus-20240229"})
    mocker.patch("ytask.anthropic.Anthropic")
    mock_app_class = _patch_app(mocker)

    with patch.object(sys, "argv", ["ytask.py", "python", "GIL"]):
        ytask.main()

    mock_app_class.assert_called_once()
    call_kwargs = mock_app_class.call_args.kwargs
    assert call_kwargs["initial_query"] == "python GIL"


def test_main_prompts_for_query_via_input_when_no_argument_given(mocker):
    mocker.patch.dict("os.environ", {"YTQA_MODEL": "claude-3-opus-20240229"})
    mocker.patch("ytask.anthropic.Anthropic")
    mock_app_class = _patch_app(mocker)
    mocker.patch("builtins.input", return_value="what is asyncio")

    with patch.object(sys, "argv", ["ytask.py"]):
        ytask.main()

    mock_app_class.assert_called_once()
    call_kwargs = mock_app_class.call_args.kwargs
    assert call_kwargs["initial_query"] == "what is asyncio"


# ===========================================================================
# Group L — run_agent_loop() session logging integration
# ===========================================================================

def test_run_agent_loop_records_claude_planning_response_in_session_log(mocker, tmp_path):
    # The planning call result must be captured to review what search strategy
    # and keywords Claude chose — essential for evaluating agent quality.
    mocker.patch("ytask.search_youtube", return_value=[])
    client = _make_client(
        _plan_response(["python GIL"], ["GIL"]),
        _make_synthesis_response("No results."),
    )
    log = SessionLog(query=SEARCH_QUERY, model=DEFAULT_MODEL, logs_dir=tmp_path)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL, session_log=log)

    planning_events = [e for e in log.events if e["type"] == "claude_planning_response"]
    assert len(planning_events) == 1
    assert "search_queries" in planning_events[0]
    assert "transcript_keywords" in planning_events[0]


def test_run_agent_loop_records_youtube_search_query_and_results_in_session_log(mocker, tmp_path):
    # Recording the query and results makes it possible to judge whether
    # Claude chose good search terms.
    search_result = VideoResult(
        video_id=SELECTED_VIDEO_ID, title=SELECTED_VIDEO_TITLE, url=SELECTED_VIDEO_URL,
        description=None, view_count=None, duration_seconds=None,
    )
    mocker.patch("ytask.search_youtube", return_value=[search_result])
    mocker.patch("ytask.load_transcript", return_value=(SELECTED_TRANSCRIPT, False, "auto-generated EN"))
    mocker.patch("ytask.load_transcript_segments", return_value=None)
    client = _make_client(
        _plan_response(["Python GIL"], ["GIL"]),
        _make_synthesis_response(SYNTHESIS_ANSWER),
    )
    log = SessionLog(query=SEARCH_QUERY, model=DEFAULT_MODEL, logs_dir=tmp_path)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL, session_log=log)

    search_events = [e for e in log.events if e["type"] == "youtube_search"]
    assert len(search_events) == 1
    assert search_events[0]["query"] == "Python GIL"
    assert any(r["video_id"] == SELECTED_VIDEO_ID for r in search_events[0]["results"])


def test_run_agent_loop_records_transcript_fetch_success_in_session_log(mocker, tmp_path):
    # Success events show which videos actually contributed to the answer.
    client, _ = _setup_synthesis_scenario(mocker)
    log = SessionLog(query=SEARCH_QUERY, model=DEFAULT_MODEL, logs_dir=tmp_path)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL, session_log=log)

    fetch_events = [e for e in log.events if e["type"] == "transcript_fetch"]
    assert any(e["status"] == "success" for e in fetch_events)
    assert any(e["video_id"] == SELECTED_VIDEO_ID for e in fetch_events)


def test_run_agent_loop_records_transcript_no_captions_in_session_log(mocker, tmp_path):
    # No-captions events explain gaps in the final answer during review.
    from youtube_transcript_api import CouldNotRetrieveTranscript

    search_result = VideoResult(
        video_id=SELECTED_VIDEO_ID, title=SELECTED_VIDEO_TITLE, url=SELECTED_VIDEO_URL,
        description=None, view_count=None, duration_seconds=None,
    )
    mocker.patch("ytask.search_youtube", return_value=[search_result])
    mocker.patch("ytask.load_transcript", side_effect=CouldNotRetrieveTranscript(SELECTED_VIDEO_ID))
    client = _make_client(
        _plan_response(["Python GIL"], ["GIL"]),
        _make_synthesis_response(SYNTHESIS_ANSWER),
    )
    log = SessionLog(query=SEARCH_QUERY, model=DEFAULT_MODEL, logs_dir=tmp_path)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL, session_log=log)

    fetch_events = [e for e in log.events if e["type"] == "transcript_fetch"]
    assert any(e["status"] == "no_captions" for e in fetch_events)


def test_run_agent_loop_records_synthesis_input_with_excerpt_context(mocker, tmp_path):
    # The excerpt context passed to synthesis must be logged so reviewers
    # can judge whether Claude had enough relevant content.
    client, _ = _setup_synthesis_scenario(mocker)
    log = SessionLog(query=SEARCH_QUERY, model=DEFAULT_MODEL, logs_dir=tmp_path)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL, session_log=log)

    synthesis_input_events = [e for e in log.events if e["type"] == "synthesis_input"]
    assert len(synthesis_input_events) == 1
    # Excerpt context should contain text from the keyword-matching segments.
    assert "Today we talk about the GIL" in synthesis_input_events[0]["transcript_context"]


def test_run_agent_loop_records_synthesis_output_answer_in_session_log(mocker, tmp_path):
    # The final answer is stored verbatim so logs are self-contained for
    # offline evaluation without re-running the agent.
    client, _ = _setup_synthesis_scenario(mocker)
    log = SessionLog(query=SEARCH_QUERY, model=DEFAULT_MODEL, logs_dir=tmp_path)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL, session_log=log)

    synthesis_output_events = [e for e in log.events if e["type"] == "synthesis_output"]
    assert len(synthesis_output_events) == 1
    assert synthesis_output_events[0]["answer"] == SYNTHESIS_ANSWER


# ===========================================================================
# Group Q — SearchSession returned by run_agent_loop()
# ===========================================================================

def test_run_agent_loop_returns_search_session_object(mocker):
    # run_agent_loop must return a SearchSession so the TUI can continue
    # the conversation with accumulated context across follow-up turns.
    client, _ = _setup_synthesis_scenario(mocker)

    result = run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    assert isinstance(result, SearchSession)


def test_run_agent_loop_search_session_answer_equals_synthesis_text(mocker):
    client, _ = _setup_synthesis_scenario(mocker)

    session = run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    assert session.answer == SYNTHESIS_ANSWER


def test_run_agent_loop_search_session_contains_transcript_keywords(mocker):
    client, _ = _setup_synthesis_scenario(mocker)

    session = run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    assert "GIL" in session.transcript_keywords


def test_run_agent_loop_search_session_contains_transcript_context(mocker):
    client, _ = _setup_synthesis_scenario(mocker)

    session = run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    assert SELECTED_VIDEO_TITLE in session.transcript_context


def test_run_agent_loop_search_session_contains_video_results(mocker):
    client, _ = _setup_synthesis_scenario(mocker)

    session = run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    assert SELECTED_VIDEO_ID in session.all_video_results


# ===========================================================================
# Group R — run_followup() basic behavior (no new searches)
# ===========================================================================

# Shared fixtures for follow-up tests.
FOLLOWUP_QUESTION = "What about GIL in Python 3.13?"
FOLLOWUP_ANSWER = "Python 3.13 introduces optional GIL disabling."
FOLLOWUP_CONTEXT = (
    f"## {SELECTED_VIDEO_TITLE}\nURL: {SELECTED_VIDEO_URL}\n\n"
    "Relevant excerpts:\n[2:15] The GIL is a mutex."
)
FOLLOWUP_CONVERSATION = [
    {"role": "user", "content": SEARCH_QUERY},
    {"role": "assistant", "content": SYNTHESIS_ANSWER},
]


def _make_followup_session() -> SearchSession:
    return SearchSession(
        answer=SYNTHESIS_ANSWER,
        segments_by_id={SELECTED_VIDEO_ID: SEGMENTS_WITH_TIMESTAMPS},
        all_video_results={
            SELECTED_VIDEO_ID: VideoResult(
                video_id=SELECTED_VIDEO_ID, title=SELECTED_VIDEO_TITLE,
                url=SELECTED_VIDEO_URL, description=None,
                view_count=None, duration_seconds=None,
            )
        },
        transcript_context=FOLLOWUP_CONTEXT,
        transcript_keywords=["GIL", "mutex"],
    )


def test_run_followup_returns_answer_string(mocker):
    # The first element of the return tuple must be the text answer Claude provides.
    client = _make_client(_make_synthesis_response(FOLLOWUP_ANSWER))

    answer, _ = run_followup(
        FOLLOWUP_QUESTION, _make_followup_session(), FOLLOWUP_CONVERSATION,
        client, DEFAULT_MODEL,
    )

    assert answer == FOLLOWUP_ANSWER


def test_run_followup_returns_updated_search_session(mocker):
    # The second element must be a SearchSession carrying accumulated context.
    client = _make_client(_make_synthesis_response(FOLLOWUP_ANSWER))

    _, updated = run_followup(
        FOLLOWUP_QUESTION, _make_followup_session(), FOLLOWUP_CONVERSATION,
        client, DEFAULT_MODEL,
    )

    assert isinstance(updated, SearchSession)


def test_run_followup_sends_conversation_history_to_claude(mocker):
    # The original query and answer must appear in the messages sent to Claude
    # so it has the full conversation context when answering the follow-up.
    client = _make_client(_make_synthesis_response(FOLLOWUP_ANSWER))

    run_followup(
        FOLLOWUP_QUESTION, _make_followup_session(), FOLLOWUP_CONVERSATION,
        client, DEFAULT_MODEL,
    )

    messages = client.messages.create.call_args.kwargs["messages"]
    user_contents = [m["content"] for m in messages if m["role"] == "user" and isinstance(m.get("content"), str)]
    assert any(SEARCH_QUERY in c for c in user_contents)


def test_run_followup_includes_existing_context_in_system_prompt(mocker):
    # Transcript excerpts from prior searches must be in the system prompt
    # so Claude can cite them without re-fetching.
    client = _make_client(_make_synthesis_response(FOLLOWUP_ANSWER))

    run_followup(
        FOLLOWUP_QUESTION, _make_followup_session(), FOLLOWUP_CONVERSATION,
        client, DEFAULT_MODEL,
    )

    system = client.messages.create.call_args.kwargs["system"]
    assert FOLLOWUP_CONTEXT in system


def test_run_followup_provides_youtube_search_tool_to_claude(mocker):
    # Claude must have the youtube_search tool available so it can fetch
    # additional content when the existing excerpts are insufficient.
    client = _make_client(_make_synthesis_response(FOLLOWUP_ANSWER))

    run_followup(
        FOLLOWUP_QUESTION, _make_followup_session(), FOLLOWUP_CONVERSATION,
        client, DEFAULT_MODEL,
    )

    tool_names = [t["name"] for t in client.messages.create.call_args.kwargs["tools"]]
    assert "youtube_search" in tool_names


# ===========================================================================
# Group S — run_followup() when Claude searches YouTube for more content
# ===========================================================================

NEW_VIDEO_ID = "new_vid_zzzzz"
NEW_VIDEO_TITLE = "GIL in Python 3.13 explained"
NEW_VIDEO_URL = f"https://www.youtube.com/watch?v={NEW_VIDEO_ID}"
NEW_VIDEO_TRANSCRIPT = "Python 3.13 allows disabling the GIL."
NEW_VIDEO_SEGMENTS = [{"start": 10.0, "text": "Python 3.13 allows disabling the GIL."}]
SEARCH_TOOL_USE_ID = "tu_search_001"


def _followup_search_response(query: str = "GIL Python 3.13") -> FakeResponse:
    return FakeResponse(
        stop_reason="tool_use",
        content=[FakeToolUseBlock(id=SEARCH_TOOL_USE_ID, name="youtube_search", input={"query": query})],
    )


def _new_video() -> VideoResult:
    return VideoResult(
        video_id=NEW_VIDEO_ID, title=NEW_VIDEO_TITLE, url=NEW_VIDEO_URL,
        description=None, view_count=None, duration_seconds=None,
    )


def test_run_followup_calls_search_youtube_with_claude_query(mocker):
    # When Claude emits a youtube_search tool call, search_youtube must be
    # invoked with the exact query Claude specified.
    mock_search = mocker.patch("ytask.search_youtube", return_value=[_new_video()])
    mocker.patch("ytask.load_transcript", return_value=(NEW_VIDEO_TRANSCRIPT, False, None))
    mocker.patch("ytask.load_transcript_segments", return_value=NEW_VIDEO_SEGMENTS)
    client = _make_client(_followup_search_response("GIL Python 3.13"), _make_synthesis_response(FOLLOWUP_ANSWER))

    run_followup(
        FOLLOWUP_QUESTION, _make_followup_session(), FOLLOWUP_CONVERSATION,
        client, DEFAULT_MODEL,
    )

    mock_search.assert_called_once()
    assert mock_search.call_args[0][0] == "GIL Python 3.13"


def test_run_followup_fetches_transcript_for_newly_found_video(mocker):
    # Transcripts for videos returned by the search must be loaded so their
    # content can be included in the updated context for the next Claude call.
    mocker.patch("ytask.search_youtube", return_value=[_new_video()])
    mock_load = mocker.patch("ytask.load_transcript", return_value=(NEW_VIDEO_TRANSCRIPT, False, None))
    mocker.patch("ytask.load_transcript_segments", return_value=NEW_VIDEO_SEGMENTS)
    client = _make_client(_followup_search_response(), _make_synthesis_response(FOLLOWUP_ANSWER))

    run_followup(
        FOLLOWUP_QUESTION, _make_followup_session(), FOLLOWUP_CONVERSATION,
        client, DEFAULT_MODEL,
    )

    mock_load.assert_called_once_with(NEW_VIDEO_ID, auto=True)


def test_run_followup_adds_new_video_to_updated_session(mocker):
    # The updated session returned must include the newly discovered video
    # so subsequent follow-ups can reference it.
    mocker.patch("ytask.search_youtube", return_value=[_new_video()])
    mocker.patch("ytask.load_transcript", return_value=(NEW_VIDEO_TRANSCRIPT, False, None))
    mocker.patch("ytask.load_transcript_segments", return_value=NEW_VIDEO_SEGMENTS)
    client = _make_client(_followup_search_response(), _make_synthesis_response(FOLLOWUP_ANSWER))

    _, updated = run_followup(
        FOLLOWUP_QUESTION, _make_followup_session(), FOLLOWUP_CONVERSATION,
        client, DEFAULT_MODEL,
    )

    assert NEW_VIDEO_ID in updated.all_video_results
