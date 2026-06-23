"""
Tests for ytsearch.py.

Group A  — VideoResult dataclass
Group B  — search_youtube() normal results
Group C  — search_youtube() error handling
Group D  — build_search_tool_definition()
Group E  — fetch_transcripts_for_videos()
Group F  — run_agent_loop() search phase
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

import ytsearch
from ytsearch import (
    VideoResult,
    SearchError,
    build_search_tool_definition,
    fetch_transcripts_for_videos,
    run_agent_loop,
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


def _end_turn_response(text: str) -> FakeResponse:
    return FakeResponse(stop_reason="end_turn", content=[FakeTextBlock(text=text)])


def _tool_use_response(tool_id: str, tool_input: dict) -> FakeResponse:
    return FakeResponse(
        stop_reason="tool_use",
        content=[FakeToolUseBlock(id=tool_id, name="youtube_search", input=tool_input)],
    )


def _make_synthesis_response(answer_text: str) -> FakeResponse:
    return FakeResponse(stop_reason="end_turn", content=[FakeTextBlock(text=answer_text)])


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
    mock_class = mocker.patch("ytsearch.yt_dlp.YoutubeDL")
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

    mock_class = mocker.patch("ytsearch.yt_dlp.YoutubeDL")
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
    mock_class = mocker.patch("ytsearch.yt_dlp.YoutubeDL")
    mock_class.return_value.__enter__.return_value = mock_instance
    mock_class.return_value.__exit__.return_value = False

    results = search_youtube("query")

    assert results == []


# ===========================================================================
# Group D — build_search_tool_definition()
# ===========================================================================

def test_build_search_tool_definition_returns_dict_with_name_youtube_search():
    tool = build_search_tool_definition()

    assert tool["name"] == "youtube_search"


def test_build_search_tool_definition_includes_query_parameter_in_input_schema():
    tool = build_search_tool_definition()

    assert "query" in tool["input_schema"]["properties"]


def test_build_search_tool_definition_marks_query_parameter_as_required():
    tool = build_search_tool_definition()

    assert "query" in tool["input_schema"]["required"]


def test_build_search_tool_definition_includes_optional_max_results_parameter():
    tool = build_search_tool_definition()

    assert "max_results" in tool["input_schema"]["properties"]


def test_build_search_tool_definition_specifies_max_results_as_integer_type():
    tool = build_search_tool_definition()

    assert tool["input_schema"]["properties"]["max_results"]["type"] == "integer"


# ===========================================================================
# Group E — fetch_transcripts_for_videos()
# ===========================================================================

TRANSCRIPT_VIDEO_ID = "vid1111aaaaa"
TRANSCRIPT_TEXT = "Hello world transcript."


def test_fetch_transcripts_for_videos_returns_dict_keyed_by_video_id(mocker):
    mocker.patch("ytsearch.load_transcript", return_value=(TRANSCRIPT_TEXT, True))

    result = fetch_transcripts_for_videos([TRANSCRIPT_VIDEO_ID])

    assert TRANSCRIPT_VIDEO_ID in result


def test_fetch_transcripts_for_videos_calls_load_transcript_once_per_video_id(mocker):
    mock_load = mocker.patch("ytsearch.load_transcript", return_value=(TRANSCRIPT_TEXT, True))

    fetch_transcripts_for_videos(["vid_a1bbbbbbb", "vid_b2ccccccc"])

    assert mock_load.call_count == 2


def test_fetch_transcripts_for_videos_skips_videos_where_transcript_fetch_fails(mocker):
    # The first video fails; the second succeeds. Only the second should appear.
    def load_side_effect(video_id):
        if video_id == "bad_vid_aaaa":
            raise RuntimeError("transcript unavailable")
        return (TRANSCRIPT_TEXT, False)

    mocker.patch("ytsearch.load_transcript", side_effect=load_side_effect)

    result = fetch_transcripts_for_videos(["bad_vid_aaaa", "good_vid_aaa"])

    assert "bad_vid_aaaa" not in result
    assert "good_vid_aaa" in result


def test_fetch_transcripts_for_videos_returns_empty_dict_for_empty_input(mocker):
    mock_load = mocker.patch("ytsearch.load_transcript")

    result = fetch_transcripts_for_videos([])

    mock_load.assert_not_called()
    assert result == {}


# ===========================================================================
# Group F — run_agent_loop() search phase
# ===========================================================================

SEARCH_QUERY = "how does Python GIL work"
TOOL_USE_ID = "tu_abc123"
DEFAULT_MODEL = "claude-3-opus-20240229"


def _make_client(*responses) -> MagicMock:
    client = MagicMock()
    client.messages.create.side_effect = list(responses)
    return client


def test_run_agent_loop_sends_youtube_search_tool_definition_to_claude_on_first_call(mocker):
    # Claude returns end_turn immediately with no videos selected.
    mocker.patch("ytsearch.search_youtube", return_value=[])
    client = _make_client(
        _end_turn_response('<selected_videos>[]</selected_videos>'),
        _make_synthesis_response("No results found."),
    )

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    first_call_kwargs = client.messages.create.call_args_list[0].kwargs
    tool_names = [t["name"] for t in first_call_kwargs["tools"]]
    assert "youtube_search" in tool_names


def test_run_agent_loop_calls_search_youtube_when_claude_returns_tool_use_block(mocker):
    mock_search = mocker.patch("ytsearch.search_youtube", return_value=[])
    client = _make_client(
        _tool_use_response(TOOL_USE_ID, {"query": "Python GIL"}),
        _end_turn_response('<selected_videos>[]</selected_videos>'),
        _make_synthesis_response("Answer here."),
    )

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    mock_search.assert_called_once_with("Python GIL", 5)


def test_run_agent_loop_sends_tool_result_back_to_claude_after_search(mocker):
    mocker.patch("ytsearch.search_youtube", return_value=[])
    client = _make_client(
        _tool_use_response(TOOL_USE_ID, {"query": "Python GIL"}),
        _end_turn_response('<selected_videos>[]</selected_videos>'),
        _make_synthesis_response("Answer."),
    )

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    # Second call to messages.create should include the tool_result in messages.
    second_call_messages = client.messages.create.call_args_list[1].kwargs["messages"]
    tool_result_messages = [
        m for m in second_call_messages
        if isinstance(m.get("content"), list)
        and any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in m["content"]
        )
    ]
    assert len(tool_result_messages) == 1


def test_run_agent_loop_loops_and_calls_search_youtube_again_on_second_tool_use(mocker):
    mock_search = mocker.patch("ytsearch.search_youtube", return_value=[])
    client = _make_client(
        _tool_use_response(TOOL_USE_ID, {"query": "Python GIL"}),
        _tool_use_response("tu_second_call", {"query": "Python GIL explained"}),
        _end_turn_response('<selected_videos>[]</selected_videos>'),
        _make_synthesis_response("Answer."),
    )

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    assert mock_search.call_count == 2


def test_run_agent_loop_stops_search_loop_when_claude_returns_end_turn_with_no_tool_use(mocker):
    mock_search = mocker.patch("ytsearch.search_youtube", return_value=[])
    client = _make_client(
        _end_turn_response('<selected_videos>[]</selected_videos>'),
        _make_synthesis_response("Nothing found."),
    )

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    mock_search.assert_not_called()


# ===========================================================================
# Group G — run_agent_loop() synthesis phase
# ===========================================================================

SELECTED_VIDEO_ID = "abc1234defg"
SELECTED_VIDEO_TITLE = "Understanding the GIL"
SELECTED_VIDEO_URL = f"https://www.youtube.com/watch?v={SELECTED_VIDEO_ID}"
SELECTED_TRANSCRIPT = "The GIL is a mutex that protects Python objects."
SYNTHESIS_ANSWER = f"The GIL protects objects. See **[{SELECTED_VIDEO_TITLE}]({SELECTED_VIDEO_URL})**"


SEGMENTS_WITH_TIMESTAMPS = [
    {"start": 0.0, "text": "Hello everyone."},
    {"start": 5.5, "text": "Today we talk about the GIL."},
    {"start": 135.0, "text": "The GIL is a mutex."},
]


def _setup_synthesis_scenario(mocker):
    """
    Claude's search phase names one video; that transcript is fetched and given
    to the synthesis call. Returns (client, mock_load_transcript).
    """
    search_result = VideoResult(
        video_id=SELECTED_VIDEO_ID,
        title=SELECTED_VIDEO_TITLE,
        url=SELECTED_VIDEO_URL,
        description=None,
        view_count=None,
        duration_seconds=None,
    )
    mocker.patch("ytsearch.search_youtube", return_value=[search_result])

    mock_load = mocker.patch(
        "ytsearch.load_transcript",
        return_value=(SELECTED_TRANSCRIPT, False),
    )
    # No JSON sidecar — fall back to plain-text transcript in synthesis.
    mocker.patch("ytsearch.load_transcript_segments", return_value=None)

    end_turn_text = (
        f'I found a great video. '
        f'<selected_videos>["{SELECTED_VIDEO_ID}"]</selected_videos>'
    )
    client = _make_client(
        _tool_use_response(TOOL_USE_ID, {"query": "Python GIL"}),
        _end_turn_response(end_turn_text),
        _make_synthesis_response(SYNTHESIS_ANSWER),
    )
    return client, mock_load


def test_run_agent_loop_fetches_transcripts_for_video_ids_mentioned_by_claude(mocker):
    client, mock_load = _setup_synthesis_scenario(mocker)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    mock_load.assert_called_once_with(SELECTED_VIDEO_ID)


def test_run_agent_loop_includes_transcript_content_in_synthesis_prompt(mocker):
    client, _mock_fetch = _setup_synthesis_scenario(mocker)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    synthesis_call_messages = client.messages.create.call_args_list[-1].kwargs["messages"]
    combined_content = " ".join(
        m["content"] for m in synthesis_call_messages if isinstance(m.get("content"), str)
    )
    assert SELECTED_TRANSCRIPT in combined_content


def test_run_agent_loop_returns_text_from_claude_synthesis_response(mocker):
    client, _mock_fetch = _setup_synthesis_scenario(mocker)

    result = run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    assert result == SYNTHESIS_ANSWER


def test_run_agent_loop_includes_video_title_and_url_in_synthesis_context(mocker):
    client, _mock_load = _setup_synthesis_scenario(mocker)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    synthesis_call_messages = client.messages.create.call_args_list[-1].kwargs["messages"]
    combined_content = " ".join(
        m["content"] for m in synthesis_call_messages if isinstance(m.get("content"), str)
    )
    assert SELECTED_VIDEO_TITLE in combined_content
    assert SELECTED_VIDEO_URL in combined_content


def test_run_agent_loop_reports_found_video_title_and_url_via_status_callback(mocker):
    # After each search, the title and URL of every found video should be
    # reported so the user can see what the agent discovered.
    search_result = VideoResult(
        video_id=SELECTED_VIDEO_ID,
        title=SELECTED_VIDEO_TITLE,
        url=SELECTED_VIDEO_URL,
        description=None,
        view_count=None,
        duration_seconds=None,
    )
    mocker.patch("ytsearch.search_youtube", return_value=[search_result])
    mocker.patch("ytsearch.load_transcript", return_value=(SELECTED_TRANSCRIPT, False))
    mocker.patch("ytsearch.load_transcript_segments", return_value=None)

    status_events: list[tuple[str, str]] = []
    client = _make_client(
        _tool_use_response(TOOL_USE_ID, {"query": "Python GIL"}),
        _end_turn_response(f'<selected_videos>["{SELECTED_VIDEO_ID}"]</selected_videos>'),
        _make_synthesis_response(SYNTHESIS_ANSWER),
    )

    run_agent_loop(
        SEARCH_QUERY, client, DEFAULT_MODEL,
        status_callback=lambda msg, kind="info": status_events.append((msg, kind)),
    )

    found_messages = [m for m, _k in status_events if SELECTED_VIDEO_TITLE in m]
    assert len(found_messages) >= 1
    assert any(SELECTED_VIDEO_URL in m for m in found_messages)


def test_run_agent_loop_reports_transcript_fetch_per_video_via_status_callback(mocker):
    # The user should see which video's transcript is being fetched before the
    # network call happens, so long waits feel informative.
    search_result = VideoResult(
        video_id=SELECTED_VIDEO_ID,
        title=SELECTED_VIDEO_TITLE,
        url=SELECTED_VIDEO_URL,
        description=None,
        view_count=None,
        duration_seconds=None,
    )
    mocker.patch("ytsearch.search_youtube", return_value=[search_result])
    mocker.patch("ytsearch.load_transcript", return_value=(SELECTED_TRANSCRIPT, False))
    mocker.patch("ytsearch.load_transcript_segments", return_value=None)

    status_messages: list[str] = []
    client = _make_client(
        _tool_use_response(TOOL_USE_ID, {"query": "Python GIL"}),
        _end_turn_response(f'<selected_videos>["{SELECTED_VIDEO_ID}"]</selected_videos>'),
        _make_synthesis_response(SYNTHESIS_ANSWER),
    )

    run_agent_loop(
        SEARCH_QUERY, client, DEFAULT_MODEL,
        status_callback=lambda msg, kind="info": status_messages.append(msg),
    )

    fetch_messages = [m for m in status_messages if "transcript" in m.lower() and SELECTED_VIDEO_TITLE in m]
    assert len(fetch_messages) >= 1


def test_run_agent_loop_uses_success_kind_when_transcript_loads(mocker):
    # A green "Loaded" signal lets the user see at a glance which videos
    # contributed to the final answer.
    client, _ = _setup_synthesis_scenario(mocker)

    status_events: list[tuple[str, str]] = []
    run_agent_loop(
        SEARCH_QUERY, client, DEFAULT_MODEL,
        status_callback=lambda msg, kind="info": status_events.append((msg, kind)),
    )

    success_events = [(m, k) for m, k in status_events if k == "success"]
    assert len(success_events) >= 1
    assert any(SELECTED_VIDEO_TITLE in m for m, _ in success_events)


def test_run_agent_loop_uses_skip_kind_when_no_captions_available(mocker):
    # A visually distinct (yellow) "skip" signal lets the user know why a
    # video that was found is absent from the answer.
    from youtube_transcript_api import CouldNotRetrieveTranscript

    search_result = VideoResult(
        video_id=SELECTED_VIDEO_ID, title=SELECTED_VIDEO_TITLE, url=SELECTED_VIDEO_URL,
        description=None, view_count=None, duration_seconds=None,
    )
    mocker.patch("ytsearch.search_youtube", return_value=[search_result])
    mocker.patch("ytsearch.load_transcript", side_effect=CouldNotRetrieveTranscript(SELECTED_VIDEO_ID))

    status_events: list[tuple[str, str]] = []
    client = _make_client(
        _tool_use_response(TOOL_USE_ID, {"query": "Python GIL"}),
        _end_turn_response(f'<selected_videos>["{SELECTED_VIDEO_ID}"]</selected_videos>'),
        _make_synthesis_response(SYNTHESIS_ANSWER),
    )

    run_agent_loop(
        SEARCH_QUERY, client, DEFAULT_MODEL,
        status_callback=lambda msg, kind="info": status_events.append((msg, kind)),
    )

    skip_events = [(m, k) for m, k in status_events if k == "skip"]
    assert len(skip_events) >= 1


def test_run_agent_loop_falls_back_to_other_discovered_videos_when_selected_transcript_unavailable(mocker):
    # When a selected video has no captions, the agent tries other videos
    # that were found during searching rather than giving up.
    from youtube_transcript_api import CouldNotRetrieveTranscript

    FALLBACK_VIDEO_ID = "fallback_vid_aa"
    selected = VideoResult(
        video_id=SELECTED_VIDEO_ID, title=SELECTED_VIDEO_TITLE, url=SELECTED_VIDEO_URL,
        description=None, view_count=None, duration_seconds=None,
    )
    fallback = VideoResult(
        video_id=FALLBACK_VIDEO_ID, title="Fallback Video",
        url=f"https://www.youtube.com/watch?v={FALLBACK_VIDEO_ID}",
        description=None, view_count=None, duration_seconds=None,
    )
    # Both appear in search results; Claude selects only the first.
    mocker.patch("ytsearch.search_youtube", return_value=[selected, fallback])

    def load_side_effect(video_id):
        if video_id == SELECTED_VIDEO_ID:
            raise CouldNotRetrieveTranscript(SELECTED_VIDEO_ID)
        return (SELECTED_TRANSCRIPT, False)

    mock_load = mocker.patch("ytsearch.load_transcript", side_effect=load_side_effect)
    mocker.patch("ytsearch.load_transcript_segments", return_value=None)

    client = _make_client(
        _tool_use_response(TOOL_USE_ID, {"query": "Python GIL"}),
        _end_turn_response(f'<selected_videos>["{SELECTED_VIDEO_ID}"]</selected_videos>'),
        _make_synthesis_response(SYNTHESIS_ANSWER),
    )

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    mock_load.assert_any_call(FALLBACK_VIDEO_ID)


def test_run_agent_loop_reports_no_captions_when_transcript_unavailable(mocker):
    # When a video has no captions the user should see "no captions available"
    # rather than a generic error, because it is a normal YouTube condition.
    from youtube_transcript_api import CouldNotRetrieveTranscript

    search_result = VideoResult(
        video_id=SELECTED_VIDEO_ID,
        title=SELECTED_VIDEO_TITLE,
        url=SELECTED_VIDEO_URL,
        description=None,
        view_count=None,
        duration_seconds=None,
    )
    mocker.patch("ytsearch.search_youtube", return_value=[search_result])
    mocker.patch(
        "ytsearch.load_transcript",
        side_effect=CouldNotRetrieveTranscript(SELECTED_VIDEO_ID),
    )

    status_events: list[tuple[str, str]] = []
    client = _make_client(
        _tool_use_response(TOOL_USE_ID, {"query": "Python GIL"}),
        _end_turn_response(f'<selected_videos>["{SELECTED_VIDEO_ID}"]</selected_videos>'),
        _make_synthesis_response(SYNTHESIS_ANSWER),
    )

    run_agent_loop(
        SEARCH_QUERY, client, DEFAULT_MODEL,
        status_callback=lambda msg, kind="info": status_events.append((msg, kind)),
    )

    skip_messages = [m for m, _k in status_events if "no captions" in m.lower()]
    assert len(skip_messages) == 1


def test_run_agent_loop_includes_timestamps_in_synthesis_when_segments_available(mocker):
    # When a JSON sidecar with timestamps exists, the synthesis prompt should
    # contain [M:SS] markers so Claude can cite specific moments in the video.
    search_result = VideoResult(
        video_id=SELECTED_VIDEO_ID,
        title=SELECTED_VIDEO_TITLE,
        url=SELECTED_VIDEO_URL,
        description=None,
        view_count=None,
        duration_seconds=None,
    )
    mocker.patch("ytsearch.search_youtube", return_value=[search_result])
    mocker.patch("ytsearch.load_transcript", return_value=(SELECTED_TRANSCRIPT, False))
    mocker.patch("ytsearch.load_transcript_segments", return_value=SEGMENTS_WITH_TIMESTAMPS)

    client = _make_client(
        _tool_use_response(TOOL_USE_ID, {"query": "Python GIL"}),
        _end_turn_response(f'<selected_videos>["{SELECTED_VIDEO_ID}"]</selected_videos>'),
        _make_synthesis_response(SYNTHESIS_ANSWER),
    )

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL)

    synthesis_messages = client.messages.create.call_args_list[-1].kwargs["messages"]
    combined_content = " ".join(
        m["content"] for m in synthesis_messages if isinstance(m.get("content"), str)
    )
    # 135 seconds → [2:15] from SEGMENTS_WITH_TIMESTAMPS
    assert "[2:15]" in combined_content


# ===========================================================================
# Group M — _create_with_retry() transient error handling
# ===========================================================================

# A fake exception that mimics the shape of Anthropic SDK HTTP errors,
# which expose the HTTP status code via a `.status_code` attribute.

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
    mocker.patch("ytsearch.time.sleep")
    client = MagicMock()
    expected = MagicMock()
    client.messages.create.return_value = expected

    result = ytsearch._create_with_retry(client, model="m", messages=[])

    assert result is expected


def test_create_with_retry_retries_on_529_and_eventually_succeeds(mocker):
    # A single 529 (overloaded) should be retried and succeed on the next attempt.
    mock_sleep = mocker.patch("ytsearch.time.sleep")
    success = MagicMock()
    client = _make_retrying_client(failing_status_code=529, fail_times=1, success_response=success)

    result = ytsearch._create_with_retry(client, model="m", messages=[])

    assert result is success
    mock_sleep.assert_called_once()


def test_create_with_retry_retries_on_429_and_eventually_succeeds(mocker):
    # 429 (rate limited) is also retryable and should be handled the same way.
    mock_sleep = mocker.patch("ytsearch.time.sleep")
    success = MagicMock()
    client = _make_retrying_client(failing_status_code=429, fail_times=1, success_response=success)

    result = ytsearch._create_with_retry(client, model="m", messages=[])

    assert result is success
    mock_sleep.assert_called_once()


def test_create_with_retry_uses_exponential_backoff(mocker):
    # Each retry waits 2^attempt seconds: 1st retry waits 1s, 2nd waits 2s.
    mock_sleep = mocker.patch("ytsearch.time.sleep")
    success = MagicMock()
    client = _make_retrying_client(failing_status_code=529, fail_times=2, success_response=success)

    ytsearch._create_with_retry(client, model="m", messages=[])

    sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
    assert sleep_calls == [1, 2]


def test_create_with_retry_raises_after_max_retries_exhausted(mocker):
    # If the API keeps returning 529 beyond the retry limit, the error propagates.
    mocker.patch("ytsearch.time.sleep")
    client = MagicMock()
    client.messages.create.side_effect = _FakeAPIError(529)

    with pytest.raises(_FakeAPIError):
        ytsearch._create_with_retry(client, model="m", messages=[])


def test_create_with_retry_does_not_retry_non_transient_errors(mocker):
    # A 400 bad request should not be retried — it is a permanent client error.
    mock_sleep = mocker.patch("ytsearch.time.sleep")
    client = MagicMock()
    client.messages.create.side_effect = _FakeAPIError(400)

    with pytest.raises(_FakeAPIError):
        ytsearch._create_with_retry(client, model="m", messages=[])

    mock_sleep.assert_not_called()


def test_create_with_retry_notifies_status_callback_before_each_retry(mocker):
    # Status callback is called so the TUI can show the user a "retrying" message.
    mocker.patch("ytsearch.time.sleep")
    success = MagicMock()
    client = _make_retrying_client(failing_status_code=529, fail_times=1, success_response=success)
    status_events: list[tuple[str, str]] = []

    ytsearch._create_with_retry(
        client,
        lambda msg, kind="info": status_events.append((msg, kind)),
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
    mock_app_class = mocker.patch("ytsearch.YtsearchApp")
    mock_app_instance = MagicMock()
    mock_app_class.return_value = mock_app_instance
    return mock_app_class


def test_main_reads_query_from_command_line_argument_when_provided(mocker):
    mocker.patch.dict("os.environ", {"YTQA_MODEL": "claude-3-opus-20240229"})
    mocker.patch("ytsearch.anthropic.Anthropic")
    mock_app_class = _patch_app(mocker)

    with patch.object(sys, "argv", ["ytsearch.py", "python", "GIL"]):
        ytsearch.main()

    mock_app_class.assert_called_once()
    call_kwargs = mock_app_class.call_args.kwargs
    assert call_kwargs["initial_query"] == "python GIL"


def test_main_prompts_for_query_via_input_when_no_argument_given(mocker):
    mocker.patch.dict("os.environ", {"YTQA_MODEL": "claude-3-opus-20240229"})
    mocker.patch("ytsearch.anthropic.Anthropic")
    mock_app_class = _patch_app(mocker)
    mocker.patch("builtins.input", return_value="what is asyncio")

    with patch.object(sys, "argv", ["ytsearch.py"]):
        ytsearch.main()

    mock_app_class.assert_called_once()
    call_kwargs = mock_app_class.call_args.kwargs
    assert call_kwargs["initial_query"] == "what is asyncio"


# ===========================================================================
# Group L — run_agent_loop() session logging integration
# ===========================================================================

def test_run_agent_loop_records_claude_search_response_in_session_log(mocker):
    # Every Claude response in the search phase must be captured so we can
    # see Claude's reasoning (text blocks) and tool calls when reviewing logs.
    mocker.patch("ytsearch.search_youtube", return_value=[])
    client = _make_client(
        _end_turn_response('<selected_videos>[]</selected_videos>'),
        _make_synthesis_response("No results."),
    )
    log = SessionLog(query=SEARCH_QUERY, model=DEFAULT_MODEL)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL, session_log=log)

    response_events = [e for e in log.events if e["type"] == "claude_search_response"]
    assert len(response_events) >= 1
    assert "stop_reason" in response_events[0]
    assert "content" in response_events[0]


def test_run_agent_loop_records_youtube_search_query_and_results_in_session_log(mocker):
    # Recording the query and results makes it possible to judge whether
    # Claude chose good search terms.
    search_result = VideoResult(
        video_id=SELECTED_VIDEO_ID, title=SELECTED_VIDEO_TITLE, url=SELECTED_VIDEO_URL,
        description=None, view_count=None, duration_seconds=None,
    )
    mocker.patch("ytsearch.search_youtube", return_value=[search_result])
    mocker.patch("ytsearch.load_transcript", return_value=(SELECTED_TRANSCRIPT, False))
    mocker.patch("ytsearch.load_transcript_segments", return_value=None)
    client = _make_client(
        _tool_use_response(TOOL_USE_ID, {"query": "Python GIL"}),
        _end_turn_response(f'<selected_videos>["{SELECTED_VIDEO_ID}"]</selected_videos>'),
        _make_synthesis_response(SYNTHESIS_ANSWER),
    )
    log = SessionLog(query=SEARCH_QUERY, model=DEFAULT_MODEL)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL, session_log=log)

    search_events = [e for e in log.events if e["type"] == "youtube_search"]
    assert len(search_events) == 1
    assert search_events[0]["query"] == "Python GIL"
    assert any(r["video_id"] == SELECTED_VIDEO_ID for r in search_events[0]["results"])


def test_run_agent_loop_records_transcript_fetch_success_in_session_log(mocker):
    # Success events let us see which videos actually contributed to the answer.
    client, _ = _setup_synthesis_scenario(mocker)
    log = SessionLog(query=SEARCH_QUERY, model=DEFAULT_MODEL)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL, session_log=log)

    fetch_events = [e for e in log.events if e["type"] == "transcript_fetch"]
    assert any(e["status"] == "success" for e in fetch_events)
    assert any(e["video_id"] == SELECTED_VIDEO_ID for e in fetch_events)


def test_run_agent_loop_records_transcript_no_captions_in_session_log(mocker):
    # No-captions events explain gaps in the final answer during review.
    from youtube_transcript_api import CouldNotRetrieveTranscript

    search_result = VideoResult(
        video_id=SELECTED_VIDEO_ID, title=SELECTED_VIDEO_TITLE, url=SELECTED_VIDEO_URL,
        description=None, view_count=None, duration_seconds=None,
    )
    mocker.patch("ytsearch.search_youtube", return_value=[search_result])
    mocker.patch("ytsearch.load_transcript", side_effect=CouldNotRetrieveTranscript(SELECTED_VIDEO_ID))
    client = _make_client(
        _tool_use_response(TOOL_USE_ID, {"query": "Python GIL"}),
        _end_turn_response(f'<selected_videos>["{SELECTED_VIDEO_ID}"]</selected_videos>'),
        _make_synthesis_response(SYNTHESIS_ANSWER),
    )
    log = SessionLog(query=SEARCH_QUERY, model=DEFAULT_MODEL)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL, session_log=log)

    fetch_events = [e for e in log.events if e["type"] == "transcript_fetch"]
    assert any(e["status"] == "no_captions" for e in fetch_events)


def test_run_agent_loop_records_synthesis_input_with_transcript_context(mocker):
    # The full transcript context passed to the synthesis call must be logged
    # so reviewers can judge whether Claude had enough information.
    client, _ = _setup_synthesis_scenario(mocker)
    log = SessionLog(query=SEARCH_QUERY, model=DEFAULT_MODEL)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL, session_log=log)

    synthesis_input_events = [e for e in log.events if e["type"] == "synthesis_input"]
    assert len(synthesis_input_events) == 1
    assert SELECTED_TRANSCRIPT in synthesis_input_events[0]["transcript_context"]


def test_run_agent_loop_records_synthesis_output_answer_in_session_log(mocker):
    # The final answer is stored verbatim so log files are self-contained
    # for offline evaluation without re-running the agent.
    client, _ = _setup_synthesis_scenario(mocker)
    log = SessionLog(query=SEARCH_QUERY, model=DEFAULT_MODEL)

    run_agent_loop(SEARCH_QUERY, client, DEFAULT_MODEL, session_log=log)

    synthesis_output_events = [e for e in log.events if e["type"] == "synthesis_output"]
    assert len(synthesis_output_events) == 1
    assert synthesis_output_events[0]["answer"] == SYNTHESIS_ANSWER
