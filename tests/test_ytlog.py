"""
Tests for ytlog.py.

Group L: SessionLog — covers construction, event recording, file persistence,
and filename conventions. These tests verify the shape of what gets written
so that log files remain useful for offline agent evaluation.
"""

import json
from pathlib import Path

import pytest

from ytlog import SessionLog, serialize_content


# ---------------------------------------------------------------------------
# Helpers — fake SDK-style content blocks
# ---------------------------------------------------------------------------

class _FakeTextBlock:
    type = "text"
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeToolUseBlock:
    type = "tool_use"
    def __init__(self, id: str, name: str, input: dict) -> None:
        self.id = id
        self.name = name
        self.input = input


# ===========================================================================
# Group L — SessionLog
# ===========================================================================

def test_session_log_stores_query_and_model():
    log = SessionLog(query="python GIL", model="claude-opus")

    assert log.query == "python GIL"
    assert log.model == "claude-opus"


def test_session_log_started_at_is_set_on_construction():
    # started_at is an ISO timestamp string, not None.
    log = SessionLog(query="q", model="m")

    assert log.started_at is not None
    assert "T" in log.started_at  # ISO format: YYYY-MM-DDTHH:MM:SS


def test_session_log_record_appends_event_with_correct_type(tmp_path):
    log = SessionLog(query="q", model="m", logs_dir=tmp_path)

    log.record("youtube_search", query="python GIL")

    assert log.events[0]["type"] == "youtube_search"


def test_session_log_record_includes_extra_kwargs_in_event(tmp_path):
    log = SessionLog(query="q", model="m", logs_dir=tmp_path)

    log.record("youtube_search", query="python GIL", result_count=5)

    assert log.events[0]["query"] == "python GIL"
    assert log.events[0]["result_count"] == 5


def test_session_log_record_adds_timestamp_to_event(tmp_path):
    log = SessionLog(query="q", model="m", logs_dir=tmp_path)

    log.record("youtube_search", query="x")

    assert "t" in log.events[0]


def test_session_log_record_persists_event_to_disk_immediately(tmp_path):
    # Events must be written to disk as they are recorded so partial logs
    # survive if the process is killed or crashes mid-run.
    log = SessionLog(query="python GIL", model="claude-opus", logs_dir=tmp_path)

    log.record("youtube_search", query="python GIL")

    # File exists and contains the event without an explicit save() call.
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert len(data["events"]) == 1


def test_session_log_record_updates_existing_file_on_each_call(tmp_path):
    # Each subsequent record() call should overwrite the file with all events
    # accumulated so far, not create a new file.
    log = SessionLog(query="python GIL", model="claude-opus", logs_dir=tmp_path)

    log.record("youtube_search", query="python GIL")
    log.record("transcript_fetch", video_id="abc", title="Video", status="success")

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert len(data["events"]) == 2


def test_session_log_save_writes_valid_json_file(tmp_path):
    log = SessionLog(query="python GIL", model="claude-opus", logs_dir=tmp_path)
    log.record("youtube_search", query="python GIL")

    path = log.save()

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["query"] == "python GIL"
    assert data["model"] == "claude-opus"
    assert len(data["events"]) == 1


def test_session_log_save_creates_logs_dir_if_absent(tmp_path):
    logs_dir = tmp_path / "logs"
    log = SessionLog(query="q", model="m", logs_dir=logs_dir)

    log.save()

    assert logs_dir.is_dir()


def test_session_log_save_filename_contains_query_slug(tmp_path):
    log = SessionLog(query="How does Python GIL work", model="m", logs_dir=tmp_path)

    path = log.save()

    assert "how-does-python-gil-work" in path.name


def test_session_log_save_filename_contains_timestamp(tmp_path):
    log = SessionLog(query="q", model="m", logs_dir=tmp_path)

    path = log.save()

    # Timestamp portion must be in the name (year prefix is enough).
    assert log.started_at[:4] in path.name


def test_session_log_save_returns_path_to_written_file(tmp_path):
    log = SessionLog(query="q", model="m", logs_dir=tmp_path)

    path = log.save()

    assert isinstance(path, Path)
    assert path.parent == tmp_path


# ---------------------------------------------------------------------------
# serialize_content
# ---------------------------------------------------------------------------

def test_serialize_content_converts_text_block_to_dict():
    blocks = [_FakeTextBlock("Hello world")]

    result = serialize_content(blocks)

    assert result == [{"type": "text", "text": "Hello world"}]


def test_serialize_content_converts_tool_use_block_to_dict():
    blocks = [_FakeToolUseBlock(id="tu_1", name="youtube_search", input={"query": "GIL"})]

    result = serialize_content(blocks)

    assert result == [{"type": "tool_use", "id": "tu_1", "name": "youtube_search", "input": {"query": "GIL"}}]


def test_serialize_content_handles_mixed_block_list():
    blocks = [
        _FakeTextBlock("Searching now."),
        _FakeToolUseBlock(id="tu_2", name="youtube_search", input={"query": "GIL"}),
    ]

    result = serialize_content(blocks)

    assert len(result) == 2
    assert result[0]["type"] == "text"
    assert result[1]["type"] == "tool_use"


def test_serialize_content_returns_empty_list_for_empty_input():
    result = serialize_content([])

    assert result == []
