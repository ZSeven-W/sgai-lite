"""Tests for history module."""

import pytest
import json
import tempfile
import os
from pathlib import Path


class TestHistoryModule:
    def test_add_and_list_entry(self, monkeypatch):
        import sgai_lite.history as history

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_history = Path(tmpdir) / "history.jsonl"
            monkeypatch.setattr(history, "HISTORY_FILE", fake_history)

            idx = history.add_entry(
                goal="test goal",
                language="python",
                model="gpt-4o",
                code="print('hello')",
                output_file="hello.py",
                temperature=0.3,
            )
            assert idx == 1

            entries = history.list_entries(n=5)
            assert len(entries) == 1
            assert entries[0]["goal"] == "test goal"
            assert entries[0]["language"] == "python"
            assert entries[0]["model"] == "gpt-4o"

    def test_add_multiple_entries(self, monkeypatch):
        import sgai_lite.history as history

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_history = Path(tmpdir) / "history.jsonl"
            monkeypatch.setattr(history, "HISTORY_FILE", fake_history)

            for i in range(5):
                history.add_entry(
                    goal=f"goal {i}",
                    language="python",
                    model="gpt-4o",
                    code=f"code {i}",
                )

            entries = history.list_entries(n=10)
            assert len(entries) == 5
            assert entries[-1]["goal"] == "goal 4"

    def test_list_n_entries(self, monkeypatch):
        import sgai_lite.history as history

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_history = Path(tmpdir) / "history.jsonl"
            monkeypatch.setattr(history, "HISTORY_FILE", fake_history)

            for i in range(15):
                history.add_entry(
                    goal=f"goal {i}",
                    language="python",
                    model="gpt-4o",
                    code=f"code {i}",
                )

            entries = history.list_entries(n=5)
            assert len(entries) == 5
            # Should be last 5
            assert entries[0]["goal"] == "goal 10"

    def test_get_entry(self, monkeypatch):
        import sgai_lite.history as history

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_history = Path(tmpdir) / "history.jsonl"
            monkeypatch.setattr(history, "HISTORY_FILE", fake_history)

            for i in range(3):
                history.add_entry(
                    goal=f"goal {i}",
                    language="python",
                    model="gpt-4o",
                    code=f"code {i}",
                )

            entry = history.get_entry(2)
            assert entry is not None
            assert entry["goal"] == "goal 1"

    def test_get_nonexistent_entry(self, monkeypatch):
        import sgai_lite.history as history

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_history = Path(tmpdir) / "history.jsonl"
            monkeypatch.setattr(history, "HISTORY_FILE", fake_history)

            entry = history.get_entry(99)
            assert entry is None

    def test_clear_history(self, monkeypatch):
        import sgai_lite.history as history

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_history = Path(tmpdir) / "history.jsonl"
            monkeypatch.setattr(history, "HISTORY_FILE", fake_history)

            history.add_entry("goal", "python", "gpt-4o", "code")
            history.clear_history()
            entries = history.list_entries()
            assert len(entries) == 0

    def test_format_entries(self, monkeypatch):
        import sgai_lite.history as history

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_history = Path(tmpdir) / "history.jsonl"
            monkeypatch.setattr(history, "HISTORY_FILE", fake_history)

            history.add_entry(
                goal="short goal",
                language="python",
                model="gpt-4o",
                code="x = 1",
            )

            entries = history.list_entries()
            formatted = history.format_entries(entries)
            assert "python" in formatted
            assert "short goal" in formatted

    def test_format_entries_empty(self):
        from sgai_lite.history import format_entries
        result = format_entries([])
        assert "No history" in result
