"""Generation history tracking."""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

TZ_SHANGHAI = timezone(timedelta(hours=8))
HISTORY_FILE = Path.home() / ".sgai-lite" / "history.jsonl"


def _get_history_dir() -> Path:
    """Get history directory, create if needed."""
    d = HISTORY_FILE.parent
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_history_file() -> Path:
    """Get history file path, create if needed."""
    path = HISTORY_FILE
    _get_history_dir()
    if not path.exists():
        path.write_text("", encoding="utf-8")
        # Restrict to user only
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return path


def add_entry(
    goal: str,
    language: str,
    model: str,
    code: str,
    output_file: str | None = None,
    temperature: float = 0.3,
) -> int:
    """Add a generation to history. Returns entry index (1-based)."""
    path = _get_history_file()
    entries = list_entries()  # loads all entries to count

    entry = {
        "index": len(entries) + 1,
        "timestamp": datetime.now(TZ_SHANGHAI).isoformat(),
        "goal": goal,
        "language": language,
        "model": model,
        "temperature": temperature,
        "output_file": output_file,
        "lines": code.count("\n") + 1,
        "chars": len(code),
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry["index"]


def list_entries(n: int = 10) -> list[dict[str, Any]]:
    """Read last N entries from history."""
    path = _get_history_file()
    if not path.exists():
        return []

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries[-n:]


def get_entry(index: int) -> dict[str, Any] | None:
    """Get a specific entry by index (1-based)."""
    entries = list_entries(n=index * 2)  # over-fetch to be safe
    for e in entries:
        if e.get("index") == index:
            return e
    return None


def clear_history() -> None:
    """Clear all history."""
    path = _get_history_file()
    path.write_text("", encoding="utf-8")


def format_entries(entries: list[dict[str, Any]]) -> str:
    """Format entries for display."""
    if not entries:
        return "No history yet."

    lines = []
    for e in entries:
        ts = e.get("timestamp", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                ts = dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                pass

        lang = e.get("language", "?")
        model = e.get("model", "?")
        lines_count = e.get("lines", "?")
        goal_preview = e.get("goal", "?")[:60]
        if len(e.get("goal", "")) > 60:
            goal_preview += "..."

        lines.append(
            f"  [{e.get('index')}] {ts} | {lang} | {model} | {lines_count} lines\n"
            f"      {goal_preview}"
        )

    return "\n".join(lines)
