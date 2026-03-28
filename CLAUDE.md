# CLAUDE.md — Sgai-Lite Developer Guide

## Project Overview

**sgai-lite** is a goal-driven single-file code generator powered by OpenAI. Give it a natural language goal, and it generates a complete, production-ready code file.

## Architecture

```
sgai-lite/
├── src/sgai_lite/
│   ├── __init__.py        — Package init, version
│   ├── __main__.py        — Entry point: python -m sgai_lite
│   ├── cli.py             — CLI argument parsing & user interaction
│   ├── generator.py       — OpenAI API calls, streaming, validation
│   ├── history.py         — JSONL-based generation history
│   ├── config.py          — Config file loading (JSON/YAML)
│   ├── languages.py       — Language detection & mapping
│   └── prompts.py         — Intent detection & system prompts
├── tests/                 — pytest test suite (~74 tests)
├── pyproject.toml         — Project metadata & dependencies
└── README.md
```

## Key Design Decisions

### Intent Detection
`prompts.py` scans the goal for keywords to select language-specific best-practice tips:
- **cli**: argparse, click, menu patterns
- **web**: Flask, FastAPI, Express, HTTP servers
- **data**: pandas, CSV, JSON processing
- **gui**: tkinter, PyQt, Electron widgets
- **script**: automation, cron
- **game**: arcade, score mechanics
- **test**: pytest fixtures, unittest

### Streaming
`generate_code_stream()` in `generator.py` uses OpenAI's streaming API. Each chunk is yielded in real-time so the user sees code as it's generated. The buffer accumulates and fences are stripped on completion.

### Validation
After generation, syntax is validated using language-specific tools:
- Python: `compile()` built-in
- Bash: `bash -n`
- JS/TS: parenthesis/brace balancing

### History
`history.py` stores entries as JSONL in `~/.sgai-lite/history.jsonl`. Each entry includes timestamp, goal, language, model, temperature, output file, line/char counts.

### Config
`config.py` checks multiple locations in order:
1. `~/.sgai-lite/config.json`
2. `~/.sgai-lite/config.yaml`
3. `~/.sgai-lite/.sgai-lite.json`
4. `~/.sgai-lite/.sgai-lite.yaml`

## Code Style

- Python 3.8+ compatible (no walrus operator, no positional-only args unless guarded)
- Use `from __future__ import annotations` for forward references
- Type hints on all public functions
- Colorama for colored terminal output (graceful fallback if not installed)
- Click for CLI argument parsing
- Keep `generator.py` free of CLI concerns — it handles generation only

## Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=src/sgai_lite --cov-report=term-missing
```

## Adding a New Language

1. Add to `LANGUAGE_MAP` in `languages.py`
2. Add validation in `VALIDATABLE_LANGUAGES` in `generator.py`
3. Add language-specific tips in `LANG_SPECIFIC_TIPS` in `prompts.py`
4. Add test cases in `test_languages.py`

## Release Process

1. Update version in `src/sgai_lite/__init__.py`
2. Update `pyproject.toml` version
3. Tag: `git tag v0.x.x && git push --tags`
4. CI automatically publishes to PyPI on tags

## Scoring Guide

Features are scored against the following criteria (~105 points total):
- Core generation (20)
- CLI UX & flags (15)
- Streaming & real-time feedback (15)
- Validation & error handling (15)
- History & config (10)
- Refinement loop (10)
- Documentation (10)
- Tests (10)
- CI/CD & automation (5)
- Polish & quality (5)
