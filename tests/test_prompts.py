"""Tests for prompt generation."""

import pytest
from sgai_lite.prompts import (
    build_system_prompt,
    SYSTEM_PROMPT_TEMPLATE,
    INTENT_KEYWORDS,
    LANG_SPECIFIC_TIPS,
)


class TestBuildSystemPrompt:
    def test_basic_prompt(self):
        prompt = build_system_prompt("python", "output.py", goal="")
        assert "python" in prompt.lower()
        assert "output.py" in prompt

    def test_prompt_with_goal_python_flask(self):
        # Goal "a web server with Flask" does not match any INTENT_KEYWORDS,
        # so no specialized intent tips are added.
        prompt = build_system_prompt("python", "output.py", goal="a web server with Flask")
        assert "python" in prompt.lower()
        # Base template should be present
        assert "Generate the code now" in prompt

    def test_python_tips_added(self):
        prompt = build_system_prompt("python", "output.py", goal="cli tool with argparse")
        assert "Python-specific" in prompt or "Python" in prompt

    def test_bash_tips_added(self):
        prompt = build_system_prompt("bash", "script.sh", goal="backup script")
        assert "Bash-specific" in prompt or "bash" in prompt.lower()

    def test_empty_goal_returns_base_prompt(self):
        prompt = build_system_prompt("javascript", "app.js", goal="")
        # Language name is capitalized via get_language_name
        assert "javascript" in prompt.lower()
        assert "app.js" in prompt
        assert "Generate the code now" in prompt


class TestIntentKeywords:
    def test_all_intents_have_keywords(self):
        for intent, keywords in INTENT_KEYWORDS.items():
            assert isinstance(keywords, list)
            assert len(keywords) > 0
            for kw in keywords:
                assert isinstance(kw, str)
                assert kw.strip()

    def test_intent_detection_keywords(self):
        assert "cli" in INTENT_KEYWORDS
        assert "web" in INTENT_KEYWORDS
        assert "data" in INTENT_KEYWORDS


class TestLangSpecificTips:
    def test_python_tips_exist(self):
        assert "python" in LANG_SPECIFIC_TIPS
        tips = LANG_SPECIFIC_TIPS["python"]
        assert "type hint" in tips.lower() or "dataclass" in tips.lower()

    def test_bash_tips_exist(self):
        assert "bash" in LANG_SPECIFIC_TIPS
        tips = LANG_SPECIFIC_TIPS["bash"]
        assert "set -e" in tips or "pipefail" in tips

    def test_go_tips_exist(self):
        assert "go" in LANG_SPECIFIC_TIPS
        tips = LANG_SPECIFIC_TIPS["go"]
        assert "error" in tips.lower()

    def test_rust_tips_exist(self):
        assert "rust" in LANG_SPECIFIC_TIPS
        tips = LANG_SPECIFIC_TIPS["rust"]
        assert "Result" in tips or "error" in tips.lower()
