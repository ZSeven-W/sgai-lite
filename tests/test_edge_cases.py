"""Edge case tests for sgai-lite: empty goal, invalid language, file write failures."""

import pytest
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

from sgai_lite.generator import (
    validate_code,
    detect_imports,
    estimate_cost,
    GenerationResult,
)
from sgai_lite.languages import detect_language, get_extension, get_language_name, LANGUAGE_MAP


class TestEmptyGoal:
    """Tests for empty goal handling."""

    def test_detect_language_empty_string(self):
        """detect_language should return default (python) for empty string."""
        lang = detect_language("")
        assert lang == "python"

    def test_get_extension_for_empty(self):
        """get_extension should return default for empty language."""
        ext = get_extension("")
        assert ext == ".py"

    def test_get_language_name_empty(self):
        """get_language_name should handle empty string."""
        name = get_language_name("")
        assert name == ""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("sgai_lite.generator.get_client")
    def test_generate_code_with_empty_goal(self, mock_get_client):
        """generate_code should handle empty goal gracefully."""
        from sgai_lite.generator import generate_code

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "```python\nprint('hello')\n```"
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 60
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = generate_code("")
        assert result.code == "print('hello')"


class TestInvalidLanguage:
    """Tests for invalid language handling."""

    def test_validate_unknown_language(self):
        """validate_code should return True for unknown languages."""
        ok, msg = validate_code("some code", "cobol")
        assert ok is True
        assert "No validator" in msg

    def test_validate_fictional_language(self):
        """validate_code should handle fictional languages gracefully."""
        ok, msg = validate_code("code here", "madeup123")
        assert ok is True

    def test_get_extension_unknown_language(self):
        """get_extension should return default for unknown language."""
        ext = get_extension("madeup_language")
        assert ext == ".py"

    def test_get_language_name_unknown(self):
        """get_language_name should handle unknown languages."""
        name = get_language_name("unknown_lang_xyz")
        assert name == "Unknown_lang_xyz"  # capitalized

    def test_all_languages_have_extension(self):
        """All languages in LANGUAGE_MAP should have non-empty extensions."""
        for lang, ext in LANGUAGE_MAP.items():
            assert ext, f"Language {lang} has empty extension"
            assert len(ext) > 0

    def test_validate_lua_not_installed(self):
        """_validate_lua should return True when lua is not available."""
        from sgai_lite.generator import _validate_lua
        with patch("shutil.which", return_value=None):
            code = "print('hello')"
            ok, msg = _validate_lua(code)
            assert ok is True
            assert "not found" in msg

    def test_validate_go_not_installed(self):
        """_validate_go should return True when gofmt is not available."""
        from sgai_lite.generator import _validate_go
        with patch("shutil.which", return_value=None):
            code = "package main\nfunc main() {}"
            ok, msg = _validate_go(code)
            assert ok is True
            assert "not found" in msg

    def test_validate_rust_not_installed(self):
        """_validate_rust should return True when rustfmt is not available."""
        from sgai_lite.generator import _validate_rust
        with patch("shutil.which", return_value=None):
            code = "fn main() {}"
            ok, msg = _validate_rust(code)
            assert ok is True
            assert "not found" in msg

    def test_validate_ruby_not_installed(self):
        """_validate_ruby should return True when ruby is not available."""
        from sgai_lite.generator import _validate_ruby
        with patch("shutil.which", return_value=None):
            code = "puts 'hello'"
            ok, msg = _validate_ruby(code)
            assert ok is True
            assert "not found" in msg

    def test_validate_php_not_installed(self):
        """_validate_php should return True when php is not available."""
        from sgai_lite.generator import _validate_php
        with patch("shutil.which", return_value=None):
            code = "<?php echo 'hello';"
            ok, msg = _validate_php(code)
            assert ok is True
            assert "not found" in msg


class TestFileWriteFailures:
    """Tests for file write failure handling."""

    def test_write_to_readonly_directory(self):
        """_write_file should return False when directory is read-only."""
        from sgai_lite.cli import _write_file

        # Create a temp dir, make it read-only
        with tempfile.TemporaryDirectory() as tmpdir:
            readonly_dir = os.path.join(tmpdir, "readonly")
            os.makedirs(readonly_dir)
            os.chmod(readonly_dir, 0o444)  # read-only

            file_path = os.path.join(readonly_dir, "output.py")
            try:
                result = _write_file(file_path, "print('hello')")
                assert result is False
            finally:
                # Restore permissions for cleanup
                os.chmod(readonly_dir, 0o755)

    def test_write_to_nonexistent_directory(self):
        """_write_file should return False when parent directory doesn't exist."""
        from sgai_lite.cli import _write_file
        result = _write_file("/nonexistent/path/to/file.txt", "content")
        assert result is False


class TestGenerationResult:
    """Tests for GenerationResult namedtuple."""

    def test_generation_result_fields(self):
        """GenerationResult should have all expected fields."""
        result = GenerationResult(
            code="print('hello')",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            estimated_cost=0.002,
        )
        assert result.code == "print('hello')"
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 20
        assert result.total_tokens == 120
        assert result.estimated_cost == 0.002

    def test_generation_result_none_values(self):
        """GenerationResult should handle None values for usage."""
        result = GenerationResult(
            code="x = 1",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            estimated_cost=None,
        )
        assert result.code == "x = 1"
        assert result.total_tokens is None


class TestCostEstimation:
    """Tests for cost estimation."""

    def test_estimate_cost_known_model(self):
        """estimate_cost should calculate cost for known models."""
        cost = estimate_cost("gpt-4o", 1000, 500)
        # gpt-4o: $2.5/1M input, $10/1M output
        # input: 1000/1M * 2.5 = 0.0025
        # output: 500/1M * 10 = 0.005
        expected = 0.0025 + 0.005
        assert abs(cost - expected) < 0.0001

    def test_estimate_cost_unknown_model(self):
        """estimate_cost should use defaults for unknown models."""
        cost = estimate_cost("unknown-model", 1000, 500)
        # Should use default: $2.5 input, $10 output per 1M
        expected = 0.0025 + 0.005
        assert abs(cost - expected) < 0.0001

    def test_estimate_cost_mini_model(self):
        """estimate_cost should handle gpt-4o-mini pricing."""
        cost = estimate_cost("gpt-4o-mini", 1000, 500)
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        expected = 0.00015 + 0.0003
        assert abs(cost - expected) < 0.0001


class TestDetectImports:
    """Tests for import detection."""

    def test_detect_third_party_imports(self):
        """detect_imports should find third-party packages."""
        code = """
import requests
import click
from rich import print
"""
        imports = detect_imports(code)
        assert "requests" in imports
        assert "click" in imports
        assert "rich" in imports

    def test_detect_ignores_stdlib(self):
        """detect_imports should ignore standard library modules."""
        code = """
import os
import sys
import json
from typing import List
from pathlib import Path
"""
        imports = detect_imports(code)
        assert "os" not in imports
        assert "sys" not in imports
        assert "json" not in imports
        assert "typing" not in imports
        assert "pathlib" not in imports

    def test_detect_from_import(self):
        """detect_imports should handle 'from X import Y' syntax."""
        code = "from click import echo, argument"
        imports = detect_imports(code)
        assert "click" in imports

    def test_detect_dotted_import(self):
        """detect_imports should handle dotted module names."""
        code = "import os.path"
        imports = detect_imports(code)
        # os is stdlib, should not be included
        assert len(imports) == 0

    def test_detect_no_imports(self):
        """detect_imports should return empty list for code with no imports."""
        code = "x = 1\nprint(x)"
        imports = detect_imports(code)
        assert imports == []


class TestLanguageHints:
    """Tests for language hint detection."""

    def test_detect_rust(self):
        """detect_language should detect Rust from keywords."""
        lang = detect_language("a rust program that prints hello")
        assert lang == "rust"

    def test_detect_go(self):
        """detect_language should detect Go from keywords."""
        lang = detect_language("a golang http server")
        assert lang == "go"

    def test_detect_ruby(self):
        """detect_language should detect Ruby from keywords."""
        lang = detect_language("a ruby script to process data")
        assert lang == "ruby"

    def test_detect_php(self):
        """detect_language should detect PHP from keywords."""
        lang = detect_language("a php script for web processing")
        assert lang == "php"

    def test_detect_typescript(self):
        """detect_language should detect TypeScript from angular."""
        lang = detect_language("an angular component")
        assert lang == "typescript"
