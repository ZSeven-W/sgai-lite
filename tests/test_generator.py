"""Tests for code generation logic."""

import pytest
import os
from unittest.mock import patch, MagicMock
from sgai_lite.generator import (
    _strip_code_fences,
    validate_code,
    _validate_python,
    _validate_bash,
    _validate_js,
    _validate_ts,
    CodeGenerationError,
    APIKeyMissingError,
    ValidationError,
    DEFAULT_MODEL,
    generate_code,
    VALIDATABLE_LANGUAGES,
)
from sgai_lite.languages import detect_language


class TestStripCodeFences:
    def test_no_fences(self):
        code = "print('hello')\nprint('world')"
        assert _strip_code_fences(code) == code

    def test_single_fence_start(self):
        code = "```python\nprint('hello')\n```"
        assert _strip_code_fences(code) == "print('hello')"

    def test_fence_with_language_tag(self):
        code = "```python\nprint('hello')\n```"
        assert _strip_code_fences(code) == "print('hello')"

    def test_fence_no_language(self):
        code = "```\nprint('hello')\n```"
        assert _strip_code_fences(code) == "print('hello')"

    def test_fence_only_at_start(self):
        code = "```python\nprint('hello')"
        assert _strip_code_fences(code) == "print('hello')"

    def test_fence_only_at_end(self):
        code = "print('hello')\n```"
        assert _strip_code_fences(code) == "print('hello')"

    def test_multiple_fences(self):
        code = "```python\nprint('a')\n```\n```python\nprint('b')\n```"
        assert _strip_code_fences(code) == "print('a')\n```\n```python\nprint('b')"

    def test_empty_after_strip(self):
        assert _strip_code_fences("```\n```") == ""
        assert _strip_code_fences("```python\n```\n```") == ""


class TestValidatePython:
    def test_valid_code(self):
        code = "print('hello world')\nx = 1"
        ok, msg = _validate_python(code)
        assert ok is True

    def test_valid_function(self):
        code = """
def add(a, b):
    return a + b

if __name__ == '__main__':
    print(add(1, 2))
"""
        ok, msg = _validate_python(code)
        assert ok is True

    def test_invalid_syntax(self):
        code = "print('hello'\n"  # missing closing paren
        ok, msg = _validate_python(code)
        assert ok is False
        assert "SyntaxError" in msg

    def test_invalid_indentation(self):
        code = "if True:\n  print('a')\n print('b')"  # inconsistent indent
        ok, msg = _validate_python(code)
        assert ok is False


class TestValidateBash:
    def test_valid_script(self):
        code = "#!/bin/bash\nset -euo pipefail\necho 'hello'"
        ok, msg = _validate_bash(code)
        assert ok is True

    def test_valid_function(self):
        code = """#!/bin/bash
greet() {
    echo "Hello, $1"
}
greet "World"
"""
        ok, msg = _validate_bash(code)
        assert ok is True


class TestValidateJS:
    def test_valid_js(self):
        code = "const x = 1;\nconsole.log(x);"
        ok, msg = _validate_js(code)
        assert ok is True

    def test_balanced_braces(self):
        code = "function test() { return 42; }\ntest();"
        ok, msg = _validate_js(code)
        assert ok is True

    def test_unbalanced_braces(self):
        code = "function test() { return 42; \ntest();"
        ok, msg = _validate_js(code)
        assert ok is False
        assert "braces" in msg.lower()


class TestValidateTS:
    def test_valid_ts(self):
        code = "const x: number = 1;\nconsole.log(x);"
        ok, msg = _validate_ts(code)
        assert ok is True


class TestValidateCode:
    def test_unknown_language_no_validation(self):
        ok, msg = validate_code("some code", "cobol")
        assert ok is True

    def test_all_validatable_languages_covered(self):
        assert "python" in VALIDATABLE_LANGUAGES
        assert "py" in VALIDATABLE_LANGUAGES
        assert "bash" in VALIDATABLE_LANGUAGES
        assert "js" in VALIDATABLE_LANGUAGES
        assert "ts" in VALIDATABLE_LANGUAGES


class TestDefaultModel:
    def test_default_model_is_set(self):
        assert DEFAULT_MODEL == "gpt-4o"


class TestGenerateCode:
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("sgai_lite.generator.get_client")
    def test_generate_code_success(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "```python\nprint('hello world')\n```"
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = generate_code("a Python script that prints hello world")
        assert result == "print('hello world')"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("sgai_lite.generator.get_client")
    def test_generate_code_empty_response_raises(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        with pytest.raises(CodeGenerationError):
            generate_code("test goal")

    def test_generate_code_no_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from sgai_lite import generator
        # Force reimport by clearing cached client
        with pytest.raises(APIKeyMissingError):
            generate_code("test")
