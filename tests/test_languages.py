"""Tests for language detection and mapping."""

import pytest
from sgai_lite.languages import (
    detect_language,
    get_extension,
    get_language_name,
    LANGUAGE_MAP,
    LANGUAGE_HINTS,
    DEFAULT_EXTENSION,
)


class TestDetectLanguage:
    def test_python_detection(self):
        assert detect_language("a Python script that prints hello") == "python"
        assert detect_language("use pip to install dependencies") == "python"
        assert detect_language("django rest api") == "python"

    def test_javascript_detection(self):
        assert detect_language("a JavaScript function for array sorting") == "javascript"
        assert detect_language("node.js server with express") == "javascript"

    def test_typescript_detection(self):
        assert detect_language("a TypeScript function with type hints") == "typescript"
        assert detect_language("angular component with TypeScript") == "typescript"

    def test_bash_detection(self):
        assert detect_language("a bash script to find large files") == "bash"
        assert detect_language("shell command to backup database") == "bash"

    def test_go_detection(self):
        assert detect_language("a Go program that fetches URLs") == "go"

    def test_rust_detection(self):
        assert detect_language("a rust program for file processing") == "rust"

    def test_default_python(self):
        """Unknown goals should default to Python."""
        assert detect_language("make something cool") == "python"
        assert detect_language("build a todo app") == "python"


class TestGetExtension:
    def test_common_extensions(self):
        assert get_extension("python") == ".py"
        assert get_extension("py") == ".py"
        assert get_extension("javascript") == ".js"
        assert get_extension("js") == ".js"
        assert get_extension("typescript") == ".ts"
        assert get_extension("bash") == ".sh"
        assert get_extension("go") == ".go"
        assert get_extension("rust") == ".rs"
        assert get_extension("ruby") == ".rb"
        assert get_extension("java") == ".java"

    def test_dockerfile(self):
        assert get_extension("dockerfile") == "Dockerfile"
        assert get_extension("docker") == "Dockerfile"

    def test_unknown_language(self):
        """Unknown languages should fall back to default extension."""
        assert get_extension("nonexistent") == DEFAULT_EXTENSION
        assert get_extension("") == DEFAULT_EXTENSION


class TestGetLanguageName:
    def test_readable_names(self):
        assert get_language_name("python") == "Python"
        assert get_language_name("py") == "Python"
        assert get_language_name("javascript") == "JavaScript"
        assert get_language_name("js") == "JavaScript"
        assert get_language_name("typescript") == "TypeScript"
        assert get_language_name("bash") == "Bash"
        assert get_language_name("go") == "Go"

    def test_case_insensitive(self):
        assert get_language_name("PYTHON") == "Python"
        assert get_language_name("JavaScript") == "JavaScript"

    def test_unknown_language(self):
        """Unknown languages should be capitalized."""
        assert get_language_name("unknown") == "Unknown"
        assert get_language_name("cobol") == "Cobol"


class TestLanguageMap:
    def test_all_languages_have_extensions(self):
        """Every language in LANGUAGE_MAP should have a non-empty extension."""
        for lang, ext in LANGUAGE_MAP.items():
            assert ext, f"Language {lang} has empty extension"
            assert ext.strip(), f"Language {lang} has whitespace extension"

    def test_language_map_keys(self):
        """LANGUAGE_MAP should contain expected languages."""
        expected = {"python", "javascript", "typescript", "bash", "go", "rust", "ruby"}
        for lang in expected:
            assert lang in LANGUAGE_MAP, f"{lang} missing from LANGUAGE_MAP"
