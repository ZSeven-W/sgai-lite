"""Core AI-powered code generation logic with streaming support."""

from __future__ import annotations

import os
import sys
import time
import random
import subprocess
import tempfile
from typing import Iterator

from openai import OpenAI, APIError, AuthenticationError, RateLimitError

from sgai_lite.languages import get_language_name, get_extension, detect_language
from sgai_lite.prompts import build_system_prompt

DEFAULT_MODEL = "gpt-4o"

# Languages that support syntax validation
VALIDATABLE_LANGUAGES = {"python", "py", "js", "javascript", "ts", "typescript", "bash", "sh", "shell"}

# OpenAI pricing (approximate, per 1M tokens)
PRICING = {
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


class CodeGenerationError(Exception):
    """Raised when code generation fails."""
    pass


class APIKeyMissingError(Exception):
    """Raised when the API key is not set."""
    pass


class ValidationError(Exception):
    """Raised when generated code fails validation."""
    pass


def detect_imports(code: str) -> list[str]:
    """Detect third-party imports from Python code."""
    import re
    # Standard library modules to ignore
    stdlib = {
        "os", "sys", "re", "json", "csv", "xml", "html", "time", "datetime",
        "random", "math", "statistics", "collections", "itertools", "functools",
        "operator", "string", "textwrap", "unicodedata", "locale", "copy",
        "abc", "dataclasses", "enum", "typing", "pathlib", "io", "os.path",
        "urllib", "http", "ftplib", "smtplib", "poplib", "imaplib", "socket",
        "argparse", "getopt", "logging", "warnings", "dateutil", "pprint",
        "struct", "codecs", "gc", "weakref", "types", "inspect", "dis",
        "ast", "platform", "errno", "ctypes", "signal", "mmap", "shutil",
        "tempfile", "glob", "fnmatch", "linecache", "tokenize", "keyword",
        "ast", "symtable", "token", "tabnanny", "py_compile", "pickle",
        " shelve", "dbm", "sqlite3", "zlib", "gzip", "bz2", "lzma", "zipfile",
        "tarfile", "fileinput", "stat", "filecmp", "difflib", "mailbox",
        "subprocess", "sysconfig", "webbrowser", "turtle", "tracemalloc",
        "concurrent", "multiprocessing", "asyncio", "queue", "threading",
        "contextvars", "contextlib", "typing_extensions", "fractions",
        "decimal", "numbers", "cmath", "cProfile", "profile", "resource",
        "test", "unittest", "doctest", "traceback", "sysconfig", "pkgutil",
        "venv", "zipapp", "__future__", "builtins",
    }

    # Regex to find import X and from X import Y
    pattern = re.compile(r'^\s*(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)', re.MULTILINE)
    candidates = pattern.findall(code)

    # Filter: only third-party (not stdlib), take first part of dotted imports
    third_party = []
    for module in candidates:
        base = module.split('.')[0].lower()
        if base not in stdlib and base not in third_party:
            third_party.append(base)

    return sorted(third_party)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate generation cost in USD."""
    prices = PRICING.get(model, {"input": 2.5, "output": 10.0})
    input_cost = (prompt_tokens / 1_000_000) * prices["input"]
    output_cost = (completion_tokens / 1_000_000) * prices["output"]
    return input_cost + output_cost


def get_client() -> OpenAI:
    """Create an OpenAI client, checking for API key."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise APIKeyMissingError(
            "OPENAI_API_KEY environment variable is not set.\n"
            "Please set it: export OPENAI_API_KEY=your_key_here\n"
            "Get your key at: https://platform.openai.com/api-keys"
        )

    base_url = os.environ.get("OPENAI_BASE_URL")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _strip_code_fences(code: str) -> str:
    """Remove markdown code fences if present."""
    lines = code.splitlines()
    while lines and lines[0].strip().startswith("```"):
        lines.pop(0)
    while lines and lines[-1].strip() == "```":
        lines.pop()
    return "\n".join(lines).strip()


def validate_code(code: str, language: str) -> tuple[bool, str]:
    """Validate generated code syntax. Returns (ok, message)."""
    lang = language.lower().split()[0]

    if lang == "python" or lang == "py":
        return _validate_python(code)
    elif lang in ("bash", "sh", "shell"):
        return _validate_bash(code)
    elif lang in ("js", "javascript"):
        return _validate_js(code)
    elif lang in ("ts", "typescript"):
        return _validate_ts(code)

    return True, "No validator available for this language"


def _validate_python(code: str) -> tuple[bool, str]:
    """Validate Python code with py_compile."""
    try:
        compile(code, "<generated>", "exec")
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def _validate_bash(code: str) -> tuple[bool, str]:
    """Validate Bash code with bash -n."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".sh", delete=False, mode="w") as f:
            f.write(code)
            f.flush()
            tmp = f.name

        result = subprocess.run(
            ["bash", "-n", tmp],
            capture_output=True,
            text=True,
            timeout=10,
        )
        os.unlink(tmp)

        if result.returncode == 0:
            return True, "Syntax OK"
        return False, result.stderr.strip()
    except Exception as e:
        return False, f"Validation error: {e}"


def _validate_js(code: str) -> tuple[bool, str]:
    """Basic JavaScript validation."""
    # Check for obvious syntax issues
    lines = code.splitlines()
    issues = []
    parens = brackets = braces = 0
    in_string = False
    string_char = None

    for i, line in enumerate(lines, 1):
        for ch in line:
            if not in_string:
                if ch in "\"'\"":
                    in_string = True
                    string_char = ch
                elif ch == "(":
                    parens += 1
                elif ch == ")":
                    parens -= 1
                elif ch == "[":
                    brackets += 1
                elif ch == "]":
                    brackets -= 1
                elif ch == "{":
                    braces += 1
                elif ch == "}":
                    braces -= 1
            else:
                if ch == string_char and (not line.count(string_char) > 1 or line.index(ch) != len(line) - 1):
                    in_string = False

    if parens != 0:
        issues.append(f"unbalanced parentheses ({'+' if parens > 0 else ''}{parens})")
    if brackets != 0:
        issues.append(f"unbalanced brackets ({'+' if brackets > 0 else ''}{brackets})")
    if braces != 0:
        issues.append(f"unbalanced braces ({'+' if braces > 0 else ''}{braces})")

    if issues:
        return False, "; ".join(issues)
    return True, "Basic syntax OK"


def _validate_ts(code: str) -> tuple[bool, str]:
    """TypeScript validation (basic, same as JS for now)."""
    return _validate_js(code)


def generate_code_stream(
    goal: str,
    language: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    skip_validation: bool = False,
    refinement: str | None = None,
) -> Iterator[tuple[str, bool]]:
    """Generate code via OpenAI with streaming, yielding (chunk, done) tuples.

    When done=True, the stream is finished.

    Args:
        goal: Natural language description of what to build
        language: Target language (auto-detected if None)
        model: OpenAI model to use
        temperature: Sampling temperature (lower = more deterministic)
        skip_validation: Skip syntax validation
        refinement: If provided, refine existing code with this instruction

    Yields:
        (code_chunk, done) tuples
    """
    client = get_client()

    if language is None:
        language = detect_language(goal)

    lang_name = get_language_name(language)
    ext = get_extension(language)
    filename = f"output{ext}"

    system_prompt = build_system_prompt(language, filename, goal)

    if refinement:
        user_prompt = (
            f"Here is the current code:\n\n{{code}}\n\n"
            f"Refinement request: {refinement}\n\n"
            "Please generate the improved code. Output ONLY the code, no fences, no explanations."
        )
        # For refinement, we need to ask user for the code or read from context
        # This is handled at a higher level
    else:
        user_prompt = f"Goal: {goal}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Retry logic with exponential backoff
    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            buffer = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    buffer += delta
                    yield delta, False

            # Final cleanup: strip fences
            full = _strip_code_fences(buffer)
            if not full:
                raise CodeGenerationError("Empty response from API. Please try again.")

            # Validation
            if not skip_validation:
                lang_key = language.lower().split()[0]
                if lang_key in VALIDATABLE_LANGUAGES:
                    ok, msg = validate_code(full, language)
                    if not ok:
                        print(f"\n⚠ Validation warning: {msg}", file=sys.stderr)

            yield "", True  # done signal
            return  # exit after successful generation

        except AuthenticationError:
            raise APIKeyMissingError(
                "Authentication failed. Check your OPENAI_API_KEY.\n"
                "Get your key at: https://platform.openai.com/api-keys"
            )
        except RateLimitError as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"\n⚠ Rate limited. Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                time.sleep(wait_time)
            else:
                raise CodeGenerationError(
                    f"Rate limit hit after {max_retries} retries. Please wait and try again.\n"
                    "Consider setting a different model: --model gpt-4o-mini"
                )
        except APIError as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"\n⚠ API error ({e}). Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                time.sleep(wait_time)
            else:
                raise CodeGenerationError(f"OpenAI API error after {max_retries} retries: {e}")


def generate_code(
    goal: str,
    language: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    skip_validation: bool = False,
    refinement: str | None = None,
    existing_code: str | None = None,
) -> str:
    """Generate code via OpenAI (non-streaming version for testing).

    Args:
        goal: Natural language description of what to build
        language: Target language (auto-detected if None)
        model: OpenAI model to use
        temperature: Sampling temperature
        skip_validation: Skip syntax validation
        refinement: Refinement instruction
        existing_code: Existing code to refine

    Returns:
        Generated code as a string
    """
    client = get_client()

    if language is None:
        language = detect_language(goal)

    lang_name = get_language_name(language)
    ext = get_extension(language)
    filename = f"output{ext}"

    system_prompt = build_system_prompt(language, filename, goal)

    if refinement and existing_code:
        user_prompt = (
            f"Here is the current code:\n\n{existing_code}\n\n"
            f"Refinement request: {refinement}\n\n"
            "Please generate the improved code. Output ONLY the code, no fences, no explanations."
        )
    elif refinement:
        user_prompt = f"Goal: {goal}\n\nRefinement: {refinement}"
    else:
        user_prompt = f"Goal: {goal}"

    # Retry logic with exponential backoff
    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )

            raw = response.choices[0].message.content or ""
            code = _strip_code_fences(raw)

            if not code:
                raise CodeGenerationError("Empty response from API. Please try again.")

            # Validation
            if not skip_validation:
                lang_key = language.lower().split()[0]
                if lang_key in VALIDATABLE_LANGUAGES:
                    ok, msg = validate_code(code, language)
                    if not ok:
                        raise ValidationError(f"Validation failed: {msg}")

            return code

        except AuthenticationError:
            raise APIKeyMissingError(
                "Authentication failed. Check your OPENAI_API_KEY.\n"
                "Get your key at: https://platform.openai.com/api-keys"
            )
        except RateLimitError as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                raise CodeGenerationError(
                    f"Rate limit hit after {max_retries} retries. Please wait and try again."
                )
        except APIError as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                raise CodeGenerationError(f"OpenAI API error after {max_retries} retries: {e}")
