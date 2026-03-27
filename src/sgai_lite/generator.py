"""Core AI-powered code generation logic with streaming support."""

from __future__ import annotations

import os
import sys
import time
import subprocess
import tempfile
from typing import Iterator

from openai import OpenAI, APIError, AuthenticationError, RateLimitError

from sgai_lite.languages import get_language_name, get_extension, detect_language
from sgai_lite.prompts import build_system_prompt

DEFAULT_MODEL = "gpt-4o"

# Languages that support syntax validation
VALIDATABLE_LANGUAGES = {"python", "py", "js", "javascript", "ts", "typescript", "bash", "sh", "shell"}


class CodeGenerationError(Exception):
    """Raised when code generation fails."""
    pass


class APIKeyMissingError(Exception):
    """Raised when the API key is not set."""
    pass


class ValidationError(Exception):
    """Raised when generated code fails validation."""
    pass


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

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

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

    except AuthenticationError:
        raise APIKeyMissingError(
            "Authentication failed. Check your OPENAI_API_KEY.\n"
            "Get your key at: https://platform.openai.com/api-keys"
        )
    except RateLimitError:
        raise CodeGenerationError(
            "Rate limit hit. Please wait a moment and try again.\n"
            "Consider setting a different model: --model gpt-4o-mini"
        )
    except APIError as e:
        raise CodeGenerationError(f"OpenAI API error: {e}")


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
    except RateLimitError:
        raise CodeGenerationError("Rate limit hit. Please wait and try again.")
    except APIError as e:
        raise CodeGenerationError(f"OpenAI API error: {e}")
