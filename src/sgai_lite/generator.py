"""Core AI-powered code generation logic with streaming support."""

from __future__ import annotations

import os
import sys
import time
from typing import Iterator

from openai import OpenAI, APIError, AuthenticationError, RateLimitError

from sgai_lite.languages import get_language_name, get_extension, detect_language
from sgai_lite.prompts import SYSTEM_PROMPT_TEMPLATE

DEFAULT_MODEL = "gpt-4o"


class CodeGenerationError(Exception):
    """Raised when code generation fails."""
    pass


class APIKeyMissingError(Exception):
    """Raised when the API key is not set."""
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
    return OpenAI(api_key=api_key)


def _strip_code_fences(code: str) -> str:
    """Remove markdown code fences if present."""
    lines = code.splitlines()
    # Remove opening ```language or ```
    while lines and lines[0].strip().startswith("```"):
        lines.pop(0)
    # Remove closing ```
    while lines and lines[-1].strip() == "```":
        lines.pop()
    return "\n".join(lines).strip()


def generate_code_stream(
    goal: str,
    language: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
) -> Iterator[str]:
    """Generate code via OpenAI with streaming, yielding chunks.

    Args:
        goal: Natural language description of what to build
        language: Target language (auto-detected if None)
        model: OpenAI model to use
        temperature: Sampling temperature (lower = more deterministic)

    Yields:
        Code chunks as they are generated
    """
    client = get_client()

    if language is None:
        language = detect_language(goal)

    lang_name = get_language_name(language)
    ext = get_extension(language)
    filename = f"output{ext}"

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        language=lang_name,
        filename=filename,
    )

    user_prompt = f"Goal: {goal}"

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            stream=True,
        )

        buffer = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                buffer += delta
                yield delta

        # Final cleanup: strip fences from the full buffer
        full = _strip_code_fences(buffer)
        if not full:
            raise CodeGenerationError("Empty response from API. Please try again.")

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
) -> str:
    """Generate code via OpenAI (non-streaming version for testing).

    Args:
        goal: Natural language description of what to build
        language: Target language (auto-detected if None)
        model: OpenAI model to use
        temperature: Sampling temperature

    Returns:
        Generated code as a string
    """
    client = get_client()

    if language is None:
        language = detect_language(goal)

    lang_name = get_language_name(language)
    ext = get_extension(language)
    filename = f"output{ext}"

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        language=lang_name,
        filename=filename,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Goal: {goal}"},
            ],
            temperature=temperature,
        )

        raw = response.choices[0].message.content or ""
        code = _strip_code_fences(raw)

        if not code:
            raise CodeGenerationError("Empty response from API. Please try again.")

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
