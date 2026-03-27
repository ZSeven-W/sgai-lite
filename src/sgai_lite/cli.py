"""CLI for sgai-lite."""

from __future__ import annotations

import os
import sys
import argparse

try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    _CYAN = Fore.CYAN
    _GREEN = Fore.GREEN
    _YELLOW = Fore.YELLOW
    _RED = Fore.RED
    _RESET = Style.RESET_ALL
    _BOLD = Style.BRIGHT
except ImportError:
    _CYAN = _GREEN = _YELLOW = _RED = _RESET = _BOLD = ""


from sgai_lite import __version__
from sgai_lite.generator import (
    generate_code_stream,
    CodeGenerationError,
    APIKeyMissingError,
    DEFAULT_MODEL,
)
from sgai_lite.languages import detect_language, get_extension, get_language_name, LANGUAGE_MAP


def _print_header(goal: str, lang: str | None):
    lang_display = lang if lang else f"auto ({get_language_name(detect_language(goal))})"
    print(f"{_BOLD}{_CYAN}sgai-lite{_RESET} — generating [{lang_display}]...")
    print(f"{_BOLD}Goal:{_RESET} {goal}")
    print()


def _print_error(msg: str):
    print(f"{_RED}Error:{_RESET} {msg}", file=sys.stderr)


def _print_warning(msg: str):
    print(f"{_YELLOW}Warning:{_RESET} {msg}", file=sys.stderr)


def _write_file(output_path: str, content: str) -> bool:
    """Write generated code to file. Returns True on success."""
    try:
        with open(output_path, "w") as f:
            f.write(content)
            f.write("\n")
        return True
    except OSError as e:
        _print_error(f"Failed to write {output_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        prog="sgai",
        description="Goal-driven single-file code generator powered by AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sgai \"a Python script that prints hello world with a timestamp\"
  sgai --lang ts \"a TypeScript function that validates email addresses\"
  sgai --output server.py \"a Python HTTP server with GET /hello\"
  sgai --model gpt-4o-mini \"a bash script to find large files\"

Environment:
  OPENAI_API_KEY   Your OpenAI API key (required)
  OPENAI_BASE_URL Custom API base URL (optional, for proxies)
        """,
    )

    parser.add_argument(
        "goal",
        nargs="?",
        help="Natural language description of what to build",
    )
    parser.add_argument(
        "-l", "--lang",
        dest="language",
        metavar="LANG",
        help=f"Target language ({', '.join(sorted(LANGUAGE_MAP.keys())[:12])}...)",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Output file path (auto-generated if not specified)",
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.3,
        help="Sampling temperature 0.0-2.0 (default: 0.3, lower=more deterministic)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show detected language and exit without generating",
    )
    parser.add_argument(
        "--list-langs",
        action="store_true",
        help="List supported languages and exit",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"sgai-lite {__version__}",
    )

    args = parser.parse_args()

    if args.list_langs:
        print(f"{_BOLD}Supported languages:{_RESET}")
        for lang in sorted(LANGUAGE_MAP.keys()):
            ext = LANGUAGE_MAP[lang]
            print(f"  {lang:15s} → {ext}")
        return 0

    if not args.goal:
        _print_error("No goal specified. Run: sgai \"your goal here\"")
        parser.print_help()
        return 1

    language = args.language

    if args.dry_run:
        detected = detect_language(args.goal)
        lang_display = language or detected
        ext = get_extension(lang_display)
        output = args.output or f"output{ext}"
        print(f"{_BOLD}Language:{_RESET} {get_language_name(lang_display)}")
        print(f"{_BOLD}Extension:{_RESET} {ext}")
        print(f"{_BOLD}Output:{_RESET} {output}")
        return 0

    # Check API key for actual generation
    if not os.environ.get("OPENAI_API_KEY"):
        _print_error(
            "OPENAI_API_KEY is not set.\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "Get your key at: https://platform.openai.com/api-keys"
        )
        return 1

    # Determine output file
    if language is None:
        detected = detect_language(args.goal)
        language = detected

    ext = get_extension(language)
    output_path = args.output or f"output{ext}"

    _print_header(args.goal, language)

    # Stream and display
    full_code = ""
    try:
        for chunk in generate_code_stream(
            goal=args.goal,
            language=language,
            model=args.model,
            temperature=args.temp,
        ):
            print(chunk, end="", flush=True)
            full_code += chunk
    except KeyboardInterrupt:
        print(f"\n{_YELLOW}Interrupted by user.{_RESET}")
        # Offer to save what was generated
        if full_code:
            save = input(f"\nSave partial output to {output_path}? [y/N]: ").strip().lower()
            if save == "y":
                _write_file(output_path, full_code)
                print(f"Saved partial output to {output_path}")
        return 130

    print()  # newline after streaming

    if not full_code.strip():
        _print_error("No code was generated. Please try again.")
        return 1

    # Write to file
    if _write_file(output_path, full_code):
        print(f"{_GREEN}{_BOLD}✓ Saved to {output_path}{_RESET}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
