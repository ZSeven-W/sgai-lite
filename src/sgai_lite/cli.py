"""CLI for sgai-lite."""

from __future__ import annotations

import os
import sys
import argparse
import shutil

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
    generate_code,
    CodeGenerationError,
    APIKeyMissingError,
    ValidationError,
    DEFAULT_MODEL,
    detect_imports,
)
from sgai_lite.languages import detect_language, get_extension, get_language_name, LANGUAGE_MAP
from sgai_lite.history import add_entry, list_entries, get_entry, format_entries, clear_history
from sgai_lite.config import load_config

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def _bold(text: str) -> str:
    return f"{_BOLD}{text}{_RESET}" if _BOLD else text


def _cyan(text: str) -> str:
    return f"{_CYAN}{text}{_RESET}" if _CYAN else text


def _green(text: str) -> str:
    return f"{_GREEN}{text}{_RESET}" if _GREEN else text


def _yellow(text: str) -> str:
    return f"{_YELLOW}{text}{_RESET}" if _YELLOW else text


def _red(text: str) -> str:
    return f"{_RED}{text}{_RESET}" if _RED else text


def _print_header(goal: str, lang: str | None):
    lang_display = lang if lang else f"auto ({get_language_name(detect_language(goal))})"
    print(f"{_bold(_cyan('sgai-lite'))} — generating [{lang_display}]...")
    print(f"{_bold('Goal:')} {goal}")
    print()


def _print_error(msg: str):
    print(f"{_red('Error:')} {msg}", file=sys.stderr)


def _print_warning(msg: str):
    print(f"{_yellow('Warning:')} {msg}", file=sys.stderr)


def _print_success(msg: str):
    print(f"{_green('✓')} {msg}")


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


def _is_in_git_repo(path: str) -> bool:
    """Check if a path is inside a git repository."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=os.path.dirname(os.path.abspath(path)) if os.path.isfile(path) else path,
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _git_commit_file(file_path: str, goal: str) -> bool:
    """Commit a generated file to git."""
    import subprocess
    try:
        abs_path = os.path.abspath(file_path)
        rel_path = os.path.basename(abs_path)
        # Truncate goal for commit message
        msg = f"feat: generated {rel_path} — {goal[:80]}"
        subprocess.run(["git", "add", abs_path], check=True, capture_output=True, timeout=10)
        subprocess.run(
            ["git", "commit", "-m", msg],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _open_file(file_path: str) -> bool:
    """Open file in default application (browser for web files, editor for code)."""
    import subprocess
    try:
        abs_path = os.path.abspath(file_path)
        if sys.platform == "darwin":
            subprocess.run(["open", abs_path], check=False, capture_output=True, timeout=10)
        elif sys.platform == "win32":
            subprocess.run(["start", "", abs_path], shell=True, check=False, capture_output=True, timeout=10)
        else:
            subprocess.run(["xdg-open", abs_path], check=False, capture_output=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _install_dependencies(code: str, language: str) -> list[str]:
    """Attempt to install detected dependencies. Returns list of installed packages."""
    import subprocess
    lang_key = language.lower().split()[0]
    if lang_key not in ("python", "py"):
        return []

    imports = detect_imports(code)
    installed = []
    for pkg in imports:
        # Map common import names to package names
        pkg_map = {
            "requests": "requests",
            "click": "click",
            "rich": "rich",
            "typer": "typer",
            "fastapi": "fastapi",
            "flask": "flask",
            "django": "django",
            "pandas": "pandas",
            "numpy": "numpy",
            "pydantic": "pydantic",
            "sqlalchemy": "sqlalchemy",
            "pytest": "pytest",
            "matplotlib": "matplotlib",
            "seaborn": "seaborn",
            "aiohttp": "aiohttp",
            "httpx": "httpx",
            "beautifulsoup4": "beautifulsoup4",
            "lxml": "lxml",
            "pyyaml": "pyyaml",
            "toml": "toml",
            "tqdm": "tqdm",
            "pillow": "pillow",
            "cryptography": "cryptography",
            "boto3": "boto3",
            "redis": "redis",
            "pymongo": "pymongo",
            "psycopg2": "psycopg2-binary",
            "mysql": "mysql-connector-python",
            "jinja2": "jinja2",
            "markdown": "markdown",
            "pytest": "pytest",
            "playwright": "playwright",
            "selenium": "selenium",
            "streamlit": "streamlit",
            "gradio": "gradio",
            "inquirer": "inquirer",
            "questionary": "questionary",
            "prompt_toolkit": "prompt-toolkit",
            "click": "click",
            "typer": "typer",
            "black": "black",
            "ruff": "ruff",
            "mypy": "mypy",
            "pylint": "pylint",
        }
        pip_name = pkg_map.get(pkg, pkg)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                installed.append(pip_name)
            else:
                pass  # silently skip failures
        except subprocess.TimeoutExpired:
            pass
    return installed


def _format_code(code: str, formatter: str | None) -> str:
    """Apply code formatter if available."""
    if not formatter:
        return code

    import tempfile, subprocess

    if formatter == "black":
        tool = shutil.which("black")
        if not tool:
            _print_warning("black not found. Install: pip install black")
            return code
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
                f.write(code)
                f.flush()
                tmp = f.name
            result = subprocess.run([tool, tmp], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return open(tmp).read()
            else:
                _print_warning(f"black failed: {result.stderr.strip()}")
        except Exception as e:
            _print_warning(f"Formatting error: {e}")
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    elif formatter == "ruff":
        tool = shutil.which("ruff")
        if not tool:
            _print_warning("ruff not found. Install: pip install ruff")
            return code
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
                f.write(code)
                f.flush()
                tmp = f.name
            result = subprocess.run([tool, "format", tmp], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return open(tmp).read()
            else:
                _print_warning(f"ruff format failed: {result.stderr.strip()}")
        except Exception as e:
            _print_warning(f"Formatting error: {e}")
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    elif formatter == "autopep8":
        tool = shutil.which("autopep8")
        if not tool:
            _print_warning("autopep8 not found. Install: pip install autopep8")
            return code
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
                f.write(code)
                f.flush()
                tmp = f.name
            result = subprocess.run([tool, "-i", tmp], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return open(tmp).read()
        except Exception as e:
            _print_warning(f"Formatting error: {e}")
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    return code


# Language color map for output
_LANG_COLORS = {
    "python": _CYAN,
    "javascript": _YELLOW,
    "typescript": Fore.BLUE if hasattr(Fore, 'BLUE') else _CYAN,
    "bash": _GREEN,
    "shell": _GREEN,
    "go": Fore.CYAN if hasattr(Fore, 'CYAN') else _CYAN,
    "rust": Fore.RED if hasattr(Fore, 'RED') else _YELLOW,
    "html": _RED,
    "css": Fore.BLUE if hasattr(Fore, 'BLUE') else _CYAN,
    "sql": _YELLOW,
    "yaml": _GREEN,
    "json": _CYAN,
    "dockerfile": _CYAN,
}


def _language_color(language: str) -> str:
    """Get color for a language."""
    lang_key = language.lower().split()[0]
    return _LANG_COLORS.get(lang_key, _CYAN)


_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_spinner_index = 0


def _print_spinner(label: str = "Generating"):
    """Print a spinning cursor with label."""
    global _spinner_index
    _spinner_index = (_spinner_index + 1) % len(_SPINNER_FRAMES)
    frame = _SPINNER_FRAMES[_spinner_index]
    # \r moves cursor to beginning, \b backspaces one char, space overwrites
    print(f"\r{_CYAN}{frame}{_RESET} {label}...", end="", flush=True)


def _clear_spinner():
    """Clear the spinner line."""
    print("\r" + " " * 60 + "\r", end="", flush=True)


def _show_history(n: int = 10):
    """Show generation history."""
    entries = list_entries(n=n)
    print(_bold(f"Last {len(entries)} generation(s):"))
    print()
    print(format_entries(entries))


def _rerun_from_history(index: int):
    """Regenerate from a history entry."""
    entry = get_entry(index)
    if not entry:
        _print_error(f"History entry #{index} not found.")
        return None
    return entry


def _interactive_refine(code: str, language: str, model: str, temperature: float, output_path: str) -> str | None:
    """Ask user if they want to refine, handle refinement loop."""
    try:
        response = input(f"\n{_yellow('Refine this code?')} (y/n) or shortcut (fix/tests/docs/faster): ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return None

    if response in ("n", "no", ""):
        return None

    shortcuts = {
        "fix": "fix all bugs and improve error handling",
        "tests": "add comprehensive unit tests using pytest",
        "docs": "add detailed docstrings and comments",
        "faster": "optimize for performance and efficiency",
    }

    if response in shortcuts:
        refinement = shortcuts[response]
    elif response == "y":
        refinement = input(f"{_cyan('What should I change?')} ").strip()
        if not refinement:
            return None
    else:
        refinement = response

    print(f"\n{_cyan('Refining...')} ({refinement})")
    try:
        new_code = generate_code(
            goal="",  # empty since we're refining
            language=language,
            model=model,
            temperature=temperature,
            refinement=refinement,
            existing_code=code,
        )
        return new_code
    except Exception as e:
        _print_error(f"Refinement failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        prog="sgai",
        description="Goal-driven single-file code generator powered by AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sgai "a Python script that prints hello world with a timestamp"
  sgai --lang ts "a TypeScript function that validates email addresses"
  sgai --output server.py "a Python HTTP server with GET /hello"
  sgai --model gpt-4o-mini "a bash script to find large files"
  sgai --history              # Show last 10 generations
  sgai --rerun 3              # Regenerate from history entry #3
  sgai --refine "add tests" --input output.py

Environment:
  OPENAI_API_KEY   Your OpenAI API key (required)
  OPENAI_BASE_URL  Custom API base URL (optional, for proxies)
  SGAI_CONFIG      Path to config file (optional)
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
        default=None,  # Will be filled from config
        help="OpenAI model",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=None,  # Will be filled from config
        help="Sampling temperature 0.0-2.0 (default: 0.3)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip syntax validation after generation",
    )
    parser.add_argument(
        "--formatter",
        choices=["black", "ruff", "autopep8", "none"],
        default=None,
        help="Auto-format generated code (Python only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show detected language and exit without generating",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show generation history",
    )
    parser.add_argument(
        "--history-count",
        type=int,
        default=10,
        metavar="N",
        help="Number of history entries to show (default: 10)",
    )
    parser.add_argument(
        "--rerun",
        type=int,
        metavar="N",
        help="Regenerate from history entry N",
    )
    parser.add_argument(
        "--refine",
        metavar="INSTRUCTION",
        help="Refine an existing file with instructions",
    )
    parser.add_argument(
        "--input",
        metavar="FILE",
        help="Input file for --refine mode",
    )
    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Clear all generation history",
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show model, language, and token info during generation",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated file in the default application",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Auto-install detected third-party dependencies (Python only)",
    )
    parser.add_argument(
        "--git-commit",
        action="store_true",
        help="Automatically commit the generated file to git (if in a git repo)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config()

    # Apply config defaults where CLI args aren't provided
    if args.model is None:
        args.model = config.get("default_model", DEFAULT_MODEL)
    if args.temp is None:
        args.temp = config.get("temperature", 0.3)
    if args.formatter is None:
        args.formatter = config.get("formatter")

    if args.list_langs:
        print(_bold("Supported languages:"))
        for lang in sorted(LANGUAGE_MAP.keys()):
            ext = LANGUAGE_MAP[lang]
            print(f"  {lang:15s} → {ext}")
        return 0

    if args.history:
        _show_history(args.history_count)
        return 0

    if args.clear_history:
        clear_history()
        _print_success("History cleared.")
        return 0

    if args.refine:
        if not args.input:
            _print_error("--input FILE is required with --refine")
            return 1
        if not os.path.exists(args.input):
            _print_error(f"File not found: {args.input}")
            return 1
        if not os.environ.get("OPENAI_API_KEY"):
            _print_error("OPENAI_API_KEY is not set.")
            return 1

        code = open(args.input).read()
        lang = args.language or detect_language(code)
        print(f"{_cyan('Refining')} {args.input} ({lang})...")
        try:
            new_code = generate_code(
                goal="",
                language=lang,
                model=args.model,
                temperature=args.temp,
                refinement=args.refine,
                existing_code=code,
            )
            if _write_file(args.input, new_code):
                _print_success(f"Refined and saved to {args.input}")
            return 0
        except Exception as e:
            _print_error(str(e))
            return 1

    if args.rerun:
        entry = _rerun_from_history(args.rerun)
        if not entry:
            return 1
        if not os.environ.get("OPENAI_API_KEY"):
            _print_error("OPENAI_API_KEY is not set.")
            return 1

        lang = args.language or entry.get("language")
        model = entry.get("model", args.model)
        goal = entry.get("goal", "")
        output_path = args.output or entry.get("output_file") or f"output{get_extension(lang)}"

        print(f"{_cyan('Regenerating')} from history #{args.rerun}...")
        _print_header(goal, lang)

        full_code = ""
        import time as _time_module
        start_time = _time_module.time()
        last_spin = 0
        try:
            for chunk, done in generate_code_stream(
                goal=goal,
                language=lang,
                model=model,
                temperature=args.temp,
                skip_validation=args.no_validate,
            ):
                if chunk:
                    _clear_spinner()
                    print(chunk, end="", flush=True)
                    full_code += chunk
                    now = _time_module.time()
                    if now - last_spin > 0.1:
                        elapsed = int(now - start_time)
                        _print_spinner(f"Regenerating ({elapsed}s)")
                        last_spin = now
                if done:
                    _clear_spinner()
                    break
        except CodeGenerationError as e:
            _print_error(str(e))
            return 1

        print()
        if not full_code.strip():
            _print_error("No code was generated.")
            return 1

        if _write_file(output_path, full_code):
            _print_success(f"Saved to {output_path}")
            line_count = full_code.count("\n") + 1
            char_count = len(full_code)
            print(f"  {line_count} lines, {char_count} chars")
            add_entry(goal, lang, model, full_code, output_path, args.temp)
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
        print(f"{_bold('Language:')} {get_language_name(lang_display)}")
        print(f"{_bold('Extension:')} {ext}")
        print(f"{_bold('Output:')} {output}")
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

    # Verbose info
    if args.verbose:
        print(f"{_bold('Model:')} {args.model}")
        print(f"{_bold('Language:')} {language}")
        print(f"{_bold('Temperature:')} {args.temp}")
        print()

    # Stream and display with spinner
    full_code = ""
    validation_ok = True
    import time as _time_module
    start_time = _time_module.time()
    last_spin = 0
    try:
        for chunk, done in generate_code_stream(
            goal=args.goal,
            language=language,
            model=args.model,
            temperature=args.temp,
            skip_validation=args.no_validate,
        ):
            if chunk:
                _clear_spinner()
                # Colorize language header
                lang_col = _language_color(language)
                print(chunk, end="", flush=True)
                full_code += chunk
                # Update spinner every 0.1s
                now = _time_module.time()
                if now - last_spin > 0.1:
                    elapsed = int(now - start_time)
                    _print_spinner(f"Generating ({elapsed}s)")
                    last_spin = now
            if done:
                _clear_spinner()
                break
    except KeyboardInterrupt:
        _clear_spinner()
        print(f"\n{_yellow('Interrupted by user.')}")
        if full_code:
            save = input(f"\nSave partial output to {output_path}? [y/N]: ").strip().lower()
            if save == "y":
                if _write_file(output_path, full_code):
                    _print_success(f"Saved partial output to {output_path}")
        return 130

    print()

    elapsed = _time_module.time() - start_time
    if not full_code.strip():
        _print_error("No code was generated. Please try again.")
        return 1

    # Format
    if args.formatter and args.formatter != "none":
        lang_key = language.lower().split()[0]
        if lang_key == "python":
            formatted = _format_code(full_code, args.formatter)
            if formatted != full_code:
                print(f"{_cyan('Formatted with')} {args.formatter}{_cyan(':')}")
                full_code = formatted

    # Write to file
    if _write_file(output_path, full_code):
        _print_success(f"Saved to {output_path}")
        line_count = full_code.count("\n") + 1
        char_count = len(full_code)
        print(f"  {line_count} lines, {char_count} chars, {elapsed:.1f}s")

        # Verbose: show detected dependencies
        if args.verbose:
            imports = detect_imports(full_code)
            if imports:
                print(f"  {_bold('Dependencies:')} {', '.join(imports)}")

        # Auto-install dependencies
        if args.install:
            lang_key = language.lower().split()[0]
            if lang_key in ("python", "py"):
                installed = _install_dependencies(full_code, language)
                if installed:
                    print(f"  {_green('Installed:')} {', '.join(installed)}")
                else:
                    print(f"  {_yellow('No third-party dependencies detected')}")

    # Git commit
    if args.git_commit:
        if _is_in_git_repo(output_path):
            if _git_commit_file(output_path, args.goal):
                _print_success("Committed to git.")
            else:
                _print_warning("Git commit failed (check git status).")
        else:
            _print_warning("Not in a git repository. Skipping commit.")

    # Auto-open
    if args.open:
        if _open_file(output_path):
            print(f"  {_cyan('Opened')} {output_path}")
        else:
            _print_warning(f"Could not open {output_path}")

    # Record history
    entry_idx = add_entry(args.goal, language, args.model, full_code, output_path, args.temp)
    print(f"  {args.output or output_path} #{entry_idx}")

    # Interactive refinement
    if sys.stdin.isatty():
        new_code = _interactive_refine(full_code, language, args.model, args.temp, output_path)
        if new_code and new_code != full_code:
            if _write_file(output_path, new_code):
                _print_success(f"Updated: {output_path}")
                add_entry(args.goal + " [refined]", language, args.model, new_code, output_path, args.temp)

    return 0


if __name__ == "__main__":
    sys.exit(main())
