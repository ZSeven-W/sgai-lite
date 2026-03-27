"""System prompts for code generation."""

# ─── Specialized prompt templates by intent ───────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are an expert code generator. Generate complete, production-ready code in a single file based on the user's goal.

IMPORTANT RULES:
1. Output ONLY the code - no explanations, no markdown fences, no comments describing what you are doing
2. The code must be complete and runnable - include all imports, error handling, and dependencies
3. Include a main guard (if __name__ == "__main__":) for executable scripts
4. Use type hints where appropriate for Python and TypeScript
5. Make the code clean, well-structured, and follow best practices for the target language
6. For scripts: handle errors gracefully and provide useful output
7. If the goal mentions a specific framework/library, use it
8. Default dependencies: for Python use standard library where possible, for JS/TS use Node.js built-ins

Target language: {language}
Output file: {filename}

Generate the code now:"""


# ─── Intent-based specialized prompts ─────────────────────────────────────────

INTENT_KEYWORDS = {
    "cli": ["cli", "command line", "terminal", "argparse", "menu", "interactive menu"],
    "web": ["web", "http", "server", "api", "rest", "html", "css", "frontend", "backend", "flask", "fastapi", "express"],
    "data": ["data", "csv", "json", "parse", "process", "analysis", "pandas", "numpy", "filter", "transform"],
    "gui": ["gui", "window", "ui", "interface", "tkinter", "pyqt", "electron", "widget"],
    "script": ["script", "automation", "schedule", "cron", "backup"],
    "game": ["game", "play", "player", "score", "arcade", "puzzle"],
    "test": ["test", "mock", "fixture", "unittest", "pytest"],
}

LANG_SPECIFIC_TIPS = {
    "python": """Python-specific best practices:
- Use 'if __name__ == "__main__":' for entry point
- Prefer pathlib over os.path
- Use dataclasses or Pydantic models for structured data
- Include comprehensive docstrings
- Use 'argparse' or 'click' for CLI tools""",
    "javascript": """JavaScript/Node.js-specific best practices:
- Use async/await over raw promises
- Include proper error handling with try/catch
- Use modern ES6+ syntax
- Handle process exit codes properly""",
    "typescript": """TypeScript-specific best practices:
- Use strict typing throughout
- Define interfaces for data structures
- Use generics where appropriate
- Include JSDoc comments for complex logic""",
    "bash": """Bash-specific best practices:
- Use 'set -euo pipefail' at the top
- Define usage function for help text
- Check for required dependencies
- Use $() over backticks for command substitution""",
    "go": """Go-specific best practices:
- Use proper error handling (no panic unless fatal)
- Include comprehensive error messages
- Use struct tags for serialization
- Follow Go idioms and conventions""",
    "rust": """Rust-specific best practices:
- Use Result types for error handling
- Include proper lifetime annotations
- Write unit tests in the same file
- Use idiomatic Rust patterns""",
}


def build_system_prompt(language: str, filename: str, goal: str = "") -> str:
    """Build a specialized system prompt based on intent detection."""
    base = SYSTEM_PROMPT_TEMPLATE.format(language=language, filename=filename)

    if not goal:
        return base

    # Detect intent
    goal_lower = goal.lower()
    detected_intents = [
        intent for intent, keywords in INTENT_KEYWORDS.items()
        if any(kw in goal_lower for kw in keywords)
    ]

    if not detected_intents:
        return base

    # Append language-specific tips if available
    lang_key = language.lower()
    tips = ""
    for lk in [lang_key, lang_key.split()[0]]:
        if lk in LANG_SPECIFIC_TIPS:
            tips = LANG_SPECIFIC_TIPS[lk]
            break

    if tips:
        base += "\n\n" + tips

    return base
