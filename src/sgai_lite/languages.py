"""Language detection and file extension mapping."""

from __future__ import annotations

LANGUAGE_MAP: dict[str, str] = {
    "python": ".py",
    "py": ".py",
    "javascript": ".js",
    "js": ".js",
    "typescript": ".ts",
    "ts": ".ts",
    "bash": ".sh",
    "shell": ".sh",
    "sh": ".sh",
    "zsh": ".sh",
    "go": ".go",
    "golang": ".go",
    "rust": ".rs",
    "ruby": ".rb",
    "php": ".php",
    "java": ".java",
    "c": ".c",
    "cpp": ".cpp",
    "c++": ".cpp",
    "csharp": ".cs",
    "c#": ".cs",
    "swift": ".swift",
    "kotlin": ".kt",
    "scala": ".scala",
    "r": ".r",
    "lua": ".lua",
    "perl": ".pl",
    "haskell": ".hs",
    "elixir": ".ex",
    "clojure": ".clj",
    "dart": ".dart",
    "vue": ".vue",
    "svelte": ".svelte",
    "html": ".html",
    "css": ".css",
    "sql": ".sql",
    "yaml": ".yaml",
    "json": ".json",
    "toml": ".toml",
    "dockerfile": "Dockerfile",
    "docker": "Dockerfile",
}

# Keywords that hint at the target language
LANGUAGE_HINTS: dict[str, list[str]] = {
    "python": ["python", "pip", "pypi", "pygame", "django", "flask", "fastapi", "pandas", "numpy"],
    "javascript": ["javascript", "node", "nodejs", "npm", "express", "react", "webpack"],
    "typescript": ["typescript", "ts-node", "angular", "nestjs"],
    "bash": ["bash", "shell", "sh script", "zsh", "unix", "linux command"],
    "go": ["golang", " go ", "go program"],
    "rust": ["rust", "cargo"],
    "ruby": ["ruby", "gem ", "rails"],
    "php": ["php"],
    "java": ["java", "jvm", "spring"],
}

DEFAULT_EXTENSION = ".py"


def detect_language(goal: str) -> str:
    """Detect the programming language from a goal description."""
    goal_lower = goal.lower()
    for lang, keywords in LANGUAGE_HINTS.items():
        for keyword in keywords:
            if keyword in goal_lower:
                return lang
    return "python"


def get_extension(lang: str) -> str:
    """Get file extension for a language."""
    return LANGUAGE_MAP.get(lang.lower(), DEFAULT_EXTENSION)


def get_language_name(lang: str) -> str:
    """Get a human-readable language name."""
    names = {
        "py": "Python",
        "python": "Python",
        "js": "JavaScript",
        "javascript": "JavaScript",
        "ts": "TypeScript",
        "typescript": "TypeScript",
        "sh": "Bash",
        "bash": "Bash",
        "shell": "Shell",
        "go": "Go",
        "golang": "Go",
        "rs": "Rust",
        "rust": "Rust",
        "rb": "Ruby",
        "ruby": "Ruby",
        "php": "PHP",
        "java": "Java",
    }
    return names.get(lang.lower(), lang.capitalize())
