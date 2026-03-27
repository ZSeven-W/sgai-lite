"""Config file support for sgai-lite."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".sgai-lite"
CONFIG_FILES = [
    CONFIG_DIR / "config.json",
    CONFIG_DIR / "config.yaml",
    CONFIG_DIR / ".sgai-lite.json",
    CONFIG_DIR / ".sgai-lite.yaml",
]

# Defaults
DEFAULTS: dict[str, Any] = {
    "default_model": "gpt-4o",
    "default_language": None,
    "temperature": 0.3,
    "validate": True,
    "formatter": None,  # "black", "ruff", etc.
}


def load_config() -> dict[str, Any]:
    """Load config from file, merging with defaults."""
    config = DEFAULTS.copy()

    for cf_path in CONFIG_FILES:
        if cf_path.exists():
            try:
                if cf_path.suffix == ".yaml" or cf_path.suffix == ".yml":
                    import yaml  # optional dependency
                    with open(cf_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                else:
                    with open(cf_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                if data:
                    config.update(data)

                # Restrict to user-only access for security
                os.chmod(cf_path, stat.S_IRUSR | stat.S_IWUSR)
                break
            except Exception:
                pass

    return config


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a specific config value."""
    cfg = load_config()
    return cfg.get(key, default)
