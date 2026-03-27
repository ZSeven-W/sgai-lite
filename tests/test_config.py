"""Tests for config module."""

import pytest
import json
import tempfile
from pathlib import Path


class TestConfigDefaults:
    def test_defaults_defined(self):
        from sgai_lite.config import DEFAULTS
        assert "default_model" in DEFAULTS
        assert "temperature" in DEFAULTS
        assert "validate" in DEFAULTS
        assert DEFAULTS["default_model"] == "gpt-4o"
        assert DEFAULTS["temperature"] == 0.3

    def test_get_config_value(self):
        from sgai_lite.config import get_config_value
        val = get_config_value("nonexistent_key", default="fallback")
        assert val == "fallback"

        # temperature exists in defaults, so returns the default value 0.3
        val = get_config_value("temperature", default=0.9)
        assert val == 0.3  # from DEFAULTS, not the passed-in default


class TestConfigLoad:
    def test_load_config_returns_dict(self):
        from sgai_lite.config import load_config
        config = load_config()
        assert isinstance(config, dict)
        assert "default_model" in config
        assert "temperature" in config

    def test_config_file_json(self, monkeypatch):
        import sgai_lite.config as config_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_config = Path(tmpdir) / "config.json"
            fake_config.write_text(json.dumps({
                "default_model": "gpt-4o-mini",
                "temperature": 0.7,
            }), encoding="utf-8")
            monkeypatch.setattr(config_mod, "CONFIG_FILES", [fake_config])

            cfg = config_mod.load_config()
            assert cfg["default_model"] == "gpt-4o-mini"
            assert cfg["temperature"] == 0.7

    def test_missing_config_falls_back_to_defaults(self):
        from sgai_lite.config import load_config, DEFAULTS
        cfg = load_config()
        assert cfg.get("default_model") == DEFAULTS["default_model"]
