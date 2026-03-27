"""Tests for CLI argument parsing and commands."""

import pytest
from unittest.mock import patch, MagicMock
from io import StringIO
import sys


class TestCLIImports:
    def test_cli_module_imports(self):
        from sgai_lite.cli import main
        assert callable(main)

    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, "argv", ["sgai", "--version"]):
                from sgai_lite.cli import main
                main()
        assert exc_info.value.code == 0

    def test_list_langs_flag(self, capsys):
        with patch.object(sys, "argv", ["sgai", "--list-langs"]):
            from sgai_lite.cli import main
            rc = main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "python" in out.lower()
        assert ".py" in out

    def test_no_goal_shows_help(self, capsys):
        with patch.object(sys, "argv", ["sgai"]):
            from sgai_lite.cli import main
            rc = main()
        assert rc == 1
        err = capsys.readouterr().err
        assert "No goal" in err or "goal" in err.lower()

    def test_dry_run_detects_language(self, capsys):
        with patch.object(sys, "argv", ["sgai", "--dry-run", "a Python script"]):
            from sgai_lite.cli import main
            rc = main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "Python" in out

    def test_dry_run_with_lang(self, capsys):
        with patch.object(sys, "argv", ["sgai", "--dry-run", "--lang", "go", "a server"]):
            from sgai_lite.cli import main
            rc = main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "Go" in out

    def test_history_flag(self, capsys):
        with patch.object(sys, "argv", ["sgai", "--history"]):
            from sgai_lite.cli import main
            rc = main()
        assert rc == 0
        out = capsys.readouterr().out
        # Should show history (may be empty)
        assert "generation" in out.lower() or "history" in out.lower() or "No history" in out

    def test_clear_history_flag(self, capsys):
        with patch.object(sys, "argv", ["sgai", "--clear-history"]):
            from sgai_lite.cli import main
            rc = main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "cleared" in out.lower() or "✓" in out


class TestCLIHistory:
    def test_rerun_nonexistent_shows_error(self, capsys):
        with patch.object(sys, "argv", ["sgai", "--rerun", "999"]):
            from sgai_lite.cli import main
            rc = main()
        assert rc == 1
        err = capsys.readouterr().err
        assert "not found" in err.lower()


class TestCLIRefine:
    def test_refine_without_input_shows_error(self, capsys):
        with patch.object(sys, "argv", ["sgai", "--refine", "add tests"]):
            from sgai_lite.cli import main
            rc = main()
        assert rc == 1
        err = capsys.readouterr().err
        assert "--input" in err


class TestCLIIntegration:
    def test_cli_returns_integer(self):
        with patch.object(sys, "argv", ["sgai", "--version"]):
            from sgai_lite.cli import main
            try:
                rc = main()
            except SystemExit as e:
                rc = e.code
        assert isinstance(rc, int)
        assert rc == 0
