"""Tests for generator features: retry logic, dependency detection, cost estimation."""

import pytest
from unittest.mock import patch, MagicMock
from openai import RateLimitError, APIError, AuthenticationError

from sgai_lite.generator import (
    detect_imports,
    estimate_cost,
    CodeGenerationError,
    APIKeyMissingError,
    generate_code,
)


class TestDetectImports:
    """Tests for detect_imports function."""

    def test_detect_third_party(self):
        code = """
import requests
from bs4 import BeautifulSoup
import click
"""
        result = detect_imports(code)
        assert "requests" in result
        assert "bs4" in result  # BeautifulSoup module name is bs4
        assert "click" in result

    def test_ignore_stdlib(self):
        code = """
import os
import sys
import json
from pathlib import Path
import typing
"""
        result = detect_imports(code)
        assert len(result) == 0

    def test_dotted_imports(self):
        code = """
import os.path
import xml.etree.ElementTree
from collections import defaultdict
"""
        result = detect_imports(code)
        assert "os" in result or "pathlib" in result or len(result) == 0

    def test_mixed_imports(self):
        code = """
import os
import requests
from bs4 import BeautifulSoup
import datetime
"""
        result = detect_imports(code)
        assert "requests" in result
        assert "bs4" in result
        assert "os" not in result
        assert "datetime" not in result

    def test_empty_code(self):
        assert detect_imports("") == []
        assert detect_imports("# no imports") == []

    def test_from_import(self):
        code = "from rich.console import Console"
        result = detect_imports(code)
        assert "rich" in result

    def test_common_packages(self):
        code = """
import pandas as pd
import numpy as np
import fastapi
import click
import typer
"""
        result = detect_imports(code)
        assert "pandas" in result
        assert "numpy" in result
        assert "fastapi" in result
        assert "click" in result
        assert "typer" in result


class TestEstimateCost:
    """Tests for estimate_cost function."""

    def test_gpt4o_pricing(self):
        cost = estimate_cost("gpt-4o", 1000, 2000)
        # gpt-4o: $2.5/1M input, $10/1M output
        expected = (1000 / 1_000_000 * 2.5) + (2000 / 1_000_000 * 10.0)
        assert abs(cost - expected) < 0.001

    def test_gpt4o_mini_pricing(self):
        cost = estimate_cost("gpt-4o-mini", 1000, 2000)
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        expected = (1000 / 1_000_000 * 0.15) + (2000 / 1_000_000 * 0.60)
        assert abs(cost - expected) < 0.001

    def test_unknown_model_default(self):
        cost = estimate_cost("unknown-model", 1000, 2000)
        # Should use default gpt-4o pricing
        assert cost > 0


class TestRetryLogic:
    """Tests for retry logic in generate_code."""

    def test_rate_limit_retries(self):
        """Should retry on RateLimitError up to 3 times."""
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "def hello(): pass\n"
        mock_response = MagicMock()
        mock_response.choices = [choice]

        mock_client.chat.completions.create.side_effect = [
            RateLimitError("rate limited", response=MagicMock(), body=None),
            RateLimitError("rate limited", response=MagicMock(), body=None),
            mock_response,
        ]

        with patch("sgai_lite.generator.get_client", return_value=mock_client):
            with patch("sgai_lite.generator.time.sleep"):  # skip delays
                result = generate_code(
                    goal="hello world",
                    language="python",
                    model="gpt-4o",
                )
        assert "hello" in result.lower()

    def test_auth_error_no_retry(self):
        """Should raise immediately on AuthenticationError."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            "auth failed", response=MagicMock(), body=None
        )

        with patch("sgai_lite.generator.get_client", return_value=mock_client):
            with pytest.raises(APIKeyMissingError):
                generate_code(goal="hello", language="python")

    def test_api_error_retries(self):
        """Should retry on generic APIError up to 3 times."""
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "def foo(): pass\n"
        mock_response = MagicMock()
        mock_response.choices = [choice]

        mock_client.chat.completions.create.side_effect = [
            APIError("api error", request=MagicMock(), body=None),
            APIError("api error", request=MagicMock(), body=None),
            mock_response,
        ]

        with patch("sgai_lite.generator.get_client", return_value=mock_client):
            with patch("sgai_lite.generator.time.sleep"):
                result = generate_code(goal="foo function", language="python")
        assert "foo" in result.lower()
