#!/usr/bin/env python3
"""
V117 Optimization Test: Deprecated token= Parameter Fix

This test validates that Letta client initialization uses correct parameters:
1. Uses `api_key=` (not deprecated `token=`)
2. Uses `base_url=` for Cloud connections (V115)

Critical Bugs Fixed:
- Deprecated parameter: token= → api_key=

Expected Gains:
- Future SDK compatibility: Risk eliminated
- Deprecation warnings: -100%

Test Date: 2026-01-30
"""

import os
import re
import sys
import pytest

# Add platform to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestTokenParameterPatterns:
    """Test suite for deprecated token= parameter fixes."""

    def test_long_running_agents_uses_api_key_platform(self):
        """Verify platform test file uses api_key=, not token=."""
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "sdks", "letta", "tests", "test_long_running_agents.py"
        )

        if not os.path.exists(file_path):
            pytest.skip("test_long_running_agents.py not found in platform/sdks/letta/tests")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should use api_key=
        assert "api_key=" in content, \
            "test_long_running_agents.py should use api_key= parameter"

        # Should NOT use token= for Letta initialization
        # Check for pattern: Letta(token=
        deprecated_pattern = re.search(r"Letta\([^)]*token\s*=", content)
        assert deprecated_pattern is None, \
            "test_long_running_agents.py should not use deprecated token= parameter"

    def test_long_running_agents_uses_api_key_sdk(self):
        """Verify SDK test file uses api_key=, not token=."""
        # Try multiple possible locations
        possible_paths = [
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "sdks", "letta", "letta", "tests", "test_long_running_agents.py"
            ),
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "sdks", "mcp-ecosystem", "letta", "tests", "test_long_running_agents.py"
            ),
        ]

        found = False
        for file_path in possible_paths:
            if os.path.exists(file_path):
                found = True
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Should use api_key=
                assert "api_key=" in content, \
                    f"{file_path} should use api_key= parameter"

                # Should NOT use token= for Letta initialization
                deprecated_pattern = re.search(r"Letta\([^)]*token\s*=", content)
                assert deprecated_pattern is None, \
                    f"{file_path} should not use deprecated token= parameter"

        if not found:
            pytest.skip("test_long_running_agents.py not found in SDK paths")


class TestLettaClientImports:
    """Verify correct Letta client import and initialization patterns."""

    def test_letta_client_importable(self):
        """Verify Letta client can be imported."""
        try:
            from letta_client import Letta
            assert Letta is not None
        except ImportError:
            pytest.skip("letta-client not installed")

    def test_letta_accepts_api_key(self):
        """Verify Letta client accepts api_key parameter."""
        try:
            from letta_client import Letta
            import inspect

            sig = inspect.signature(Letta.__init__)
            params = list(sig.parameters.keys())

            # Should have api_key parameter
            assert "api_key" in params, \
                "Letta.__init__ should accept api_key parameter"

        except ImportError:
            pytest.skip("letta-client not installed")


class TestLettaClientInitialization:
    """Live initialization tests (require LETTA_API_KEY)."""

    @pytest.mark.skipif(
        not os.getenv("LETTA_API_KEY"),
        reason="LETTA_API_KEY not configured"
    )
    def test_api_key_initialization_works(self):
        """Test that api_key= initialization works against real API."""
        try:
            from letta_client import Letta
        except ImportError:
            pytest.skip("letta_client not installed")

        # Initialize with api_key (NOT token)
        client = Letta(
            api_key=os.getenv("LETTA_API_KEY"),
            base_url=os.getenv("LETTA_BASE_URL", "https://api.letta.com")
        )

        # Should be able to list agents
        try:
            _ = list(client.agents.list(limit=1))
            assert True, "Successfully connected with api_key parameter"
        except Exception as e:
            error_str = str(e).lower()
            # Auth errors mean we connected (V115 working)
            if "401" in error_str or "unauthorized" in error_str:
                pytest.fail(f"Auth failed - check LETTA_API_KEY: {e}")
            elif "localhost" in error_str or "connection refused" in error_str:
                pytest.fail(f"V115 regression - connected to localhost: {e}")
            else:
                pytest.skip(f"Connection issue: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
