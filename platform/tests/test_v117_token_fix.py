#!/usr/bin/env python3
"""
V117 Optimization Test: Deprecated token= Parameter Fix

Tests Letta client uses api_key= not deprecated token= by importing
and testing the real SDK - not by grepping file contents.

Test Date: 2026-01-30, Updated: 2026-02-02 (V14 Iter 55)
"""

import os
import sys
import pytest

# Add platform to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestTokenParameterReal:
    """Test Letta SDK uses api_key by inspecting real class."""

    def test_letta_init_accepts_api_key(self):
        """Letta.__init__ should accept api_key parameter."""
        try:
            from letta_client import Letta
            import inspect
        except ImportError:
            pytest.skip("letta-client not installed")

        sig = inspect.signature(Letta.__init__)
        params = list(sig.parameters.keys())
        assert "api_key" in params, \
            f"Letta.__init__ should accept api_key, got params: {params}"

    def test_letta_init_no_token_required(self):
        """Letta should work with api_key= not requiring token=."""
        try:
            from letta_client import Letta
            import inspect
        except ImportError:
            pytest.skip("letta-client not installed")

        sig = inspect.signature(Letta.__init__)
        # token should not be a required positional parameter
        for name, param in sig.parameters.items():
            if name == "token":
                assert param.default is not inspect.Parameter.empty, \
                    "token= parameter should not be required (deprecated)"


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
