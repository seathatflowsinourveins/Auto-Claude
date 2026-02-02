#!/usr/bin/env python3
"""
Integration Tests - Explicit Error Handling
Part of V33 Architecture - Phase 9 Production Fix.

Tests that all V33 layers properly raise SDKNotAvailableError and
SDKConfigurationError instead of returning stubs or silent fallbacks.
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch


# =============================================================================
# Test Observability Layer
# =============================================================================

class TestObservabilityExplicitErrors:
    """Test that observability layer raises explicit errors."""

    def test_sdk_not_available_error_exists(self):
        """Test that SDKNotAvailableError is properly exported."""
        from core.observability import SDKNotAvailableError
        assert SDKNotAvailableError is not None

    def test_sdk_configuration_error_exists(self):
        """Test that SDKConfigurationError is properly exported."""
        from core.observability import SDKConfigurationError
        assert SDKConfigurationError is not None

    def test_sdk_not_available_error_has_install_cmd(self):
        """Test that SDKNotAvailableError includes install instructions."""
        from core.observability import SDKNotAvailableError

        error = SDKNotAvailableError(
            sdk_name="test_sdk",
            install_cmd="pip install test-sdk>=1.0.0",
            docs_url="https://test-sdk.dev/"
        )

        assert "test_sdk" in str(error)
        assert "pip install" in str(error)
        assert error.install_cmd == "pip install test-sdk>=1.0.0"
        assert error.docs_url == "https://test-sdk.dev/"

    def test_sdk_configuration_error_has_missing_config(self):
        """Test that SDKConfigurationError includes missing config list."""
        from core.observability import SDKConfigurationError

        error = SDKConfigurationError(
            sdk_name="test_sdk",
            missing_config=["API_KEY", "SECRET_KEY"],
            example="API_KEY=xxx"
        )

        assert "test_sdk" in str(error)
        assert "API_KEY" in str(error)
        assert error.missing_config == ["API_KEY", "SECRET_KEY"]

    def test_langfuse_getter_raises_on_unavailable(self):
        """Test that get_langfuse_tracer raises when langfuse not installed."""
        from core.observability import (
            get_langfuse_tracer,
            LANGFUSE_AVAILABLE,
            SDKNotAvailableError,
        )

        if not LANGFUSE_AVAILABLE:
            with pytest.raises(SDKNotAvailableError) as exc_info:
                get_langfuse_tracer()

            assert "langfuse" in str(exc_info.value)
            assert "pip install" in str(exc_info.value)

    def test_langfuse_getter_raises_on_missing_config(self):
        """Test that get_langfuse_tracer raises when config missing."""
        from core.observability import (
            get_langfuse_tracer,
            LANGFUSE_AVAILABLE,
            SDKConfigurationError,
        )

        if LANGFUSE_AVAILABLE:
            # Clear config to test
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(SDKConfigurationError) as exc_info:
                    get_langfuse_tracer()

                assert "LANGFUSE_PUBLIC_KEY" in str(exc_info.value)

    def test_get_available_sdks_returns_dict(self):
        """Test that get_available_sdks returns proper dictionary."""
        from core.observability import get_available_sdks

        status = get_available_sdks()
        assert isinstance(status, dict)
        assert "langfuse" in status
        assert "phoenix" in status
        assert all(isinstance(v, bool) for v in status.values())


# =============================================================================
# Test Memory Layer
# =============================================================================

class TestMemoryExplicitErrors:
    """Test that memory layer raises explicit errors."""

    def test_sdk_not_available_error_re_exported(self):
        """Test that SDKNotAvailableError is re-exported from memory."""
        from core.memory import SDKNotAvailableError
        assert SDKNotAvailableError is not None

    def test_mem0_getter_raises_on_unavailable(self):
        """Test that get_mem0_client raises when mem0 not installed."""
        from core.memory import (
            get_mem0_client,
            MEM0_AVAILABLE,
            SDKNotAvailableError,
        )

        if not MEM0_AVAILABLE:
            with pytest.raises(SDKNotAvailableError) as exc_info:
                get_mem0_client()

            assert "mem0" in str(exc_info.value)
            assert "pip install" in str(exc_info.value)

    def test_zep_getter_raises_on_missing_config(self):
        """Test that get_zep_client raises when API key missing."""
        from core.memory import (
            get_zep_client,
            ZEP_AVAILABLE,
            SDKConfigurationError,
        )

        if ZEP_AVAILABLE:
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(SDKConfigurationError) as exc_info:
                    get_zep_client()

                assert "ZEP_API_KEY" in str(exc_info.value)

    def test_cross_session_raises_on_unavailable(self):
        """Test that get_cross_session_context raises explicitly."""
        from core.memory import (
            get_cross_session_context,
            CROSS_SESSION_AVAILABLE,
            SDKNotAvailableError,
        )

        if not CROSS_SESSION_AVAILABLE:
            with pytest.raises(SDKNotAvailableError) as exc_info:
                get_cross_session_context()

            # Should NOT return a stub string like "Cross-session memory not available"
            assert "cross_session" in str(exc_info.value).lower()


# =============================================================================
# Test Orchestration Layer
# =============================================================================

class TestOrchestrationExplicitErrors:
    """Test that orchestration layer raises explicit errors."""

    def test_sdk_not_available_error_re_exported(self):
        """Test that SDKNotAvailableError is re-exported from orchestration."""
        from core.orchestration import SDKNotAvailableError
        assert SDKNotAvailableError is not None

    def test_temporal_getter_raises_on_unavailable(self):
        """Test that get_temporal_orchestrator raises when temporal not installed."""
        from core.orchestration import (
            get_temporal_orchestrator,
            TEMPORAL_AVAILABLE,
            SDKNotAvailableError,
        )

        if not TEMPORAL_AVAILABLE:
            with pytest.raises(SDKNotAvailableError) as exc_info:
                get_temporal_orchestrator()

            assert "temporal" in str(exc_info.value).lower()

    def test_claude_flow_getter_raises_on_missing_config(self):
        """Test that get_claude_flow_orchestrator raises when API key missing."""
        from core.orchestration import (
            get_claude_flow_orchestrator,
            CLAUDE_FLOW_AVAILABLE,
            SDKConfigurationError,
        )

        if CLAUDE_FLOW_AVAILABLE:
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(SDKConfigurationError) as exc_info:
                    get_claude_flow_orchestrator()

                assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_getter_functions_exported(self):
        """Test that all getter functions are properly exported."""
        from core.orchestration import (
            get_temporal_orchestrator,
            get_langgraph_orchestrator,
            get_claude_flow_orchestrator,
            get_crewai_manager,
            get_autogen_orchestrator,
        )

        assert callable(get_temporal_orchestrator)
        assert callable(get_langgraph_orchestrator)
        assert callable(get_claude_flow_orchestrator)
        assert callable(get_crewai_manager)
        assert callable(get_autogen_orchestrator)


# =============================================================================
# Test Structured Layer
# =============================================================================

class TestStructuredExplicitErrors:
    """Test that structured output layer raises explicit errors."""

    def test_sdk_not_available_error_re_exported(self):
        """Test that SDKNotAvailableError is re-exported from structured."""
        from core.structured import SDKNotAvailableError
        assert SDKNotAvailableError is not None

    def test_instructor_getter_raises_on_unavailable(self):
        """Test that get_instructor_client raises when instructor not installed."""
        from core.structured import (
            get_instructor_client,
            INSTRUCTOR_AVAILABLE,
            SDKNotAvailableError,
        )

        if not INSTRUCTOR_AVAILABLE:
            with pytest.raises(SDKNotAvailableError) as exc_info:
                get_instructor_client()

            assert "instructor" in str(exc_info.value)

    def test_instructor_getter_raises_on_missing_config(self):
        """Test that get_instructor_client raises when API key missing."""
        from core.structured import (
            get_instructor_client,
            INSTRUCTOR_AVAILABLE,
            SDKConfigurationError,
        )

        if INSTRUCTOR_AVAILABLE:
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(SDKConfigurationError) as exc_info:
                    get_instructor_client(provider="anthropic")

                assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_getter_functions_exported(self):
        """Test that all getter functions are properly exported."""
        from core.structured import (
            get_instructor_client,
            get_baml_client,
            get_outlines_generator,
            get_pydantic_agent,
        )

        assert callable(get_instructor_client)
        assert callable(get_baml_client)
        assert callable(get_outlines_generator)
        assert callable(get_pydantic_agent)


# =============================================================================
# Test Validator
# =============================================================================

class TestValidator:
    """Test the SDK validator module."""

    def test_validate_all_returns_result(self):
        """Test that validate_all returns a proper result."""
        from core.validator import validate_all, FullValidationResult

        result = validate_all()
        assert isinstance(result, FullValidationResult)
        assert result.total_sdks > 0
        assert result.status is not None

    def test_validate_layer_returns_result(self):
        """Test that validate_layer returns a proper result."""
        from core.validator import validate_layer, LayerValidationResult

        for layer in ["observability", "memory", "orchestration", "structured"]:
            result = validate_layer(layer)
            assert isinstance(result, LayerValidationResult)
            assert result.layer == layer
            assert result.total > 0

    def test_validate_sdk_returns_result(self):
        """Test that validate_sdk returns a proper result."""
        from core.validator import validate_sdk, SDKValidationResult

        result = validate_sdk("langfuse", "observability")
        assert isinstance(result, SDKValidationResult)
        assert result.name == "langfuse"
        assert isinstance(result.available, bool)
        assert isinstance(result.configured, bool)

    def test_result_to_dict_is_serializable(self):
        """Test that validation result can be converted to dict."""
        import json
        from core.validator import validate_all

        result = validate_all()
        result_dict = result.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        assert isinstance(json_str, str)

        # Should contain expected keys
        parsed = json.loads(json_str)
        assert "status" in parsed
        assert "layers" in parsed
        assert "total_sdks" in parsed


# =============================================================================
# Test No Stubs Pattern
# =============================================================================

class TestNoStubsPattern:
    """Test that no stub patterns exist in the codebase."""

    def test_observability_no_stub_class(self):
        """Test that observability doesn't have stub classes."""
        import core.observability as obs

        # Check that there are no "Stub" or "Fallback" classes
        module_attrs = dir(obs)
        stub_classes = [
            attr for attr in module_attrs
            if "Stub" in attr or "Fallback" in attr or "Mock" in attr
        ]
        assert len(stub_classes) == 0, f"Found stub classes: {stub_classes}"

    def test_memory_no_stub_class(self):
        """Test that memory doesn't have stub classes."""
        import core.memory as mem

        module_attrs = dir(mem)
        stub_classes = [
            attr for attr in module_attrs
            if "Stub" in attr or "Fallback" in attr or "Mock" in attr
        ]
        assert len(stub_classes) == 0, f"Found stub classes: {stub_classes}"

    def test_orchestration_no_stub_class(self):
        """Test that orchestration doesn't have stub classes."""
        import core.orchestration as orch

        module_attrs = dir(orch)
        stub_classes = [
            attr for attr in module_attrs
            if "Stub" in attr or "Fallback" in attr or "Mock" in attr
        ]
        assert len(stub_classes) == 0, f"Found stub classes: {stub_classes}"

    def test_structured_no_stub_class(self):
        """Test that structured doesn't have stub classes."""
        import core.structured as struct

        module_attrs = dir(struct)
        stub_classes = [
            attr for attr in module_attrs
            if "Stub" in attr or "Fallback" in attr or "Mock" in attr
        ]
        assert len(stub_classes) == 0, f"Found stub classes: {stub_classes}"


# =============================================================================
# Test Cross-Layer Consistency
# =============================================================================

class TestCrossLayerConsistency:
    """Test that error patterns are consistent across layers."""

    def test_all_layers_export_sdk_not_available_error(self):
        """Test that all layers export SDKNotAvailableError."""
        from core.observability import SDKNotAvailableError as ObsError
        from core.memory import SDKNotAvailableError as MemError
        from core.orchestration import SDKNotAvailableError as OrchError
        from core.structured import SDKNotAvailableError as StructError

        # All should be the same class (re-exported from observability)
        assert ObsError is MemError
        assert MemError is OrchError
        assert OrchError is StructError

    def test_all_layers_export_sdk_configuration_error(self):
        """Test that all layers export SDKConfigurationError."""
        from core.observability import SDKConfigurationError as ObsError
        from core.memory import SDKConfigurationError as MemError
        from core.orchestration import SDKConfigurationError as OrchError
        from core.structured import SDKConfigurationError as StructError

        # All should be the same class
        assert ObsError is MemError
        assert MemError is OrchError
        assert OrchError is StructError

    def test_all_layers_have_availability_flags(self):
        """Test that all layers expose availability flags."""
        # Observability
        from core.observability import LANGFUSE_AVAILABLE, PHOENIX_AVAILABLE

        # Memory
        from core.memory import MEM0_AVAILABLE, ZEP_AVAILABLE

        # Orchestration
        from core.orchestration import TEMPORAL_AVAILABLE, LANGGRAPH_AVAILABLE

        # Structured
        from core.structured import INSTRUCTOR_AVAILABLE, BAML_AVAILABLE

        # All should be booleans
        assert all(isinstance(flag, bool) for flag in [
            LANGFUSE_AVAILABLE, PHOENIX_AVAILABLE,
            MEM0_AVAILABLE, ZEP_AVAILABLE,
            TEMPORAL_AVAILABLE, LANGGRAPH_AVAILABLE,
            INSTRUCTOR_AVAILABLE, BAML_AVAILABLE,
        ])
