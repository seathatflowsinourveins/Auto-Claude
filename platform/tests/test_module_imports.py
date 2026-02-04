#!/usr/bin/env python
"""
Module Import Validation Test
=============================

Validates that all major platform modules can be imported without errors.
Tests:
- platform.core.rag (all exports)
- platform.core.memory (all exports)
- platform.adapters (registry, key adapters)
- platform.core.health_check
- platform.core.logging_config

IMPORTANT: The 'platform' directory name conflicts with Python's stdlib 'platform' module.
This script handles this by directly loading modules from file paths.

Run with: python platform/tests/test_module_imports.py
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

# Get project root
SCRIPT_DIR = Path(__file__).parent.absolute()
PLATFORM_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PLATFORM_DIR.parent


@dataclass
class ImportResult:
    """Result of a single import attempt."""
    module_path: str
    success: bool
    error: Optional[str] = None
    traceback_info: Optional[str] = None
    loaded_items: List[str] = field(default_factory=list)
    missing_items: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Complete validation report."""
    results: List[ImportResult] = field(default_factory=list)
    successful_imports: int = 0
    failed_imports: int = 0
    missing_dependencies: Set[str] = field(default_factory=set)
    partial_success: int = 0

    def add_result(self, result: ImportResult) -> None:
        """Add a result and update counters."""
        self.results.append(result)
        if result.success:
            self.successful_imports += 1
            if result.missing_items:
                self.partial_success += 1
        else:
            self.failed_imports += 1
            # Track missing dependencies
            if result.error and "No module named" in result.error:
                dep = result.error.split("No module named")[-1].strip().strip("'\"")
                if dep:
                    self.missing_dependencies.add(dep)

    def print_report(self) -> None:
        """Print a formatted report."""
        print("\n" + "=" * 70)
        print("MODULE IMPORT VALIDATION REPORT")
        print("=" * 70)

        # Summary
        total = self.successful_imports + self.failed_imports
        print(f"\nSummary: {self.successful_imports}/{total} modules imported successfully")
        if self.partial_success > 0:
            print(f"         ({self.partial_success} with some items unavailable)")

        # Successful imports
        if self.successful_imports > 0:
            print("\n[SUCCESSFUL IMPORTS]")
            for result in self.results:
                if result.success:
                    items_str = f" ({len(result.loaded_items)} items loaded)"
                    print(f"  [OK] {result.module_path}{items_str}")
                    if result.missing_items:
                        print(f"       Missing items: {', '.join(result.missing_items[:5])}")
                        if len(result.missing_items) > 5:
                            print(f"       ... and {len(result.missing_items) - 5} more")

        # Failed imports
        if self.failed_imports > 0:
            print("\n[FAILED IMPORTS]")
            for result in self.results:
                if not result.success:
                    print(f"  [FAIL] {result.module_path}")
                    print(f"         Error: {result.error}")
                    if result.traceback_info:
                        # Print last few lines of traceback
                        lines = result.traceback_info.strip().split("\n")[-4:]
                        for line in lines:
                            if line.strip():
                                print(f"         {line}")

        # Missing dependencies
        if self.missing_dependencies:
            print("\n[MISSING DEPENDENCIES]")
            for dep in sorted(self.missing_dependencies):
                print(f"  - {dep}")
            print("\n  Install with: pip install <package_name>")

        print("\n" + "=" * 70)
        status = "PASS" if self.failed_imports == 0 else "FAIL"
        print(f"RESULT: {status}")
        print("=" * 70 + "\n")


def load_module_from_file(module_name: str, file_path: Path) -> Any:
    """
    Load a module directly from a file path.

    This bypasses the normal import system to avoid conflicts with
    the stdlib 'platform' module.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {file_path}")

    module = importlib.util.module_from_spec(spec)

    # Add to sys.modules BEFORE exec to handle internal imports
    old_module = sys.modules.get(module_name)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        # Restore old module if loading fails
        if old_module is not None:
            sys.modules[module_name] = old_module
        else:
            sys.modules.pop(module_name, None)
        raise


def setup_platform_package():
    """
    Set up the platform package hierarchy in sys.modules.
    This allows internal relative imports to work correctly.
    """
    import types

    # Create platform package if needed (avoiding conflict with stdlib)
    if "platform" in sys.modules and hasattr(sys.modules["platform"], "python_version"):
        # stdlib platform is loaded, we need to be careful
        # Create our package under a different internal name but alias it
        pass

    # Add project root to path
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def safe_import(
    display_name: str,
    file_path: Path,
    items: Optional[List[str]] = None
) -> ImportResult:
    """
    Safely attempt to import a module and optionally specific items.

    Args:
        display_name: Display name for the module (e.g., "platform.core.rag")
        file_path: Absolute path to the module __init__.py file
        items: Optional list of specific items to check for

    Returns:
        ImportResult with success status and any errors
    """
    result = ImportResult(module_path=display_name, success=False)

    if not file_path.exists():
        result.error = f"File not found: {file_path}"
        return result

    try:
        # Load the module
        module = load_module_from_file(display_name, file_path)
        result.success = True

        # Get __all__ if items not specified
        if items is None:
            items = getattr(module, "__all__", [])

        # Validate specific items can be accessed
        loaded = []
        missing = []
        for item in items:
            try:
                obj = getattr(module, item)
                if obj is not None:
                    loaded.append(item)
                else:
                    # Item exists but is None (optional import failed)
                    missing.append(item)
            except AttributeError:
                missing.append(item)

        result.loaded_items = loaded
        result.missing_items = missing

    except ImportError as e:
        result.error = str(e)
        result.traceback_info = traceback.format_exc()
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        result.traceback_info = traceback.format_exc()

    return result


def validate_core_module() -> ImportResult:
    """Validate platform/core/__init__.py module."""
    file_path = PLATFORM_DIR / "core" / "__init__.py"

    key_items = [
        # Memory
        "MemorySystem",
        "CoreMemory",
        # Cooperation
        "CooperationManager",
        # Executor
        "AgentExecutor",
        "create_executor",
        # Thinking
        "ThinkingEngine",
        # Orchestrator
        "Orchestrator",
        "create_orchestrator",
        # Resilience
        "CircuitBreaker",
        "RetryPolicy",
        "RateLimiter",
        # Availability flags
        "V3_ULTIMATE_AVAILABLE",
        "V3_MEMORY_AVAILABLE",
        "LOGGING_CONFIG_AVAILABLE",
    ]

    return safe_import("platform.core", file_path, key_items)


def validate_rag_module() -> ImportResult:
    """Validate platform/core/rag/__init__.py module."""
    file_path = PLATFORM_DIR / "core" / "rag" / "__init__.py"

    key_items = [
        # Semantic Chunking
        "SemanticChunker",
        "Chunk",
        # Reranking
        "SemanticReranker",
        "Document",
        "ScoredDocument",
        # Self-RAG
        "SelfRAG",
        "SelfRAGConfig",
        # Corrective RAG
        "CorrectiveRAG",
        "CRAGConfig",
        # HyDE
        "HyDERetriever",
        "HyDEConfig",
        # RAPTOR
        "RAPTOR",
        "RAPTORConfig",
        # Evaluation
        "RAGEvaluator",
        "EvaluationConfig",
        # Agentic RAG
        "AgenticRAG",
        "AgenticRAGConfig",
        # Pipeline
        "RAGPipeline",
        "create_pipeline",
    ]

    return safe_import("platform.core.rag", file_path, key_items)


def validate_memory_module() -> ImportResult:
    """Validate platform/core/memory/__init__.py module."""
    file_path = PLATFORM_DIR / "core" / "memory" / "__init__.py"

    key_items = [
        # V36 Canonical types
        "MemoryEntry",
        "MemoryTier",
        "MemoryPriority",
        "TierBackend",
        # Backends
        "InMemoryTierBackend",
        # System classes
        "MemoryTierManager",
        "UnifiedMemoryGateway",
        # V40 Features
        "ForgettingCurve",
        "BiTemporalMemory",
        "ProceduralMemory",
        # V41 Unified
        "UnifiedMemory",
        # Factory functions
        "create_memory_gateway",
        "create_tier_manager",
        # Legacy compatibility
        "MemorySystem",
        "CoreMemory",
    ]

    return safe_import("platform.core.memory", file_path, key_items)


def validate_memory_backends() -> ImportResult:
    """Validate platform/core/memory/backends/__init__.py module."""
    file_path = PLATFORM_DIR / "core" / "memory" / "backends" / "__init__.py"

    key_items = [
        "MemoryEntry",
        "MemoryTier",
        "MemoryPriority",
        "MemoryAccessPattern",
        "TierBackend",
        "MemoryBackend",
        "InMemoryTierBackend",
    ]

    return safe_import("platform.core.memory.backends", file_path, key_items)


def validate_adapters_module() -> ImportResult:
    """Validate platform/adapters/__init__.py module."""
    file_path = PLATFORM_DIR / "adapters" / "__init__.py"

    key_items = [
        # Registry
        "REGISTRY_AVAILABLE",
        "AdapterRegistry",
        "get_registry",
        # Factory functions
        "get_exa_adapter",
        "get_tavily_adapter",
        "get_jina_adapter",
        "get_perplexity_adapter",
        "get_firecrawl_adapter",
        "get_letta_adapter",
        "get_dspy_adapter",
        "get_langgraph_adapter",
        # Status functions
        "get_adapter_status",
        "check_sdk_version",
    ]

    return safe_import("platform.adapters", file_path, key_items)


def validate_adapter_registry() -> ImportResult:
    """Validate platform/adapters/registry.py module."""
    file_path = PLATFORM_DIR / "adapters" / "registry.py"

    key_items = [
        "AdapterRegistry",
        "AdapterInfo",
        "AdapterLoadStatus",
        "HealthCheckResult",
        "get_registry",
        "register_adapter",
    ]

    return safe_import("platform.adapters.registry", file_path, key_items)


def validate_health_check_module() -> ImportResult:
    """Validate platform/core/health_check.py module."""
    file_path = PLATFORM_DIR / "core" / "health_check.py"

    key_items = [
        "HealthChecker",
        "HealthStatus",
        "ComponentCategory",
        "ComponentHealth",
        "HealthReport",
        "PrometheusMetrics",
        "get_health_checker",
        "quick_health_check",
        "create_health_endpoint",
        "create_prometheus_endpoint",
    ]

    return safe_import("platform.core.health_check", file_path, key_items)


def validate_logging_config_module() -> ImportResult:
    """Validate platform/core/logging_config.py module."""
    file_path = PLATFORM_DIR / "core" / "logging_config.py"

    key_items = [
        "StructuredLogger",
        "StructuredJSONFormatter",
        "LogContext",
        "get_logger",
        "configure_logging",
        "reset_logging",
        "ComponentLogLevel",
        "COMPONENT_LOG_LEVELS",
        "set_component_log_level",
        "get_component_log_level",
        "logged",
        "generate_correlation_id",
        "generate_request_id",
        "generate_trace_id",
    ]

    return safe_import("platform.core.logging_config", file_path, key_items)


def run_validation() -> ValidationReport:
    """Run all validation tests and return a report."""
    report = ValidationReport()

    print("\nValidating module imports...\n")

    # Setup platform package
    setup_platform_package()

    # Logging config (standalone, no dependencies)
    print("  Checking platform.core.logging_config...")
    report.add_result(validate_logging_config_module())

    # Health check (standalone)
    print("  Checking platform.core.health_check...")
    report.add_result(validate_health_check_module())

    # Adapter registry (standalone)
    print("  Checking platform.adapters.registry...")
    report.add_result(validate_adapter_registry())

    # Memory backends (depends on base types)
    print("  Checking platform.core.memory.backends...")
    report.add_result(validate_memory_backends())

    # Memory module (depends on backends)
    print("  Checking platform.core.memory...")
    report.add_result(validate_memory_module())

    # RAG module (depends on memory)
    print("  Checking platform.core.rag...")
    report.add_result(validate_rag_module())

    # Adapters module (depends on registry)
    print("  Checking platform.adapters...")
    report.add_result(validate_adapters_module())

    # Core module (depends on everything)
    print("  Checking platform.core...")
    report.add_result(validate_core_module())

    return report


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("PLATFORM MODULE IMPORT VALIDATION")
    print("=" * 70)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Platform dir: {PLATFORM_DIR}")

    report = run_validation()
    report.print_report()

    # Return exit code based on results
    sys.exit(0 if report.failed_imports == 0 else 1)


if __name__ == "__main__":
    main()
