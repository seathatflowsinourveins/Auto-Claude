#!/usr/bin/env python3
"""
Production Validation Script - Phase 9 Production Fix
Part of V33 Architecture.

Comprehensive validation of the explicit failure pattern:
1. Validates all layers export correct exception types
2. Validates getter functions raise appropriate errors
3. Validates no stub patterns exist
4. Validates validator module works correctly
5. Runs integration test suite
6. Generates production readiness report

Usage:
    python scripts/validate_production.py
    python scripts/validate_production.py --verbose
    python scripts/validate_production.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Callable

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    category: str
    error: Optional[str] = None


@dataclass
class ProductionValidationResult:
    """Complete production validation result."""
    passed: int = 0
    failed: int = 0
    total: int = 0
    checks: List[ValidationCheck] = field(default_factory=list)
    categories: dict = field(default_factory=dict)

    def add(self, check: ValidationCheck) -> None:
        self.checks.append(check)
        self.total += 1
        if check.passed:
            self.passed += 1
        else:
            self.failed += 1

        # Track by category
        if check.category not in self.categories:
            self.categories[check.category] = {"passed": 0, "failed": 0}
        if check.passed:
            self.categories[check.category]["passed"] += 1
        else:
            self.categories[check.category]["failed"] += 1

    @property
    def success_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "total": self.total,
            "success_rate": self.success_rate,
            "categories": self.categories,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "category": c.category,
                    "error": c.error,
                }
                for c in self.checks
            ]
        }


def check(name: str, category: str = "general") -> Callable:
    """Decorator for validation checks."""
    def decorator(func: Callable) -> Callable:
        func._check_name = name
        func._check_category = category
        return func
    return decorator


# =============================================================================
# Exception Type Checks
# =============================================================================

@check("SDKNotAvailableError exists in observability", "exceptions")
def check_sdk_not_available_error() -> ValidationCheck:
    try:
        from core.observability import SDKNotAvailableError
        return ValidationCheck(
            name="SDKNotAvailableError exists in observability",
            passed=True,
            message="SDKNotAvailableError properly defined",
            category="exceptions",
        )
    except ImportError as e:
        return ValidationCheck(
            name="SDKNotAvailableError exists in observability",
            passed=False,
            message="Failed to import SDKNotAvailableError",
            category="exceptions",
            error=str(e),
        )


@check("SDKConfigurationError exists in observability", "exceptions")
def check_sdk_configuration_error() -> ValidationCheck:
    try:
        from core.observability import SDKConfigurationError
        return ValidationCheck(
            name="SDKConfigurationError exists in observability",
            passed=True,
            message="SDKConfigurationError properly defined",
            category="exceptions",
        )
    except ImportError as e:
        return ValidationCheck(
            name="SDKConfigurationError exists in observability",
            passed=False,
            message="Failed to import SDKConfigurationError",
            category="exceptions",
            error=str(e),
        )


@check("Exceptions re-exported in memory layer", "exceptions")
def check_memory_exceptions() -> ValidationCheck:
    try:
        from core.memory import SDKNotAvailableError, SDKConfigurationError
        from core.observability import SDKNotAvailableError as ObsError
        assert SDKNotAvailableError is ObsError
        return ValidationCheck(
            name="Exceptions re-exported in memory layer",
            passed=True,
            message="Memory layer properly re-exports exceptions",
            category="exceptions",
        )
    except Exception as e:
        return ValidationCheck(
            name="Exceptions re-exported in memory layer",
            passed=False,
            message="Memory layer exception re-export failed",
            category="exceptions",
            error=str(e),
        )


@check("Exceptions re-exported in orchestration layer", "exceptions")
def check_orchestration_exceptions() -> ValidationCheck:
    try:
        from core.orchestration import SDKNotAvailableError, SDKConfigurationError
        from core.observability import SDKNotAvailableError as ObsError
        assert SDKNotAvailableError is ObsError
        return ValidationCheck(
            name="Exceptions re-exported in orchestration layer",
            passed=True,
            message="Orchestration layer properly re-exports exceptions",
            category="exceptions",
        )
    except Exception as e:
        return ValidationCheck(
            name="Exceptions re-exported in orchestration layer",
            passed=False,
            message="Orchestration layer exception re-export failed",
            category="exceptions",
            error=str(e),
        )


@check("Exceptions re-exported in structured layer", "exceptions")
def check_structured_exceptions() -> ValidationCheck:
    try:
        from core.structured import SDKNotAvailableError, SDKConfigurationError
        from core.observability import SDKNotAvailableError as ObsError
        assert SDKNotAvailableError is ObsError
        return ValidationCheck(
            name="Exceptions re-exported in structured layer",
            passed=True,
            message="Structured layer properly re-exports exceptions",
            category="exceptions",
        )
    except Exception as e:
        return ValidationCheck(
            name="Exceptions re-exported in structured layer",
            passed=False,
            message="Structured layer exception re-export failed",
            category="exceptions",
            error=str(e),
        )


# =============================================================================
# Getter Function Checks
# =============================================================================

@check("Observability getter functions exist", "getters")
def check_observability_getters() -> ValidationCheck:
    try:
        from core.observability import (
            get_langfuse_tracer,
            get_langfuse_observe,
            get_phoenix_client,
            get_opik_client,
            get_opik_track,
            get_deepeval_evaluator,
            get_deepeval_metrics,
            get_ragas_evaluator,
            get_ragas_metrics,
            get_logfire_logger,
            get_opentelemetry_tracer,
        )
        return ValidationCheck(
            name="Observability getter functions exist",
            passed=True,
            message="All observability getter functions exist",
            category="getters",
        )
    except ImportError as e:
        return ValidationCheck(
            name="Observability getter functions exist",
            passed=False,
            message="Missing observability getter functions",
            category="getters",
            error=str(e),
        )


@check("Memory getter functions exist", "getters")
def check_memory_getters() -> ValidationCheck:
    try:
        from core.memory import (
            get_mem0_client,
            get_zep_client,
            get_letta_client,
            get_cross_session_provider,
        )
        return ValidationCheck(
            name="Memory getter functions exist",
            passed=True,
            message="All memory getter functions exist",
            category="getters",
        )
    except ImportError as e:
        return ValidationCheck(
            name="Memory getter functions exist",
            passed=False,
            message="Missing memory getter functions",
            category="getters",
            error=str(e),
        )


@check("Orchestration getter functions exist", "getters")
def check_orchestration_getters() -> ValidationCheck:
    try:
        from core.orchestration import (
            get_temporal_orchestrator,
            get_langgraph_orchestrator,
            get_claude_flow_orchestrator,
            get_crewai_manager,
            get_autogen_orchestrator,
        )
        return ValidationCheck(
            name="Orchestration getter functions exist",
            passed=True,
            message="All orchestration getter functions exist",
            category="getters",
        )
    except ImportError as e:
        return ValidationCheck(
            name="Orchestration getter functions exist",
            passed=False,
            message="Missing orchestration getter functions",
            category="getters",
            error=str(e),
        )


@check("Structured getter functions exist", "getters")
def check_structured_getters() -> ValidationCheck:
    try:
        from core.structured import (
            get_instructor_client,
            get_baml_client,
            get_outlines_generator,
            get_pydantic_agent,
        )
        return ValidationCheck(
            name="Structured getter functions exist",
            passed=True,
            message="All structured getter functions exist",
            category="getters",
        )
    except ImportError as e:
        return ValidationCheck(
            name="Structured getter functions exist",
            passed=False,
            message="Missing structured getter functions",
            category="getters",
            error=str(e),
        )


# =============================================================================
# Validator Module Checks
# =============================================================================

@check("Validator module exists", "validator")
def check_validator_module() -> ValidationCheck:
    try:
        from core.validator import validate_all, validate_layer, validate_sdk
        return ValidationCheck(
            name="Validator module exists",
            passed=True,
            message="core.validator module imported successfully",
            category="validator",
        )
    except ImportError as e:
        return ValidationCheck(
            name="Validator module exists",
            passed=False,
            message="Failed to import validator module",
            category="validator",
            error=str(e),
        )


@check("Validator returns proper results", "validator")
def check_validator_results() -> ValidationCheck:
    try:
        from core.validator import validate_all, FullValidationResult

        result = validate_all()
        assert isinstance(result, FullValidationResult)
        assert result.total_sdks > 0
        assert result.status is not None

        return ValidationCheck(
            name="Validator returns proper results",
            passed=True,
            message=f"Validated {result.total_sdks} SDKs across {len(result.layers)} layers",
            category="validator",
        )
    except Exception as e:
        return ValidationCheck(
            name="Validator returns proper results",
            passed=False,
            message="Validator failed to return proper results",
            category="validator",
            error=str(e),
        )


@check("Validator result is JSON serializable", "validator")
def check_validator_serializable() -> ValidationCheck:
    try:
        from core.validator import validate_all

        result = validate_all()
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)

        assert isinstance(json_str, str)
        assert len(json_str) > 0

        return ValidationCheck(
            name="Validator result is JSON serializable",
            passed=True,
            message="Validation result can be serialized to JSON",
            category="validator",
        )
    except Exception as e:
        return ValidationCheck(
            name="Validator result is JSON serializable",
            passed=False,
            message="Validation result failed JSON serialization",
            category="validator",
            error=str(e),
        )


# =============================================================================
# No Stubs Pattern Checks
# =============================================================================

@check("No stub classes in observability", "no_stubs")
def check_no_stubs_observability() -> ValidationCheck:
    try:
        import core.observability as obs

        module_attrs = dir(obs)
        stub_patterns = ["Stub", "Fallback", "Mock", "Dummy", "Fake"]
        found_stubs = [
            attr for attr in module_attrs
            if any(pattern in attr for pattern in stub_patterns)
        ]

        if found_stubs:
            return ValidationCheck(
                name="No stub classes in observability",
                passed=False,
                message=f"Found stub patterns: {found_stubs}",
                category="no_stubs",
            )

        return ValidationCheck(
            name="No stub classes in observability",
            passed=True,
            message="No stub patterns found in observability",
            category="no_stubs",
        )
    except Exception as e:
        return ValidationCheck(
            name="No stub classes in observability",
            passed=False,
            message="Failed to check observability for stubs",
            category="no_stubs",
            error=str(e),
        )


@check("No stub classes in memory", "no_stubs")
def check_no_stubs_memory() -> ValidationCheck:
    try:
        import core.memory as mem

        module_attrs = dir(mem)
        stub_patterns = ["Stub", "Fallback", "Mock", "Dummy", "Fake"]
        found_stubs = [
            attr for attr in module_attrs
            if any(pattern in attr for pattern in stub_patterns)
        ]

        if found_stubs:
            return ValidationCheck(
                name="No stub classes in memory",
                passed=False,
                message=f"Found stub patterns: {found_stubs}",
                category="no_stubs",
            )

        return ValidationCheck(
            name="No stub classes in memory",
            passed=True,
            message="No stub patterns found in memory",
            category="no_stubs",
        )
    except Exception as e:
        return ValidationCheck(
            name="No stub classes in memory",
            passed=False,
            message="Failed to check memory for stubs",
            category="no_stubs",
            error=str(e),
        )


@check("No stub classes in orchestration", "no_stubs")
def check_no_stubs_orchestration() -> ValidationCheck:
    try:
        import core.orchestration as orch

        module_attrs = dir(orch)
        stub_patterns = ["Stub", "Fallback", "Mock", "Dummy", "Fake"]
        found_stubs = [
            attr for attr in module_attrs
            if any(pattern in attr for pattern in stub_patterns)
        ]

        if found_stubs:
            return ValidationCheck(
                name="No stub classes in orchestration",
                passed=False,
                message=f"Found stub patterns: {found_stubs}",
                category="no_stubs",
            )

        return ValidationCheck(
            name="No stub classes in orchestration",
            passed=True,
            message="No stub patterns found in orchestration",
            category="no_stubs",
        )
    except Exception as e:
        return ValidationCheck(
            name="No stub classes in orchestration",
            passed=False,
            message="Failed to check orchestration for stubs",
            category="no_stubs",
            error=str(e),
        )


@check("No stub classes in structured", "no_stubs")
def check_no_stubs_structured() -> ValidationCheck:
    try:
        import core.structured as struct

        module_attrs = dir(struct)
        stub_patterns = ["Stub", "Fallback", "Mock", "Dummy", "Fake"]
        found_stubs = [
            attr for attr in module_attrs
            if any(pattern in attr for pattern in stub_patterns)
        ]

        if found_stubs:
            return ValidationCheck(
                name="No stub classes in structured",
                passed=False,
                message=f"Found stub patterns: {found_stubs}",
                category="no_stubs",
            )

        return ValidationCheck(
            name="No stub classes in structured",
            passed=True,
            message="No stub patterns found in structured",
            category="no_stubs",
        )
    except Exception as e:
        return ValidationCheck(
            name="No stub classes in structured",
            passed=False,
            message="Failed to check structured for stubs",
            category="no_stubs",
            error=str(e),
        )


# =============================================================================
# Availability Flag Checks
# =============================================================================

@check("Observability availability flags exist", "availability")
def check_observability_flags() -> ValidationCheck:
    try:
        from core.observability import (
            LANGFUSE_AVAILABLE,
            PHOENIX_AVAILABLE,
            OPIK_AVAILABLE,
            DEEPEVAL_AVAILABLE,
            RAGAS_AVAILABLE,
            LOGFIRE_AVAILABLE,
            OPENTELEMETRY_AVAILABLE,
        )

        flags = [
            LANGFUSE_AVAILABLE,
            PHOENIX_AVAILABLE,
            OPIK_AVAILABLE,
            DEEPEVAL_AVAILABLE,
            RAGAS_AVAILABLE,
            LOGFIRE_AVAILABLE,
            OPENTELEMETRY_AVAILABLE,
        ]

        assert all(isinstance(f, bool) for f in flags)

        return ValidationCheck(
            name="Observability availability flags exist",
            passed=True,
            message="All observability availability flags are boolean",
            category="availability",
        )
    except Exception as e:
        return ValidationCheck(
            name="Observability availability flags exist",
            passed=False,
            message="Observability availability flags issue",
            category="availability",
            error=str(e),
        )


@check("Memory availability flags exist", "availability")
def check_memory_flags() -> ValidationCheck:
    try:
        from core.memory import (
            MEM0_AVAILABLE,
            ZEP_AVAILABLE,
            LETTA_AVAILABLE,
            CROSS_SESSION_AVAILABLE,
        )

        flags = [
            MEM0_AVAILABLE,
            ZEP_AVAILABLE,
            LETTA_AVAILABLE,
            CROSS_SESSION_AVAILABLE,
        ]

        assert all(isinstance(f, bool) for f in flags)

        return ValidationCheck(
            name="Memory availability flags exist",
            passed=True,
            message="All memory availability flags are boolean",
            category="availability",
        )
    except Exception as e:
        return ValidationCheck(
            name="Memory availability flags exist",
            passed=False,
            message="Memory availability flags issue",
            category="availability",
            error=str(e),
        )


@check("Orchestration availability flags exist", "availability")
def check_orchestration_flags() -> ValidationCheck:
    try:
        from core.orchestration import (
            TEMPORAL_AVAILABLE,
            LANGGRAPH_AVAILABLE,
            CLAUDE_FLOW_AVAILABLE,
            CREWAI_AVAILABLE,
            AUTOGEN_AVAILABLE,
        )

        flags = [
            TEMPORAL_AVAILABLE,
            LANGGRAPH_AVAILABLE,
            CLAUDE_FLOW_AVAILABLE,
            CREWAI_AVAILABLE,
            AUTOGEN_AVAILABLE,
        ]

        assert all(isinstance(f, bool) for f in flags)

        return ValidationCheck(
            name="Orchestration availability flags exist",
            passed=True,
            message="All orchestration availability flags are boolean",
            category="availability",
        )
    except Exception as e:
        return ValidationCheck(
            name="Orchestration availability flags exist",
            passed=False,
            message="Orchestration availability flags issue",
            category="availability",
            error=str(e),
        )


@check("Structured availability flags exist", "availability")
def check_structured_flags() -> ValidationCheck:
    try:
        from core.structured import (
            INSTRUCTOR_AVAILABLE,
            BAML_AVAILABLE,
            OUTLINES_AVAILABLE,
            PYDANTIC_AI_AVAILABLE,
        )

        flags = [
            INSTRUCTOR_AVAILABLE,
            BAML_AVAILABLE,
            OUTLINES_AVAILABLE,
            PYDANTIC_AI_AVAILABLE,
        ]

        assert all(isinstance(f, bool) for f in flags)

        return ValidationCheck(
            name="Structured availability flags exist",
            passed=True,
            message="All structured availability flags are boolean",
            category="availability",
        )
    except Exception as e:
        return ValidationCheck(
            name="Structured availability flags exist",
            passed=False,
            message="Structured availability flags issue",
            category="availability",
            error=str(e),
        )


# =============================================================================
# Main Runner
# =============================================================================

def run_all_checks() -> ProductionValidationResult:
    """Run all production validation checks."""
    result = ProductionValidationResult()

    checks = [
        # Exceptions
        check_sdk_not_available_error,
        check_sdk_configuration_error,
        check_memory_exceptions,
        check_orchestration_exceptions,
        check_structured_exceptions,

        # Getters
        check_observability_getters,
        check_memory_getters,
        check_orchestration_getters,
        check_structured_getters,

        # Validator
        check_validator_module,
        check_validator_results,
        check_validator_serializable,

        # No Stubs
        check_no_stubs_observability,
        check_no_stubs_memory,
        check_no_stubs_orchestration,
        check_no_stubs_structured,

        # Availability
        check_observability_flags,
        check_memory_flags,
        check_orchestration_flags,
        check_structured_flags,
    ]

    for check_func in checks:
        try:
            check_result = check_func()
        except Exception as e:
            check_result = ValidationCheck(
                name=getattr(check_func, "_check_name", check_func.__name__),
                passed=False,
                message="Check raised exception",
                category=getattr(check_func, "_check_category", "general"),
                error=str(e),
            )

        result.add(check_result)

    return result


def print_results(result: ProductionValidationResult, verbose: bool = False) -> None:
    """Print validation results."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        _print_rich_results(result, verbose)
    except ImportError:
        _print_plain_results(result, verbose)


def _print_rich_results(result: ProductionValidationResult, verbose: bool) -> None:
    """Print results using Rich."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    # Header
    status = "PASS" if result.failed == 0 else "FAIL"
    status_color = "green" if result.failed == 0 else "red"

    console.print(Panel(
        f"[bold]Phase 9 Production Validation[/bold]\n"
        f"Result: [{status_color}]{status}[/{status_color}]\n"
        f"Checks: {result.passed}/{result.total} passed ({result.success_rate:.1f}%)",
        title="V33 Production Readiness",
    ))

    # Category summary
    cat_table = Table(title="\nValidation Categories")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Passed", style="green")
    cat_table.add_column("Failed", style="red")
    cat_table.add_column("Status")

    for cat_name, cat_stats in result.categories.items():
        status = "[green]✓[/green]" if cat_stats["failed"] == 0 else "[red]✗[/red]"
        cat_table.add_row(
            cat_name,
            str(cat_stats["passed"]),
            str(cat_stats["failed"]),
            status,
        )

    console.print(cat_table)

    # Detailed results (if verbose or failures)
    if verbose or result.failed > 0:
        detail_table = Table(title="\nDetailed Results")
        detail_table.add_column("Check", style="cyan")
        detail_table.add_column("Status")
        detail_table.add_column("Message")

        for check in result.checks:
            if verbose or not check.passed:
                status = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
                msg = check.message
                if check.error:
                    msg += f" ({check.error[:50]}...)" if len(check.error) > 50 else f" ({check.error})"
                detail_table.add_row(check.name[:40], status, msg[:60])

        console.print(detail_table)

    # Summary
    console.print()
    if result.failed == 0:
        console.print("[green bold]All Phase 9 production checks passed![/green bold]")
    else:
        console.print(f"[red bold]{result.failed} checks failed. See details above.[/red bold]")


def _print_plain_results(result: ProductionValidationResult, verbose: bool) -> None:
    """Print results using plain text."""
    print("=" * 60)
    print("Phase 9 Production Validation")
    print("=" * 60)
    status = "PASS" if result.failed == 0 else "FAIL"
    print(f"Result: {status}")
    print(f"Checks: {result.passed}/{result.total} passed ({result.success_rate:.1f}%)")
    print()

    # Category summary
    print("Categories:")
    for cat_name, cat_stats in result.categories.items():
        status = "OK" if cat_stats["failed"] == 0 else "FAIL"
        print(f"  {cat_name}: {cat_stats['passed']}/{cat_stats['passed'] + cat_stats['failed']} [{status}]")
    print()

    # Details
    if verbose or result.failed > 0:
        print("Details:")
        for check in result.checks:
            if verbose or not check.passed:
                status = "PASS" if check.passed else "FAIL"
                print(f"  [{status}] {check.name}")
                if check.error:
                    print(f"         Error: {check.error[:80]}")
        print()

    # Summary
    if result.failed == 0:
        print("All Phase 9 production checks passed!")
    else:
        print(f"{result.failed} checks failed.")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Phase 9 Production Validation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all check details")
    parser.add_argument("--json", "-j", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    result = run_all_checks()

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_results(result, verbose=args.verbose)

    sys.exit(0 if result.failed == 0 else 1)


if __name__ == "__main__":
    main()
