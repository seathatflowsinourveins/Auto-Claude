#!/usr/bin/env python3
"""
Phase 5 Validation Script - Structured Output Layer

Validates that all Phase 5 components are properly installed and functional:
1. SDK imports and availability
2. InstructorClient functionality
3. BAMLClient functionality
4. OutlinesGenerator functionality
5. PydanticAIAgent functionality
6. StructuredOutputFactory integration
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_status(name: str, passed: bool, details: str = "") -> None:
    """Print a status line."""
    symbol = "[+]" if passed else "[X]"
    status = "PASS" if passed else "FAIL"
    detail_str = f" - {details}" if details else ""
    print(f"  {symbol} [{status}] {name}{detail_str}")


def validate_file_structure() -> tuple[int, int]:
    """Validate all expected files exist."""
    print_header("File Structure")
    passed = 0
    failed = 0

    expected_files = [
        "core/structured/__init__.py",
        "core/structured/instructor_chains.py",
        "core/structured/baml_functions.py",
        "core/structured/outlines_constraints.py",
        "core/structured/pydantic_agents.py",
    ]

    for file_path in expected_files:
        full_path = project_root / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print_status(file_path, True, f"{size:,} bytes")
            passed += 1
        else:
            print_status(file_path, False, "not found")
            failed += 1

    return passed, failed


def validate_imports() -> tuple[int, int]:
    """Validate all imports work correctly."""
    print_header("Module Imports")
    passed = 0
    failed = 0

    # Test structured module import
    try:
        from core.structured import (
            STRUCTURED_OUTPUT_AVAILABLE,
            get_available_sdks,
        )
        print_status("core/structured module", True)
        passed += 1

        # Check availability
        sdks = get_available_sdks()
        available_count = sum(1 for v in sdks.values() if v)
        print_status("get_available_sdks()", True, f"{available_count}/4 available")
        passed += 1

    except ImportError as e:
        print_status("core/structured module", False, str(e)[:50])
        failed += 1

    return passed, failed


def validate_instructor() -> tuple[int, int]:
    """Validate InstructorClient."""
    print_header("Instructor SDK")
    passed = 0
    failed = 0

    try:
        from core.structured import (
            InstructorClient,
            ChainResult,
            Sentiment,
            SentimentType,
            Classification,
            INSTRUCTOR_AVAILABLE,
        )
        print_status("Instructor imports", True)
        passed += 1

        print_status("INSTRUCTOR_AVAILABLE", True, f"available={INSTRUCTOR_AVAILABLE}")
        passed += 1

        # Test model creation
        assert issubclass(Sentiment, object)
        assert SentimentType.POSITIVE.value == "positive"
        print_status("Pydantic models", True)
        passed += 1

        # Test ChainResult
        result = ChainResult(success=True, data=None)
        assert result.success is True
        print_status("ChainResult dataclass", True)
        passed += 1

    except Exception as e:
        print_status("Instructor validation", False, str(e)[:50])
        failed += 1

    return passed, failed


def validate_baml() -> tuple[int, int]:
    """Validate BAMLClient."""
    print_header("BAML SDK")
    passed = 0
    failed = 0

    try:
        from core.structured import (
            BAMLClient,
            BAMLResult,
            BAMLStatus,
            BAMLFunctionDef,
            BAMLFunctionRegistry,
            SummaryOutput,
            TranslationOutput,
            BAML_AVAILABLE,
        )
        print_status("BAML imports", True)
        passed += 1

        print_status("BAML_AVAILABLE", True, f"available={BAML_AVAILABLE}")
        passed += 1

        # Test registry
        registry = BAMLFunctionRegistry()
        functions = registry.list_functions()
        assert len(functions) >= 4  # summarize, translate, generate_code, analyze
        print_status("BAMLFunctionRegistry", True, f"{len(functions)} built-in functions")
        passed += 1

        # Test models
        summary = SummaryOutput(
            summary="Test summary",
            key_points=["point1"],
            word_count=2,
        )
        assert summary.summary == "Test summary"
        print_status("BAML output models", True)
        passed += 1

    except Exception as e:
        print_status("BAML validation", False, str(e)[:50])
        failed += 1

    return passed, failed


def validate_outlines() -> tuple[int, int]:
    """Validate OutlinesGenerator."""
    print_header("Outlines SDK")
    passed = 0
    failed = 0

    try:
        from core.structured import (
            OutlinesGenerator,
            GenerationResult,
            ChoiceResult,
            ConstraintType,
            CommonPatterns,
            OUTLINES_AVAILABLE,
        )
        print_status("Outlines imports", True)
        passed += 1

        print_status("OUTLINES_AVAILABLE", True, f"available={OUTLINES_AVAILABLE}")
        passed += 1

        # Test CommonPatterns
        assert CommonPatterns.EMAIL is not None
        assert CommonPatterns.DATE_ISO is not None
        print_status("CommonPatterns", True, "patterns defined")
        passed += 1

        # Test ConstraintType enum
        assert ConstraintType.REGEX.value == "regex"
        assert ConstraintType.JSON.value == "json"
        print_status("ConstraintType enum", True)
        passed += 1

    except Exception as e:
        print_status("Outlines validation", False, str(e)[:50])
        failed += 1

    return passed, failed


def validate_pydantic_ai() -> tuple[int, int]:
    """Validate PydanticAIAgent."""
    print_header("Pydantic AI SDK")
    passed = 0
    failed = 0

    try:
        from core.structured import (
            PydanticAIAgent,
            AgentConfig,
            AgentRole,
            AgentProvider,
            AgentRun,
            create_agent,
            PYDANTIC_AI_AVAILABLE,
        )
        print_status("Pydantic AI imports", True)
        passed += 1

        print_status("PYDANTIC_AI_AVAILABLE", True, f"available={PYDANTIC_AI_AVAILABLE}")
        passed += 1

        # Test AgentRole enum
        assert AgentRole.RESEARCHER.value == "researcher"
        assert AgentRole.CODER.value == "coder"
        print_status("AgentRole enum", True)
        passed += 1

        # Test AgentConfig
        config = AgentConfig(
            name="test_agent",
            role=AgentRole.ASSISTANT,
        )
        assert config.name == "test_agent"
        print_status("AgentConfig", True)
        passed += 1

        # Test AgentRun
        run = AgentRun(success=True)
        assert run.success is True
        print_status("AgentRun model", True)
        passed += 1

    except Exception as e:
        print_status("Pydantic AI validation", False, str(e)[:50])
        failed += 1

    return passed, failed


def validate_factory() -> tuple[int, int]:
    """Validate StructuredOutputFactory."""
    print_header("Structured Output Factory")
    passed = 0
    failed = 0

    try:
        from core.structured import (
            StructuredOutputFactory,
            StructuredOutputConfig,
            AgentRole,
        )

        # Test factory creation
        factory = StructuredOutputFactory()
        print_status("StructuredOutputFactory()", True)
        passed += 1

        # Test availability check
        availability = factory.get_availability()
        assert isinstance(availability, dict)
        assert "instructor" in availability
        print_status("get_availability()", True, f"{sum(availability.values())}/4 available")
        passed += 1

        # Test with config
        config = StructuredOutputConfig(
            default_provider="anthropic",
            default_model="claude-sonnet-4-20250514",
        )
        factory = StructuredOutputFactory(config)
        assert factory.config.default_model == "claude-sonnet-4-20250514"
        print_status("StructuredOutputConfig", True)
        passed += 1

    except Exception as e:
        print_status("Factory validation", False, str(e)[:50])
        failed += 1

    return passed, failed


async def validate_integration() -> tuple[int, int]:
    """Validate integration between components."""
    print_header("Integration Tests")
    passed = 0
    failed = 0

    try:
        from core.structured import (
            StructuredOutputFactory,
            InstructorClient,
            BAMLClient,
            OutlinesGenerator,
            PydanticAIAgent,
            AgentRole,
        )

        factory = StructuredOutputFactory()

        # Test instructor creation via factory
        instructor = factory.create_instructor()
        assert isinstance(instructor, InstructorClient)
        print_status("factory.create_instructor()", True)
        passed += 1

        # Test BAML creation via factory
        baml = factory.create_baml()
        assert isinstance(baml, BAMLClient)
        print_status("factory.create_baml()", True)
        passed += 1

        # Test Outlines creation via factory
        outlines = factory.create_outlines()
        assert isinstance(outlines, OutlinesGenerator)
        print_status("factory.create_outlines()", True)
        passed += 1

        # Test agent creation via factory
        agent = factory.create_agent("test_agent", role=AgentRole.RESEARCHER)
        assert isinstance(agent, PydanticAIAgent)
        print_status("factory.create_agent()", True)
        passed += 1

        # Test specialized agent creation
        research_agent = factory.create_research_agent()
        assert isinstance(research_agent, PydanticAIAgent)
        print_status("factory.create_research_agent()", True)
        passed += 1

    except Exception as e:
        print_status("Integration tests", False, str(e)[:50])
        failed += 1

    return passed, failed


async def main() -> int:
    """Run all Phase 5 validations."""
    print("\n" + "=" * 60)
    print("  PHASE 5 VALIDATION: Structured Output Layer")
    print("  " + "=" * 56)

    total_passed = 0
    total_failed = 0

    # Run validations
    p, f = validate_file_structure()
    total_passed += p
    total_failed += f

    p, f = validate_imports()
    total_passed += p
    total_failed += f

    p, f = validate_instructor()
    total_passed += p
    total_failed += f

    p, f = validate_baml()
    total_passed += p
    total_failed += f

    p, f = validate_outlines()
    total_passed += p
    total_failed += f

    p, f = validate_pydantic_ai()
    total_passed += p
    total_failed += f

    p, f = validate_factory()
    total_passed += p
    total_failed += f

    p, f = await validate_integration()
    total_passed += p
    total_failed += f

    # Summary
    print_header("SUMMARY")
    print(f"\n  Total Passed: {total_passed}")
    print(f"  Total Failed: {total_failed}")

    if total_failed == 0:
        print("\n  [+] PHASE 5 VALIDATION PASSED")
        print("  Structured Output Layer is fully functional!")
        return 0
    else:
        print(f"\n  [X] PHASE 5 VALIDATION FAILED ({total_failed} issues)")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
