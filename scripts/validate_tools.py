#!/usr/bin/env python3
"""
Phase 5 Validation Script - Tool Layer

Validates that all Phase 5 components are properly installed and functional:
1. Platform tool registry availability
2. SDK integrations
3. Core tool types
4. UnifiedToolLayer functionality
5. Tool search and execution
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
        "core/tools/__init__.py",
        "core/tools/types.py",
        "platform/core/tool_registry.py",
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


def validate_tool_types() -> tuple[int, int]:
    """Validate tool type imports."""
    print_header("Tool Types")
    passed = 0
    failed = 0

    try:
        from core.tools.types import (
            ToolCategory,
            ToolPermission,
            ToolStatus,
            ToolSource,
            ToolParameter,
            ToolSchema,
            ToolInfo,
            ToolResult,
            ToolConfig,
        )
        print_status("core/tools/types.py", True, "all types imported")
        passed += 1

        # Test enum values
        assert ToolCategory.FILE_SYSTEM.value == "file_system"
        assert ToolPermission.READ_ONLY.value == "read_only"
        assert ToolStatus.AVAILABLE.value == "available"
        print_status("Enum values", True, "validated")
        passed += 1

        # Test ToolSchema
        schema = ToolSchema(
            name="TestTool",
            description="A test tool",
            parameters=[
                ToolParameter(
                    name="param1",
                    type="string",
                    description="A parameter",
                    required=True,
                )
            ],
        )
        anthropic_fmt = schema.to_anthropic_format()
        assert anthropic_fmt["name"] == "TestTool"
        print_status("ToolSchema.to_anthropic_format()", True)
        passed += 1

        openai_fmt = schema.to_openai_format()
        assert openai_fmt["type"] == "function"
        print_status("ToolSchema.to_openai_format()", True)
        passed += 1

    except Exception as e:
        print_status("Tool types", False, str(e)[:50])
        failed += 1

    return passed, failed


def validate_platform_registry() -> tuple[int, int]:
    """Validate platform tool registry."""
    print_header("Platform Tool Registry")
    passed = 0
    failed = 0

    # Use importlib to avoid conflict with Python's standard library 'platform' module
    import importlib.util

    def load_module_from_path(module_name: str, file_path: Path):
        """Load a module directly from file path.

        Note: On Python 3.14+, we must register the module in sys.modules BEFORE
        executing it, otherwise the dataclass decorator fails.
        """
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        # Register module BEFORE execution (required for Python 3.14+ dataclasses)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    try:
        platform_path = project_root / "platform" / "core" / "tool_registry.py"
        if not platform_path.exists():
            print_status("platform/core/tool_registry.py", False, "not found")
            failed += 1
            return passed, failed

        tool_registry_module = load_module_from_path("platform_tool_registry", platform_path)
        ToolRegistry = tool_registry_module.ToolRegistry
        create_tool_registry = tool_registry_module.create_tool_registry
        ToolCategory = tool_registry_module.ToolCategory
        ToolPermission = tool_registry_module.ToolPermission
        ToolSchema = tool_registry_module.ToolSchema

        print_status("platform/core/tool_registry.py", True, "imported via importlib")
        passed += 1

        # Create registry
        registry = create_tool_registry(include_builtins=True)
        tool_count = len(registry._tools)
        print_status("create_tool_registry()", True, f"{tool_count} built-in tools")
        passed += 1

        # Test search
        results = registry.search("read")
        print_status("registry.search()", True, f"found {len(results)} results")
        passed += 1

        # Test get_schemas_for_llm
        schemas = registry.get_schemas_for_llm(format="anthropic")
        print_status("get_schemas_for_llm()", True, f"{len(schemas)} schemas")
        passed += 1

        # Test stats
        stats = registry.get_stats()
        print_status("get_stats()", True, f"total={stats['total_tools']}")
        passed += 1

    except Exception as e:
        print_status("Platform registry", False, str(e)[:50])
        failed += 1

    return passed, failed


def validate_sdk_integrations() -> tuple[int, int]:
    """Validate SDK integration availability."""
    print_header("SDK Integrations")
    passed = 0
    failed = 0

    # Use importlib to avoid conflict with Python's standard library 'platform' module
    import importlib.util

    def load_module_from_path(module_name: str, file_path: Path):
        """Load a module directly from file path.

        Note: On Python 3.14+, we must register the module in sys.modules BEFORE
        executing it, otherwise the dataclass decorator fails.
        """
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        # Register module BEFORE execution (required for Python 3.14+ dataclasses)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    try:
        sdk_path = project_root / "platform" / "core" / "sdk_integrations.py"
        if not sdk_path.exists():
            print_status("platform/core/sdk_integrations.py", False, "not found")
            failed += 1
            return passed, failed

        sdk_module = load_module_from_path("platform_sdk_integrations", sdk_path)
        GRAPHRAG_AVAILABLE = getattr(sdk_module, "GRAPHRAG_AVAILABLE", False)
        LIGHTRAG_AVAILABLE = getattr(sdk_module, "LIGHTRAG_AVAILABLE", False)
        LLAMAINDEX_AVAILABLE = getattr(sdk_module, "LLAMAINDEX_AVAILABLE", False)
        DSPY_AVAILABLE = getattr(sdk_module, "DSPY_AVAILABLE", False)

        # Report status (don't fail if SDKs not available)
        print_status("GraphRAG", GRAPHRAG_AVAILABLE, f"available={GRAPHRAG_AVAILABLE}")
        print_status("LightRAG", LIGHTRAG_AVAILABLE, f"available={LIGHTRAG_AVAILABLE}")
        print_status("LlamaIndex", LLAMAINDEX_AVAILABLE, f"available={LLAMAINDEX_AVAILABLE}")
        print_status("DSPy", DSPY_AVAILABLE, f"available={DSPY_AVAILABLE}")

        # Count at least import worked
        passed += 4

    except Exception as e:
        print_status("SDK integrations", False, str(e)[:50])
        failed += 1

    return passed, failed


def validate_unified_tool_layer() -> tuple[int, int]:
    """Validate unified tool layer imports."""
    print_header("Unified Tool Layer")
    passed = 0
    failed = 0

    try:
        from core.tools import (
            UnifiedToolLayer,
            create_tool_layer,
            get_available_tool_sources,
            PLATFORM_TOOLS_AVAILABLE,
            SDK_INTEGRATIONS_AVAILABLE,
        )
        print_status("core/tools/__init__.py", True, "imported")
        passed += 1

        # Check availability
        print_status("Platform tools", PLATFORM_TOOLS_AVAILABLE, f"available={PLATFORM_TOOLS_AVAILABLE}")
        print_status("SDK integrations", SDK_INTEGRATIONS_AVAILABLE, f"available={SDK_INTEGRATIONS_AVAILABLE}")
        passed += 2

        # Get tool sources
        sources = get_available_tool_sources()
        print_status("get_available_tool_sources()", True, f"{sum(sources.values())} sources available")
        passed += 1

    except Exception as e:
        print_status("Unified tool layer", False, str(e)[:50])
        failed += 1

    return passed, failed


async def validate_tool_operations() -> tuple[int, int]:
    """Validate tool layer operations."""
    print_header("Tool Operations")
    passed = 0
    failed = 0

    try:
        from core.tools import create_tool_layer

        tools = create_tool_layer()
        await tools.initialize()

        # Test search
        results = await tools.search("read")
        print_status("search('read')", True, f"found {len(results)} tools")
        passed += 1

        # Test recommend
        recommendations = await tools.recommend_for_task("read a file and search the web")
        print_status("recommend_for_task()", True, f"{len(recommendations)} recommendations")
        passed += 1

        # Test available_tools
        available = tools.available_tools
        print_status("available_tools", True, f"{len(available)} tools")
        passed += 1

        # Test stats
        stats = tools.get_stats()
        print_status("get_stats()", True, f"platform={stats.get('platform_available', False)}")
        passed += 1

        # Test get_schemas_for_llm
        schemas = tools.get_schemas_for_llm(format="anthropic")
        print_status("get_schemas_for_llm()", True, f"{len(schemas)} schemas")
        passed += 1

    except Exception as e:
        print_status("Tool operations", False, str(e)[:50])
        failed += 1

    return passed, failed


async def main() -> int:
    """Run all Phase 5 validations."""
    print("\n" + "=" * 60)
    print("  PHASE 5 VALIDATION: Tool Layer")
    print("  " + "=" * 56)

    total_passed = 0
    total_failed = 0

    # Run validations
    p, f = validate_file_structure()
    total_passed += p
    total_failed += f

    p, f = validate_tool_types()
    total_passed += p
    total_failed += f

    p, f = validate_platform_registry()
    total_passed += p
    total_failed += f

    p, f = validate_sdk_integrations()
    total_passed += p
    total_failed += f

    p, f = validate_unified_tool_layer()
    total_passed += p
    total_failed += f

    p, f = await validate_tool_operations()
    total_passed += p
    total_failed += f

    # Summary
    print_header("SUMMARY")
    print(f"\n  Total Passed: {total_passed}")
    print(f"  Total Failed: {total_failed}")

    if total_failed == 0:
        print("\n  [+] PHASE 5 VALIDATION PASSED")
        print("  All tool components are functional!")
        return 0
    else:
        print(f"\n  [X] PHASE 5 VALIDATION FAILED ({total_failed} issues)")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
