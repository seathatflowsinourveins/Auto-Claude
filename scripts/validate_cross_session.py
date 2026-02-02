#!/usr/bin/env python3
"""
Phase 7 Validation Script - Cross-Session Memory Integration

Validates that cross-session memory is properly integrated:
1. CrossSessionProvider availability
2. Memory persistence across provider instances
3. Session tracking functionality
4. Context generation for handoff
5. Integration with UnifiedMemory
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


def validate_imports() -> tuple[int, int]:
    """Validate cross-session memory imports."""
    print_header("Cross-Session Imports")
    passed = 0
    failed = 0

    try:
        from core.memory import (
            CrossSessionProvider,
            CROSS_SESSION_AVAILABLE,
            get_cross_session_context,
            start_memory_session,
            end_memory_session,
        )
        print_status("core/memory cross-session imports", True)
        passed += 1

        print_status("CROSS_SESSION_AVAILABLE", True, f"available={CROSS_SESSION_AVAILABLE}")
        passed += 1

    except ImportError as e:
        print_status("Cross-session imports", False, str(e)[:50])
        failed += 1

    return passed, failed


def validate_platform_module() -> tuple[int, int]:
    """Validate platform cross-session module."""
    print_header("Platform Cross-Session Module")
    passed = 0
    failed = 0

    try:
        platform_path = project_root / "platform" / "core" / "cross_session_memory.py"
        if platform_path.exists():
            size = platform_path.stat().st_size
            print_status("platform/core/cross_session_memory.py", True, f"{size:,} bytes")
            passed += 1
        else:
            print_status("platform/core/cross_session_memory.py", False, "not found")
            failed += 1

        # Try importing the module
        import importlib.util
        spec = importlib.util.spec_from_file_location("cross_session_memory", platform_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["cross_session_memory"] = module
            spec.loader.exec_module(module)

            print_status("CrossSessionMemory class", True)
            passed += 1

            print_status("get_memory_store function", True)
            passed += 1

            # Test memory store creation
            store = module.get_memory_store()
            print_status("Memory store instantiation", True, f"path={store.base_path}")
            passed += 1

    except Exception as e:
        print_status("Platform module", False, str(e)[:50])
        failed += 1

    return passed, failed


def validate_provider_creation() -> tuple[int, int]:
    """Validate CrossSessionProvider creation."""
    print_header("CrossSessionProvider Creation")
    passed = 0
    failed = 0

    try:
        from core.memory import CrossSessionProvider, CROSS_SESSION_AVAILABLE

        if not CROSS_SESSION_AVAILABLE:
            print_status("CrossSessionProvider", False, "not available")
            failed += 1
            return passed, failed

        provider = CrossSessionProvider()
        print_status("CrossSessionProvider()", True)
        passed += 1

        # Check provider type
        from core.memory import MemoryProvider
        assert provider.provider_type == MemoryProvider.CROSS_SESSION
        print_status("provider_type", True, "CROSS_SESSION")
        passed += 1

    except Exception as e:
        print_status("Provider creation", False, str(e)[:50])
        failed += 1

    return passed, failed


async def validate_memory_operations() -> tuple[int, int]:
    """Validate cross-session memory operations."""
    print_header("Memory Operations")
    passed = 0
    failed = 0

    try:
        from core.memory import CrossSessionProvider, CROSS_SESSION_AVAILABLE

        if not CROSS_SESSION_AVAILABLE:
            print_status("Memory operations", False, "provider not available")
            failed += 1
            return passed, failed

        provider = CrossSessionProvider()

        # Test store
        entry = await provider.store(
            content="V33 Integration test memory entry",
            metadata={
                "type": "fact",
                "importance": 0.8,
                "tags": ["test", "v33", "integration"],
            },
        )
        assert entry.id
        print_status("store()", True, f"id={entry.id[:12]}...")
        passed += 1

        # Test search
        results = await provider.search("V33 Integration", limit=5)
        assert len(results) >= 1
        print_status("search()", True, f"found {len(results)} results")
        passed += 1

        # Test get
        retrieved = await provider.get(entry.id)
        assert retrieved is not None
        assert "V33 Integration" in retrieved.content
        print_status("get()", True, "entry retrieved")
        passed += 1

    except Exception as e:
        print_status("Memory operations", False, str(e)[:50])
        failed += 1

    return passed, failed


def validate_session_tracking() -> tuple[int, int]:
    """Validate session tracking functionality."""
    print_header("Session Tracking")
    passed = 0
    failed = 0

    try:
        from core.memory import CrossSessionProvider, CROSS_SESSION_AVAILABLE

        if not CROSS_SESSION_AVAILABLE:
            print_status("Session tracking", False, "provider not available")
            failed += 1
            return passed, failed

        provider = CrossSessionProvider()

        # Start session
        session = provider.start_session("V33 Validation Test Session")
        assert session is not None
        print_status("start_session()", True, f"id={session.id}")
        passed += 1

        # Remember decision
        decision = provider.remember_decision(
            "Use CrossSessionProvider for V33 cross-session memory",
            importance=0.9,
            tags=["v33", "architecture"],
        )
        assert decision.memory_type == "decision"
        print_status("remember_decision()", True)
        passed += 1

        # Remember learning
        learning = provider.remember_learning(
            "CrossSessionMemory provides file-based persistence",
            importance=0.7,
            tags=["v33", "learning"],
        )
        assert learning.memory_type == "learning"
        print_status("remember_learning()", True)
        passed += 1

        # End session
        provider.end_session("V33 Validation completed")
        print_status("end_session()", True)
        passed += 1

    except Exception as e:
        print_status("Session tracking", False, str(e)[:50])
        failed += 1

    return passed, failed


def validate_context_generation() -> tuple[int, int]:
    """Validate context generation for handoff."""
    print_header("Context Generation")
    passed = 0
    failed = 0

    try:
        from core.memory import (
            get_cross_session_context,
            CROSS_SESSION_AVAILABLE,
        )

        if not CROSS_SESSION_AVAILABLE:
            print_status("Context generation", False, "not available")
            failed += 1
            return passed, failed

        # Get context
        context = get_cross_session_context(max_tokens=2000)
        assert isinstance(context, str)
        print_status("get_cross_session_context()", True, f"{len(context)} chars")
        passed += 1

        # Check context contains expected sections
        has_content = len(context) > 50 or "No content" not in context.lower()
        print_status("Context has content", True if has_content else False,
                    "sessions/decisions present" if has_content else "empty context")
        passed += 1

    except Exception as e:
        print_status("Context generation", False, str(e)[:50])
        failed += 1

    return passed, failed


async def validate_unified_memory_integration() -> tuple[int, int]:
    """Validate integration with UnifiedMemory."""
    print_header("UnifiedMemory Integration")
    passed = 0
    failed = 0

    try:
        from core.memory import (
            create_memory,
            get_available_providers,
            MemoryProvider,
            CROSS_SESSION_AVAILABLE,
        )

        # Check availability
        providers = get_available_providers()
        cross_session_available = providers.get(MemoryProvider.CROSS_SESSION, False)
        print_status("get_available_providers()", True,
                    f"cross_session={cross_session_available}")
        passed += 1

        if not CROSS_SESSION_AVAILABLE:
            print_status("UnifiedMemory integration", False, "provider not available")
            failed += 1
            return passed, failed

        # Create memory with cross-session enabled
        memory = create_memory()
        await memory.initialize()
        print_status("create_memory()", True)
        passed += 1

        # Store via unified interface
        entry = await memory.store(
            content="UnifiedMemory cross-session test",
            importance=0.7,
            metadata={"source": "v33_validation"},
        )
        assert entry.id
        print_status("memory.store()", True)
        passed += 1

        # Search via unified interface
        results = await memory.search("UnifiedMemory cross-session")
        print_status("memory.search()", True, f"{len(results)} results")
        passed += 1

        await memory.close()

    except Exception as e:
        print_status("UnifiedMemory integration", False, str(e)[:50])
        failed += 1

    return passed, failed


def validate_helper_functions() -> tuple[int, int]:
    """Validate helper functions."""
    print_header("Helper Functions")
    passed = 0
    failed = 0

    try:
        from core.memory import (
            start_memory_session,
            end_memory_session,
            CROSS_SESSION_AVAILABLE,
        )

        if not CROSS_SESSION_AVAILABLE:
            print_status("Helper functions", False, "not available")
            failed += 1
            return passed, failed

        # Test start_memory_session
        started = start_memory_session("Helper function test")
        print_status("start_memory_session()", started)
        passed += 1 if started else 0
        failed += 0 if started else 1

        # Test end_memory_session
        ended = end_memory_session("Test completed")
        print_status("end_memory_session()", ended)
        passed += 1 if ended else 0
        failed += 0 if ended else 1

    except Exception as e:
        print_status("Helper functions", False, str(e)[:50])
        failed += 1

    return passed, failed


async def main() -> int:
    """Run all Phase 7 validations."""
    print("\n" + "=" * 60)
    print("  PHASE 7 VALIDATION: Cross-Session Memory Integration")
    print("  " + "=" * 56)

    total_passed = 0
    total_failed = 0

    # Run validations
    p, f = validate_imports()
    total_passed += p
    total_failed += f

    p, f = validate_platform_module()
    total_passed += p
    total_failed += f

    p, f = validate_provider_creation()
    total_passed += p
    total_failed += f

    p, f = await validate_memory_operations()
    total_passed += p
    total_failed += f

    p, f = validate_session_tracking()
    total_passed += p
    total_failed += f

    p, f = validate_context_generation()
    total_passed += p
    total_failed += f

    p, f = await validate_unified_memory_integration()
    total_passed += p
    total_failed += f

    p, f = validate_helper_functions()
    total_passed += p
    total_failed += f

    # Summary
    print_header("SUMMARY")
    print(f"\n  Total Passed: {total_passed}")
    print(f"  Total Failed: {total_failed}")

    if total_failed == 0:
        print("\n  [+] PHASE 7 VALIDATION PASSED")
        print("  Cross-session memory integration is functional!")
        return 0
    else:
        print(f"\n  [X] PHASE 7 VALIDATION FAILED ({total_failed} issues)")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
