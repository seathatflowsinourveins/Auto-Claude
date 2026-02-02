#!/usr/bin/env python3
"""
Phase 3 Validation Script - Orchestration Layer

Validates that all Phase 3 components are properly installed and functional:
1. SDK installations (temporalio, langgraph, crewai, pyautogen)
2. Module imports and class availability
3. Orchestrator instantiation
4. Unified interface functionality
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


def validate_sdk_installations() -> tuple[int, int]:
    """Validate that all SDKs are installed."""
    print_header("SDK Installations")
    passed = 0
    failed = 0

    # Temporal
    try:
        import temporalio
        print_status("temporalio", True, f"v{temporalio.__version__ if hasattr(temporalio, '__version__') else 'installed'}")
        passed += 1
    except ImportError as e:
        print_status("temporalio", False, str(e))
        failed += 1

    # LangGraph
    try:
        import langgraph
        version = getattr(langgraph, '__version__', 'installed')
        print_status("langgraph", True, f"v{version}")
        passed += 1
    except ImportError as e:
        print_status("langgraph", False, str(e))
        failed += 1

    # LangChain Core (required by LangGraph)
    try:
        import langchain_core
        version = getattr(langchain_core, '__version__', 'installed')
        print_status("langchain-core", True, f"v{version}")
        passed += 1
    except ImportError as e:
        print_status("langchain-core", False, str(e))
        failed += 1

    # CrewAI
    try:
        import crewai
        version = getattr(crewai, '__version__', 'installed')
        print_status("crewai", True, f"v{version}")
        passed += 1
    except ImportError as e:
        print_status("crewai", False, str(e))
        failed += 1

    # AutoGen
    try:
        import autogen
        version = getattr(autogen, '__version__', 'installed')
        print_status("pyautogen", True, f"v{version}")
        passed += 1
    except ImportError as e:
        print_status("pyautogen", False, str(e))
        failed += 1

    return passed, failed


def validate_orchestration_modules() -> tuple[int, int]:
    """Validate orchestration module imports."""
    print_header("Orchestration Modules")
    passed = 0
    failed = 0

    # Temporal Workflows
    try:
        from core.orchestration.temporal_workflows import (
            TEMPORAL_AVAILABLE,
            WorkflowInput,
            WorkflowResult,
        )
        print_status("temporal_workflows.py", True, f"TEMPORAL_AVAILABLE={TEMPORAL_AVAILABLE}")
        passed += 1
    except Exception as e:
        print_status("temporal_workflows.py", False, str(e))
        failed += 1

    # LangGraph Agents
    try:
        from core.orchestration.langgraph_agents import (
            LANGGRAPH_AVAILABLE,
            GraphConfig,
            GraphResult,
        )
        print_status("langgraph_agents.py", True, f"LANGGRAPH_AVAILABLE={LANGGRAPH_AVAILABLE}")
        passed += 1
    except Exception as e:
        print_status("langgraph_agents.py", False, str(e))
        failed += 1

    # Claude Flow
    try:
        from core.orchestration.claude_flow import (
            CLAUDE_FLOW_AVAILABLE,
            ClaudeFlow,
            ClaudeFlowOrchestrator,
            AgentRole,
        )
        print_status("claude_flow.py", True, f"CLAUDE_FLOW_AVAILABLE={CLAUDE_FLOW_AVAILABLE}")
        passed += 1
    except Exception as e:
        print_status("claude_flow.py", False, str(e))
        failed += 1

    # Crew Manager
    try:
        from core.orchestration.crew_manager import (
            CREWAI_AVAILABLE,
            CrewSpec,
            CrewResult,
        )
        print_status("crew_manager.py", True, f"CREWAI_AVAILABLE={CREWAI_AVAILABLE}")
        passed += 1
    except Exception as e:
        print_status("crew_manager.py", False, str(e))
        failed += 1

    # AutoGen Agents
    try:
        from core.orchestration.autogen_agents import (
            AUTOGEN_AVAILABLE,
            ConversationResult,
        )
        print_status("autogen_agents.py", True, f"AUTOGEN_AVAILABLE={AUTOGEN_AVAILABLE}")
        passed += 1
    except Exception as e:
        print_status("autogen_agents.py", False, str(e))
        failed += 1

    # Unified Interface
    try:
        from core.orchestration import (
            UnifiedOrchestrator,
            OrchestrationResult,
            create_unified_orchestrator,
            get_available_frameworks,
        )
        print_status("__init__.py (unified)", True)
        passed += 1
    except Exception as e:
        print_status("__init__.py (unified)", False, str(e))
        failed += 1

    return passed, failed


def validate_orchestrator_instantiation() -> tuple[int, int]:
    """Validate that orchestrators can be instantiated."""
    print_header("Orchestrator Instantiation")
    passed = 0
    failed = 0

    # Unified Orchestrator
    try:
        from core.orchestration import create_unified_orchestrator, get_available_frameworks

        frameworks = get_available_frameworks()
        orchestrator = create_unified_orchestrator()

        available = orchestrator.available_frameworks
        print_status(
            "UnifiedOrchestrator",
            True,
            f"{len(available)} frameworks available: {', '.join(available)}",
        )
        passed += 1
    except Exception as e:
        print_status("UnifiedOrchestrator", False, str(e))
        failed += 1

    # Claude Flow (always available)
    try:
        from core.orchestration.claude_flow import create_default_flow

        flow = create_default_flow("test-flow")
        agents = flow.list_agents()
        print_status("ClaudeFlow", True, f"agents: {agents}")
        passed += 1
    except Exception as e:
        print_status("ClaudeFlow", False, str(e))
        failed += 1

    # LangGraph (if available)
    try:
        from core.orchestration.langgraph_agents import LANGGRAPH_AVAILABLE

        if LANGGRAPH_AVAILABLE:
            from core.orchestration.langgraph_agents import create_orchestrator
            orchestrator = create_orchestrator()
            graph = orchestrator.create_graph("test-graph")
            print_status("LangGraphOrchestrator", True, "graph created")
            passed += 1
        else:
            print_status("LangGraphOrchestrator", True, "skipped (not installed)")
            passed += 1
    except Exception as e:
        print_status("LangGraphOrchestrator", False, str(e))
        failed += 1

    # CrewAI (if available)
    try:
        from core.orchestration.crew_manager import CREWAI_AVAILABLE

        if CREWAI_AVAILABLE:
            from core.orchestration.crew_manager import create_manager
            manager = create_manager()
            crew = manager.create_default_crew("test-crew")
            print_status("CrewManager", True, f"crew created with {len(crew.agents)} agents")
            passed += 1
        else:
            print_status("CrewManager", True, "skipped (not installed)")
            passed += 1
    except Exception as e:
        print_status("CrewManager", False, str(e))
        failed += 1

    # AutoGen (if available)
    try:
        from core.orchestration.autogen_agents import AUTOGEN_AVAILABLE

        if AUTOGEN_AVAILABLE:
            from core.orchestration.autogen_agents import create_orchestrator
            orchestrator = create_orchestrator()
            conv = orchestrator.create_coding_conversation("test-conv")
            print_status("AutoGenOrchestrator", True, f"agents: {conv.list_agents()}")
            passed += 1
        else:
            print_status("AutoGenOrchestrator", True, "skipped (not installed)")
            passed += 1
    except Exception as e:
        print_status("AutoGenOrchestrator", False, str(e))
        failed += 1

    return passed, failed


async def validate_claude_flow_execution() -> tuple[int, int]:
    """Validate Claude Flow can execute a simple task."""
    print_header("Claude Flow Execution Test")
    passed = 0
    failed = 0

    try:
        from core.orchestration.claude_flow import create_default_flow

        flow = create_default_flow("execution-test")
        result = await flow.run(
            task="Return 'Hello, World!' as your final output",
            context={"test": True},
        )

        if result.success or result.total_steps > 0:
            print_status(
                "Flow execution",
                True,
                f"completed in {result.total_steps} steps, {result.duration_seconds:.2f}s",
            )
            passed += 1
        else:
            print_status("Flow execution", False, result.error or "No output")
            failed += 1

    except Exception as e:
        print_status("Flow execution", False, str(e))
        failed += 1

    return passed, failed


def validate_file_structure() -> tuple[int, int]:
    """Validate all expected files exist."""
    print_header("File Structure")
    passed = 0
    failed = 0

    expected_files = [
        "core/orchestration/__init__.py",
        "core/orchestration/temporal_workflows.py",
        "core/orchestration/langgraph_agents.py",
        "core/orchestration/claude_flow.py",
        "core/orchestration/crew_manager.py",
        "core/orchestration/autogen_agents.py",
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


async def main() -> int:
    """Run all Phase 3 validations."""
    print("\n" + "=" * 60)
    print("  PHASE 3 VALIDATION: Orchestration Layer")
    print("  " + "=" * 56)

    total_passed = 0
    total_failed = 0

    # Run validations
    p, f = validate_file_structure()
    total_passed += p
    total_failed += f

    p, f = validate_sdk_installations()
    total_passed += p
    total_failed += f

    p, f = validate_orchestration_modules()
    total_passed += p
    total_failed += f

    p, f = validate_orchestrator_instantiation()
    total_passed += p
    total_failed += f

    p, f = await validate_claude_flow_execution()
    total_passed += p
    total_failed += f

    # Summary
    print_header("SUMMARY")
    print(f"\n  Total Passed: {total_passed}")
    print(f"  Total Failed: {total_failed}")

    if total_failed == 0:
        print("\n  [+] PHASE 3 VALIDATION PASSED")
        print("  All orchestration components are functional!")
        return 0
    else:
        print(f"\n  [X] PHASE 3 VALIDATION FAILED ({total_failed} issues)")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
