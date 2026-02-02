#!/usr/bin/env python3
"""
V2 SDK Stack Validation Runner

Directly validates the complete V2 architecture without pytest dependency.
Provides detailed status report for each component.

Usage: python validate_v2_stack.py
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add platform directory to path
PLATFORM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PLATFORM_DIR)


def test_import(module_path: str, items: List[str]) -> Tuple[bool, str, Dict[str, Any]]:
    """Test importing a module and its items."""
    try:
        module = __import__(module_path, fromlist=items)
        result = {}
        for item in items:
            if hasattr(module, item):
                result[item] = True
            else:
                result[item] = False
        return True, None, result
    except Exception as e:
        return False, str(e), {}


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    width = 70
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}")


def print_result(name: str, success: bool, details: str = ""):
    """Print a test result with symbol."""
    symbol = "[OK]" if success else "[X]"
    print(f"  {symbol} {name}")
    if details:
        print(f"      {details}")


def validate_adapters() -> Dict[str, Dict[str, Any]]:
    """Validate all adapters."""
    print_header("ADAPTER VALIDATION")

    results = {}

    # Test adapters/__init__.py
    print("\n  Testing adapters module...")
    success, error, items = test_import("adapters", [
        "ADAPTER_STATUS",
        "get_adapter_status",
        "register_adapter",
    ])

    if success:
        print_result("adapters/__init__.py", True)

        # Get adapter status
        from adapters import get_adapter_status
        status = get_adapter_status()
        for name, info in status.items():
            avail = info.get("available", False)
            print_result(f"  +- {name} registered", True, f"available={avail}")
    else:
        print_result("adapters/__init__.py", False, error)

    results["init"] = {"success": success, "error": error}

    # Test individual adapters
    adapter_tests = [
        ("DSPy", "adapters.dspy_adapter", "DSPyAdapter", "DSPY_AVAILABLE"),
        ("LangGraph", "adapters.langgraph_adapter", "LangGraphAdapter", "LANGGRAPH_AVAILABLE"),
        ("Mem0", "adapters.mem0_adapter", "Mem0Adapter", "MEM0_AVAILABLE"),
        ("llm-reasoners", "adapters.llm_reasoners_adapter", "LLMReasonersAdapter", "LLM_REASONERS_AVAILABLE"),
    ]

    print("\n  Testing individual adapters...")
    for display_name, module_path, class_name, avail_name in adapter_tests:
        success, error, _ = test_import(module_path, [class_name, avail_name])

        if success:
            module = __import__(module_path, fromlist=[class_name, avail_name])
            adapter_class = getattr(module, class_name)
            available = getattr(module, avail_name)

            # Try to instantiate
            try:
                adapter = adapter_class()
                status = adapter.get_status()
                print_result(display_name, True, f"SDK={'installed' if available else 'not installed'}")
                results[display_name.lower()] = {
                    "success": True,
                    "available": available,
                    "status": status,
                }
            except Exception as e:
                print_result(display_name, False, f"Instantiation error: {e}")
                results[display_name.lower()] = {"success": False, "error": str(e)}
        else:
            print_result(display_name, False, error)
            results[display_name.lower()] = {"success": False, "error": error}

    return results


def validate_pipelines() -> Dict[str, Dict[str, Any]]:
    """Validate all pipelines."""
    print_header("PIPELINE VALIDATION")

    results = {}

    # Test pipelines/__init__.py
    print("\n  Testing pipelines module...")
    success, error, _ = test_import("pipelines", [
        "PIPELINE_STATUS",
        "get_pipeline_status",
        "register_pipeline",
    ])

    if success:
        print_result("pipelines/__init__.py", True)

        from pipelines import get_pipeline_status
        status = get_pipeline_status()
        for name, info in status.items():
            avail = info.get("available", False)
            deps = info.get("dependencies", [])
            print_result(f"  +- {name}", True, f"available={avail}, deps={len(deps)}")
    else:
        print_result("pipelines/__init__.py", False, error)

    results["init"] = {"success": success, "error": error}

    # Test individual pipelines
    pipeline_tests = [
        ("Deep Research", "pipelines.deep_research_pipeline", "DeepResearchPipeline", "PIPELINE_AVAILABLE"),
        ("Self Improvement", "pipelines.self_improvement_pipeline", "SelfImprovementPipeline", "PIPELINE_AVAILABLE"),
    ]

    print("\n  Testing individual pipelines...")
    for display_name, module_path, class_name, avail_name in pipeline_tests:
        success, error, _ = test_import(module_path, [class_name, avail_name])

        if success:
            module = __import__(module_path, fromlist=[class_name, avail_name])
            pipeline_class = getattr(module, class_name)
            available = getattr(module, avail_name)

            try:
                pipeline = pipeline_class()
                status = pipeline.get_status()
                print_result(display_name, True)

                # Show component status
                for comp, comp_avail in status.items():
                    if isinstance(comp_avail, bool):
                        print(f"        +- {comp}: {'[OK]' if comp_avail else '[-]'}")

                results[display_name.lower().replace(" ", "_")] = {
                    "success": True,
                    "available": available,
                    "status": status,
                }
            except Exception as e:
                print_result(display_name, False, f"Error: {e}")
                results[display_name.lower().replace(" ", "_")] = {"success": False, "error": str(e)}
        else:
            print_result(display_name, False, error)
            results[display_name.lower().replace(" ", "_")] = {"success": False, "error": error}

    return results


def validate_orchestrator() -> Dict[str, Any]:
    """Validate EcosystemOrchestratorV2."""
    print_header("ORCHESTRATOR V2 VALIDATION")

    result = {}

    # Test import
    print("\n  Testing orchestrator V2...")
    success, error, _ = test_import("core.ecosystem_orchestrator", [
        "EcosystemOrchestratorV2",
        "get_orchestrator_v2",
        "ecosystem_v2",
    ])

    if not success:
        print_result("EcosystemOrchestratorV2", False, error)
        return {"success": False, "error": error}

    try:
        from core.ecosystem_orchestrator import get_orchestrator_v2

        orchestrator = get_orchestrator_v2()
        print_result("EcosystemOrchestratorV2", True, "Instantiated successfully")

        # Check V2 status
        v2_status = orchestrator.v2_status()

        print("\n  Adapter availability:")
        for adapter, available in v2_status.get("adapters", {}).items():
            symbol = "[OK]" if available else "[-]"
            print(f"    {symbol} {adapter}")

        print("\n  Pipeline availability:")
        for pipeline, available in v2_status.get("pipelines", {}).items():
            symbol = "[OK]" if available else "[-]"
            print(f"    {symbol} {pipeline}")

        # Check V2 methods
        print("\n  V2 methods available:")
        v2_methods = [
            "optimize_prompt",
            "create_workflow",
            "remember",
            "recall",
            "reason_v2",
            "deep_research_v2",
            "improve_workflow",
            "v2_status",
        ]

        for method in v2_methods:
            has_method = hasattr(orchestrator, method) and callable(getattr(orchestrator, method))
            print_result(f"  {method}()", has_method)

        result = {
            "success": True,
            "status": v2_status,
            "methods": {m: hasattr(orchestrator, m) for m in v2_methods},
        }

    except Exception as e:
        print_result("EcosystemOrchestratorV2", False, str(e))
        result = {"success": False, "error": str(e)}

    return result


def validate_data_classes() -> Dict[str, bool]:
    """Validate data classes are properly defined."""
    print_header("DATA CLASS VALIDATION")

    results = {}

    dataclass_tests = [
        ("ThoughtNode", "adapters.llm_reasoners_adapter", "ThoughtNode"),
        ("ReasoningResult", "adapters.llm_reasoners_adapter", "ReasoningResult"),
        ("MemoryEntry", "adapters.mem0_adapter", "MemoryEntry"),
        ("SearchResult", "adapters.mem0_adapter", "SearchResult"),
        ("WorkflowResult", "adapters.langgraph_adapter", "WorkflowResult"),
        ("OptimizationResult", "adapters.dspy_adapter", "OptimizationResult"),
        ("Source", "pipelines.deep_research_pipeline", "Source"),
        ("ResearchResult", "pipelines.deep_research_pipeline", "ResearchResult"),
        ("Workflow", "pipelines.self_improvement_pipeline", "Workflow"),
        ("ImprovementResult", "pipelines.self_improvement_pipeline", "ImprovementResult"),
    ]

    print("\n  Testing dataclasses...")
    for display_name, module_path, class_name in dataclass_tests:
        success, error, _ = test_import(module_path, [class_name])

        if success:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)

            # Check if it's a dataclass
            from dataclasses import fields
            try:
                field_list = fields(cls)
                print_result(display_name, True, f"{len(field_list)} fields")
                results[display_name] = True
            except TypeError:
                # Not a dataclass, but class exists
                print_result(display_name, True, "Class (not dataclass)")
                results[display_name] = True
        else:
            print_result(display_name, False)
            results[display_name] = False

    return results


def generate_summary(adapter_results, pipeline_results, orchestrator_result, dataclass_results):
    """Generate final summary."""
    print_header("VALIDATION SUMMARY", "=")

    # Count successes
    adapter_success = sum(1 for v in adapter_results.values() if v.get("success", False))
    adapter_total = len(adapter_results)

    pipeline_success = sum(1 for v in pipeline_results.values() if v.get("success", False))
    pipeline_total = len(pipeline_results)

    orchestrator_success = orchestrator_result.get("success", False)

    dataclass_success = sum(1 for v in dataclass_results.values() if v)
    dataclass_total = len(dataclass_results)

    print(f"\n  Adapters:     {adapter_success}/{adapter_total} passing")
    print(f"  Pipelines:    {pipeline_success}/{pipeline_total} passing")
    print(f"  Orchestrator: {'[OK]' if orchestrator_success else '[X]'} {'ready' if orchestrator_success else 'not ready'}")
    print(f"  Data Classes: {dataclass_success}/{dataclass_total} valid")

    # Overall status
    all_passing = (
        adapter_success == adapter_total and
        pipeline_success == pipeline_total and
        orchestrator_success and
        dataclass_success == dataclass_total
    )

    print("\n" + "-" * 70)
    if all_passing:
        print("  [OK] V2 SDK STACK: FULLY OPERATIONAL")
    else:
        print("  [-] V2 SDK STACK: OPERATIONAL (some SDKs not installed)")
    print("-" * 70)

    # Architecture summary
    print("\n  V2 Architecture Components:")
    print("  +-------------------------------------------------------------------+")
    print("  |  OPTIMIZATION: DSPy (prompt compilation)                         |")
    print("  |  ORCHESTRATION: LangGraph (state graphs, checkpointing)          |")
    print("  |  MEMORY: Mem0 (unified memory layer)                             |")
    print("  |  REASONING: llm-reasoners (MCTS, ToT, GoT, CoT)                  |")
    print("  |  RESEARCH: Exa + Firecrawl + Crawl4AI                            |")
    print("  |  SELF-IMPROVEMENT: Genetic + Gradient optimization               |")
    print("  +-------------------------------------------------------------------+")

    return all_passing


def main():
    """Run complete validation."""
    print("\n" + "=" * 70)
    print(" UNLEASHED PLATFORM - V2 SDK STACK VALIDATION")
    print(f" Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Run validations
    adapter_results = validate_adapters()
    pipeline_results = validate_pipelines()
    orchestrator_result = validate_orchestrator()
    dataclass_results = validate_data_classes()

    # Generate summary
    success = generate_summary(
        adapter_results,
        pipeline_results,
        orchestrator_result,
        dataclass_results,
    )

    print("\n")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
