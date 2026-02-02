#!/usr/bin/env python3
"""
V35 Final Validation - All 36 SDKs (27 Native + 9 Compatibility Layers)

This script validates 100% SDK availability on Python 3.14.0 through:
- 27 native SDK installations
- 5 Phase 11 compat layers (broken SDKs on 3.14)
- 4 Phase 12 compat layers (impossible-to-install SDKs on 3.14)

Usage:
    python scripts/validate_v35_final.py
"""
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# All 36 SDK validation tests
TESTS = [
    # ==========================================================================
    # L0: Protocol Layer (3 native)
    # ==========================================================================
    ("L0", "anthropic", "import anthropic", False),
    ("L0", "openai", "import openai", False),
    ("L0", "mcp", "import mcp", False),

    # ==========================================================================
    # L1: Orchestration Layer (5 native + 1 compat)
    # ==========================================================================
    ("L1", "langgraph", "import langgraph", False),
    ("L1", "controlflow", "import controlflow", False),
    ("L1", "pydantic_ai", "import pydantic_ai", False),
    ("L1", "instructor", "import instructor", False),
    ("L1", "autogen_agentchat", "import autogen_agentchat", False),
    ("L1", "crewai_compat", "from core.orchestration.crewai_compat import CrewCompat, CREWAI_COMPAT_AVAILABLE", True),

    # ==========================================================================
    # L2: Memory Layer (3 native + 1 compat)
    # ==========================================================================
    ("L2", "mem0", "import mem0", False),
    ("L2", "graphiti_core", "import graphiti_core", False),
    ("L2", "letta", "import letta", False),
    ("L2", "zep_compat", "from core.memory.zep_compat import ZepCompat, ZEP_COMPAT_AVAILABLE", True),

    # ==========================================================================
    # L3: Structured Output (4 native + 1 compat)
    # ==========================================================================
    ("L3", "pydantic", "import pydantic", False),
    ("L3", "guidance", "import guidance", False),
    ("L3", "mirascope", "import mirascope", False),
    ("L3", "ell", "import ell", False),
    ("L3", "outlines_compat", "from core.structured.outlines_compat import OutlinesCompat, OUTLINES_COMPAT_AVAILABLE", True),

    # ==========================================================================
    # L4: Reasoning Layer (1 native + 1 compat)
    # ==========================================================================
    ("L4", "dspy", "import dspy", False),
    ("L4", "agentlite_compat", "from core.reasoning.agentlite_compat import AgentLiteCompat, AGENTLITE_COMPAT_AVAILABLE", True),

    # ==========================================================================
    # L5: Observability Layer (5 native + 2 compat)
    # ==========================================================================
    ("L5", "opik", "import opik", False),
    ("L5", "deepeval", "import deepeval", False),
    ("L5", "ragas", "import ragas", False),
    ("L5", "logfire", "import logfire", False),
    ("L5", "opentelemetry", "import opentelemetry", False),
    ("L5", "langfuse_compat", "from core.observability.langfuse_compat import LangfuseCompat, LANGFUSE_COMPAT_AVAILABLE", True),
    ("L5", "phoenix_compat", "from core.observability.phoenix_compat import PhoenixCompat, PHOENIX_COMPAT_AVAILABLE", True),

    # ==========================================================================
    # L6: Safety Layer (2 compat)
    # ==========================================================================
    ("L6", "llm_guard_compat", "from core.safety.scanner_compat import ScannerCompat, SCANNER_COMPAT_AVAILABLE", True),
    ("L6", "nemoguardrails_compat", "from core.safety.rails_compat import RailsCompat, RAILS_COMPAT_AVAILABLE", True),

    # ==========================================================================
    # L7: Processing Layer (2 native + 1 compat)
    # ==========================================================================
    ("L7", "docling", "import docling", False),
    ("L7", "markitdown", "import markitdown", False),
    ("L7", "aider_compat", "from core.processing.aider_compat import AiderCompat, AIDER_COMPAT_AVAILABLE", True),

    # ==========================================================================
    # L8: Knowledge Layer (4 native)
    # ==========================================================================
    ("L8", "llama_index", "import llama_index", False),
    ("L8", "haystack", "import haystack", False),
    ("L8", "firecrawl", "import firecrawl", False),
    ("L8", "lightrag", "import lightrag", False),
]


def run_tests(verbose: bool = True) -> dict:
    """Run all SDK validation tests."""
    results = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "timestamp": datetime.now().isoformat(),
        "total_sdks": len(TESTS),
        "native_sdks": sum(1 for t in TESTS if not t[3]),
        "compat_layers": sum(1 for t in TESTS if t[3]),
        "passed": 0,
        "failed": 0,
        "layers": {},
        "details": [],
    }

    if verbose:
        print("=" * 70)
        print("V35 FINAL VALIDATION - 36 SDKs (27 Native + 9 Compat Layers)")
        print(f"Python Version: {results['python_version']}")
        print("=" * 70)
        print()

    current_layer = None

    for layer, name, import_stmt, is_compat in TESTS:
        # Track layer statistics
        if layer not in results["layers"]:
            results["layers"][layer] = {"total": 0, "passed": 0, "failed": [], "compat_count": 0}
        results["layers"][layer]["total"] += 1
        if is_compat:
            results["layers"][layer]["compat_count"] += 1

        # Print layer header
        if verbose and layer != current_layer:
            current_layer = layer
            print(f"\n--- {layer} ---")

        # Run test
        try:
            exec(import_stmt)
            results["passed"] += 1
            results["layers"][layer]["passed"] += 1
            status = "[OK]"
            suffix = " (compat)" if is_compat else ""
            if verbose:
                print(f"  {status} {name}{suffix}")
            results["details"].append({
                "layer": layer,
                "name": name,
                "is_compat": is_compat,
                "passed": True,
                "error": None
            })
        except Exception as e:
            results["failed"] += 1
            error_msg = str(e)[:100]
            results["layers"][layer]["failed"].append(name)
            status = "[FAIL]"
            suffix = " (compat)" if is_compat else ""
            if verbose:
                print(f"  {status} {name}{suffix}: {error_msg}")
            results["details"].append({
                "layer": layer,
                "name": name,
                "is_compat": is_compat,
                "passed": False,
                "error": error_msg
            })

    # Calculate percentages
    results["percentage"] = round(100 * results["passed"] / results["total_sdks"], 1) if results["total_sdks"] > 0 else 0

    if verbose:
        print()
        print("=" * 70)
        print(f"V35 RESULT: {results['passed']}/{results['total_sdks']} ({results['percentage']}%)")
        print(f"  Native SDKs: {results['native_sdks']}")
        print(f"  Compat Layers: {results['compat_layers']}")
        print("=" * 70)

        if results["failed"] > 0:
            print(f"\n[!] {results['failed']} SDK(s) failed:")
            for layer, layer_data in results["layers"].items():
                if layer_data["failed"]:
                    for name in layer_data["failed"]:
                        print(f"    - {layer}/{name}")

        if results["passed"] == results["total_sdks"]:
            print()
            print("*" * 70)
            print("*  V35 COMPLETE - 100% SDK AVAILABILITY ON PYTHON 3.14!           *")
            print("*  27 Native SDKs + 9 Compatibility Layers = 36 Total SDKs        *")
            print("*" * 70)

    return results


def save_results(results: dict, output_path: Path = None):
    """Save validation results to JSON."""
    if output_path is None:
        output_path = PROJECT_ROOT / "validation_v35_result.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    results = run_tests(verbose=True)
    save_results(results)

    # Exit with appropriate code
    if results["passed"] == results["total_sdks"]:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
