#!/usr/bin/env python3
"""
Comprehensive Import Validation Script for UNLEASH Platform
============================================================

This script validates all imports across the platform modules:
1. RAG modules from core.rag
2. Memory modules from core.memory
3. Adapters from platform.adapters
4. Platform startup and API facade
5. Platform CLI

For each import, it reports:
- Success or failure
- Missing dependencies
- Circular import issues

Run: python -m platform.tests.test_import_validation
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ImportResult:
    """Result of an import attempt."""
    module: str
    success: bool
    error: Optional[str] = None
    error_type: Optional[str] = None
    time_ms: float = 0.0
    attributes_found: List[str] = field(default_factory=list)
    missing_dependencies: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Overall validation report."""
    timestamp: str
    total_modules: int = 0
    successful: int = 0
    failed: int = 0
    missing_deps: int = 0
    circular_imports: int = 0
    total_time_ms: float = 0.0
    results: List[ImportResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_modules": self.total_modules,
                "successful": self.successful,
                "failed": self.failed,
                "missing_dependencies": self.missing_deps,
                "circular_imports": self.circular_imports,
                "total_time_ms": round(self.total_time_ms, 2),
            },
            "results": [
                {
                    "module": r.module,
                    "success": r.success,
                    "error": r.error,
                    "error_type": r.error_type,
                    "time_ms": round(r.time_ms, 2),
                    "missing_dependencies": r.missing_dependencies,
                }
                for r in self.results
                if not r.success or r.missing_dependencies
            ],
            "successful_modules": [r.module for r in self.results if r.success],
        }


def detect_circular_import(error_str: str) -> bool:
    """Detect if an error indicates a circular import."""
    circular_indicators = [
        "circular import",
        "partially initialized module",
        "cannot import name",
        "most likely due to a circular import",
    ]
    return any(indicator in error_str.lower() for indicator in circular_indicators)


def extract_missing_dep(error_str: str) -> List[str]:
    """Extract missing dependency names from import error."""
    missing = []
    if "No module named" in error_str:
        # Extract module name
        parts = error_str.split("No module named")
        if len(parts) > 1:
            dep = parts[1].strip().strip("'\"")
            # Get root package
            missing.append(dep.split(".")[0])
    elif "cannot import name" in error_str:
        parts = error_str.split("cannot import name")
        if len(parts) > 1:
            name_part = parts[1].strip().split("from")[0].strip().strip("'\"")
            missing.append(name_part)
    return missing


def try_import(module_name: str) -> ImportResult:
    """Try to import a module and capture results."""
    start = time.time()
    result = ImportResult(module=module_name, success=False)

    try:
        # Clear module from cache to ensure fresh import
        # (but don't actually remove to avoid breaking things)
        module = importlib.import_module(module_name)
        result.success = True
        result.time_ms = (time.time() - start) * 1000

        # Get exported attributes
        if hasattr(module, "__all__"):
            result.attributes_found = list(module.__all__)[:20]  # First 20
        else:
            result.attributes_found = [
                name for name in dir(module)
                if not name.startswith("_")
            ][:20]

    except ImportError as e:
        result.error = str(e)
        result.error_type = "ImportError"
        result.missing_dependencies = extract_missing_dep(str(e))
        result.time_ms = (time.time() - start) * 1000

    except Exception as e:
        result.error = str(e)
        result.error_type = type(e).__name__
        result.time_ms = (time.time() - start) * 1000

        if detect_circular_import(str(e)):
            result.error_type = "CircularImport"

    return result


def validate_rag_modules() -> List[ImportResult]:
    """Validate all RAG module imports."""
    results = []

    # Main RAG package
    results.append(try_import("platform.core.rag"))

    # Individual RAG modules
    rag_modules = [
        "platform.core.rag.semantic_chunker",
        "platform.core.rag.self_rag",
        "platform.core.rag.corrective_rag",
        "platform.core.rag.hyde",
        "platform.core.rag.reranker",
        "platform.core.rag.raptor",
        "platform.core.rag.evaluation",
        "platform.core.rag.graph_rag",
        "platform.core.rag.metrics",
        "platform.core.rag.context_manager",
        "platform.core.rag.query_rewriter",
        "platform.core.rag.colbert_retriever",
        "platform.core.rag.agentic_rag",
        "platform.core.rag.cache_warmer",
        "platform.core.rag.streaming",
        "platform.core.rag.result_cache",
        "platform.core.rag.complexity_analyzer",
        "platform.core.rag.hallucination_guard",
        "platform.core.rag.contextual_retrieval",
        "platform.core.rag.dual_level_retrieval",
        "platform.core.rag.pipeline",
    ]

    for module in rag_modules:
        results.append(try_import(module))

    return results


def validate_memory_modules() -> List[ImportResult]:
    """Validate all memory module imports."""
    results = []

    # Main memory package
    results.append(try_import("platform.core.memory"))

    # Backend modules
    memory_modules = [
        "platform.core.memory.backends",
        "platform.core.memory.backends.base",
        "platform.core.memory.backends.in_memory",
        "platform.core.memory.backends.letta",
        "platform.core.memory.backends.sqlite",
        "platform.core.memory.backends.hnsw",
        "platform.core.memory.backends.graphiti",
        "platform.core.memory.hooks",
        "platform.core.memory.quality",
        "platform.core.memory.forgetting",
        "platform.core.memory.temporal",
        "platform.core.memory.procedural",
        "platform.core.memory.compression",
        "platform.core.memory.unified",
        "platform.core.memory.compaction",
    ]

    for module in memory_modules:
        results.append(try_import(module))

    return results


def validate_adapter_modules() -> List[ImportResult]:
    """Validate all adapter module imports."""
    results = []

    # Main adapters package
    results.append(try_import("platform.adapters"))

    # Research adapters
    adapter_modules = [
        "platform.adapters.exa_adapter",
        "platform.adapters.tavily_adapter",
        "platform.adapters.perplexity_adapter",
        "platform.adapters.jina_adapter",
        "platform.adapters.firecrawl_adapter",
        "platform.adapters.context7_adapter",
        "platform.adapters.serper_adapter",
        # Memory adapters
        "platform.adapters.letta_adapter",
        "platform.adapters.cognee_adapter",
        "platform.adapters.cognee_v36_adapter",
        "platform.adapters.mem0_adapter",
        "platform.adapters.graphiti_adapter",
        # Orchestration adapters
        "platform.adapters.dspy_adapter",
        "platform.adapters.langgraph_adapter",
        "platform.adapters.openai_agents_adapter",
        "platform.adapters.strands_agents_adapter",
        "platform.adapters.a2a_protocol_adapter",
        # Code/Reasoning adapters
        "platform.adapters.aider_adapter",
        "platform.adapters.llm_reasoners_adapter",
        "platform.adapters.textgrad_adapter",
        # Observability adapters
        "platform.adapters.opik_tracing_adapter",
        "platform.adapters.braintrust_adapter",
        "platform.adapters.portkey_gateway_adapter",
        "platform.adapters.observability_adapter",
        # Workflow/Utility adapters
        "platform.adapters.temporal_workflow_activities",
        "platform.adapters.ragflow_adapter",
        "platform.adapters.simplemem_adapter",
        "platform.adapters.ragatouille_adapter",
        "platform.adapters.chonkie_adapter",
        "platform.adapters.quality_diversity_adapter",
        "platform.adapters.agent_mesh",
        "platform.adapters.letta_voyage_adapter",
        "platform.adapters.dspy_voyage_retriever",
        "platform.adapters.mcp_apps_adapter",
        "platform.adapters.async_letta",
        # Infrastructure
        "platform.adapters.registry",
        "platform.adapters.retry",
        "platform.adapters.http_pool",
        "platform.adapters.circuit_breaker_manager",
        "platform.adapters.model_router",
        "platform.adapters.safety_adapter",
        "platform.adapters.token_optimizer",
        "platform.adapters.knowledge_adapter",
        "platform.adapters.platform_orchestrator_v2",
        "platform.adapters.rag_evaluator",
    ]

    for module in adapter_modules:
        results.append(try_import(module))

    return results


def validate_platform_modules() -> List[ImportResult]:
    """Validate platform startup and facade modules."""
    results = []

    platform_modules = [
        # Main platform package
        "platform",
        # Startup
        "platform.startup",
        # CLI
        "platform.cli",
        # Core platform modules
        "platform.core",
        "platform.core.api_validator",
    ]

    for module in platform_modules:
        results.append(try_import(module))

    return results


def run_validation() -> ValidationReport:
    """Run the full import validation."""
    report = ValidationReport(
        timestamp=datetime.now(timezone.utc).isoformat()
    )

    start_time = time.time()

    print("=" * 70)
    print("UNLEASH Platform Import Validation")
    print("=" * 70)
    print(f"Started: {report.timestamp}")
    print()

    # Validate each category
    categories = [
        ("RAG Modules (platform.core.rag)", validate_rag_modules),
        ("Memory Modules (platform.core.memory)", validate_memory_modules),
        ("Adapters (platform.adapters)", validate_adapter_modules),
        ("Platform Startup/CLI/Core", validate_platform_modules),
    ]

    for category_name, validator in categories:
        print(f"\n--- {category_name} ---")
        results = validator()

        for r in results:
            report.results.append(r)
            report.total_modules += 1

            if r.success:
                report.successful += 1
                status = "[OK]"
            else:
                report.failed += 1
                status = "[FAIL]"

                if r.error_type == "CircularImport":
                    report.circular_imports += 1
                if r.missing_dependencies:
                    report.missing_deps += 1

            # Print result
            print(f"  {status} {r.module.split('.')[-1]:<35} ({r.time_ms:.1f}ms)", end="")
            if r.error:
                print(f" - {r.error_type}: {r.error[:50]}...", end="")
            print()

    report.total_time_ms = (time.time() - start_time) * 1000

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Modules:        {report.total_modules}")
    print(f"Successful:           {report.successful}")
    print(f"Failed:               {report.failed}")
    print(f"Missing Dependencies: {report.missing_deps}")
    print(f"Circular Imports:     {report.circular_imports}")
    print(f"Total Time:           {report.total_time_ms:.1f}ms")
    print()

    # Success rate
    if report.total_modules > 0:
        success_rate = (report.successful / report.total_modules) * 100
        print(f"Success Rate:         {success_rate:.1f}%")

    # List failed modules
    if report.failed > 0:
        print()
        print("Failed Modules:")
        print("-" * 40)
        for r in report.results:
            if not r.success:
                print(f"  - {r.module}")
                print(f"    Error: {r.error}")
                if r.missing_dependencies:
                    print(f"    Missing: {', '.join(r.missing_dependencies)}")

    # List missing dependencies
    all_missing = set()
    for r in report.results:
        all_missing.update(r.missing_dependencies)

    if all_missing:
        print()
        print("All Missing Dependencies:")
        print("-" * 40)
        for dep in sorted(all_missing):
            print(f"  - {dep}")

    print()
    print("=" * 70)

    return report


if __name__ == "__main__":
    report = run_validation()

    # Exit with error code if there are failures
    if report.failed > 0:
        sys.exit(1)
    sys.exit(0)
