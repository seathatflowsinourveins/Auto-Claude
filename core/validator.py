#!/usr/bin/env python3
"""
SDK Validator - Comprehensive Health Check System
Part of V33 Architecture - Complete 8-Layer Stack.

Provides unified validation across all V33 layers (L0-L8):
- L1 Orchestration: temporal, langgraph, claude_flow, crewai, autogen
- L2 Memory: mem0, zep, letta, cross_session
- L3 Structured: instructor, baml, outlines, pydantic_ai
- L4 Reasoning: dspy, serena
- L5 Observability: langfuse, phoenix, opik, deepeval, ragas, logfire, opentelemetry
- L6 Safety: guardrails, llm_guard, nemo_guardrails
- L7 Processing: crawl4ai, firecrawl, aider, ast_grep
- L8 Knowledge: graphrag, pyribs

NO STUBS: Returns explicit validation results with actionable fixes.

Usage:
    from core.validator import (
        validate_all,
        validate_layer,
        SDKValidationResult,
        LayerValidationResult,
        FullValidationResult,
    )

    # Full system validation
    result = validate_all()
    print(f"Overall: {result.status}")
    for layer_result in result.layers.values():
        print(f"  {layer_result.layer}: {layer_result.available}/{layer_result.total}")

    # Single layer validation
    obs_result = validate_layer("observability")
    for sdk, sdk_result in obs_result.sdks.items():
        if not sdk_result.available:
            print(f"  Install: {sdk_result.install_cmd}")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable


# =============================================================================
# Validation Result Types
# =============================================================================

class ValidationStatus(str, Enum):
    """Overall validation status."""
    HEALTHY = "healthy"           # All required SDKs available and configured
    PARTIAL = "partial"           # Some optional SDKs missing
    DEGRADED = "degraded"         # Core functionality affected
    CRITICAL = "critical"         # System cannot function


@dataclass
class SDKValidationResult:
    """Result of validating a single SDK."""
    name: str
    available: bool
    configured: bool
    required: bool
    error: Optional[str] = None
    install_cmd: str = ""
    docs_url: str = ""
    missing_config: List[str] = field(default_factory=list)


@dataclass
class LayerValidationResult:
    """Result of validating a V33 layer."""
    layer: str
    total: int
    available: int
    configured: int
    status: ValidationStatus
    sdks: Dict[str, SDKValidationResult] = field(default_factory=dict)

    @property
    def all_available(self) -> bool:
        return self.available == self.total

    @property
    def all_configured(self) -> bool:
        return self.configured == self.total


@dataclass
class FullValidationResult:
    """Result of validating the entire V33 system."""
    status: ValidationStatus
    layers: Dict[str, LayerValidationResult] = field(default_factory=dict)
    total_sdks: int = 0
    available_sdks: int = 0
    configured_sdks: int = 0
    critical_missing: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "total_sdks": self.total_sdks,
            "available_sdks": self.available_sdks,
            "configured_sdks": self.configured_sdks,
            "critical_missing": self.critical_missing,
            "layers": {
                name: {
                    "layer": layer.layer,
                    "total": layer.total,
                    "available": layer.available,
                    "configured": layer.configured,
                    "status": layer.status.value,
                    "sdks": {
                        sdk_name: {
                            "name": sdk.name,
                            "available": sdk.available,
                            "configured": sdk.configured,
                            "required": sdk.required,
                            "error": sdk.error,
                            "install_cmd": sdk.install_cmd,
                        }
                        for sdk_name, sdk in layer.sdks.items()
                    }
                }
                for name, layer in self.layers.items()
            }
        }


# =============================================================================
# SDK Registry - All SDKs with metadata
# =============================================================================

SDK_REGISTRY = {
    # =========================================================================
    # L5: Observability Layer
    # =========================================================================
    "observability": {
        "langfuse": {
            "required": False,
            "install_cmd": "pip install langfuse>=2.0.0",
            "docs_url": "https://langfuse.com/docs/sdk/python",
            "env_vars": ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"],
            "check_import": "langfuse",
        },
        "phoenix": {
            "required": False,
            "install_cmd": "pip install arize-phoenix>=4.0.0",
            "docs_url": "https://docs.arize.com/phoenix",
            "env_vars": [],  # Works without config
            "check_import": "phoenix",
        },
        "opik": {
            "required": False,
            "install_cmd": "pip install opik>=1.0.0",
            "docs_url": "https://www.comet.com/docs/opik/",
            "env_vars": ["OPIK_API_KEY"],
            "check_import": "opik",
        },
        "deepeval": {
            "required": False,
            "install_cmd": "pip install deepeval>=1.0.0",
            "docs_url": "https://docs.confident-ai.com/",
            "env_vars": [],  # Works locally
            "check_import": "deepeval",
        },
        "ragas": {
            "required": False,
            "install_cmd": "pip install ragas>=0.2.0",
            "docs_url": "https://docs.ragas.io/",
            "env_vars": [],  # Works locally
            "check_import": "ragas",
        },
        "logfire": {
            "required": False,
            "install_cmd": "pip install logfire>=0.30.0",
            "docs_url": "https://logfire.pydantic.dev/docs/",
            "env_vars": ["LOGFIRE_TOKEN"],
            "check_import": "logfire",
        },
        "opentelemetry": {
            "required": False,
            "install_cmd": "pip install opentelemetry-api opentelemetry-sdk>=1.20.0",
            "docs_url": "https://opentelemetry.io/docs/instrumentation/python/",
            "env_vars": [],  # Works without config
            "check_import": "opentelemetry",
        },
    },

    # =========================================================================
    # L2: Memory Layer
    # =========================================================================
    "memory": {
        "mem0": {
            "required": False,
            "install_cmd": "pip install mem0ai>=0.1.0",
            "docs_url": "https://docs.mem0.ai/overview",
            "env_vars": [],  # Can work locally
            "check_import": "mem0",
        },
        "zep": {
            "required": False,
            "install_cmd": "pip install zep-python>=2.0.0",
            "docs_url": "https://docs.getzep.com/sdk/python/",
            "env_vars": ["ZEP_API_KEY"],
            "check_import": "zep_python",
        },
        "letta": {
            "required": False,
            "install_cmd": "pip install letta>=0.4.0",
            "docs_url": "https://docs.letta.com/",
            "env_vars": [],  # Can work locally
            "check_import": "letta",
        },
    },

    # =========================================================================
    # L1: Orchestration Layer
    # =========================================================================
    "orchestration": {
        "temporal": {
            "required": False,
            "install_cmd": "pip install temporalio>=1.5.0",
            "docs_url": "https://docs.temporal.io/dev-guide/python",
            "env_vars": [],  # Uses address config
            "check_import": "temporalio",
        },
        "langgraph": {
            "required": False,
            "install_cmd": "pip install langgraph>=0.2.0 langchain-core>=0.2.0",
            "docs_url": "https://python.langchain.com/docs/langgraph",
            "env_vars": [],  # Uses model-specific keys
            "check_import": "langgraph",
        },
        "claude_flow": {
            "required": False,
            "install_cmd": "pip install anthropic>=0.30.0",
            "docs_url": "https://docs.anthropic.com/claude/reference",
            "env_vars": ["ANTHROPIC_API_KEY"],
            "check_import": "anthropic",
        },
        "crewai": {
            "required": False,
            "install_cmd": "pip install crewai>=0.50.0",
            "docs_url": "https://docs.crewai.com/",
            "env_vars": [],  # Uses model-specific keys
            "check_import": "crewai",
        },
        "autogen": {
            "required": False,
            "install_cmd": "pip install pyautogen>=0.3.0",
            "docs_url": "https://microsoft.github.io/autogen/",
            "env_vars": [],  # Uses model-specific keys
            "check_import": "autogen",
        },
    },

    # =========================================================================
    # L3: Structured Output Layer
    # =========================================================================
    "structured": {
        "instructor": {
            "required": False,
            "install_cmd": "pip install instructor>=1.0.0",
            "docs_url": "https://python.useinstructor.com/",
            "env_vars": [],  # Uses model-specific keys
            "check_import": "instructor",
        },
        "baml": {
            "required": False,
            "install_cmd": "pip install baml>=0.55.0",
            "docs_url": "https://docs.boundaryml.com/",
            "env_vars": [],  # Uses model-specific keys
            "check_import": "baml",
        },
        "outlines": {
            "required": False,
            "install_cmd": "pip install outlines>=0.0.36",
            "docs_url": "https://outlines-dev.github.io/outlines/",
            "env_vars": [],  # Uses local models
            "check_import": "outlines",
        },
        "pydantic_ai": {
            "required": False,
            "install_cmd": "pip install pydantic-ai>=0.0.1",
            "docs_url": "https://ai.pydantic.dev/",
            "env_vars": [],  # Uses model-specific keys
            "check_import": "pydantic_ai",
        },
    },

    # =========================================================================
    # L4: Reasoning Layer
    # =========================================================================
    "reasoning": {
        "dspy": {
            "required": False,
            "install_cmd": "pip install dspy>=2.5.0",
            "docs_url": "https://dspy.ai/",
            "env_vars": [],  # Uses model-specific keys
            "check_import": "dspy",
        },
        "serena": {
            "required": False,
            "install_cmd": "pip install serena-mcp>=1.0.0",
            "docs_url": "https://github.com/serena-ai/serena",
            "env_vars": [],  # Uses local config
            "check_import": "serena",
        },
    },

    # =========================================================================
    # L6: Safety Layer
    # =========================================================================
    "safety": {
        "guardrails": {
            "required": False,
            "install_cmd": "pip install guardrails-ai>=0.5.0",
            "docs_url": "https://docs.guardrailsai.com/",
            "env_vars": [],  # Uses local validators
            "check_import": "guardrails",
        },
        "llm_guard": {
            "required": False,
            "install_cmd": "pip install llm-guard>=0.3.0",
            "docs_url": "https://llm-guard.com/",
            "env_vars": [],  # Uses local scanners
            "check_import": "llm_guard",
        },
        "nemo_guardrails": {
            "required": False,
            "install_cmd": "pip install nemoguardrails>=0.10.0",
            "docs_url": "https://docs.nvidia.com/nemo/guardrails/",
            "env_vars": [],  # Uses Colang config
            "check_import": "nemoguardrails",
        },
    },

    # =========================================================================
    # L7: Processing Layer
    # =========================================================================
    "processing": {
        "crawl4ai": {
            "required": False,
            "install_cmd": "pip install crawl4ai>=0.3.0",
            "docs_url": "https://github.com/unclecode/crawl4ai",
            "env_vars": [],  # Works locally
            "check_import": "crawl4ai",
        },
        "firecrawl": {
            "required": False,
            "install_cmd": "pip install firecrawl-py>=1.0.0",
            "docs_url": "https://docs.firecrawl.dev/",
            "env_vars": ["FIRECRAWL_API_KEY"],
            "check_import": "firecrawl",
        },
        "aider": {
            "required": False,
            "install_cmd": "pip install aider-chat>=0.50.0",
            "docs_url": "https://aider.chat/docs/",
            "env_vars": [],  # Uses model-specific keys
            "check_import": "aider",
        },
        "ast_grep": {
            "required": False,
            "install_cmd": "pip install ast-grep-py>=0.20.0",
            "docs_url": "https://ast-grep.github.io/",
            "env_vars": [],  # Works locally
            "check_import": "ast_grep",
        },
    },

    # =========================================================================
    # L8: Knowledge Layer
    # =========================================================================
    "knowledge": {
        "graphrag": {
            "required": False,
            "install_cmd": "pip install graphrag>=0.3.0",
            "docs_url": "https://microsoft.github.io/graphrag/",
            "env_vars": [],  # Uses model-specific keys
            "check_import": "graphrag",
        },
        "pyribs": {
            "required": False,
            "install_cmd": "pip install ribs>=0.7.0",
            "docs_url": "https://docs.pyribs.org/",
            "env_vars": [],  # Works locally
            "check_import": "ribs",
        },
    },
}

# Required core dependencies (system fails without these)
REQUIRED_CORE = {
    "pydantic": {
        "install_cmd": "pip install pydantic>=2.0.0",
        "check_import": "pydantic",
    },
    "httpx": {
        "install_cmd": "pip install httpx>=0.24.0",
        "check_import": "httpx",
    },
    "click": {
        "install_cmd": "pip install click>=8.0.0",
        "check_import": "click",
    },
    "rich": {
        "install_cmd": "pip install rich>=13.0.0",
        "check_import": "rich",
    },
}


# =============================================================================
# Validation Functions
# =============================================================================

def _check_import(module_name: str) -> tuple[bool, Optional[str]]:
    """Check if a module can be imported.

    Catches ALL exceptions, not just ImportError, to handle:
    - Pydantic v1 compatibility errors on Python 3.14+
    - Other runtime initialization errors in SDK dependencies
    """
    try:
        __import__(module_name)
        return True, None
    except Exception as e:
        return False, str(e)


def _check_config(env_vars: List[str]) -> tuple[bool, List[str]]:
    """Check if required environment variables are set."""
    missing = [var for var in env_vars if not os.getenv(var)]
    return len(missing) == 0, missing


def validate_sdk(
    name: str,
    layer: str,
) -> SDKValidationResult:
    """
    Validate a single SDK.

    Args:
        name: SDK name
        layer: V33 layer name

    Returns:
        SDKValidationResult with availability and configuration status
    """
    if layer not in SDK_REGISTRY:
        return SDKValidationResult(
            name=name,
            available=False,
            configured=False,
            required=False,
            error=f"Unknown layer: {layer}",
        )

    if name not in SDK_REGISTRY[layer]:
        return SDKValidationResult(
            name=name,
            available=False,
            configured=False,
            required=False,
            error=f"Unknown SDK: {name} in layer {layer}",
        )

    sdk_info = SDK_REGISTRY[layer][name]

    # Check import availability
    available, import_error = _check_import(sdk_info["check_import"])

    # Check configuration (only if available)
    if available:
        configured, missing_config = _check_config(sdk_info["env_vars"])
    else:
        configured = False
        missing_config = sdk_info["env_vars"]  # All config missing if not installed

    return SDKValidationResult(
        name=name,
        available=available,
        configured=configured,
        required=sdk_info["required"],
        error=import_error,
        install_cmd=sdk_info["install_cmd"],
        docs_url=sdk_info["docs_url"],
        missing_config=missing_config,
    )


def validate_layer(layer: str) -> LayerValidationResult:
    """
    Validate all SDKs in a V33 layer.

    Args:
        layer: Layer name (observability, memory, orchestration, structured)

    Returns:
        LayerValidationResult with all SDK statuses
    """
    if layer not in SDK_REGISTRY:
        return LayerValidationResult(
            layer=layer,
            total=0,
            available=0,
            configured=0,
            status=ValidationStatus.CRITICAL,
        )

    sdks = {}
    available_count = 0
    configured_count = 0
    required_missing = False

    for sdk_name in SDK_REGISTRY[layer]:
        result = validate_sdk(sdk_name, layer)
        sdks[sdk_name] = result

        if result.available:
            available_count += 1
        if result.configured:
            configured_count += 1
        if result.required and not result.available:
            required_missing = True

    total = len(SDK_REGISTRY[layer])

    # Determine status
    if required_missing:
        status = ValidationStatus.CRITICAL
    elif available_count == 0:
        status = ValidationStatus.DEGRADED
    elif available_count < total:
        status = ValidationStatus.PARTIAL
    else:
        status = ValidationStatus.HEALTHY

    return LayerValidationResult(
        layer=layer,
        total=total,
        available=available_count,
        configured=configured_count,
        status=status,
        sdks=sdks,
    )


def validate_core() -> Dict[str, SDKValidationResult]:
    """
    Validate core dependencies required for system operation.

    Returns:
        Dictionary of core dependency validation results
    """
    results = {}
    for name, info in REQUIRED_CORE.items():
        available, error = _check_import(info["check_import"])
        results[name] = SDKValidationResult(
            name=name,
            available=available,
            configured=available,  # Core deps don't need config
            required=True,
            error=error,
            install_cmd=info["install_cmd"],
        )
    return results


def validate_all() -> FullValidationResult:
    """
    Validate the entire V33 system.

    Returns:
        FullValidationResult with comprehensive status of all layers and SDKs
    """
    layers = {}
    total_sdks = 0
    available_sdks = 0
    configured_sdks = 0
    critical_missing = []

    # Validate core dependencies first
    core_results = validate_core()
    for name, result in core_results.items():
        if not result.available:
            critical_missing.append(f"core:{name}")

    # Validate each layer
    for layer_name in SDK_REGISTRY:
        layer_result = validate_layer(layer_name)
        layers[layer_name] = layer_result

        total_sdks += layer_result.total
        available_sdks += layer_result.available
        configured_sdks += layer_result.configured

        # Track critical missing SDKs
        for sdk_name, sdk_result in layer_result.sdks.items():
            if sdk_result.required and not sdk_result.available:
                critical_missing.append(f"{layer_name}:{sdk_name}")

    # Determine overall status
    if critical_missing:
        status = ValidationStatus.CRITICAL
    elif available_sdks == 0:
        status = ValidationStatus.DEGRADED
    elif available_sdks < total_sdks:
        status = ValidationStatus.PARTIAL
    else:
        status = ValidationStatus.HEALTHY

    return FullValidationResult(
        status=status,
        layers=layers,
        total_sdks=total_sdks,
        available_sdks=available_sdks,
        configured_sdks=configured_sdks,
        critical_missing=critical_missing,
    )


# =============================================================================
# Pretty Print Functions
# =============================================================================

def print_validation_report(result: FullValidationResult) -> None:
    """
    Print a formatted validation report.

    Args:
        result: FullValidationResult to display
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        _print_rich_report(result)
    except ImportError:
        _print_plain_report(result)


def _print_rich_report(result: FullValidationResult) -> None:
    """Print validation report using Rich."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    # Header
    status_color = {
        ValidationStatus.HEALTHY: "green",
        ValidationStatus.PARTIAL: "yellow",
        ValidationStatus.DEGRADED: "orange1",
        ValidationStatus.CRITICAL: "red",
    }[result.status]

    console.print(Panel(
        f"[bold]V33 SDK Validation Report[/bold]\n"
        f"Status: [{status_color}]{result.status.value.upper()}[/{status_color}]\n"
        f"SDKs: {result.available_sdks}/{result.total_sdks} available, "
        f"{result.configured_sdks}/{result.total_sdks} configured",
        title="System Health",
    ))

    # Critical missing
    if result.critical_missing:
        console.print("\n[red bold]Critical Missing Dependencies:[/red bold]")
        for item in result.critical_missing:
            console.print(f"  [red]✗[/red] {item}")

    # Layer details
    for layer_name, layer in result.layers.items():
        table = Table(title=f"\n{layer_name.title()} Layer")
        table.add_column("SDK", style="cyan")
        table.add_column("Available")
        table.add_column("Configured")
        table.add_column("Install Command")

        for sdk_name, sdk in layer.sdks.items():
            avail = "[green]✓[/green]" if sdk.available else "[red]✗[/red]"
            conf = "[green]✓[/green]" if sdk.configured else "[yellow]○[/yellow]"
            install = "" if sdk.available else sdk.install_cmd

            table.add_row(sdk_name, avail, conf, install)

        console.print(table)


def _print_plain_report(result: FullValidationResult) -> None:
    """Print validation report using plain text."""
    print("=" * 60)
    print("V33 SDK Validation Report")
    print("=" * 60)
    print(f"Status: {result.status.value.upper()}")
    print(f"SDKs: {result.available_sdks}/{result.total_sdks} available")
    print(f"      {result.configured_sdks}/{result.total_sdks} configured")
    print()

    if result.critical_missing:
        print("CRITICAL MISSING:")
        for item in result.critical_missing:
            print(f"  X {item}")
        print()

    for layer_name, layer in result.layers.items():
        print(f"\n{layer_name.upper()} LAYER ({layer.available}/{layer.total}):")
        print("-" * 40)

        for sdk_name, sdk in layer.sdks.items():
            avail = "OK" if sdk.available else "MISSING"
            conf = "OK" if sdk.configured else "NEEDS CONFIG"
            print(f"  {sdk_name}: {avail} / {conf}")
            if not sdk.available:
                print(f"    Install: {sdk.install_cmd}")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Result types
    "ValidationStatus",
    "SDKValidationResult",
    "LayerValidationResult",
    "FullValidationResult",

    # Validation functions
    "validate_sdk",
    "validate_layer",
    "validate_core",
    "validate_all",

    # Display
    "print_validation_report",

    # Registry (for programmatic access)
    "SDK_REGISTRY",
    "REQUIRED_CORE",
]


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    result = validate_all()
    print_validation_report(result)
