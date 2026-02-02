#!/usr/bin/env python3
"""
Unified SDK Access Layer for Unleash V35
Provides seamless access to all 36 SDKs across 9 layers.

Usage:
    from core.sdk_access import sdk, quick

    # Get any SDK by name
    anthropic = sdk.get("anthropic")
    pydantic = sdk.get("pydantic")

    # Quick operations
    result = quick.validate({"x": 42}, schema={"x": int})
    doc = quick.document("Hello world")
    safe = quick.scan("user input")
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__version__ = "35.0.0"


@dataclass
class SDKInfo:
    """Information about an available SDK."""
    name: str
    layer: str
    category: str
    module: str
    is_compat: bool = False
    loaded: bool = False
    instance: Any = None
    error: str | None = None


# SDK Registry - All 36 SDKs
SDK_REGISTRY: dict[str, SDKInfo] = {
    # L0: Protocol Layer (6 SDKs)
    "anthropic": SDKInfo("anthropic", "L0", "protocol", "anthropic"),
    "openai": SDKInfo("openai", "L0", "protocol", "openai"),
    "cohere": SDKInfo("cohere", "L0", "protocol", "cohere"),
    "litellm": SDKInfo("litellm", "L0", "protocol", "litellm"),
    "tokenizers": SDKInfo("tokenizers", "L0", "protocol", "tokenizers"),
    "tiktoken": SDKInfo("tiktoken", "L0", "protocol", "tiktoken"),

    # L1: Orchestration Layer (5 SDKs)
    "langgraph": SDKInfo("langgraph", "L1", "orchestration", "langgraph"),
    "pydantic_ai": SDKInfo("pydantic_ai", "L1", "orchestration", "pydantic_ai"),
    "letta": SDKInfo("letta", "L1", "orchestration", "letta"),
    "crewai_compat": SDKInfo("crewai_compat", "L1", "orchestration", "core.orchestration.crewai_compat", is_compat=True),
    "controlflow": SDKInfo("controlflow", "L1", "orchestration", "controlflow"),

    # L2: Memory Layer (4 SDKs)
    "mem0": SDKInfo("mem0", "L2", "memory", "mem0"),
    "chromadb": SDKInfo("chromadb", "L2", "memory", "chromadb"),
    "qdrant": SDKInfo("qdrant", "L2", "memory", "qdrant_client"),
    "zep_compat": SDKInfo("zep_compat", "L2", "memory", "core.memory.zep_compat", is_compat=True),

    # L3: Structured Layer (4 SDKs)
    "instructor": SDKInfo("instructor", "L3", "structured", "instructor"),
    "pydantic": SDKInfo("pydantic", "L3", "structured", "pydantic"),
    "marvin": SDKInfo("marvin", "L3", "structured", "marvin"),
    "outlines_compat": SDKInfo("outlines_compat", "L3", "structured", "core.structured.outlines_compat", is_compat=True),

    # L4: Agent Layer (4 SDKs)
    "dspy": SDKInfo("dspy", "L4", "agents", "dspy"),
    "smolagents": SDKInfo("smolagents", "L4", "agents", "smolagents"),
    "agentlite_compat": SDKInfo("agentlite_compat", "L4", "agents", "core.reasoning.agentlite_compat", is_compat=True),
    "browser_use": SDKInfo("browser_use", "L4", "agents", "browser_use"),

    # L5: Observability Layer (4 SDKs)
    "opentelemetry": SDKInfo("opentelemetry", "L5", "observability", "opentelemetry"),
    "logfire": SDKInfo("logfire", "L5", "observability", "logfire"),
    "langfuse_compat": SDKInfo("langfuse_compat", "L5", "observability", "core.observability.langfuse_compat", is_compat=True),
    "phoenix_compat": SDKInfo("phoenix_compat", "L5", "observability", "core.observability.phoenix_compat", is_compat=True),

    # L6: Safety Layer (4 SDKs)
    "guardrails": SDKInfo("guardrails", "L6", "safety", "guardrails"),
    "deepeval": SDKInfo("deepeval", "L6", "safety", "deepeval"),
    "scanner_compat": SDKInfo("scanner_compat", "L6", "safety", "core.safety.scanner_compat", is_compat=True),
    "rails_compat": SDKInfo("rails_compat", "L6", "safety", "core.safety.rails_compat", is_compat=True),

    # L7: Testing Layer (2 SDKs)
    "pytest": SDKInfo("pytest", "L7", "testing", "pytest"),
    "hypothesis": SDKInfo("hypothesis", "L7", "testing", "hypothesis"),

    # L8: Knowledge Layer (3 SDKs)
    "llama_index": SDKInfo("llama_index", "L8", "knowledge", "llama_index.core"),
    "haystack": SDKInfo("haystack", "L8", "knowledge", "haystack"),
    "aider_compat": SDKInfo("aider_compat", "L8", "knowledge", "core.processing.aider_compat", is_compat=True),
}


class SDKAccessLayer:
    """Unified access to all V35 SDKs."""

    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._loaded: dict[str, bool] = {}

    def get(self, name: str, default: Any = None) -> Any:
        """Get an SDK module by name.

        Args:
            name: SDK name (e.g., 'anthropic', 'pydantic', 'langfuse_compat')
            default: Value to return if SDK not found

        Returns:
            The loaded SDK module or default
        """
        if name in self._cache:
            return self._cache[name]

        if name not in SDK_REGISTRY:
            return default

        info = SDK_REGISTRY[name]
        try:
            import importlib
            module = importlib.import_module(info.module)
            self._cache[name] = module
            self._loaded[name] = True
            return module
        except ImportError as e:
            self._loaded[name] = False
            SDK_REGISTRY[name].error = str(e)
            return default

    def get_class(self, sdk_name: str, class_name: str) -> Any:
        """Get a specific class from an SDK.

        Args:
            sdk_name: SDK name
            class_name: Class to import (e.g., 'Anthropic', 'BaseModel')

        Returns:
            The class or None
        """
        module = self.get(sdk_name)
        if module:
            return getattr(module, class_name, None)
        return None

    def available(self) -> list[str]:
        """List all available SDK names."""
        return list(SDK_REGISTRY.keys())

    def by_layer(self, layer: str) -> list[str]:
        """Get SDK names by layer (L0-L8)."""
        return [name for name, info in SDK_REGISTRY.items() if info.layer == layer]

    def by_category(self, category: str) -> list[str]:
        """Get SDK names by category."""
        return [name for name, info in SDK_REGISTRY.items() if info.category == category]

    def status(self) -> dict[str, dict]:
        """Get status of all SDKs."""
        result = {}
        for name, info in SDK_REGISTRY.items():
            result[name] = {
                "layer": info.layer,
                "category": info.category,
                "is_compat": info.is_compat,
                "loaded": self._loaded.get(name, False),
                "error": info.error,
            }
        return result

    def ensure_loaded(self, *names: str) -> dict[str, bool]:
        """Ensure multiple SDKs are loaded.

        Returns:
            Dict mapping SDK names to load success status
        """
        results = {}
        for name in names:
            module = self.get(name)
            results[name] = module is not None
        return results


class QuickOperations:
    """Quick convenience operations using loaded SDKs."""

    def __init__(self, sdk: SDKAccessLayer):
        self._sdk = sdk

    def validate(self, data: dict, schema: type | None = None) -> Any:
        """Validate data using Pydantic.

        Args:
            data: Dict to validate
            schema: Optional Pydantic model class

        Returns:
            Validated model instance
        """
        pydantic = self._sdk.get("pydantic")
        if not pydantic:
            raise ImportError("Pydantic not available")

        if schema is None:
            # Create dynamic model from dict
            from pydantic import create_model
            fields = {k: (type(v), ...) for k, v in data.items()}
            DynamicModel = create_model("DynamicModel", **fields)
            return DynamicModel(**data)

        return schema(**data)

    def document(self, text: str, metadata: dict | None = None):
        """Create a LlamaIndex document.

        Args:
            text: Document content
            metadata: Optional metadata dict

        Returns:
            LlamaIndex Document
        """
        llama = self._sdk.get("llama_index")
        if not llama:
            raise ImportError("LlamaIndex not available")

        from llama_index.core import Document
        return Document(text=text, metadata=metadata or {})

    def scan(self, text: str) -> bool:
        """Scan text for safety using scanner_compat.

        Args:
            text: Text to scan

        Returns:
            True if safe, False if flagged
        """
        scanner = self._sdk.get("scanner_compat")
        if not scanner:
            # Fallback: always safe if scanner not available
            return True

        from core.safety.scanner_compat import InputScanner
        result = InputScanner().scan(text)
        return result.is_safe

    def trace(self, name: str = "default"):
        """Get an OpenTelemetry tracer.

        Args:
            name: Tracer name

        Returns:
            OTel tracer
        """
        otel = self._sdk.get("opentelemetry")
        if not otel:
            raise ImportError("OpenTelemetry not available")

        from opentelemetry import trace
        return trace.get_tracer(name)

    def client(self, provider: str = "anthropic"):
        """Get an LLM client.

        Args:
            provider: 'anthropic', 'openai', 'cohere', or 'litellm'

        Returns:
            LLM client instance
        """
        module = self._sdk.get(provider)
        if not module:
            raise ImportError(f"{provider} not available")

        if provider == "anthropic":
            return module.Anthropic()
        elif provider == "openai":
            return module.OpenAI()
        elif provider == "cohere":
            return module.Client()
        elif provider == "litellm":
            return module

        raise ValueError(f"Unknown provider: {provider}")

    def structured(self, client, response_model: type):
        """Patch a client for structured output using Instructor.

        Args:
            client: LLM client to patch
            response_model: Pydantic model for response

        Returns:
            Patched client
        """
        instructor = self._sdk.get("instructor")
        if not instructor:
            raise ImportError("Instructor not available")

        return instructor.from_anthropic(client)


# Global instances
sdk = SDKAccessLayer()
quick = QuickOperations(sdk)


def inventory() -> str:
    """Get a formatted inventory of all SDKs."""
    lines = [
        "=" * 60,
        "UNLEASH V35 SDK INVENTORY",
        "=" * 60,
        "",
    ]

    by_layer = {}
    for name, info in SDK_REGISTRY.items():
        if info.layer not in by_layer:
            by_layer[info.layer] = []
        by_layer[info.layer].append((name, info))

    for layer in sorted(by_layer.keys()):
        lines.append(f"{layer}: {SDK_REGISTRY[by_layer[layer][0][0]].category.upper()}")
        lines.append("-" * 40)
        for name, info in by_layer[layer]:
            compat = " (compat)" if info.is_compat else ""
            lines.append(f"  - {name}{compat}")
        lines.append("")

    lines.extend([
        "=" * 60,
        f"TOTAL: {len(SDK_REGISTRY)} SDKs (27 native + 9 compat)",
        "=" * 60,
    ])

    return "\n".join(lines)


def test_all() -> dict[str, bool]:
    """Test loading all SDKs.

    Returns:
        Dict mapping SDK names to load success
    """
    results = {}
    for name in SDK_REGISTRY:
        try:
            module = sdk.get(name)
            results[name] = module is not None
        except Exception:
            results[name] = False

    return results


if __name__ == "__main__":
    print(inventory())
    print()
    print("Testing SDK Access...")
    results = test_all()
    passed = sum(1 for v in results.values() if v)
    print(f"Result: {passed}/{len(results)} SDKs accessible")
