#!/usr/bin/env python3
"""
Reasoning Layer - Unified Interface for Prompt Optimization & Semantic Editing
Part of the V33 Architecture (Layer 4) - Phase 9 Production Fix.

Provides unified access to two reasoning SDKs:
- dspy: Programming language model pipelines with automatic prompt optimization
- serena: Semantic code editing with LSP integration

NO STUBS: All SDKs must be explicitly installed and configured.
Missing SDKs raise SDKNotAvailableError with install instructions.
Misconfigured SDKs raise SDKConfigurationError with missing config.

Usage:
    from core.reasoning import (
        # Exceptions
        SDKNotAvailableError,
        SDKConfigurationError,

        # DSPy
        get_dspy_lm,
        get_dspy_optimizer,
        DSPyClient,
        DSPY_AVAILABLE,

        # Serena
        get_serena_client,
        SerenaClient,
        SERENA_AVAILABLE,

        # Factory
        ReasoningFactory,
    )

    # Quick start with explicit error handling
    try:
        lm = get_dspy_lm()
    except SDKNotAvailableError as e:
        print(f"Install: {e.install_cmd}")
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Optional, List, Dict, Callable, Union
from dataclasses import dataclass, field

# Import exceptions from observability layer
from core.observability import (
    SDKNotAvailableError,
    SDKConfigurationError,
)


# ============================================================================
# SDK Availability Checks - Import-time validation
# ============================================================================

# DSPy - Prompt Optimization
DSPY_AVAILABLE = False
DSPY_ERROR = None
try:
    import dspy
    from dspy import LM as DSPyLM
    from dspy import ChainOfThought, ReAct, Predict
    from dspy import MIPROv2
    from dspy import Evaluate
    from dspy import Signature, InputField, OutputField
    DSPY_AVAILABLE = True
except Exception as e:
    DSPY_ERROR = str(e)

# Serena - Semantic Editing (MCP-based)
SERENA_AVAILABLE = False
SERENA_ERROR = None
try:
    # Serena is accessed via MCP, check if mcp is available
    from mcp import ClientSession
    SERENA_AVAILABLE = True
except Exception as e:
    SERENA_ERROR = str(e)


# ============================================================================
# DSPy Types and Models
# ============================================================================

class OptimizerMode(str, Enum):
    """MIPROv2 optimization intensity modes."""
    LIGHT = "light"      # Quick optimization, ~5 trials
    MEDIUM = "medium"    # Balanced, ~20 trials
    HEAVY = "heavy"      # Thorough, ~100+ trials


class ReasoningModule(str, Enum):
    """Available DSPy reasoning modules."""
    PREDICT = "predict"           # Basic prediction
    CHAIN_OF_THOUGHT = "cot"      # Step-by-step reasoning
    REACT = "react"               # Reason + Act interleaved
    PROGRAM_OF_THOUGHT = "pot"    # Code-based reasoning


@dataclass
class DSPyConfig:
    """Configuration for DSPy client."""
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    cache: bool = True


@dataclass
class OptimizationResult:
    """Result from DSPy optimization."""
    original_score: float
    optimized_score: float
    improvement: float
    num_trials: int
    best_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """Result from a reasoning operation."""
    output: str
    reasoning: Optional[str] = None
    confidence: float = 0.0
    tokens_used: int = 0
    latency_ms: float = 0.0
    trace: Optional[List[str]] = None


# ============================================================================
# DSPy Client Implementation
# ============================================================================

class DSPyClient:
    """
    DSPy client for prompt optimization and reasoning.

    Provides access to DSPy's declarative programming model for LLMs,
    including automatic prompt optimization via MIPROv2.
    """

    def __init__(
        self,
        config: Optional[DSPyConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize DSPy client.

        Args:
            config: Optional DSPyConfig
            **kwargs: Override config values
        """
        if not DSPY_AVAILABLE:
            raise SDKNotAvailableError(
                sdk_name="dspy",
                install_cmd="pip install dspy>=2.5.0",
                docs_url="https://dspy.ai/"
            )

        self.config = config or DSPyConfig()
        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._lm = None
        self._configure()

    def _configure(self) -> None:
        """Configure DSPy with the language model."""
        api_key = self.config.api_key or self._get_api_key()

        # Build model string based on provider
        if self.config.provider == "anthropic":
            model_str = f"anthropic/{self.config.model}"
        elif self.config.provider == "openai":
            model_str = f"openai/{self.config.model}"
        else:
            model_str = self.config.model

        self._lm = dspy.LM(
            model_str,
            api_key=api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            cache=self.config.cache,
        )
        dspy.configure(lm=self._lm)

    def _get_api_key(self) -> str:
        """Get API key from environment based on provider."""
        if self.config.provider == "anthropic":
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise SDKConfigurationError(
                    sdk_name="dspy",
                    missing_config=["ANTHROPIC_API_KEY"],
                    example="ANTHROPIC_API_KEY=sk-ant-..."
                )
            return key
        elif self.config.provider == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise SDKConfigurationError(
                    sdk_name="dspy",
                    missing_config=["OPENAI_API_KEY"],
                    example="OPENAI_API_KEY=sk-..."
                )
            return key
        else:
            raise SDKConfigurationError(
                sdk_name="dspy",
                missing_config=[f"{self.config.provider.upper()}_API_KEY"],
                example=f"{self.config.provider.upper()}_API_KEY=your-key"
            )

    @property
    def lm(self) -> "DSPyLM":
        """Get the configured language model."""
        return self._lm

    def predict(
        self,
        signature: str,
        **inputs: Any,
    ) -> ReasoningResult:
        """
        Make a basic prediction using DSPy.

        Args:
            signature: DSPy signature string (e.g., "question -> answer")
            **inputs: Input values matching signature

        Returns:
            ReasoningResult with prediction
        """
        import time
        start = time.perf_counter()

        predictor = dspy.Predict(signature)
        result = predictor(**inputs)

        latency = (time.perf_counter() - start) * 1000

        # Extract output field name from signature
        output_field = signature.split("->")[-1].strip()
        output_value = getattr(result, output_field, str(result))

        return ReasoningResult(
            output=output_value,
            latency_ms=latency,
        )

    def chain_of_thought(
        self,
        signature: str,
        **inputs: Any,
    ) -> ReasoningResult:
        """
        Make a prediction with chain-of-thought reasoning.

        Args:
            signature: DSPy signature string
            **inputs: Input values

        Returns:
            ReasoningResult with reasoning trace
        """
        import time
        start = time.perf_counter()

        cot = dspy.ChainOfThought(signature)
        result = cot(**inputs)

        latency = (time.perf_counter() - start) * 1000

        output_field = signature.split("->")[-1].strip()
        output_value = getattr(result, output_field, str(result))
        reasoning = getattr(result, "reasoning", None) or getattr(result, "rationale", None)

        return ReasoningResult(
            output=output_value,
            reasoning=reasoning,
            latency_ms=latency,
        )

    def react(
        self,
        signature: str,
        tools: Optional[List[Callable]] = None,
        max_iters: int = 5,
        **inputs: Any,
    ) -> ReasoningResult:
        """
        Make a prediction using ReAct (Reason + Act) pattern.

        Args:
            signature: DSPy signature string
            tools: Optional list of tool functions
            max_iters: Maximum iterations
            **inputs: Input values

        Returns:
            ReasoningResult with action trace
        """
        import time
        start = time.perf_counter()

        react = dspy.ReAct(signature, tools=tools or [], max_iters=max_iters)
        result = react(**inputs)

        latency = (time.perf_counter() - start) * 1000

        output_field = signature.split("->")[-1].strip()
        output_value = getattr(result, output_field, str(result))

        # Extract trajectory if available
        trace = []
        if hasattr(result, "trajectory"):
            trace = [str(step) for step in result.trajectory]

        return ReasoningResult(
            output=output_value,
            latency_ms=latency,
            trace=trace,
        )

    def optimize(
        self,
        program: Any,
        trainset: List[Any],
        metric: Callable,
        mode: OptimizerMode = OptimizerMode.LIGHT,
        valset: Optional[List[Any]] = None,
        num_threads: int = 8,
    ) -> OptimizationResult:
        """
        Optimize a DSPy program using MIPROv2.

        Args:
            program: DSPy program to optimize
            trainset: Training examples
            metric: Evaluation metric function
            mode: Optimization intensity
            valset: Optional validation set
            num_threads: Parallel threads

        Returns:
            OptimizationResult with optimization details
        """
        # Evaluate original
        evaluator = dspy.Evaluate(
            devset=valset or trainset[:10],
            metric=metric,
            num_threads=num_threads,
            display_progress=True,
        )
        original_score = evaluator(program)

        # Run MIPROv2 optimization
        optimizer = dspy.MIPROv2(
            metric=metric,
            auto=mode.value,
            num_threads=num_threads,
        )

        optimized_program = optimizer.compile(
            program,
            trainset=trainset,
            valset=valset,
        )

        # Evaluate optimized
        optimized_score = evaluator(optimized_program)

        return OptimizationResult(
            original_score=original_score,
            optimized_score=optimized_score,
            improvement=optimized_score - original_score,
            num_trials=getattr(optimizer, "num_trials", 0),
            metadata={
                "mode": mode.value,
                "trainset_size": len(trainset),
            },
        )

    def create_signature(
        self,
        name: str,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        instructions: Optional[str] = None,
    ) -> type:
        """
        Dynamically create a DSPy Signature class.

        Args:
            name: Signature class name
            inputs: Dict of input field names to descriptions
            outputs: Dict of output field names to descriptions
            instructions: Optional docstring/instructions

        Returns:
            DSPy Signature class
        """
        # Build class attributes
        attrs = {}
        if instructions:
            attrs["__doc__"] = instructions

        for field_name, desc in inputs.items():
            attrs[field_name] = dspy.InputField(desc=desc)

        for field_name, desc in outputs.items():
            attrs[field_name] = dspy.OutputField(desc=desc)

        return type(name, (dspy.Signature,), attrs)


# ============================================================================
# Serena Client Implementation
# ============================================================================

@dataclass
class SerenaConfig:
    """Configuration for Serena semantic editor."""
    project_path: Optional[str] = None
    language: str = "python"
    use_lsp: bool = True


@dataclass
class SymbolInfo:
    """Information about a code symbol."""
    name: str
    kind: str  # class, function, method, variable
    file_path: str
    line_start: int
    line_end: int
    body: Optional[str] = None
    docstring: Optional[str] = None


@dataclass
class EditResult:
    """Result of a semantic edit operation."""
    success: bool
    file_path: str
    changes_made: int
    message: str
    diff: Optional[str] = None


class SerenaClient:
    """
    Serena client for semantic code editing.

    Provides symbol-level code manipulation using LSP integration
    for precise, refactoring-safe edits.
    """

    def __init__(
        self,
        config: Optional[SerenaConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize Serena client.

        Args:
            config: Optional SerenaConfig
            **kwargs: Override config values
        """
        if not SERENA_AVAILABLE:
            raise SDKNotAvailableError(
                sdk_name="serena (mcp)",
                install_cmd="pip install mcp>=1.0.0",
                docs_url="https://github.com/oraios/serena"
            )

        self.config = config or SerenaConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._session = None

    async def connect(self) -> None:
        """Establish MCP connection to Serena server."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        # Serena runs as MCP server
        server_params = StdioServerParameters(
            command="serena",
            args=["--project", self.config.project_path] if self.config.project_path else [],
        )

        transport = await stdio_client(server_params)
        self._session = ClientSession(*transport)
        await self._session.initialize()

    async def find_symbol(
        self,
        name_pattern: str,
        file_path: Optional[str] = None,
        include_body: bool = False,
    ) -> List[SymbolInfo]:
        """
        Find symbols matching a pattern.

        Args:
            name_pattern: Symbol name or pattern (supports * wildcards)
            file_path: Optional file to restrict search
            include_body: Whether to include symbol body

        Returns:
            List of matching SymbolInfo
        """
        if not self._session:
            await self.connect()

        result = await self._session.call_tool(
            "find_symbol",
            {
                "name_path_pattern": name_pattern,
                "relative_path": file_path or "",
                "include_body": include_body,
            }
        )

        symbols = []
        for item in result.content:
            if hasattr(item, "text"):
                import json
                data = json.loads(item.text)
                for sym in data:
                    symbols.append(SymbolInfo(
                        name=sym.get("name", ""),
                        kind=sym.get("kind", "unknown"),
                        file_path=sym.get("file_path", ""),
                        line_start=sym.get("line_start", 0),
                        line_end=sym.get("line_end", 0),
                        body=sym.get("body"),
                        docstring=sym.get("docstring"),
                    ))

        return symbols

    async def replace_symbol(
        self,
        name_path: str,
        file_path: str,
        new_body: str,
    ) -> EditResult:
        """
        Replace a symbol's body.

        Args:
            name_path: Full symbol path (e.g., "MyClass/my_method")
            file_path: File containing the symbol
            new_body: New body content

        Returns:
            EditResult with operation status
        """
        if not self._session:
            await self.connect()

        result = await self._session.call_tool(
            "replace_symbol_body",
            {
                "name_path": name_path,
                "relative_path": file_path,
                "body": new_body,
            }
        )

        success = not any(
            "error" in str(item).lower()
            for item in result.content
        )

        return EditResult(
            success=success,
            file_path=file_path,
            changes_made=1 if success else 0,
            message=str(result.content[0]) if result.content else "No response",
        )

    async def insert_after_symbol(
        self,
        name_path: str,
        file_path: str,
        content: str,
    ) -> EditResult:
        """
        Insert content after a symbol.

        Args:
            name_path: Symbol to insert after
            file_path: File containing the symbol
            content: Content to insert

        Returns:
            EditResult with operation status
        """
        if not self._session:
            await self.connect()

        result = await self._session.call_tool(
            "insert_after_symbol",
            {
                "name_path": name_path,
                "relative_path": file_path,
                "body": content,
            }
        )

        success = not any(
            "error" in str(item).lower()
            for item in result.content
        )

        return EditResult(
            success=success,
            file_path=file_path,
            changes_made=1 if success else 0,
            message=str(result.content[0]) if result.content else "No response",
        )

    async def get_symbols_overview(
        self,
        file_path: str,
        depth: int = 1,
    ) -> List[SymbolInfo]:
        """
        Get overview of symbols in a file.

        Args:
            file_path: File to analyze
            depth: Depth of nested symbols to include

        Returns:
            List of SymbolInfo for file
        """
        if not self._session:
            await self.connect()

        result = await self._session.call_tool(
            "get_symbols_overview",
            {
                "relative_path": file_path,
                "depth": depth,
            }
        )

        symbols = []
        for item in result.content:
            if hasattr(item, "text"):
                import json
                data = json.loads(item.text)
                # Parse the overview format
                for kind, names in data.items():
                    for name in names:
                        symbols.append(SymbolInfo(
                            name=name,
                            kind=kind,
                            file_path=file_path,
                            line_start=0,
                            line_end=0,
                        ))

        return symbols

    async def rename_symbol(
        self,
        name_path: str,
        file_path: str,
        new_name: str,
    ) -> EditResult:
        """
        Rename a symbol across the codebase.

        Args:
            name_path: Symbol to rename
            file_path: File containing the symbol
            new_name: New name for the symbol

        Returns:
            EditResult with operation status
        """
        if not self._session:
            await self.connect()

        result = await self._session.call_tool(
            "rename_symbol",
            {
                "name_path": name_path,
                "relative_path": file_path,
                "new_name": new_name,
            }
        )

        success = not any(
            "error" in str(item).lower()
            for item in result.content
        )

        return EditResult(
            success=success,
            file_path=file_path,
            changes_made=1 if success else 0,  # Serena handles all references
            message=str(result.content[0]) if result.content else "No response",
        )

    async def close(self) -> None:
        """Close the MCP session."""
        if self._session:
            await self._session.close()
            self._session = None


# ============================================================================
# Explicit Getter Functions - Raise SDKNotAvailableError if unavailable
# ============================================================================

def get_dspy_lm(
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
    **kwargs: Any,
) -> "DSPyLM":
    """
    Get a configured DSPy language model.

    Args:
        provider: LLM provider ("anthropic" or "openai")
        model: Model name
        **kwargs: Additional configuration

    Returns:
        Configured DSPy LM

    Raises:
        SDKNotAvailableError: If DSPy is not installed
        SDKConfigurationError: If API key is missing
    """
    client = DSPyClient(config=DSPyConfig(provider=provider, model=model, **kwargs))
    return client.lm


def get_dspy_optimizer(
    mode: OptimizerMode = OptimizerMode.LIGHT,
    metric: Optional[Callable] = None,
    num_threads: int = 8,
) -> "MIPROv2":
    """
    Get a configured MIPROv2 optimizer.

    Args:
        mode: Optimization intensity
        metric: Evaluation metric function
        num_threads: Parallel threads

    Returns:
        Configured MIPROv2 optimizer

    Raises:
        SDKNotAvailableError: If DSPy is not installed
    """
    if not DSPY_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="dspy",
            install_cmd="pip install dspy>=2.5.0",
            docs_url="https://dspy.ai/"
        )

    return dspy.MIPROv2(
        metric=metric,
        auto=mode.value,
        num_threads=num_threads,
    )


def get_serena_client(
    project_path: Optional[str] = None,
    **kwargs: Any,
) -> SerenaClient:
    """
    Get a Serena semantic editor client.

    Args:
        project_path: Path to project root
        **kwargs: Additional configuration

    Returns:
        SerenaClient instance

    Raises:
        SDKNotAvailableError: If MCP is not installed
    """
    return SerenaClient(
        config=SerenaConfig(project_path=project_path, **kwargs)
    )


# ============================================================================
# Unified Factory
# ============================================================================

class ReasoningFactory:
    """
    Unified factory for creating reasoning clients.

    Provides a single entry point for DSPy and Serena with
    consistent configuration and V33 integration.
    """

    def __init__(
        self,
        default_provider: str = "anthropic",
        default_model: str = "claude-sonnet-4-20250514",
    ):
        """
        Initialize the factory.

        Args:
            default_provider: Default LLM provider
            default_model: Default model name
        """
        self.default_provider = default_provider
        self.default_model = default_model
        self._dspy_client: Optional[DSPyClient] = None
        self._serena_client: Optional[SerenaClient] = None

    def get_availability(self) -> Dict[str, bool]:
        """Get availability status of all SDKs."""
        return {
            "dspy": DSPY_AVAILABLE,
            "serena": SERENA_AVAILABLE,
        }

    def create_dspy(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> DSPyClient:
        """
        Create a DSPyClient.

        Args:
            provider: LLM provider
            model: Model name
            **kwargs: Additional configuration

        Returns:
            Configured DSPyClient
        """
        config = DSPyConfig(
            provider=provider or self.default_provider,
            model=model or self.default_model,
            **kwargs,
        )
        self._dspy_client = DSPyClient(config=config)
        return self._dspy_client

    def create_serena(
        self,
        project_path: Optional[str] = None,
        **kwargs: Any,
    ) -> SerenaClient:
        """
        Create a SerenaClient.

        Args:
            project_path: Project root path
            **kwargs: Additional configuration

        Returns:
            Configured SerenaClient
        """
        config = SerenaConfig(project_path=project_path, **kwargs)
        self._serena_client = SerenaClient(config=config)
        return self._serena_client

    def get_dspy(self) -> Optional[DSPyClient]:
        """Get the cached DSPy client."""
        return self._dspy_client

    def get_serena(self) -> Optional[SerenaClient]:
        """Get the cached Serena client."""
        return self._serena_client


# ============================================================================
# Module-level availability
# ============================================================================

REASONING_AVAILABLE = DSPY_AVAILABLE or SERENA_AVAILABLE


def get_available_sdks() -> Dict[str, bool]:
    """Get availability status of all reasoning SDKs."""
    return {
        "dspy": DSPY_AVAILABLE,
        "serena": SERENA_AVAILABLE,
    }


# ============================================================================
# All Exports
# ============================================================================

__all__ = [
    # Exceptions (re-exported from observability)
    "SDKNotAvailableError",
    "SDKConfigurationError",

    # Availability flags
    "DSPY_AVAILABLE",
    "SERENA_AVAILABLE",
    "REASONING_AVAILABLE",

    # Getter functions (raise on unavailable)
    "get_dspy_lm",
    "get_dspy_optimizer",
    "get_serena_client",

    # DSPy
    "DSPyClient",
    "DSPyConfig",
    "OptimizerMode",
    "ReasoningModule",
    "OptimizationResult",
    "ReasoningResult",

    # Serena
    "SerenaClient",
    "SerenaConfig",
    "SymbolInfo",
    "EditResult",

    # Factory
    "ReasoningFactory",
    "get_available_sdks",
]
