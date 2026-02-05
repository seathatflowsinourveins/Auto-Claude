"""
Value Objects - Immutable Domain Concepts

V65 Modular Decomposition - Extracted from ultimate_orchestrator.py

Contains:
- CircuitState: Circuit breaker states
- SDKLayer: SDK layer enumeration (39 layers)
- ExecutionPriority: Request priority levels
- SDKConfig: SDK configuration
- ExecutionContext: Execution context for requests
- ExecutionResult: Execution result with metadata
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto
from typing import Any, Dict, List, Optional


class CircuitState(IntEnum):
    """Circuit breaker states."""
    CLOSED = 0      # Normal operation
    OPEN = 1        # Failing, reject calls
    HALF_OPEN = 2   # Testing recovery


class SDKLayer(Enum):
    """
    SDK Layers for the Ultimate Orchestrator.

    V13-V28 Research-Backed Layer Definitions:
    - Core layers: OPTIMIZATION, ORCHESTRATION, MEMORY, REASONING, RESEARCH, SELF_IMPROVEMENT
    - V18: STREAMING, MULTI_MODAL, SAFETY
    - V19: PERSISTENCE, TOOL_USE, CODE_GEN
    - V20: INFERENCE, FINE_TUNING, EMBEDDING, OBSERVABILITY
    - V21: STRUCTURED_OUTPUT, AGENT_SWARM
    - V22: BROWSER_AUTOMATION, COMPUTER_USE, MULTIMODAL_REASONING
    - V23: SEMANTIC_ROUTER, FUNCTION_CALLING, WORKFLOW_ENGINE, MODEL_SERVING, AGENTIC_DATABASE
    - V24: CODE_INTERPRETER, DATA_TRANSFORMATION, PROMPT_CACHING, AGENT_TESTING, API_GATEWAY
    - V25: SYNTHETIC_DATA, MODEL_QUANTIZATION, VOICE_SYNTHESIS, MULTI_AGENT_SIM, AGENTIC_RAG
    - V26: DOCUMENT_PROCESSING, CROSS_SESSION_MEMORY, AUTONOMOUS_TOOLS, MULTI_AGENT_ORCHESTRATION, CODE_SANDBOX_V2
    - V27: PRODUCTION_OPTIMIZATION, CONTEXT_COMPRESSION, CODE_VALIDATION, SECURITY_TESTING,
           DURABLE_EXECUTION, STRUCTURED_GENERATION, FAST_CHUNKING, OBSERVABILITY_V2
    """
    # Core layers (V1-V4)
    OPTIMIZATION = auto()        # DSPy, AdalFlow, PromptTune++
    ORCHESTRATION = auto()       # LangGraph, CrewAI, mcp-agent
    MEMORY = auto()              # Zep, Cognee, Mem0
    REASONING = auto()           # LiteLLM, AGoT, LightZero
    RESEARCH = auto()            # Firecrawl, Crawl4AI, Exa
    SELF_IMPROVEMENT = auto()    # pyribs, EvoTorch, QDax, TensorNEAT

    # V18 layers (Ralph Loop Iteration 15)
    STREAMING = auto()           # LLMRTC, LiveKit Agents
    MULTI_MODAL = auto()         # NeMo ASR, BLIP-2
    SAFETY = auto()              # Bifrost, NeMo Guardrails

    # V19 layers (Ralph Loop Iteration 16)
    PERSISTENCE = auto()         # AutoGen Core, AgentCore, MetaGPT
    TOOL_USE = auto()            # Tool Search, Parallel Executor
    CODE_GEN = auto()            # Verdent, Augment Code

    # V20 layers (Ralph Loop Iteration 17)
    INFERENCE = auto()           # vLLM, llama.cpp
    FINE_TUNING = auto()         # Unsloth, PEFT
    EMBEDDING = auto()           # ColBERT, BGE-M3
    OBSERVABILITY = auto()       # Phoenix/Arize

    # V21 layers (Ralph Loop Iteration 18)
    STRUCTURED_OUTPUT = auto()   # Guidance, Outlines
    AGENT_SWARM = auto()         # Strands-agents

    # V22 layers (Ralph Loop Iteration 19)
    BROWSER_AUTOMATION = auto()  # Browser-Use
    COMPUTER_USE = auto()        # Open Interpreter
    MULTIMODAL_REASONING = auto() # InternVL3, Phi-4

    # V23 layers (Ralph Loop Iteration 20)
    SEMANTIC_ROUTER = auto()     # semantic-router
    FUNCTION_CALLING = auto()    # instructor
    WORKFLOW_ENGINE = auto()     # Prefect 3.x
    MODEL_SERVING = auto()       # BentoML 1.0
    AGENTIC_DATABASE = auto()    # LanceDB

    # V24 layers (Ralph Loop Iteration 21)
    CODE_INTERPRETER = auto()    # E2B
    DATA_TRANSFORMATION = auto() # Polars AI
    PROMPT_CACHING = auto()      # Redis-Stack AI
    AGENT_TESTING = auto()       # AgentBench
    API_GATEWAY = auto()         # Portkey

    # V25 layers (Ralph Loop Iteration 22)
    SYNTHETIC_DATA = auto()      # SDV
    MODEL_QUANTIZATION = auto()  # AWQ
    VOICE_SYNTHESIS = auto()     # Coqui TTS
    MULTI_AGENT_SIM = auto()     # PettingZoo
    AGENTIC_RAG = auto()         # RAGFlow

    # V26 layers (Ralph Loop Iteration 23)
    DOCUMENT_PROCESSING = auto()      # Docling, Unstructured
    CROSS_SESSION_MEMORY = auto()     # MemGPT/Letta
    AUTONOMOUS_TOOLS = auto()         # AnyTool, fast-agent
    MULTI_AGENT_ORCHESTRATION = auto() # CrewAI, agent-squad
    CODE_SANDBOX_V2 = auto()          # Modal

    # V27 layers (Ralph Loop Iteration 26)
    PRODUCTION_OPTIMIZATION = auto()  # TensorZero
    CONTEXT_COMPRESSION = auto()      # LLMLingua
    CODE_VALIDATION = auto()          # ast-grep
    SECURITY_TESTING = auto()         # promptfoo
    DURABLE_EXECUTION = auto()        # Temporal
    STRUCTURED_GENERATION = auto()    # SGLang
    FAST_CHUNKING = auto()            # Chonkie
    OBSERVABILITY_V2 = auto()         # Langfuse v3


class ExecutionPriority(Enum):
    """
    Execution priority levels for request scheduling.

    Lower values = higher priority.
    """
    CRITICAL = 1   # Real-time requirements
    HIGH = 2       # Important operations
    NORMAL = 3     # Standard operations
    LOW = 4        # Background tasks
    BACKGROUND = 5 # Batch processing


@dataclass
class SDKConfig:
    """
    Configuration for an SDK adapter.

    Contains the SDK name, layer, priority, caching settings, and metadata.
    """
    name: str
    layer: SDKLayer
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    cache_ttl_seconds: int = 3600
    timeout_ms: int = 30000
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("SDK name cannot be empty")


@dataclass
class ExecutionContext:
    """
    Context for an execution request.

    Contains request ID, layer, priority, and timing information.
    """
    request_id: str
    layer: SDKLayer
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    start_time: float = field(default_factory=time.time)
    deadline_ms: Optional[float] = None
    trace_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000

    @property
    def remaining_ms(self) -> Optional[float]:
        """Get remaining time before deadline, if set."""
        if self.deadline_ms is None:
            return None
        return max(0, self.deadline_ms - self.elapsed_ms)

    @property
    def is_expired(self) -> bool:
        """Check if deadline has passed."""
        if self.deadline_ms is None:
            return False
        return self.elapsed_ms >= self.deadline_ms


@dataclass
class ExecutionResult:
    """
    Result of an SDK execution.

    Contains success status, data, error, timing, and metadata.
    """
    success: bool
    data: Any = None
    error: Optional[str] = None
    layer: Optional[SDKLayer] = None
    adapter_name: Optional[str] = None
    latency_ms: float = 0.0
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Ensure metadata is always a dict."""
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def success_result(
        cls,
        data: Any,
        layer: Optional[SDKLayer] = None,
        adapter_name: Optional[str] = None,
        latency_ms: float = 0.0,
        cached: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "ExecutionResult":
        """Create a successful result."""
        return cls(
            success=True,
            data=data,
            layer=layer,
            adapter_name=adapter_name,
            latency_ms=latency_ms,
            cached=cached,
            metadata=metadata or {}
        )

    @classmethod
    def failure_result(
        cls,
        error: str,
        layer: Optional[SDKLayer] = None,
        adapter_name: Optional[str] = None,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "ExecutionResult":
        """Create a failed result."""
        return cls(
            success=False,
            error=error,
            layer=layer,
            adapter_name=adapter_name,
            latency_ms=latency_ms,
            metadata=metadata or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "layer": self.layer.name if self.layer else None,
            "adapter_name": self.adapter_name,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
