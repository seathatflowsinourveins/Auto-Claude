#!/usr/bin/env python3
"""
Structured Output Layer - Unified Interface
Part of the V33 Architecture (Layer 3) - Phase 9 Production Fix.

Provides unified access to four structured output SDKs:
- instructor: Pydantic-validated LLM responses with automatic retries
- baml: Type-safe LLM function definitions with schema validation
- outlines: Constrained text generation with regex/grammar enforcement
- pydantic-ai: Agent framework with memory integration

NO STUBS: All SDKs must be explicitly installed and configured.
Missing SDKs raise SDKNotAvailableError with install instructions.
Misconfigured SDKs raise SDKConfigurationError with missing config.

Usage:
    from core.structured import (
        # Exceptions
        SDKNotAvailableError,
        SDKConfigurationError,

        # Instructor
        get_instructor_client,  # Raises on unavailable
        InstructorClient,
        INSTRUCTOR_AVAILABLE,

        # BAML
        get_baml_client,  # Raises on unavailable
        BAMLClient,
        BAML_AVAILABLE,

        # Outlines
        get_outlines_generator,  # Raises on unavailable
        OutlinesGenerator,
        OUTLINES_AVAILABLE,

        # Pydantic AI Agents
        get_pydantic_agent,  # Raises on unavailable
        PydanticAIAgent,
        PYDANTIC_AI_AVAILABLE,

        # Factory
        StructuredOutputFactory,
    )

    # Quick start with explicit error handling
    try:
        instructor = get_instructor_client()
    except SDKNotAvailableError as e:
        print(f"Install: {e.install_cmd}")
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Optional, Type, Union
from dataclasses import dataclass

from pydantic import BaseModel

# Import exceptions from observability layer
from core.observability import (
    SDKNotAvailableError,
    SDKConfigurationError,
)


# ============================================================================
# Instructor Exports
# ============================================================================

from .instructor_chains import (
    InstructorClient,
    ChainResult,
    Sentiment,
    SentimentType,
    Classification,
    Entity,
    ExtractionResult,
    create_instructor_client,
    INSTRUCTOR_AVAILABLE,
)


# ============================================================================
# BAML Exports
# ============================================================================

from .baml_functions import (
    BAMLClient,
    BAMLResult,
    BAMLStatus,
    BAMLFunctionDef,
    BAMLFunctionRegistry,
    SummaryOutput,
    TranslationOutput,
    CodeGenerationOutput,
    AnalysisOutput as BAMLAnalysisOutput,
    create_baml_client,
    BAML_AVAILABLE,
)


# ============================================================================
# Outlines Exports
# ============================================================================

from .outlines_constraints import (
    OutlinesGenerator,
    GenerationResult,
    ChoiceResult,
    Constraint,
    ConstraintType,
    CommonPatterns,
    create_outlines_generator,
    OUTLINES_AVAILABLE,
)


# ============================================================================
# Pydantic AI Exports
# ============================================================================

from .pydantic_agents import (
    PydanticAIAgent,
    AgentConfig,
    AgentDeps,
    AgentRole,
    AgentProvider,
    AgentRun,
    AgentMessage,
    ResearchOutput,
    AnalysisOutput as AgentAnalysisOutput,
    PlanOutput,
    CodeOutput,
    create_agent,
    create_research_agent,
    create_analyst_agent,
    create_planner_agent,
    create_coder_agent,
    PYDANTIC_AI_AVAILABLE,
)


# ============================================================================
# Explicit Getter Functions - Raise SDKNotAvailableError if unavailable
# ============================================================================

def get_instructor_client(
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
    **kwargs: Any,
) -> "InstructorClient":
    """
    Get an InstructorClient. Raises explicit error if unavailable.

    Args:
        provider: LLM provider ("anthropic" or "openai")
        model: Model name
        **kwargs: Additional configuration

    Returns:
        Configured InstructorClient

    Raises:
        SDKNotAvailableError: If instructor is not installed
        SDKConfigurationError: If API key is missing for selected provider
    """
    if not INSTRUCTOR_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="instructor",
            install_cmd="pip install instructor>=1.0.0",
            docs_url="https://python.useinstructor.com/"
        )

    # Check API key based on provider
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise SDKConfigurationError(
            sdk_name="instructor",
            missing_config=["ANTHROPIC_API_KEY"],
            example="""
ANTHROPIC_API_KEY=sk-ant-...
"""
        )
    elif provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise SDKConfigurationError(
            sdk_name="instructor",
            missing_config=["OPENAI_API_KEY"],
            example="""
OPENAI_API_KEY=sk-...
"""
        )

    return InstructorClient(provider=provider, model=model, **kwargs)


def get_baml_client(
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> "BAMLClient":
    """
    Get a BAMLClient. Raises explicit error if unavailable.

    Args:
        api_key: Optional API key
        **kwargs: Additional configuration

    Returns:
        Configured BAMLClient

    Raises:
        SDKNotAvailableError: If BAML is not installed
    """
    if not BAML_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="baml",
            install_cmd="pip install baml>=0.55.0",
            docs_url="https://docs.boundaryml.com/"
        )

    return BAMLClient(api_key=api_key, **kwargs)


def get_outlines_generator(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    **kwargs: Any,
) -> "OutlinesGenerator":
    """
    Get an OutlinesGenerator. Raises explicit error if unavailable.

    Args:
        model_name: HuggingFace model name
        **kwargs: Additional configuration

    Returns:
        Configured OutlinesGenerator

    Raises:
        SDKNotAvailableError: If outlines is not installed
    """
    if not OUTLINES_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="outlines",
            install_cmd="pip install outlines>=0.0.36",
            docs_url="https://outlines-dev.github.io/outlines/"
        )

    return OutlinesGenerator(model_name=model_name, **kwargs)


def get_pydantic_agent(
    name: str,
    role: "AgentRole" = None,
    provider: str = "anthropic",
    **kwargs: Any,
) -> "PydanticAIAgent":
    """
    Get a PydanticAIAgent. Raises explicit error if unavailable.

    Args:
        name: Agent name
        role: Agent role (defaults to ASSISTANT)
        provider: LLM provider
        **kwargs: Additional configuration

    Returns:
        Configured PydanticAIAgent

    Raises:
        SDKNotAvailableError: If pydantic-ai is not installed
        SDKConfigurationError: If API key is missing for selected provider
    """
    if not PYDANTIC_AI_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="pydantic-ai",
            install_cmd="pip install pydantic-ai>=0.0.1",
            docs_url="https://ai.pydantic.dev/"
        )

    # Check API key based on provider
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise SDKConfigurationError(
            sdk_name="pydantic-ai",
            missing_config=["ANTHROPIC_API_KEY"],
            example="""
ANTHROPIC_API_KEY=sk-ant-...
"""
        )
    elif provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise SDKConfigurationError(
            sdk_name="pydantic-ai",
            missing_config=["OPENAI_API_KEY"],
            example="""
OPENAI_API_KEY=sk-...
"""
        )

    # Use default role if not provided
    if role is None:
        role = AgentRole.ASSISTANT

    return create_agent(name=name, role=role, **kwargs)


# ============================================================================
# Unified Factory
# ============================================================================

class StructuredOutputType(str, Enum):
    """Types of structured output providers."""
    INSTRUCTOR = "instructor"
    BAML = "baml"
    OUTLINES = "outlines"
    PYDANTIC_AI = "pydantic_ai"


@dataclass
class StructuredOutputConfig:
    """Configuration for structured output factory."""
    default_provider: str = "anthropic"
    default_model: str = "claude-sonnet-4-20250514"
    api_key: Optional[str] = None
    enable_memory: bool = True
    memory_namespace: str = "structured"


class StructuredOutputFactory:
    """
    Unified factory for creating structured output clients.

    Provides a single entry point for all four SDKs with
    consistent configuration and V33 integration.
    """

    def __init__(self, config: Optional[StructuredOutputConfig] = None):
        """
        Initialize the factory.

        Args:
            config: Optional configuration
        """
        self.config = config or StructuredOutputConfig()
        self._memory = None
        self._tools = None

    def attach_memory(self, memory: Any) -> None:
        """Attach V33 UnifiedMemory for agent integration."""
        self._memory = memory

    def attach_tools(self, tools: Any) -> None:
        """Attach V33 ToolLayer for agent integration."""
        self._tools = tools

    def get_availability(self) -> dict[str, bool]:
        """Get availability status of all SDKs."""
        return {
            "instructor": INSTRUCTOR_AVAILABLE,
            "baml": BAML_AVAILABLE,
            "outlines": OUTLINES_AVAILABLE,
            "pydantic_ai": PYDANTIC_AI_AVAILABLE,
        }

    def create_instructor(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> InstructorClient:
        """
        Create an InstructorClient.

        Args:
            provider: LLM provider ("anthropic" or "openai")
            model: Model name
            **kwargs: Additional configuration

        Returns:
            Configured InstructorClient
        """
        return InstructorClient(
            provider=provider or self.config.default_provider,
            model=model or self.config.default_model,
            api_key=self.config.api_key,
            **kwargs,
        )

    def create_baml(
        self,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> BAMLClient:
        """
        Create a BAMLClient.

        Args:
            api_key: Optional API key
            **kwargs: Additional configuration

        Returns:
            Configured BAMLClient
        """
        return BAMLClient(
            api_key=api_key or self.config.api_key,
            default_model=self.config.default_model,
            **kwargs,
        )

    def create_outlines(
        self,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> OutlinesGenerator:
        """
        Create an OutlinesGenerator.

        Args:
            model_name: HuggingFace model name
            **kwargs: Additional configuration

        Returns:
            Configured OutlinesGenerator
        """
        return OutlinesGenerator(
            model_name=model_name or "mistralai/Mistral-7B-Instruct-v0.2",
            **kwargs,
        )

    def create_agent(
        self,
        name: str,
        role: AgentRole = AgentRole.ASSISTANT,
        **kwargs: Any,
    ) -> PydanticAIAgent:
        """
        Create a PydanticAIAgent with V33 integration.

        Args:
            name: Agent name
            role: Agent role
            **kwargs: Additional configuration

        Returns:
            Configured PydanticAIAgent with memory attached
        """
        agent = create_agent(
            name=name,
            role=role,
            provider=AgentProvider.ANTHROPIC if "anthropic" in self.config.default_provider else AgentProvider.OPENAI,
            model=self.config.default_model,
            enable_memory=self.config.enable_memory,
            memory_namespace=self.config.memory_namespace,
            **kwargs,
        )

        if self._memory:
            agent.attach_memory(self._memory)
        if self._tools:
            agent.attach_tools(self._tools)

        return agent

    def create_research_agent(self, name: str = "researcher", **kwargs) -> PydanticAIAgent:
        """Create a research-focused agent."""
        agent = create_research_agent(name=name, memory=self._memory, **kwargs)
        if self._tools:
            agent.attach_tools(self._tools)
        return agent

    def create_analyst_agent(self, name: str = "analyst", **kwargs) -> PydanticAIAgent:
        """Create an analysis-focused agent."""
        agent = create_analyst_agent(name=name, memory=self._memory, **kwargs)
        if self._tools:
            agent.attach_tools(self._tools)
        return agent

    def create_planner_agent(self, name: str = "planner", **kwargs) -> PydanticAIAgent:
        """Create a planning-focused agent."""
        agent = create_planner_agent(name=name, memory=self._memory, **kwargs)
        if self._tools:
            agent.attach_tools(self._tools)
        return agent

    def create_coder_agent(self, name: str = "coder", **kwargs) -> PydanticAIAgent:
        """Create a coding-focused agent."""
        agent = create_coder_agent(name=name, memory=self._memory, **kwargs)
        if self._tools:
            agent.attach_tools(self._tools)
        return agent


# ============================================================================
# Module-level availability
# ============================================================================

STRUCTURED_OUTPUT_AVAILABLE = True


def get_available_sdks() -> dict[str, bool]:
    """Get availability status of all structured output SDKs."""
    return {
        "instructor": INSTRUCTOR_AVAILABLE,
        "baml": BAML_AVAILABLE,
        "outlines": OUTLINES_AVAILABLE,
        "pydantic_ai": PYDANTIC_AI_AVAILABLE,
    }


# ============================================================================
# All Exports
# ============================================================================

__all__ = [
    # Exceptions (re-exported from observability)
    "SDKNotAvailableError",
    "SDKConfigurationError",

    # Getter functions (raise on unavailable)
    "get_instructor_client",
    "get_baml_client",
    "get_outlines_generator",
    "get_pydantic_agent",

    # Instructor
    "InstructorClient",
    "ChainResult",
    "Sentiment",
    "SentimentType",
    "Classification",
    "Entity",
    "ExtractionResult",
    "create_instructor_client",
    "INSTRUCTOR_AVAILABLE",

    # BAML
    "BAMLClient",
    "BAMLResult",
    "BAMLStatus",
    "BAMLFunctionDef",
    "BAMLFunctionRegistry",
    "SummaryOutput",
    "TranslationOutput",
    "CodeGenerationOutput",
    "BAMLAnalysisOutput",
    "create_baml_client",
    "BAML_AVAILABLE",

    # Outlines
    "OutlinesGenerator",
    "GenerationResult",
    "ChoiceResult",
    "Constraint",
    "ConstraintType",
    "CommonPatterns",
    "create_outlines_generator",
    "OUTLINES_AVAILABLE",

    # Pydantic AI Agents
    "PydanticAIAgent",
    "AgentConfig",
    "AgentDeps",
    "AgentRole",
    "AgentProvider",
    "AgentRun",
    "AgentMessage",
    "ResearchOutput",
    "AgentAnalysisOutput",
    "PlanOutput",
    "CodeOutput",
    "create_agent",
    "create_research_agent",
    "create_analyst_agent",
    "create_planner_agent",
    "create_coder_agent",
    "PYDANTIC_AI_AVAILABLE",

    # Factory
    "StructuredOutputFactory",
    "StructuredOutputConfig",
    "StructuredOutputType",
    "STRUCTURED_OUTPUT_AVAILABLE",
    "get_available_sdks",
]
