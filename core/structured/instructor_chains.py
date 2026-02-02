#!/usr/bin/env python3
"""
Instructor Chains - Pydantic-Validated LLM Responses
Part of the V33 Structured Output Layer.

Uses instructor to patch LLM clients for automatic Pydantic validation,
retry logic, and streaming structured outputs.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, TypeVar, Generic, Optional, Type, Union
from datetime import datetime

from pydantic import BaseModel, Field
import instructor

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Type variable for generic response models
T = TypeVar("T", bound=BaseModel)


# ============================================================================
# Response Models
# ============================================================================

class SentimentType(str, Enum):
    """Sentiment classification types."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class Sentiment(BaseModel):
    """Sentiment analysis result."""
    sentiment: SentimentType = Field(description="The detected sentiment")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(description="Brief explanation for the classification")


class Classification(BaseModel):
    """Multi-label classification result."""
    labels: list[str] = Field(description="Assigned labels/categories")
    scores: dict[str, float] = Field(description="Confidence score for each label")
    primary_label: str = Field(description="The most confident label")


class Entity(BaseModel):
    """Named entity extraction result."""
    text: str = Field(description="The entity text as it appears")
    label: str = Field(description="Entity type (PERSON, ORG, LOCATION, etc.)")
    start: Optional[int] = Field(default=None, description="Start character offset")
    end: Optional[int] = Field(default=None, description="End character offset")


class ExtractionResult(BaseModel):
    """Generic extraction result with entities and metadata."""
    entities: list[Entity] = Field(default_factory=list)
    summary: Optional[str] = Field(default=None, description="Optional summary")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChainResult(BaseModel, Generic[T]):
    """Wrapper for chain execution results."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    model: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0


# ============================================================================
# Instructor Client
# ============================================================================

class InstructorClient:
    """
    Unified Instructor client for structured LLM outputs.

    Supports both Anthropic (Claude) and OpenAI models with automatic
    Pydantic validation, retries, and streaming.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        max_retries: int = 3,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Instructor client.

        Args:
            provider: LLM provider ("anthropic" or "openai")
            model: Model name (defaults to claude-3-5-sonnet or gpt-4o)
            max_retries: Maximum retry attempts for validation failures
            api_key: Optional API key (uses env var if not provided)
        """
        self.provider = provider.lower()
        self.max_retries = max_retries
        self._client = None
        self._model = model

        if self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package not installed")
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            base_client = anthropic.Anthropic(api_key=api_key)
            self._client = instructor.from_anthropic(base_client)
            self._model = model or "claude-sonnet-4-20250514"

        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed")
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            base_client = openai.OpenAI(api_key=api_key)
            self._client = instructor.from_openai(base_client)
            self._model = model or "gpt-4o"

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._model

    async def extract(
        self,
        text: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ChainResult[T]:
        """
        Extract structured data from text.

        Args:
            text: Input text to extract from
            response_model: Pydantic model for the expected output
            system_prompt: Optional system prompt override
            **kwargs: Additional parameters for the LLM call

        Returns:
            ChainResult containing the extracted data or error
        """
        start_time = datetime.now()

        try:
            messages = [{"role": "user", "content": text}]

            if self.provider == "anthropic":
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=4096,
                    messages=messages,
                    response_model=response_model,
                    max_retries=self.max_retries,
                    system=system_prompt or "Extract the requested information accurately.",
                    **kwargs,
                )
            else:
                system_msgs = [{"role": "system", "content": system_prompt or "Extract the requested information accurately."}]
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=system_msgs + messages,
                    response_model=response_model,
                    max_retries=self.max_retries,
                    **kwargs,
                )

            latency = (datetime.now() - start_time).total_seconds() * 1000

            return ChainResult(
                success=True,
                data=response,
                model=self._model,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            return ChainResult(
                success=False,
                error=str(e),
                model=self._model,
                latency_ms=latency,
            )

    async def classify(
        self,
        text: str,
        labels: list[str],
        multi_label: bool = False,
        **kwargs: Any,
    ) -> ChainResult[Classification]:
        """
        Classify text into one or more categories.

        Args:
            text: Input text to classify
            labels: List of possible labels
            multi_label: Allow multiple labels if True
            **kwargs: Additional parameters

        Returns:
            ChainResult with Classification data
        """
        system_prompt = f"""Classify the following text into one or more of these categories: {', '.join(labels)}.
{'Select all applicable labels.' if multi_label else 'Select the single most appropriate label.'}
Provide confidence scores for each relevant label."""

        return await self.extract(
            text=text,
            response_model=Classification,
            system_prompt=system_prompt,
            **kwargs,
        )

    async def analyze_sentiment(
        self,
        text: str,
        **kwargs: Any,
    ) -> ChainResult[Sentiment]:
        """
        Analyze the sentiment of text.

        Args:
            text: Input text to analyze
            **kwargs: Additional parameters

        Returns:
            ChainResult with Sentiment data
        """
        system_prompt = """Analyze the sentiment of the following text.
Classify as positive, negative, neutral, or mixed.
Provide a confidence score and brief reasoning."""

        return await self.extract(
            text=text,
            response_model=Sentiment,
            system_prompt=system_prompt,
            **kwargs,
        )

    async def extract_entities(
        self,
        text: str,
        entity_types: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChainResult[ExtractionResult]:
        """
        Extract named entities from text.

        Args:
            text: Input text to extract from
            entity_types: Optional list of entity types to extract
            **kwargs: Additional parameters

        Returns:
            ChainResult with ExtractionResult data
        """
        types_str = ", ".join(entity_types) if entity_types else "PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PRODUCT"
        system_prompt = f"""Extract all named entities from the text.
Focus on these entity types: {types_str}.
Include the exact text, entity type, and character offsets when possible."""

        return await self.extract(
            text=text,
            response_model=ExtractionResult,
            system_prompt=system_prompt,
            **kwargs,
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def create_instructor_client(
    provider: str = "anthropic",
    model: Optional[str] = None,
    **kwargs: Any,
) -> InstructorClient:
    """
    Factory function to create an InstructorClient.

    Args:
        provider: LLM provider ("anthropic" or "openai")
        model: Optional model name
        **kwargs: Additional configuration

    Returns:
        Configured InstructorClient instance
    """
    return InstructorClient(provider=provider, model=model, **kwargs)


# Check availability
INSTRUCTOR_AVAILABLE = True
