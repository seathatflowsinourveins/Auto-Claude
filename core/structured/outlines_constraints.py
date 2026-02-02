#!/usr/bin/env python3
"""
Outlines Constraints - Constrained Text Generation
Part of the V33 Structured Output Layer.

Uses Outlines for regex-constrained and grammar-guided text generation
with finite state machine enforcement.
"""

from __future__ import annotations

import os
import re
from enum import Enum
from typing import Any, Optional, TypeVar, Generic, Union, Pattern
from datetime import datetime
from dataclasses import dataclass

from pydantic import BaseModel, Field

try:
    import outlines
    from outlines import generate, models
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
    outlines = None
    generate = None
    models = None


# Type variable for generic outputs
T = TypeVar("T")


# ============================================================================
# Constraint Types
# ============================================================================

class ConstraintType(str, Enum):
    """Types of generation constraints."""
    REGEX = "regex"
    JSON = "json"
    CHOICE = "choice"
    GRAMMAR = "grammar"
    FORMAT = "format"


@dataclass
class Constraint:
    """Base constraint definition."""
    type: ConstraintType
    pattern: Optional[str] = None
    schema: Optional[type] = None
    choices: Optional[list[str]] = None
    grammar: Optional[str] = None


# ============================================================================
# Common Patterns
# ============================================================================

class CommonPatterns:
    """Pre-defined regex patterns for common use cases."""

    # Basic types
    INTEGER = r"-?\d+"
    FLOAT = r"-?\d+\.?\d*"
    BOOLEAN = r"(true|false|True|False)"

    # Identifiers
    EMAIL = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    PHONE = r"\+?[\d\s\-\(\)]{10,}"
    URL = r"https?://[^\s]+"
    UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"

    # Dates and times
    DATE_ISO = r"\d{4}-\d{2}-\d{2}"
    TIME_24H = r"\d{2}:\d{2}(:\d{2})?"
    DATETIME_ISO = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})?"

    # Code-related
    PYTHON_VAR = r"[a-z_][a-z0-9_]*"
    CLASS_NAME = r"[A-Z][a-zA-Z0-9]*"
    FUNCTION_CALL = r"[a-z_][a-z0-9_]*\([^)]*\)"

    # JSON structures
    JSON_OBJECT = r"\{[^{}]*\}"
    JSON_ARRAY = r"\[[^\[\]]*\]"

    # Natural language
    SENTENCE = r"[A-Z][^.!?]*[.!?]"
    PARAGRAPH = r"[A-Z][^.!?]*[.!?](\s+[A-Z][^.!?]*[.!?])*"

    @classmethod
    def list_number(cls, min_items: int = 1, max_items: int = 10) -> str:
        """Generate pattern for numbered list items."""
        item_pattern = r"\d+\.\s+[^\n]+"
        return f"({item_pattern}\n){{{min_items},{max_items}}}"

    @classmethod
    def bullet_list(cls, min_items: int = 1, max_items: int = 10) -> str:
        """Generate pattern for bullet list items."""
        item_pattern = r"[-*]\s+[^\n]+"
        return f"({item_pattern}\n){{{min_items},{max_items}}}"


# ============================================================================
# Output Models
# ============================================================================

class GenerationResult(BaseModel, Generic[T]):
    """Result from constrained generation."""
    success: bool
    output: Optional[T] = None
    raw_text: str = ""
    constraint_type: ConstraintType = ConstraintType.REGEX
    error: Optional[str] = None
    latency_ms: float = 0.0
    tokens_generated: int = 0


class ChoiceResult(BaseModel):
    """Result from choice-based generation."""
    choice: str
    index: int
    confidence: Optional[float] = None


# ============================================================================
# Outlines Generator
# ============================================================================

class OutlinesGenerator:
    """
    Constrained text generation using Outlines.

    Provides regex-constrained, grammar-guided, and schema-validated
    text generation with finite state machine enforcement.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str = "auto",
        quantize: bool = True,
    ):
        """
        Initialize the Outlines generator.

        Args:
            model_name: HuggingFace model name or path
            device: Device to run on ("auto", "cuda", "cpu")
            quantize: Whether to use 4-bit quantization
        """
        self.model_name = model_name
        self.device = device
        self.quantize = quantize
        self._model = None
        self._generators: dict[str, Any] = {}

    def _load_model(self):
        """Lazy-load the model."""
        if self._model is None and OUTLINES_AVAILABLE:
            self._model = models.transformers(
                self.model_name,
                device=self.device if self.device != "auto" else None,
            )
        return self._model

    async def generate_regex(
        self,
        prompt: str,
        pattern: str,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> GenerationResult[str]:
        """
        Generate text constrained by a regex pattern.

        Args:
            prompt: Input prompt
            pattern: Regex pattern to constrain output
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with the generated text
        """
        start_time = datetime.now()

        try:
            if not OUTLINES_AVAILABLE:
                # Fallback: validate post-generation
                return await self._fallback_regex(prompt, pattern, max_tokens)

            model = self._load_model()

            # Get or create generator for this pattern
            cache_key = f"regex:{pattern}"
            if cache_key not in self._generators:
                self._generators[cache_key] = generate.regex(model, pattern)

            generator = self._generators[cache_key]
            output = generator(prompt, max_tokens=max_tokens)

            latency = (datetime.now() - start_time).total_seconds() * 1000

            return GenerationResult(
                success=True,
                output=output,
                raw_text=output,
                constraint_type=ConstraintType.REGEX,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            return GenerationResult(
                success=False,
                error=str(e),
                constraint_type=ConstraintType.REGEX,
                latency_ms=latency,
            )

    async def generate_json(
        self,
        prompt: str,
        schema: type,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> GenerationResult[BaseModel]:
        """
        Generate JSON constrained by a Pydantic schema.

        Args:
            prompt: Input prompt
            schema: Pydantic model class for output schema
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with the parsed Pydantic model
        """
        start_time = datetime.now()

        try:
            if not OUTLINES_AVAILABLE:
                return await self._fallback_json(prompt, schema, max_tokens)

            model = self._load_model()

            # Get or create generator for this schema
            cache_key = f"json:{schema.__name__}"
            if cache_key not in self._generators:
                self._generators[cache_key] = generate.json(model, schema)

            generator = self._generators[cache_key]
            output = generator(prompt, max_tokens=max_tokens)

            latency = (datetime.now() - start_time).total_seconds() * 1000

            return GenerationResult(
                success=True,
                output=output,
                raw_text=output.model_dump_json() if hasattr(output, 'model_dump_json') else str(output),
                constraint_type=ConstraintType.JSON,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            return GenerationResult(
                success=False,
                error=str(e),
                constraint_type=ConstraintType.JSON,
                latency_ms=latency,
            )

    async def generate_choice(
        self,
        prompt: str,
        choices: list[str],
        **kwargs: Any,
    ) -> GenerationResult[ChoiceResult]:
        """
        Generate constrained to one of the given choices.

        Args:
            prompt: Input prompt
            choices: List of valid choices
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with the selected choice
        """
        start_time = datetime.now()

        try:
            if not OUTLINES_AVAILABLE:
                return await self._fallback_choice(prompt, choices)

            model = self._load_model()

            # Get or create generator for these choices
            cache_key = f"choice:{','.join(sorted(choices))}"
            if cache_key not in self._generators:
                self._generators[cache_key] = generate.choice(model, choices)

            generator = self._generators[cache_key]
            output = generator(prompt)

            latency = (datetime.now() - start_time).total_seconds() * 1000

            return GenerationResult(
                success=True,
                output=ChoiceResult(
                    choice=output,
                    index=choices.index(output),
                ),
                raw_text=output,
                constraint_type=ConstraintType.CHOICE,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            return GenerationResult(
                success=False,
                error=str(e),
                constraint_type=ConstraintType.CHOICE,
                latency_ms=latency,
            )

    async def generate_format(
        self,
        prompt: str,
        format_type: str,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> GenerationResult[str]:
        """
        Generate using predefined format patterns.

        Args:
            prompt: Input prompt
            format_type: One of the CommonPatterns attributes
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with formatted output
        """
        pattern = getattr(CommonPatterns, format_type.upper(), None)
        if pattern is None:
            return GenerationResult(
                success=False,
                error=f"Unknown format type: {format_type}",
                constraint_type=ConstraintType.FORMAT,
            )

        result = await self.generate_regex(prompt, pattern, max_tokens, **kwargs)
        result.constraint_type = ConstraintType.FORMAT
        return result

    # ========================================================================
    # Fallback Methods (when Outlines not available)
    # ========================================================================

    async def _fallback_regex(
        self,
        prompt: str,
        pattern: str,
        max_tokens: int,
    ) -> GenerationResult[str]:
        """Fallback regex generation using post-validation."""
        # Placeholder - would use another LLM and validate
        return GenerationResult(
            success=False,
            error="Outlines not available for regex constraint",
            constraint_type=ConstraintType.REGEX,
        )

    async def _fallback_json(
        self,
        prompt: str,
        schema: type,
        max_tokens: int,
    ) -> GenerationResult[BaseModel]:
        """Fallback JSON generation using instructor."""
        try:
            from .instructor_chains import InstructorClient

            client = InstructorClient()
            result = await client.extract(prompt, schema)

            if result.success:
                return GenerationResult(
                    success=True,
                    output=result.data,
                    raw_text=result.data.model_dump_json() if result.data else "",
                    constraint_type=ConstraintType.JSON,
                    latency_ms=result.latency_ms,
                )
            else:
                return GenerationResult(
                    success=False,
                    error=result.error,
                    constraint_type=ConstraintType.JSON,
                )
        except Exception as e:
            return GenerationResult(
                success=False,
                error=str(e),
                constraint_type=ConstraintType.JSON,
            )

    async def _fallback_choice(
        self,
        prompt: str,
        choices: list[str],
    ) -> GenerationResult[ChoiceResult]:
        """Fallback choice generation."""
        # Simple heuristic fallback
        return GenerationResult(
            success=False,
            error="Outlines not available for choice constraint",
            constraint_type=ConstraintType.CHOICE,
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def create_outlines_generator(
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> OutlinesGenerator:
    """
    Factory function to create an OutlinesGenerator.

    Args:
        model_name: Optional model name
        **kwargs: Additional configuration

    Returns:
        Configured OutlinesGenerator instance
    """
    if model_name:
        kwargs["model_name"] = model_name
    return OutlinesGenerator(**kwargs)


# Export availability
__all__ = [
    "OutlinesGenerator",
    "GenerationResult",
    "ChoiceResult",
    "Constraint",
    "ConstraintType",
    "CommonPatterns",
    "create_outlines_generator",
    "OUTLINES_AVAILABLE",
]
