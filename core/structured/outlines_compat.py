"""Outlines compatibility layer using guidance + regex + JSON schema.

Outlines requires PyO3 Rust bindings which max out at Python 3.13.
This layer provides equivalent constrained generation capabilities.

Usage:
    from core.structured.outlines_compat import OutlinesCompat

    # Choice constraint
    sentiment = await OutlinesCompat.generate(
        "Is this positive or negative?",
        OutlinesCompat.choice(["positive", "negative"])
    )

    # JSON schema constraint
    from pydantic import BaseModel
    class Person(BaseModel):
        name: str
        age: int

    person = await OutlinesCompat.generate(
        "Generate a person",
        OutlinesCompat.json_schema(Person)
    )
"""
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union, get_type_hints
from abc import ABC, abstractmethod

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None


class Constraint(ABC):
    """Base class for generation constraints."""

    @abstractmethod
    def __call__(self, text: str) -> Any:
        """Apply constraint to generated text."""
        pass

    @abstractmethod
    def get_prompt_hint(self) -> str:
        """Get prompt hint for this constraint."""
        pass


@dataclass
class Choice(Constraint):
    """Constrained choice generator."""
    options: List[str]
    case_sensitive: bool = False

    def __call__(self, text: str) -> str:
        """Find best matching option from text."""
        text_cmp = text if self.case_sensitive else text.lower()

        # Try exact match first
        for opt in self.options:
            opt_cmp = opt if self.case_sensitive else opt.lower()
            if opt_cmp == text_cmp.strip():
                return opt

        # Try contains match
        for opt in self.options:
            opt_cmp = opt if self.case_sensitive else opt.lower()
            if opt_cmp in text_cmp:
                return opt

        # Return first option as default
        return self.options[0] if self.options else ""

    def get_prompt_hint(self) -> str:
        return f"Respond with exactly one of: {', '.join(self.options)}"


@dataclass
class Regex(Constraint):
    """Regex-constrained generation."""
    pattern: str
    flags: int = 0

    def __call__(self, text: str) -> Optional[str]:
        """Extract first match from text."""
        match = re.search(self.pattern, text, self.flags)
        return match.group(0) if match else None

    def get_prompt_hint(self) -> str:
        return f"Your response must match the pattern: {self.pattern}"


@dataclass
class Integer(Constraint):
    """Integer extraction constraint."""
    min_value: Optional[int] = None
    max_value: Optional[int] = None

    def __call__(self, text: str) -> Optional[int]:
        """Extract integer from text."""
        match = re.search(r'-?\d+', text)
        if match:
            value = int(match.group(0))
            if self.min_value is not None and value < self.min_value:
                return self.min_value
            if self.max_value is not None and value > self.max_value:
                return self.max_value
            return value
        return None

    def get_prompt_hint(self) -> str:
        parts = ["Respond with an integer"]
        if self.min_value is not None:
            parts.append(f"minimum {self.min_value}")
        if self.max_value is not None:
            parts.append(f"maximum {self.max_value}")
        return ", ".join(parts)


@dataclass
class Float(Constraint):
    """Float extraction constraint."""
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def __call__(self, text: str) -> Optional[float]:
        """Extract float from text."""
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            value = float(match.group(0))
            if self.min_value is not None and value < self.min_value:
                return self.min_value
            if self.max_value is not None and value > self.max_value:
                return self.max_value
            return value
        return None

    def get_prompt_hint(self) -> str:
        parts = ["Respond with a number"]
        if self.min_value is not None:
            parts.append(f"minimum {self.min_value}")
        if self.max_value is not None:
            parts.append(f"maximum {self.max_value}")
        return ", ".join(parts)


class JsonGenerator(Constraint):
    """JSON schema-constrained generation."""

    def __init__(self, schema: Union[Dict, Type]):
        if PYDANTIC_AVAILABLE and isinstance(schema, type) and issubclass(schema, BaseModel):
            self.schema = schema.model_json_schema()
            self.model_class = schema
        else:
            self.schema = schema if isinstance(schema, dict) else {}
            self.model_class = None

    def __call__(self, text: str) -> Any:
        """Extract and parse JSON from text."""
        # Try to find JSON object or array
        patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested objects
            r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # Nested arrays
            r'\{[^{}]*\}',  # Simple object
            r'\[[^\[\]]*\]',  # Simple array
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if self.model_class:
                        return self.model_class(**data)
                    return data
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue

        return None

    def get_prompt_hint(self) -> str:
        return f"Respond with valid JSON matching this schema:\n{json.dumps(self.schema, indent=2)}"


class OutlinesCompat:
    """Outlines-compatible constrained generation interface."""

    @staticmethod
    def choice(options: List[str], case_sensitive: bool = False) -> Choice:
        """Create a choice constraint."""
        return Choice(options=options, case_sensitive=case_sensitive)

    @staticmethod
    def regex(pattern: str, flags: int = 0) -> Regex:
        """Create a regex constraint."""
        return Regex(pattern=pattern, flags=flags)

    @staticmethod
    def integer(min_value: Optional[int] = None, max_value: Optional[int] = None) -> Integer:
        """Create an integer constraint."""
        return Integer(min_value=min_value, max_value=max_value)

    @staticmethod
    def float_num(min_value: Optional[float] = None, max_value: Optional[float] = None) -> Float:
        """Create a float constraint."""
        return Float(min_value=min_value, max_value=max_value)

    @staticmethod
    def json_schema(schema: Union[Dict, Type]) -> JsonGenerator:
        """Create a JSON schema constraint."""
        return JsonGenerator(schema)

    @staticmethod
    async def generate(
        prompt: str,
        constraint: Constraint,
        llm_provider=None,
        max_retries: int = 3
    ) -> Any:
        """Generate constrained output using LLM."""
        if llm_provider is None:
            try:
                from core.providers.anthropic_provider import AnthropicProvider
                llm_provider = AnthropicProvider()
            except ImportError:
                raise RuntimeError("No LLM provider available")

        # Add constraint hint to prompt
        enhanced_prompt = f"{prompt}\n\n{constraint.get_prompt_hint()}"

        for attempt in range(max_retries):
            try:
                response = await llm_provider.complete(enhanced_prompt)
                result = constraint(response)
                if result is not None:
                    return result
            except Exception:
                if attempt == max_retries - 1:
                    raise

        return None


OUTLINES_COMPAT_AVAILABLE = True
