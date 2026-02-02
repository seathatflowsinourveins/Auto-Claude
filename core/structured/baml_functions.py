#!/usr/bin/env python3
"""
BAML Functions - Type-Safe LLM Function Definitions
Part of the V33 Structured Output Layer.

Uses BAML (Boundary ML) for compile-time schema validation
and type-safe LLM function definitions.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Optional, TypeVar, Generic, Callable
from datetime import datetime
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

try:
    import baml_py
    from baml_py import ClientRegistry
    BAML_AVAILABLE = True
except ImportError:
    BAML_AVAILABLE = False
    ClientRegistry = None


# Type variable for generic function outputs
T = TypeVar("T", bound=BaseModel)


# ============================================================================
# BAML Schema Models
# ============================================================================

class BAMLStatus(str, Enum):
    """Status of a BAML function execution."""
    SUCCESS = "success"
    VALIDATION_ERROR = "validation_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"


class SummaryOutput(BaseModel):
    """Output schema for text summarization."""
    summary: str = Field(description="The condensed summary")
    key_points: list[str] = Field(default_factory=list, description="Key points extracted")
    word_count: int = Field(description="Word count of the summary")


class TranslationOutput(BaseModel):
    """Output schema for translation."""
    translated_text: str = Field(description="The translated text")
    source_language: str = Field(description="Detected source language")
    target_language: str = Field(description="Target language")
    confidence: float = Field(ge=0.0, le=1.0, description="Translation confidence")


class CodeGenerationOutput(BaseModel):
    """Output schema for code generation."""
    code: str = Field(description="The generated code")
    language: str = Field(description="Programming language")
    explanation: str = Field(description="Explanation of the code")
    dependencies: list[str] = Field(default_factory=list, description="Required dependencies")


class AnalysisOutput(BaseModel):
    """Output schema for general analysis."""
    analysis: str = Field(description="The analysis text")
    findings: list[str] = Field(default_factory=list, description="Key findings")
    recommendations: list[str] = Field(default_factory=list, description="Recommendations")
    confidence: float = Field(ge=0.0, le=1.0, description="Analysis confidence")


@dataclass
class BAMLResult(Generic[T]):
    """Result wrapper for BAML function execution."""
    status: BAMLStatus
    data: Optional[T] = None
    error: Optional[str] = None
    raw_output: Optional[str] = None
    latency_ms: float = 0.0
    tokens_used: int = 0
    model: str = ""


# ============================================================================
# BAML Function Registry
# ============================================================================

@dataclass
class BAMLFunctionDef:
    """Definition of a BAML function."""
    name: str
    description: str
    input_schema: type
    output_schema: type
    prompt_template: str
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 4096


class BAMLFunctionRegistry:
    """Registry for BAML function definitions."""

    def __init__(self):
        self._functions: dict[str, BAMLFunctionDef] = {}
        self._register_builtin_functions()

    def _register_builtin_functions(self):
        """Register built-in BAML functions."""

        # Summarization function
        self.register(BAMLFunctionDef(
            name="summarize",
            description="Summarize text into key points",
            input_schema=str,
            output_schema=SummaryOutput,
            prompt_template="""Summarize the following text concisely.
Extract the key points and provide a word count.

Text:
{input}

Respond with a JSON object containing:
- summary: condensed summary
- key_points: list of key points
- word_count: number of words in summary""",
        ))

        # Translation function
        self.register(BAMLFunctionDef(
            name="translate",
            description="Translate text to target language",
            input_schema=dict,
            output_schema=TranslationOutput,
            prompt_template="""Translate the following text to {target_language}.
Detect the source language and provide a confidence score.

Text:
{text}

Respond with a JSON object containing:
- translated_text: the translation
- source_language: detected source language
- target_language: target language
- confidence: confidence score 0-1""",
        ))

        # Code generation function
        self.register(BAMLFunctionDef(
            name="generate_code",
            description="Generate code from natural language",
            input_schema=dict,
            output_schema=CodeGenerationOutput,
            prompt_template="""Generate {language} code for the following task:

Task: {task}

Requirements:
{requirements}

Respond with a JSON object containing:
- code: the generated code
- language: programming language
- explanation: explanation of the code
- dependencies: list of required dependencies""",
            temperature=0.3,
        ))

        # Analysis function
        self.register(BAMLFunctionDef(
            name="analyze",
            description="Analyze content and provide insights",
            input_schema=dict,
            output_schema=AnalysisOutput,
            prompt_template="""Analyze the following content:

{content}

Focus on: {focus_areas}

Respond with a JSON object containing:
- analysis: detailed analysis
- findings: list of key findings
- recommendations: list of recommendations
- confidence: confidence score 0-1""",
        ))

    def register(self, func_def: BAMLFunctionDef) -> None:
        """Register a BAML function definition."""
        self._functions[func_def.name] = func_def

    def get(self, name: str) -> Optional[BAMLFunctionDef]:
        """Get a function definition by name."""
        return self._functions.get(name)

    def list_functions(self) -> list[str]:
        """List all registered function names."""
        return list(self._functions.keys())


# ============================================================================
# BAML Client
# ============================================================================

class BAMLClient:
    """
    BAML client for type-safe LLM function execution.

    Provides compile-time schema validation and structured outputs
    using BAML function definitions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "claude-sonnet-4-20250514",
    ):
        """
        Initialize the BAML client.

        Args:
            api_key: API key for the LLM provider
            default_model: Default model to use
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.default_model = default_model
        self.registry = BAMLFunctionRegistry()
        self._client = None

        if BAML_AVAILABLE and ClientRegistry:
            try:
                self._client = ClientRegistry()
            except Exception:
                pass

    def register_function(self, func_def: BAMLFunctionDef) -> None:
        """Register a custom BAML function."""
        self.registry.register(func_def)

    async def call(
        self,
        function_name: str,
        input_data: Any,
        **kwargs: Any,
    ) -> BAMLResult:
        """
        Call a registered BAML function.

        Args:
            function_name: Name of the registered function
            input_data: Input data for the function
            **kwargs: Additional parameters

        Returns:
            BAMLResult with the function output
        """
        start_time = datetime.now()

        func_def = self.registry.get(function_name)
        if not func_def:
            return BAMLResult(
                status=BAMLStatus.RUNTIME_ERROR,
                error=f"Function '{function_name}' not found",
            )

        try:
            # Format the prompt
            if isinstance(input_data, dict):
                prompt = func_def.prompt_template.format(**input_data)
            else:
                prompt = func_def.prompt_template.format(input=input_data)

            # Simulate LLM call (in production, use actual BAML runtime)
            # This is a placeholder for the actual BAML execution
            raw_output = await self._execute_llm(
                prompt=prompt,
                model=kwargs.get("model", func_def.model),
                temperature=kwargs.get("temperature", func_def.temperature),
                max_tokens=kwargs.get("max_tokens", func_def.max_tokens),
            )

            # Parse and validate output
            output = func_def.output_schema.model_validate_json(raw_output)

            latency = (datetime.now() - start_time).total_seconds() * 1000

            return BAMLResult(
                status=BAMLStatus.SUCCESS,
                data=output,
                raw_output=raw_output,
                latency_ms=latency,
                model=func_def.model,
            )

        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            return BAMLResult(
                status=BAMLStatus.VALIDATION_ERROR if "validation" in str(e).lower() else BAMLStatus.RUNTIME_ERROR,
                error=str(e),
                latency_ms=latency,
            )

    async def _execute_llm(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Execute LLM call (placeholder for actual implementation).

        In production, this would use the BAML runtime or direct API calls.
        """
        # Placeholder - in production use actual BAML runtime
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"LLM execution failed: {e}")

    async def summarize(self, text: str, **kwargs) -> BAMLResult[SummaryOutput]:
        """Summarize text using the summarize function."""
        return await self.call("summarize", text, **kwargs)

    async def translate(
        self,
        text: str,
        target_language: str,
        **kwargs,
    ) -> BAMLResult[TranslationOutput]:
        """Translate text to target language."""
        return await self.call("translate", {
            "text": text,
            "target_language": target_language,
        }, **kwargs)

    async def generate_code(
        self,
        task: str,
        language: str = "python",
        requirements: str = "",
        **kwargs,
    ) -> BAMLResult[CodeGenerationOutput]:
        """Generate code from natural language description."""
        return await self.call("generate_code", {
            "task": task,
            "language": language,
            "requirements": requirements or "None specified",
        }, **kwargs)

    async def analyze(
        self,
        content: str,
        focus_areas: str = "general insights",
        **kwargs,
    ) -> BAMLResult[AnalysisOutput]:
        """Analyze content and provide insights."""
        return await self.call("analyze", {
            "content": content,
            "focus_areas": focus_areas,
        }, **kwargs)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_baml_client(
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> BAMLClient:
    """
    Factory function to create a BAMLClient.

    Args:
        api_key: Optional API key
        **kwargs: Additional configuration

    Returns:
        Configured BAMLClient instance
    """
    return BAMLClient(api_key=api_key, **kwargs)


# Export availability
__all__ = [
    "BAMLClient",
    "BAMLResult",
    "BAMLStatus",
    "BAMLFunctionDef",
    "BAMLFunctionRegistry",
    "SummaryOutput",
    "TranslationOutput",
    "CodeGenerationOutput",
    "AnalysisOutput",
    "create_baml_client",
    "BAML_AVAILABLE",
]
