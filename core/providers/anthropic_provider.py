#!/usr/bin/env python3
"""
Anthropic Claude Provider
Direct integration with Anthropic's Claude API.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Optional
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment
load_dotenv()

logger = structlog.get_logger(__name__)

# Import Anthropic SDK
try:
    import anthropic
    from anthropic import AsyncAnthropic, Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic SDK not available - install with: pip install anthropic")


class ClaudeMessage(BaseModel):
    """A message for Claude API."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ClaudeResponse(BaseModel):
    """Response from Claude API."""
    content: str
    model: str
    stop_reason: Optional[str] = None
    usage: dict[str, int] = Field(default_factory=dict)
    raw_response: Optional[dict[str, Any]] = None


@dataclass
class AnthropicProvider:
    """
    Direct Anthropic Claude API provider.

    Provides native access to Claude models without going through LiteLLM.
    Useful for advanced features like tool use, streaming, and vision.

    Usage:
        provider = AnthropicProvider()
        response = await provider.complete(
            messages=[ClaudeMessage(role="user", content="Hello!")],
            model="claude-3-5-sonnet-20241022"
        )
    """

    api_key: Optional[str] = None
    default_model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        """Initialize the provider with API key validation."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic SDK not installed")

        self.api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self._client = Anthropic(api_key=self.api_key)
        self._async_client = AsyncAnthropic(api_key=self.api_key)

        logger.info("anthropic_provider_initialized", model=self.default_model)

    async def complete(
        self,
        messages: list[ClaudeMessage],
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ClaudeResponse:
        """
        Generate a completion from Claude.

        Args:
            messages: List of conversation messages
            model: Model ID (defaults to claude-3-5-sonnet)
            system: System prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional API parameters

        Returns:
            ClaudeResponse with content and metadata
        """
        model = model or self.default_model
        max_tokens = max_tokens or self.max_tokens

        try:
            response = await self._async_client.messages.create(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                system=system or "You are a helpful assistant.",
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            logger.info(
                "claude_completion_success",
                model=model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            return ClaudeResponse(
                content=response.content[0].text,
                model=model,
                stop_reason=response.stop_reason,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            logger.error("claude_completion_failed", model=model, error=str(e))
            raise

    async def stream(
        self,
        messages: list[ClaudeMessage],
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens from Claude.

        Args:
            messages: List of conversation messages
            model: Model ID
            system: System prompt
            max_tokens: Maximum tokens
            **kwargs: Additional API parameters

        Yields:
            Individual content tokens
        """
        model = model or self.default_model
        max_tokens = max_tokens or self.max_tokens

        try:
            async with self._async_client.messages.stream(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                system=system or "You are a helpful assistant.",
                max_tokens=max_tokens,
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error("claude_stream_failed", model=model, error=str(e))
            raise

    def complete_sync(
        self,
        messages: list[ClaudeMessage],
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ClaudeResponse:
        """
        Synchronous completion for non-async contexts.

        Args:
            messages: List of conversation messages
            model: Model ID
            system: System prompt
            max_tokens: Maximum tokens
            **kwargs: Additional API parameters

        Returns:
            ClaudeResponse with content and metadata
        """
        model = model or self.default_model
        max_tokens = max_tokens or self.max_tokens

        response = self._client.messages.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            system=system or "You are a helpful assistant.",
            max_tokens=max_tokens,
            **kwargs,
        )

        return ClaudeResponse(
            content=response.content[0].text,
            model=model,
            stop_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        )

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using Anthropic's tokenizer.

        Args:
            text: Text to tokenize

        Returns:
            Token count
        """
        # Use the messages API to count tokens
        response = await self._async_client.messages.count_tokens(
            model=self.default_model,
            messages=[{"role": "user", "content": text}],
        )
        return response.input_tokens

    @property
    def available_models(self) -> list[str]:
        """List of available Claude models."""
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]


if __name__ == "__main__":
    import asyncio

    async def main():
        """Test the Anthropic provider."""
        try:
            provider = AnthropicProvider()
            print(f"Available models: {provider.available_models}")

            response = await provider.complete(
                messages=[ClaudeMessage(role="user", content="Say 'Anthropic Provider Ready!' in 3 words")],
            )
            print(f"Response: {response.content}")
            print(f"Usage: {response.usage}")

        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(main())
