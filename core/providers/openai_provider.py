#!/usr/bin/env python3
"""
OpenAI Provider
Direct integration with OpenAI's API.
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

# Import OpenAI SDK
try:
    import openai
    from openai import AsyncOpenAI, OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai SDK not available - install with: pip install openai")


class OpenAIMessage(BaseModel):
    """A message for OpenAI API."""
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class OpenAIResponse(BaseModel):
    """Response from OpenAI API."""
    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: dict[str, int] = Field(default_factory=dict)
    raw_response: Optional[dict[str, Any]] = None


@dataclass
class OpenAIProvider:
    """
    Direct OpenAI API provider.

    Provides native access to OpenAI models including GPT-4, GPT-4 Turbo, and o1.
    Useful for advanced features like function calling, vision, and JSON mode.

    Usage:
        provider = OpenAIProvider()
        response = await provider.complete(
            messages=[OpenAIMessage(role="user", content="Hello!")],
            model="gpt-4o"
        )
    """

    api_key: Optional[str] = None
    default_model: str = "gpt-4o"
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        """Initialize the provider with API key validation."""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai SDK not installed")

        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self._client = OpenAI(api_key=self.api_key)
        self._async_client = AsyncOpenAI(api_key=self.api_key)

        logger.info("openai_provider_initialized", model=self.default_model)

    async def complete(
        self,
        messages: list[OpenAIMessage],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> OpenAIResponse:
        """
        Generate a completion from OpenAI.

        Args:
            messages: List of conversation messages
            model: Model ID (defaults to gpt-4o)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional API parameters

        Returns:
            OpenAIResponse with content and metadata
        """
        model = model or self.default_model
        max_tokens = max_tokens or self.max_tokens

        try:
            response = await self._async_client.chat.completions.create(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            usage = response.usage
            logger.info(
                "openai_completion_success",
                model=model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )

            return OpenAIResponse(
                content=response.choices[0].message.content or "",
                model=model,
                finish_reason=response.choices[0].finish_reason,
                usage={
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                },
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            logger.error("openai_completion_failed", model=model, error=str(e))
            raise

    async def stream(
        self,
        messages: list[OpenAIMessage],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens from OpenAI.

        Args:
            messages: List of conversation messages
            model: Model ID
            max_tokens: Maximum tokens
            **kwargs: Additional API parameters

        Yields:
            Individual content tokens
        """
        model = model or self.default_model
        max_tokens = max_tokens or self.max_tokens

        try:
            stream = await self._async_client.chat.completions.create(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error("openai_stream_failed", model=model, error=str(e))
            raise

    def complete_sync(
        self,
        messages: list[OpenAIMessage],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> OpenAIResponse:
        """
        Synchronous completion for non-async contexts.

        Args:
            messages: List of conversation messages
            model: Model ID
            max_tokens: Maximum tokens
            **kwargs: Additional API parameters

        Returns:
            OpenAIResponse with content and metadata
        """
        model = model or self.default_model
        max_tokens = max_tokens or self.max_tokens

        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=max_tokens,
            **kwargs,
        )

        usage = response.usage
        return OpenAIResponse(
            content=response.choices[0].message.content or "",
            model=model,
            finish_reason=response.choices[0].finish_reason,
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
        )

    async def embeddings(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small",
    ) -> list[list[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            model: Embedding model ID

        Returns:
            List of embedding vectors
        """
        try:
            response = await self._async_client.embeddings.create(
                model=model,
                input=texts,
            )

            return [item.embedding for item in response.data]

        except Exception as e:
            logger.error("openai_embeddings_failed", model=model, error=str(e))
            raise

    @property
    def available_models(self) -> list[str]:
        """List of available OpenAI models."""
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini",
        ]

    @property
    def embedding_models(self) -> list[str]:
        """List of available embedding models."""
        return [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]


if __name__ == "__main__":
    import asyncio

    async def main():
        """Test the OpenAI provider."""
        try:
            provider = OpenAIProvider()
            print(f"Available models: {provider.available_models}")

            response = await provider.complete(
                messages=[OpenAIMessage(role="user", content="Say 'OpenAI Provider Ready!' in 3 words")],
            )
            print(f"Response: {response.content}")
            print(f"Usage: {response.usage}")

        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(main())
