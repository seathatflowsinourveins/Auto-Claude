#!/usr/bin/env python3
"""
Unified LLM Gateway via LiteLLM
Provides a consistent interface to 100+ LLM providers.
"""

from __future__ import annotations

import os
import time
from typing import Any, AsyncIterator, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import litellm
from litellm import acompletion, completion

# V41: Import CircuitBreaker from claude_flow_v3
try:
    from core.orchestration.claude_flow_v3 import CircuitBreaker, CircuitState
    V41_CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    V41_CIRCUIT_BREAKER_AVAILABLE = False

# V42: Import Opik for observability tracing
V42_OPIK_AVAILABLE = False
_opik_module: Any = None
try:
    import opik as _opik_module
    V42_OPIK_AVAILABLE = True
except ImportError:
    pass

    # Fallback CircuitBreaker if import fails
    class CircuitState(str, Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"

    @dataclass
    class CircuitBreaker:
        failure_threshold: int = 5
        recovery_timeout: float = 30.0
        state: CircuitState = CircuitState.CLOSED
        failures: int = 0
        last_failure_time: Optional[float] = None

        def record_success(self) -> None:
            self.failures = 0
            self.state = CircuitState.CLOSED

        def record_failure(self) -> None:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN

        def can_execute(self) -> bool:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    return True
                return False
            return True

# Load environment variables
load_dotenv()

# Configure logging
logger = structlog.get_logger(__name__)

# Configure LiteLLM
litellm.set_verbose = os.getenv("DEBUG_MODE", "false").lower() == "true"


class Provider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE = "azure"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    OLLAMA = "ollama"


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    provider: Provider
    model_id: str
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    timeout: int = Field(default=60, ge=1)

    @property
    def litellm_model(self) -> str:
        """Get the LiteLLM model string."""
        if self.provider == Provider.ANTHROPIC:
            return self.model_id
        elif self.provider == Provider.OPENAI:
            return self.model_id
        elif self.provider == Provider.OLLAMA:
            return f"ollama/{self.model_id}"
        return f"{self.provider.value}/{self.model_id}"


class Message(BaseModel):
    """A chat message."""
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class CompletionResponse(BaseModel):
    """Standardized completion response."""
    content: str
    model: str
    provider: Provider
    usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    raw_response: Optional[dict[str, Any]] = None


@dataclass
class LLMGateway:
    """
    Unified LLM Gateway using LiteLLM.

    Provides a consistent interface across 100+ LLM providers.

    V41 ENHANCEMENT: Circuit breaker integration for resilience.
    - 5 failures -> circuit opens
    - 30s recovery timeout -> half-open
    - 3 successes in half-open -> closes

    Usage:
        gateway = LLMGateway()
        response = await gateway.complete(
            messages=[Message(role="user", content="Hello!")],
            model_config=ModelConfig(provider=Provider.ANTHROPIC, model_id="claude-3-5-sonnet-20241022")
        )
    """

    default_provider: Provider = Provider.ANTHROPIC
    default_model: str = "claude-3-5-sonnet-20241022"
    fallback_models: list[tuple[Provider, str]] = field(default_factory=lambda: [
        (Provider.OPENAI, "gpt-4o"),
        (Provider.ANTHROPIC, "claude-3-haiku-20240307"),
    ])

    # V41: Circuit breakers per provider
    circuit_breakers: Dict[str, CircuitBreaker] = field(default_factory=lambda: {
        provider.value: CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
        for provider in Provider
    })

    # V41: Metrics tracking
    metrics: Dict[str, Any] = field(default_factory=lambda: {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "circuit_breaker_trips": 0,
        "provider_usage": {},
    })

    def __post_init__(self) -> None:
        """Validate API keys are configured."""
        self._validate_keys()
        logger.info("llm_gateway_initialized",
                   default_provider=self.default_provider.value,
                   v41_circuit_breaker="enabled")

    def _validate_keys(self) -> None:
        """Check that required API keys are set."""
        if self.default_provider == Provider.ANTHROPIC:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
        elif self.default_provider == Provider.OPENAI:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not set in environment")

    def _get_default_config(self) -> ModelConfig:
        """Get default model configuration."""
        return ModelConfig(
            provider=self.default_provider,
            model_id=self.default_model,
        )

    async def complete(
        self,
        messages: list[Message],
        model_config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Generate a completion from the LLM.

        V41 ENHANCEMENT: Circuit breaker integration.
        - Checks if provider circuit is open before attempting
        - Records success/failure for circuit state management
        - Tracks metrics for observability

        Args:
            messages: List of chat messages
            model_config: Model configuration (uses defaults if not provided)
            **kwargs: Additional arguments passed to LiteLLM

        Returns:
            Standardized completion response

        Raises:
            RuntimeError: If circuit breaker is open for the provider
        """
        config = model_config or self._get_default_config()
        provider_key = config.provider.value

        # V41: Check circuit breaker before execution
        circuit = self.circuit_breakers.get(provider_key)
        if circuit and not circuit.can_execute():
            self.metrics["circuit_breaker_trips"] += 1
            logger.warning(
                "circuit_breaker_open",
                provider=provider_key,
                state=circuit.state.value if hasattr(circuit.state, 'value') else str(circuit.state),
                failures=circuit.failures
            )
            raise RuntimeError(f"Circuit breaker open for provider {provider_key}")

        self.metrics["total_requests"] += 1

        # V42: Start Opik trace for observability
        opik_trace = None
        if V42_OPIK_AVAILABLE and _opik_module:
            try:
                opik_trace = _opik_module.trace(
                    name=f"llm_completion_{provider_key}",
                    input={"model": config.model_id, "messages_count": len(messages)},
                    metadata={
                        "provider": provider_key,
                        "max_tokens": config.max_tokens,
                        "temperature": config.temperature,
                    }
                )
            except Exception as opik_err:
                logger.debug(f"Opik trace init failed: {opik_err}")

        try:
            response = await acompletion(
                model=config.litellm_model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                timeout=config.timeout,
                **kwargs,
            )

            # V41: Record success
            if circuit:
                circuit.record_success()

            self.metrics["successful_requests"] += 1
            self.metrics["provider_usage"][provider_key] = self.metrics["provider_usage"].get(provider_key, 0) + 1

            # V42: Record Opik trace success
            if opik_trace:
                try:
                    opik_trace.end(
                        output={"tokens": response.usage.total_tokens if response.usage else 0},
                        metadata={"status": "success", "finish_reason": response.choices[0].finish_reason}
                    )
                except Exception:
                    pass

            logger.info(
                "llm_completion_success",
                model=config.model_id,
                provider=config.provider.value,
                tokens=response.usage.total_tokens if response.usage else 0,
            )

            return CompletionResponse(
                content=response.choices[0].message.content,
                model=config.model_id,
                provider=config.provider,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=response.choices[0].finish_reason,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            # V41: Record failure
            if circuit:
                circuit.record_failure()

            self.metrics["failed_requests"] += 1

            # V42: Record Opik trace failure
            if opik_trace:
                try:
                    opik_trace.end(
                        output={"error": str(e)},
                        metadata={"status": "failed", "error_type": type(e).__name__}
                    )
                except Exception:
                    pass

            logger.error(
                "llm_completion_failed",
                model=config.model_id,
                provider=config.provider.value,
                error=str(e),
                circuit_failures=circuit.failures if circuit else 0,
            )
            raise

    async def complete_with_fallback(
        self,
        messages: list[Message],
        model_config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Generate completion with automatic fallback to backup models.

        Args:
            messages: List of chat messages
            model_config: Primary model configuration
            **kwargs: Additional arguments

        Returns:
            Completion response from first successful model
        """
        config = model_config or self._get_default_config()

        # Try primary model first
        try:
            return await self.complete(messages, config, **kwargs)
        except Exception as primary_error:
            logger.warning(
                "primary_model_failed_trying_fallback",
                primary_model=config.model_id,
                error=str(primary_error),
            )

        # Try fallback models
        for fallback_provider, fallback_model in self.fallback_models:
            try:
                fallback_config = ModelConfig(
                    provider=fallback_provider,
                    model_id=fallback_model,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )
                return await self.complete(messages, fallback_config, **kwargs)
            except Exception as fallback_error:
                logger.warning(
                    "fallback_model_failed",
                    fallback_model=fallback_model,
                    error=str(fallback_error),
                )
                continue

        raise RuntimeError("All models failed - no successful completion")

    async def stream(
        self,
        messages: list[Message],
        model_config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens from the LLM.

        Args:
            messages: List of chat messages
            model_config: Model configuration
            **kwargs: Additional arguments

        Yields:
            Individual content tokens as they arrive
        """
        config = model_config or self._get_default_config()

        try:
            response = await acompletion(
                model=config.litellm_model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                stream=True,
                **kwargs,
            )

            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(
                "llm_stream_failed",
                model=config.model_id,
                error=str(e),
            )
            raise

    def complete_sync(
        self,
        messages: list[Message],
        model_config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Synchronous completion (for non-async contexts).

        Args:
            messages: List of chat messages
            model_config: Model configuration
            **kwargs: Additional arguments

        Returns:
            Completion response
        """
        config = model_config or self._get_default_config()

        response = completion(
            model=config.litellm_model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            **kwargs,
        )

        return CompletionResponse(
            content=response.choices[0].message.content,
            model=config.model_id,
            provider=config.provider,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            finish_reason=response.choices[0].finish_reason,
        )

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on the gateway.

        Returns:
            Health status with provider availability
        """
        results: dict[str, Any] = {
            "status": "healthy",
            "providers": {},
        }

        # Test Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                await self.complete(
                    messages=[Message(role="user", content="ping")],
                    model_config=ModelConfig(
                        provider=Provider.ANTHROPIC,
                        model_id="claude-3-haiku-20240307",
                        max_tokens=10,
                    ),
                )
                results["providers"]["anthropic"] = "available"
            except Exception as e:
                results["providers"]["anthropic"] = f"error: {str(e)}"
        else:
            results["providers"]["anthropic"] = "not_configured"

        # Test OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                await self.complete(
                    messages=[Message(role="user", content="ping")],
                    model_config=ModelConfig(
                        provider=Provider.OPENAI,
                        model_id="gpt-4o-mini",
                        max_tokens=10,
                    ),
                )
                results["providers"]["openai"] = "available"
            except Exception as e:
                results["providers"]["openai"] = f"error: {str(e)}"
        else:
            results["providers"]["openai"] = "not_configured"

        # Set overall status
        if all(v == "not_configured" for v in results["providers"].values()):
            results["status"] = "no_providers"
        elif any("error" in str(v) for v in results["providers"].values()):
            results["status"] = "degraded"

        return results


# Convenience function for quick completions
async def quick_complete(
    prompt: str,
    system: Optional[str] = None,
    model: str = "claude-3-5-sonnet-20241022",
    provider: Provider = Provider.ANTHROPIC,
) -> str:
    """
    Quick completion helper for simple use cases.

    Args:
        prompt: User prompt
        system: Optional system message
        model: Model ID
        provider: Provider to use

    Returns:
        Completion content as string
    """
    gateway = LLMGateway(default_provider=provider, default_model=model)

    messages = []
    if system:
        messages.append(Message(role="system", content=system))
    messages.append(Message(role="user", content=prompt))

    response = await gateway.complete(messages)
    return response.content


if __name__ == "__main__":
    import asyncio

    async def main():
        """Test the LLM Gateway."""
        gateway = LLMGateway()

        print("Testing LLM Gateway...")
        print("-" * 40)

        # Health check
        health = await gateway.health_check()
        print(f"Health Status: {health['status']}")
        for provider, status in health["providers"].items():
            print(f"  - {provider}: {status}")

        print("-" * 40)

        # Test completion
        if health["status"] != "no_providers":
            response = await gateway.complete(
                messages=[
                    Message(role="system", content="You are a helpful assistant."),
                    Message(role="user", content="Say 'Protocol Layer Ready!' in exactly 3 words."),
                ]
            )
            print(f"Response: {response.content}")
            print(f"Model: {response.model}")
            print(f"Tokens: {response.usage}")

    asyncio.run(main())
