# Phase 2: Protocol Layer Setup
## Self-Contained Claude Code CLI Executable Prompt

> **Version**: 1.0
> **Created**: 2026-01-24
> **Phase**: 2 of 8
> **Estimated Duration**: 3-5 hours
> **Dependencies**: Phase 1 COMPLETE

---

## Instructions for Claude Code CLI

Copy everything below the line and paste into Claude Code CLI as a single prompt:

---

```
You are tasked with implementing the Protocol Layer (Layer 0) for the Unleash SDK platform.
This is Phase 2 of an 8-phase implementation plan.

## Context

Phase 1 (Environment Setup) has been completed. The following are now available:
- Virtual environment at `.venv/`
- Base dependencies installed (structlog, httpx, pydantic, python-dotenv, rich)
- Environment configuration in `.env`
- Validation script at `scripts/validate_environment.py`

## Phase 2 Objectives

Implement the Protocol Layer with 5 core SDKs:
1. **mcp-python-sdk** - MCP protocol client/server
2. **fastmcp** - Pythonic MCP development
3. **litellm** - Universal LLM gateway (100+ providers)
4. **anthropic** - Official Claude SDK
5. **openai-sdk** - Official OpenAI SDK

## Pre-Flight Validation

### Step 0.1: Verify Phase 1 Completion

```bash
python scripts/validate_environment.py
```

**Expected**: All checks pass. If any fail, complete Phase 1 first.

### Step 0.2: Verify API Keys

```bash
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')
openai_key = os.getenv('OPENAI_API_KEY', '')
print(f'ANTHROPIC_API_KEY: {'✓ Set' if anthropic_key and anthropic_key != 'your-anthropic-api-key' else '✗ Missing'}')
print(f'OPENAI_API_KEY: {'✓ Set' if openai_key and openai_key != 'your-openai-api-key' else '✗ Missing'}')
"
```

**Expected**: Both keys show "✓ Set"
**If missing**: Edit `.env` with valid API keys before proceeding.

### Step 0.3: Install Phase 2 Dependencies

```bash
# Activate virtual environment first
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate

# Install protocol layer dependencies
uv pip install mcp anthropic openai litellm fastmcp

# Verify installation
python -c "import mcp; import anthropic; import openai; import litellm; print('Protocol SDKs installed')"
```

## Step 1: Create LLM Gateway

Create file `core/llm_gateway.py`:

```python
#!/usr/bin/env python3
"""
Unified LLM Gateway via LiteLLM
Provides a consistent interface to 100+ LLM providers.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Optional
from dataclasses import dataclass, field
from enum import Enum

import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import litellm
from litellm import acompletion, completion

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
    
    def __post_init__(self) -> None:
        """Validate API keys are configured."""
        self._validate_keys()
        logger.info("llm_gateway_initialized", default_provider=self.default_provider.value)
    
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
        
        Args:
            messages: List of chat messages
            model_config: Model configuration (uses defaults if not provided)
            **kwargs: Additional arguments passed to LiteLLM
            
        Returns:
            Standardized completion response
        """
        config = model_config or self._get_default_config()
        
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
            logger.error(
                "llm_completion_failed",
                model=config.model_id,
                provider=config.provider.value,
                error=str(e),
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
        results = {
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
```

## Step 2: Create MCP Server

Create file `core/mcp_server.py`:

```python
#!/usr/bin/env python3
"""
MCP Server with FastMCP
Provides tools and resources via Model Context Protocol.
"""

from __future__ import annotations

import os
import json
from typing import Any, Optional
from datetime import datetime

import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment
load_dotenv()

logger = structlog.get_logger(__name__)

# Import fastmcp - the Pythonic MCP framework
try:
    from fastmcp import FastMCP, Context
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    logger.warning("fastmcp not available - install with: pip install fastmcp")


class ToolResult(BaseModel):
    """Standardized tool execution result."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


def create_mcp_server(name: str = "unleash") -> "FastMCP":
    """
    Create and configure the MCP server.
    
    Args:
        name: Server name/identifier
        
    Returns:
        Configured FastMCP server instance
    """
    if not FASTMCP_AVAILABLE:
        raise ImportError("fastmcp required - install with: pip install fastmcp")
    
    # Initialize MCP server
    mcp = FastMCP(name)
    
    # ============================================
    # Tool: Platform Status
    # ============================================
    @mcp.tool()
    async def platform_status(ctx: Context) -> dict[str, Any]:
        """
        Get current platform status and health information.
        
        Returns system status, loaded SDKs, and configuration state.
        """
        logger.info("tool_called", tool="platform_status")
        
        status = {
            "platform": "unleash",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "environment": {
                "python_version": os.sys.version,
                "debug_mode": os.getenv("DEBUG_MODE", "false"),
            },
            "sdks": {
                "protocol_layer": {
                    "mcp": "available",
                    "fastmcp": "available" if FASTMCP_AVAILABLE else "not_installed",
                    "litellm": "available",
                    "anthropic": "available" if os.getenv("ANTHROPIC_API_KEY") else "not_configured",
                    "openai": "available" if os.getenv("OPENAI_API_KEY") else "not_configured",
                }
            },
            "status": "operational",
        }
        
        return status
    
    # ============================================
    # Tool: LLM Complete
    # ============================================
    @mcp.tool()
    async def llm_complete(
        ctx: Context,
        prompt: str,
        system: str = "You are a helpful assistant.",
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        Complete a prompt using the LLM Gateway.
        
        Args:
            prompt: The user prompt to complete
            system: System message for the LLM
            model: Model identifier to use
            max_tokens: Maximum tokens in response
            
        Returns:
            Completion result with content and metadata
        """
        logger.info("tool_called", tool="llm_complete", model=model)
        
        try:
            # Import gateway here to avoid circular imports
            from core.llm_gateway import LLMGateway, Message, ModelConfig, Provider
            
            gateway = LLMGateway()
            
            messages = [
                Message(role="system", content=system),
                Message(role="user", content=prompt),
            ]
            
            # Determine provider from model name
            provider = Provider.OPENAI if "gpt" in model.lower() else Provider.ANTHROPIC
            
            config = ModelConfig(
                provider=provider,
                model_id=model,
                max_tokens=max_tokens,
            )
            
            response = await gateway.complete(messages, config)
            
            return ToolResult(
                success=True,
                data={
                    "content": response.content,
                    "model": response.model,
                    "provider": response.provider.value,
                    "usage": response.usage,
                },
            ).model_dump()
            
        except Exception as e:
            logger.error("llm_complete_failed", error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
            ).model_dump()
    
    # ============================================
    # Tool: Read File
    # ============================================
    @mcp.tool()
    async def read_file(ctx: Context, path: str) -> dict[str, Any]:
        """
        Read contents of a file in the workspace.
        
        Args:
            path: Relative or absolute path to file
            
        Returns:
            File contents or error
        """
        logger.info("tool_called", tool="read_file", path=path)
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            return ToolResult(
                success=True,
                data={
                    "path": path,
                    "content": content,
                    "size": len(content),
                    "lines": content.count("\n") + 1,
                },
            ).model_dump()
            
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error=f"File not found: {path}",
            ).model_dump()
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
            ).model_dump()
    
    # ============================================
    # Tool: Write File
    # ============================================
    @mcp.tool()
    async def write_file(
        ctx: Context,
        path: str,
        content: str,
        create_dirs: bool = True,
    ) -> dict[str, Any]:
        """
        Write content to a file.
        
        Args:
            path: Path to write to
            content: Content to write
            create_dirs: Create parent directories if needed
            
        Returns:
            Write result with file info
        """
        logger.info("tool_called", tool="write_file", path=path)
        
        try:
            if create_dirs:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return ToolResult(
                success=True,
                data={
                    "path": path,
                    "bytes_written": len(content.encode("utf-8")),
                },
            ).model_dump()
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
            ).model_dump()
    
    # ============================================
    # Tool: List Directory
    # ============================================
    @mcp.tool()
    async def list_directory(
        ctx: Context,
        path: str = ".",
        recursive: bool = False,
    ) -> dict[str, Any]:
        """
        List contents of a directory.
        
        Args:
            path: Directory path
            recursive: Include subdirectories
            
        Returns:
            List of files and directories
        """
        logger.info("tool_called", tool="list_directory", path=path)
        
        try:
            items = []
            
            if recursive:
                for root, dirs, files in os.walk(path):
                    for d in dirs:
                        items.append({
                            "path": os.path.join(root, d),
                            "type": "directory",
                        })
                    for f in files:
                        full_path = os.path.join(root, f)
                        items.append({
                            "path": full_path,
                            "type": "file",
                            "size": os.path.getsize(full_path),
                        })
            else:
                for item in os.listdir(path):
                    full_path = os.path.join(path, item)
                    items.append({
                        "path": full_path,
                        "type": "directory" if os.path.isdir(full_path) else "file",
                        "size": os.path.getsize(full_path) if os.path.isfile(full_path) else None,
                    })
            
            return ToolResult(
                success=True,
                data={
                    "path": path,
                    "count": len(items),
                    "items": items,
                },
            ).model_dump()
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
            ).model_dump()
    
    # ============================================
    # Tool: Execute Python
    # ============================================
    @mcp.tool()
    async def execute_python(ctx: Context, code: str) -> dict[str, Any]:
        """
        Execute Python code in a sandboxed environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution result or error
        """
        logger.info("tool_called", tool="execute_python")
        
        try:
            # Create a restricted globals dict
            restricted_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "tuple": tuple,
                    "set": set,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "sorted": sorted,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "round": round,
                    "isinstance": isinstance,
                    "type": type,
                    "True": True,
                    "False": False,
                    "None": None,
                },
            }
            
            # Capture output
            import io
            import sys
            
            stdout_capture = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout_capture
            
            try:
                exec(code, restricted_globals)
                output = stdout_capture.getvalue()
            finally:
                sys.stdout = old_stdout
            
            return ToolResult(
                success=True,
                data={
                    "output": output,
                    "globals": list(restricted_globals.keys()),
                },
            ).model_dump()
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
            ).model_dump()
    
    # ============================================
    # Resource: Configuration
    # ============================================
    @mcp.resource("config://platform")
    async def get_platform_config() -> str:
        """Get platform configuration as JSON."""
        config = {
            "name": "unleash",
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "features": {
                "memory": os.getenv("ENABLE_MEMORY", "true") == "true",
                "guardrails": os.getenv("ENABLE_GUARDRAILS", "true") == "true",
                "observability": os.getenv("ENABLE_OBSERVABILITY", "true") == "true",
            },
            "paths": {
                "sdk_base": os.getenv("SDK_BASE_PATH", "./sdks"),
                "stack": os.getenv("STACK_PATH", "./stack"),
                "core": os.getenv("CORE_PATH", "./core"),
                "platform": os.getenv("PLATFORM_PATH", "./platform"),
            },
        }
        return json.dumps(config, indent=2)
    
    # ============================================
    # Resource: SDK List
    # ============================================
    @mcp.resource("sdks://list")
    async def get_sdk_list() -> str:
        """Get list of available SDKs."""
        sdks = {
            "layer_0_protocol": [
                {"name": "mcp-python-sdk", "status": "available"},
                {"name": "fastmcp", "status": "available" if FASTMCP_AVAILABLE else "not_installed"},
                {"name": "litellm", "status": "available"},
                {"name": "anthropic", "status": "configured" if os.getenv("ANTHROPIC_API_KEY") else "not_configured"},
                {"name": "openai", "status": "configured" if os.getenv("OPENAI_API_KEY") else "not_configured"},
            ],
            "layer_1_orchestration": [
                {"name": "temporal", "status": "pending_phase_3"},
                {"name": "langgraph", "status": "pending_phase_3"},
                {"name": "crewai", "status": "pending_phase_3"},
            ],
            "layer_2_memory": [
                {"name": "letta", "status": "pending_phase_4"},
                {"name": "zep", "status": "pending_phase_4"},
                {"name": "mem0", "status": "pending_phase_4"},
            ],
        }
        return json.dumps(sdks, indent=2)
    
    logger.info("mcp_server_created", name=name, tools=5, resources=2)
    return mcp


# Global server instance
_server: Optional["FastMCP"] = None


def get_server() -> "FastMCP":
    """Get or create the global MCP server instance."""
    global _server
    if _server is None:
        _server = create_mcp_server()
    return _server


def run_server(transport: str = "stdio") -> None:
    """
    Run the MCP server.
    
    Args:
        transport: Transport type ("stdio" or "sse")
    """
    server = get_server()
    logger.info("mcp_server_starting", transport=transport)
    server.run(transport=transport)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unleash MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol",
    )
    
    args = parser.parse_args()
    run_server(args.transport)
```

## Step 3: Create Provider Integrations

### 3.1 Create Providers Directory

```bash
mkdir -p core/providers
touch core/providers/__init__.py
```

### 3.2 Create Anthropic Provider

Create file `core/providers/anthropic_provider.py`:

```python
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
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import anthropic
from anthropic import AsyncAnthropic, Anthropic

load_dotenv()
logger = structlog.get_logger(__name__)


class ClaudeMessage(BaseModel):
    """A Claude chat message."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ClaudeResponse(BaseModel):
    """Claude API response."""
    content: str
    model: str
    stop_reason: Optional[str] = None
    usage: dict[str, int] = Field(default_factory=dict)


@dataclass
class AnthropicProvider:
    """
    Direct Anthropic Claude provider.
    
    Use this for Claude-specific features not available through LiteLLM.
    """
    
    api_key: Optional[str] = None
    default_model: str = "claude-3-5-sonnet-20241022"
    max_retries: int = 3
    
    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        self._client = Anthropic(api_key=self.api_key, max_retries=self.max_retries)
        self._async_client = AsyncAnthropic(api_key=self.api_key, max_retries=self.max_retries)
        
        logger.info("anthropic_provider_initialized", model=self.default_model)
    
    async def complete(
        self,
        messages: list[ClaudeMessage],
        system: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ClaudeResponse:
        """
        Generate a completion using Claude.
        
        Args:
            messages: Conversation messages
            system: System prompt
            model: Model to use
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            **kwargs: Additional API parameters
            
        Returns:
            Claude response
        """
        model = model or self.default_model
        
        try:
            response = await self._async_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are a helpful assistant.",
                messages=[{"role": m.role, "content": m.content} for m in messages],
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
                model=response.model,
                stop_reason=response.stop_reason,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )
            
        except anthropic.APIError as e:
            logger.error("claude_api_error", error=str(e), model=model)
            raise
    
    async def stream(
        self,
        messages: list[ClaudeMessage],
        system: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens from Claude.
        
        Args:
            messages: Conversation messages
            system: System prompt
            model: Model to use
            max_tokens: Maximum response tokens
            **kwargs: Additional parameters
            
        Yields:
            Text tokens as they arrive
        """
        model = model or self.default_model
        
        async with self._async_client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a helpful assistant.",
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield text
    
    def complete_sync(
        self,
        messages: list[ClaudeMessage],
        system: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> ClaudeResponse:
        """Synchronous completion."""
        model = model or self.default_model
        
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a helpful assistant.",
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        )
        
        return ClaudeResponse(
            content=response.content[0].text,
            model=response.model,
            stop_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )
    
    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count
            model: Model for tokenization
            
        Returns:
            Token count
        """
        model = model or self.default_model
        count = await self._async_client.count_tokens(text)
        return count


# Convenience function
async def claude_complete(
    prompt: str,
    system: Optional[str] = None,
    model: str = "claude-3-5-sonnet-20241022",
) -> str:
    """Quick completion with Claude."""
    provider = AnthropicProvider(default_model=model)
    response = await provider.complete(
        messages=[ClaudeMessage(role="user", content=prompt)],
        system=system,
    )
    return response.content


if __name__ == "__main__":
    import asyncio
    
    async def main():
        provider = AnthropicProvider()
        
        response = await provider.complete(
            messages=[ClaudeMessage(role="user", content="Say 'Anthropic Provider Ready!'")],
            system="You are concise.",
        )
        
        print(f"Response: {response.content}")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
    
    asyncio.run(main())
```

### 3.3 Create OpenAI Provider

Create file `core/providers/openai_provider.py`:

```python
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
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import openai
from openai import AsyncOpenAI, OpenAI

load_dotenv()
logger = structlog.get_logger(__name__)


class OpenAIMessage(BaseModel):
    """An OpenAI chat message."""
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class OpenAIResponse(BaseModel):
    """OpenAI API response."""
    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: dict[str, int] = Field(default_factory=dict)


@dataclass
class OpenAIProvider:
    """
    Direct OpenAI provider.
    
    Use this for OpenAI-specific features not available through LiteLLM.
    """
    
    api_key: Optional[str] = None
    default_model: str = "gpt-4o"
    max_retries: int = 3
    
    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self._client = OpenAI(api_key=self.api_key, max_retries=self.max_retries)
        self._async_client = AsyncOpenAI(api_key=self.api_key, max_retries=self.max_retries)
        
        logger.info("openai_provider_initialized", model=self.default_model)
    
    async def complete(
        self,
        messages: list[OpenAIMessage],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> OpenAIResponse:
        """
        Generate a completion using OpenAI.
        
        Args:
            messages: Conversation messages
            model: Model to use
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            **kwargs: Additional API parameters
            
        Returns:
            OpenAI response
        """
        model = model or self.default_model
        
        try:
            response = await self._async_client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                **kwargs,
            )
            
            logger.info(
                "openai_completion_success",
                model=model,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
            )
            
            return OpenAIResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            )
            
        except openai.APIError as e:
            logger.error("openai_api_error", error=str(e), model=model)
            raise
    
    async def stream(
        self,
        messages: list[OpenAIMessage],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens from OpenAI.
        
        Args:
            messages: Conversation messages
            model: Model to use
            max_tokens: Maximum response tokens
            **kwargs: Additional parameters
            
        Yields:
            Text tokens as they arrive
        """
        model = model or self.default_model
        
        stream = await self._async_client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            stream=True,
            **kwargs,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def complete_sync(
        self,
        messages: list[OpenAIMessage],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> OpenAIResponse:
        """Synchronous completion."""
        model = model or self.default_model
        
        response = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        )
        
        return OpenAIResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
        )
    
    async def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small",
    ) -> list[float]:
        """
        Create text embedding.
        
        Args:
            text: Text to embed
            model: Embedding model
            
        Returns:
            Embedding vector
        """
        response = await self._async_client.embeddings.create(
            model=model,
            input=text,
        )
        return response.data[0].embedding


# Convenience function
async def gpt_complete(
    prompt: str,
    system: Optional[str] = None,
    model: str = "gpt-4o",
) -> str:
    """Quick completion with GPT."""
    provider = OpenAIProvider(default_model=model)
    
    messages = []
    if system:
        messages.append(OpenAIMessage(role="system", content=system))
    messages.append(OpenAIMessage(role="user", content=prompt))
    
    response = await provider.complete(messages=messages)
    return response.content


if __name__ == "__main__":
    import asyncio
    
    async def main():
        provider = OpenAIProvider()
        
        response = await provider.complete(
            messages=[
                OpenAIMessage(role="system", content="You are concise."),
                OpenAIMessage(role="user", content="Say 'OpenAI Provider Ready!'"),
            ],
        )
        
        print(f"Response: {response.content}")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
    
    asyncio.run(main())
```

### 3.4 Create Providers __init__.py

Create file `core/providers/__init__.py`:

```python
"""
Provider integrations for Unleash platform.
"""

from .anthropic_provider import AnthropicProvider, ClaudeMessage, ClaudeResponse, claude_complete
from .openai_provider import OpenAIProvider, OpenAIMessage, OpenAIResponse, gpt_complete

__all__ = [
    "AnthropicProvider",
    "ClaudeMessage",
    "ClaudeResponse",
    "claude_complete",
    "OpenAIProvider",
    "OpenAIMessage",
    "OpenAIResponse",
    "gpt_complete",
]
```

## Step 4: Update MCP Configuration

The `platform/.mcp.json` already exists. Update it with the Unleash server:

```bash
# Read current config
cat platform/.mcp.json

# Backup and update
cp platform/.mcp.json platform/.mcp.json.backup
```

Add the unleash server entry to `platform/.mcp.json`:

```json
{
  "$schema": "https://raw.githubusercontent.com/anthropics/claude-code/main/schemas/mcp-config.json",
  "mcpServers": {
    "unleash": {
      "command": "python",
      "args": ["-m", "core.mcp_server"],
      "cwd": "Z:/insider/AUTO CLAUDE/unleash",
      "description": "Unleash platform MCP server with LLM gateway integration"
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-filesystem", "Z:/insider/AUTO CLAUDE/unleash"],
      "description": "File operations for unleash platform"
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-memory"],
      "description": "Key-value memory store"
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-sequential-thinking"],
      "description": "Extended reasoning chains"
    },
    "letta": {
      "command": "curl",
      "args": ["-s", "http://localhost:8500/health"],
      "description": "Letta memory persistence server",
      "env": {
        "LETTA_URL": "http://localhost:8500"
      }
    },
    "firecrawl": {
      "command": "npx",
      "args": ["-y", "firecrawl-mcp"],
      "description": "AI-powered web scraping",
      "env": {
        "FIRECRAWL_API_KEY": "${FIRECRAWL_API_KEY}"
      }
    },
    "exa": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-exa"],
      "description": "AI-powered web search",
      "env": {
        "EXA_API_KEY": "${EXA_API_KEY}"
      }
    }
  },
  "globalShortcuts": {
    "platform-status": "python -c \"import asyncio; from core.mcp_server import get_server; s = get_server(); print('MCP Server Ready')\"",
    "llm-health": "python -c \"import asyncio; from core.llm_gateway import LLMGateway; asyncio.run(LLMGateway().health_check())\"",
    "validate": "python scripts/validate_environment.py"
  }
}
```

## Step 5: Create Validation Script

Create file `scripts/validate_phase2.py`:

```python
#!/usr/bin/env python3
"""
Phase 2 Validation Script
Validates Protocol Layer (Layer 0) implementation.
"""

import os
import sys
import asyncio
from typing import Tuple

from rich.console import Console
from rich.table import Table


def check_sdk_import(name: str, import_path: str) -> Tuple[bool, str]:
    """Check if SDK can be imported."""
    try:
        exec(f"import {import_path}")
        return True, "importable"
    except ImportError as e:
        return False, f"import error: {e}"


def check_gateway_import() -> Tuple[bool, str]:
    """Check LLM Gateway imports."""
    try:
        from core.llm_gateway import LLMGateway, Message, ModelConfig, Provider
        return True, "all imports OK"
    except ImportError as e:
        return False, f"import error: {e}"


def check_mcp_server_import() -> Tuple[bool, str]:
    """Check MCP Server imports."""
    try:
        from core.mcp_server import create_mcp_server, get_server
        return True, "all imports OK"
    except ImportError as e:
        return False, f"import error: {e}"


def check_providers_import() -> Tuple[bool, str]:
    """Check provider imports."""
    try:
        from core.providers import AnthropicProvider, OpenAIProvider
        return True, "all imports OK"
    except ImportError as e:
        return False, f"import error: {e}"


async def check_gateway_health() -> Tuple[bool, str]:
    """Check LLM Gateway health."""
    try:
        from core.llm_gateway import LLMGateway
        gateway = LLMGateway()
        health = await gateway.health_check()
        return health["status"] != "no_providers", f"status: {health['status']}"
    except Exception as e:
        return False, f"error: {e}"


def check_file_exists(path: str) -> Tuple[bool, str]:
    """Check if file exists."""
    if os.path.exists(path):
        return True, "exists"
    return False, "missing"


def main():
    """Run Phase 2 validation."""
    console = Console()
    console.print("\n[bold blue]Phase 2: Protocol Layer Validation[/bold blue]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    checks = [
        # SDK Imports
        ("SDK: mcp", lambda: check_sdk_import("mcp", "mcp")),
        ("SDK: anthropic", lambda: check_sdk_import("anthropic", "anthropic")),
        ("SDK: openai", lambda: check_sdk_import("openai", "openai")),
        ("SDK: litellm", lambda: check_sdk_import("litellm", "litellm")),
        ("SDK: fastmcp", lambda: check_sdk_import("fastmcp", "fastmcp")),
        
        # Core Files
        ("File: core/llm_gateway.py", lambda: check_file_exists("core/llm_gateway.py")),
        ("File: core/mcp_server.py", lambda: check_file_exists("core/mcp_server.py")),
        ("File: core/providers/__init__.py", lambda: check_file_exists("core/providers/__init__.py")),
        ("File: core/providers/anthropic_provider.py", lambda: check_file_exists("core/providers/anthropic_provider.py")),
        ("File: core/providers/openai_provider.py", lambda: check_file_exists("core/providers/openai_provider.py")),
        ("File: platform/.mcp.json", lambda: check_file_exists("platform/.mcp.json")),
        
        # Module Imports
        ("Import: LLM Gateway", check_gateway_import),
        ("Import: MCP Server", check_mcp_server_import),
        ("Import: Providers", check_providers_import),
    ]

    all_passed = True
    for name, check_fn in checks:
        passed, details = check_fn()
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(name, status, details)
        if not passed:
            all_passed = False

    console.print(table)
    console.print()

    # Async health check
    console.print("[bold]Running async health checks...[/bold]\n")
    
    async def run_async_checks():
        health_passed, health_details = await check_gateway_health()
        return health_passed, health_details

    health_passed, health_details = asyncio.run(run_async_checks())
    
    health_table = Table(show_header=True, header_style="bold magenta")
    health_table.add_column("Check", style="cyan")
    health_table.add_column("Status", style="green")
    health_table.add_column("Details")
    
    status = "[green]PASS[/green]" if health_passed else "[yellow]WARN[/yellow]"
    health_table.add_row("LLM Gateway Health", status, health_details)
    
    console.print(health_table)
    console.print()

    if all_passed:
        console.print("[bold green]Phase 2 Complete! Protocol Layer is operational.[/bold green]")
        return 0
    else:
        console.print("[bold red]Phase 2 has failures. Please fix before proceeding.[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

## Step 6: Run Validation

```bash
# Validate Phase 2 implementation
python scripts/validate_phase2.py
```

**Expected Output**:
```
Phase 2: Protocol Layer Validation

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Check                                   ┃ Status ┃ Details            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ SDK: mcp                                │ PASS   │ importable         │
│ SDK: anthropic                          │ PASS   │ importable         │
│ SDK: openai                             │ PASS   │ importable         │
│ SDK: litellm                            │ PASS   │ importable         │
│ SDK: fastmcp                            │ PASS   │ importable         │
│ File: core/llm_gateway.py               │ PASS   │ exists             │
│ File: core/mcp_server.py                │ PASS   │ exists             │
│ File: core/providers/__init__.py        │ PASS   │ exists             │
│ File: core/providers/anthropic_provider │ PASS   │ exists             │
│ File: core/providers/openai_provider    │ PASS   │ exists             │
│ File: platform/.mcp.json                │ PASS   │ exists             │
│ Import: LLM Gateway                     │ PASS   │ all imports OK     │
│ Import: MCP Server                      │ PASS   │ all imports OK     │
│ Import: Providers                       │ PASS   │ all imports OK     │
└─────────────────────────────────────────┴────────┴────────────────────┘

Running async health checks...

┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Check                ┃ Status ┃ Details            ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ LLM Gateway Health   │ PASS   │ status: healthy    │
└──────────────────────┴────────┴────────────────────┘

Phase 2 Complete! Protocol Layer is operational.
```

## Success Criteria

Phase 2 is complete when:

- [ ] All 5 Protocol SDKs installed and importable
  - [ ] mcp
  - [ ] anthropic
  - [ ] openai
  - [ ] litellm
  - [ ] fastmcp
- [ ] `core/llm_gateway.py` created with LiteLLM integration
- [ ] `core/mcp_server.py` created with FastMCP tools
- [ ] `core/providers/anthropic_provider.py` created
- [ ] `core/providers/openai_provider.py` created
- [ ] `platform/.mcp.json` updated with unleash server
- [ ] `scripts/validate_phase2.py` passes all checks
- [ ] LLM Gateway health check returns "healthy"

## Rollback Procedure

If something goes wrong:

```bash
# Remove Phase 2 files
rm -f core/llm_gateway.py
rm -f core/mcp_server.py
rm -rf core/providers/
rm -f scripts/validate_phase2.py

# Restore MCP config
mv platform/.mcp.json.backup platform/.mcp.json

# Uninstall Phase 2 dependencies (optional)
uv pip uninstall mcp anthropic openai litellm fastmcp

echo "Phase 2 rolled back"
```

## Files Created in Phase 2

| File | Purpose | Lines |
|------|---------|-------|
| `core/llm_gateway.py` | Unified LLM interface via LiteLLM | ~300 |
| `core/mcp_server.py` | MCP server with FastMCP tools | ~350 |
| `core/providers/__init__.py` | Provider exports | ~15 |
| `core/providers/anthropic_provider.py` | Direct Anthropic integration | ~180 |
| `core/providers/openai_provider.py` | Direct OpenAI integration | ~170 |
| `scripts/validate_phase2.py` | Phase 2 validation script | ~120 |

## Next Phase

Once validation passes, proceed to **Phase 3: Orchestration Layer Setup**:

```bash
# Install Phase 3 dependencies
uv pip install temporalio langgraph crewai pyautogen
```

---

*End of Phase 2 Executable Prompt*
```
