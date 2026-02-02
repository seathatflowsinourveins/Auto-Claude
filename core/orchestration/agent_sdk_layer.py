"""
Agent SDK Layer - Unified agent execution using Claude Agent SDK or Anthropic API.

Part of L1 Orchestration layer. Provides:
- create_agent(): Create an agent with specified configuration
- run_agent_loop(): Execute an agent loop with a prompt

Uses claude_agent_sdk if available, falls back to anthropic API with tool use.

NO STUBS - Real implementations only.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import structlog

# Configure logging
logger = structlog.get_logger(__name__)

# Try to import Claude Agent SDK (official)
CLAUDE_AGENT_SDK_AVAILABLE = False
try:
    from claude_agent_sdk import (
        query as claude_query,
        ClaudeAgentOptions,
        AssistantMessage,
        ToolUseBlock,
        TextBlock,
    )
    CLAUDE_AGENT_SDK_AVAILABLE = True
    logger.info("claude_agent_sdk_available", version="official")
except ImportError:
    logger.debug("claude_agent_sdk_not_available", fallback="anthropic_api")

# Always import Anthropic as fallback
ANTHROPIC_AVAILABLE = False
try:
    import anthropic
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
    logger.info("anthropic_sdk_available", version=anthropic.__version__)
except ImportError:
    logger.warning("anthropic_sdk_not_available")


@dataclass
class AgentConfig:
    """Configuration for creating an agent."""
    name: str
    model: str = "claude-sonnet-4-20250514"
    system_prompt: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    max_tokens: int = 4096
    temperature: float = 0.7
    permission_mode: str = "acceptEdits"
    cwd: Optional[str] = None


@dataclass
class AgentResult:
    """Result from agent execution."""
    output: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class Agent:
    """
    Unified agent wrapper supporting Claude Agent SDK or direct Anthropic API.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._client: Optional[AsyncAnthropic] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the agent's API client."""
        if self._initialized:
            return

        if ANTHROPIC_AVAILABLE:
            self._client = AsyncAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
            self._initialized = True
            logger.info("agent_initialized", name=self.config.name, model=self.config.model)
        else:
            raise RuntimeError("Neither claude_agent_sdk nor anthropic SDK available")

    async def run(self, prompt: str, max_turns: int = 10) -> AgentResult:
        """
        Run the agent with the given prompt.

        Args:
            prompt: The task/prompt for the agent
            max_turns: Maximum conversation turns (for tool use loops)

        Returns:
            AgentResult with output and metadata
        """
        await self.initialize()

        # Use Claude Agent SDK if available
        if CLAUDE_AGENT_SDK_AVAILABLE:
            return await self._run_with_claude_sdk(prompt, max_turns)

        # Fall back to direct Anthropic API
        return await self._run_with_anthropic(prompt, max_turns)

    async def _run_with_claude_sdk(self, prompt: str, max_turns: int) -> AgentResult:
        """Run using official Claude Agent SDK."""
        logger.info("running_with_claude_sdk", agent=self.config.name, prompt=prompt[:50])

        options = ClaudeAgentOptions(
            allowed_tools=self.config.tools or ["Read", "Write", "Bash"],
            permission_mode=self.config.permission_mode,
            cwd=self.config.cwd or os.getcwd(),
            model=self.config.model,
        )

        if self.config.system_prompt:
            options.system_prompt = self.config.system_prompt

        output_parts = []
        tool_calls = []
        messages = []

        try:
            async for message in claude_query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            output_parts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            tool_calls.append({
                                "name": block.name,
                                "input": block.input,
                            })
                    messages.append({"role": "assistant", "content": message.content})

            return AgentResult(
                output="\n".join(output_parts),
                tool_calls=tool_calls,
                messages=messages,
                metadata={"sdk": "claude_agent_sdk", "model": self.config.model},
                success=True,
            )

        except Exception as e:
            logger.error("claude_sdk_error", error=str(e))
            return AgentResult(
                output="",
                error=str(e),
                success=False,
                metadata={"sdk": "claude_agent_sdk", "error_type": type(e).__name__},
            )

    async def _run_with_anthropic(self, prompt: str, max_turns: int) -> AgentResult:
        """Run using direct Anthropic API with tool use loop."""
        logger.info("running_with_anthropic", agent=self.config.name, prompt=prompt[:50])

        messages = [{"role": "user", "content": prompt}]
        all_tool_calls = []
        output_parts = []

        # Build tools list (simplified - in production would have full tool definitions)
        tools = self._build_tools()

        for turn in range(max_turns):
            try:
                response = await self._client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    system=self.config.system_prompt or "You are a helpful AI assistant.",
                    messages=messages,
                    tools=tools if tools else None,
                )

                # Process response
                assistant_content = []
                has_tool_use = False

                for block in response.content:
                    if block.type == "text":
                        output_parts.append(block.text)
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        has_tool_use = True
                        tool_call = {
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                        all_tool_calls.append(tool_call)
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })

                messages.append({"role": "assistant", "content": assistant_content})

                # If no tool use, we're done
                if not has_tool_use or response.stop_reason == "end_turn":
                    break

                # Execute tools and add results
                tool_results = await self._execute_tools(all_tool_calls[-len([b for b in response.content if b.type == "tool_use"]):])
                messages.append({"role": "user", "content": tool_results})

            except Exception as e:
                logger.error("anthropic_api_error", error=str(e), turn=turn)
                return AgentResult(
                    output="\n".join(output_parts),
                    tool_calls=all_tool_calls,
                    messages=messages,
                    error=str(e),
                    success=False,
                    metadata={"sdk": "anthropic", "turns": turn + 1},
                )

        return AgentResult(
            output="\n".join(output_parts),
            tool_calls=all_tool_calls,
            messages=messages,
            metadata={"sdk": "anthropic", "model": self.config.model, "turns": len(messages) // 2},
            success=True,
        )

    def _build_tools(self) -> List[Dict[str, Any]]:
        """Build tool definitions for Anthropic API.
        
        Available tools:
        - Read: Read file contents
        - Write: Write file contents (full replacement)
        - Edit: Partial file editing with search/replace
        - Bash: Execute shell commands
        - Grep: Search for patterns in files
        - Glob: Find files by pattern
        - ListDir: List directory contents
        - WebFetch: Fetch and process web content
        - SaveMemory: Save observations to memory
        - SearchMemory: Search memory for context
        """
        tool_definitions = {
            "Read": {
                "name": "read_file",
                "description": "Read contents of a file. Returns file content as string.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute or relative file path to read"}
                    },
                    "required": ["path"]
                }
            },
            "Write": {
                "name": "write_file",
                "description": "Write contents to a file (creates or overwrites). Creates parent directories if needed.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to write"},
                        "content": {"type": "string", "description": "Content to write to file"}
                    },
                    "required": ["path", "content"]
                }
            },
            "Edit": {
                "name": "edit_file",
                "description": "Edit a file by replacing specific text. Use for partial modifications without rewriting entire file.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to edit"},
                        "old_string": {"type": "string", "description": "Exact text to find and replace"},
                        "new_string": {"type": "string", "description": "Text to replace with"},
                        "replace_all": {"type": "boolean", "description": "Replace all occurrences (default: false)", "default": False}
                    },
                    "required": ["path", "old_string", "new_string"]
                }
            },
            "Bash": {
                "name": "execute_bash",
                "description": "Execute a bash/shell command. Returns stdout, stderr, and return code.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds (default: 120)", "default": 120}
                    },
                    "required": ["command"]
                }
            },
            "Grep": {
                "name": "grep_files",
                "description": "Search for a pattern in files. Returns matching lines with file paths and line numbers.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern to search for"},
                        "path": {"type": "string", "description": "Directory or file to search in", "default": "."},
                        "file_pattern": {"type": "string", "description": "Glob pattern for files (e.g., '*.py')", "default": "*"},
                        "max_results": {"type": "integer", "description": "Maximum results to return", "default": 50}
                    },
                    "required": ["pattern"]
                }
            },
            "Glob": {
                "name": "glob_files",
                "description": "Find files matching a glob pattern. Returns list of matching file paths.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.py', 'src/**/*.ts')"},
                        "path": {"type": "string", "description": "Base directory to search from", "default": "."}
                    },
                    "required": ["pattern"]
                }
            },
            "ListDir": {
                "name": "list_directory",
                "description": "List contents of a directory. Returns files and subdirectories.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path to list", "default": "."},
                        "recursive": {"type": "boolean", "description": "List recursively", "default": False},
                        "max_depth": {"type": "integer", "description": "Max recursion depth", "default": 3}
                    },
                    "required": []
                }
            },
            "WebFetch": {
                "name": "web_fetch",
                "description": "Fetch content from a URL. Returns text content or error.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                        "extract_text": {"type": "boolean", "description": "Extract text from HTML", "default": True},
                        "max_length": {"type": "integer", "description": "Max content length", "default": 50000}
                    },
                    "required": ["url"]
                }
            },
            "SaveMemory": {
                "name": "save_memory",
                "description": "Save an observation or learning to persistent memory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to save"},
                        "category": {"type": "string", "description": "Category (decision, pattern, learning, info)", "default": "info"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for retrieval", "default": []}
                    },
                    "required": ["content"]
                }
            },
            "SearchMemory": {
                "name": "search_memory",
                "description": "Search memory for relevant context.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results", "default": 10},
                        "category": {"type": "string", "description": "Filter by category (optional)"}
                    },
                    "required": ["query"]
                }
            },
        }

        tools = []
        for tool_name in self.config.tools:
            if tool_name in tool_definitions:
                tools.append(tool_definitions[tool_name])
            else:
                logger.warning("unknown_tool_requested", tool=tool_name)

        return tools

    async def _execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls and return results.
        
        Supports: read_file, write_file, edit_file, execute_bash, grep_files,
                  glob_files, list_directory, web_fetch, save_memory, search_memory
        """
        import glob as glob_module
        import re
        from pathlib import Path
        
        results = []

        for call in tool_calls:
            tool_name = call.get("name", "")
            tool_input = call.get("input", {})
            tool_id = call.get("id", "")

            try:
                result: Dict[str, Any] = {}
                
                if tool_name == "read_file":
                    path = tool_input.get("path", "")
                    if os.path.exists(path):
                        with open(path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        result = {"content": content[:50000]}  # Limit content size
                    else:
                        result = {"error": f"File not found: {path}"}

                elif tool_name == "write_file":
                    path = tool_input.get("path", "")
                    content = tool_input.get("content", "")
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                    result = {"success": True, "path": path, "bytes_written": len(content)}

                elif tool_name == "edit_file":
                    path = tool_input.get("path", "")
                    old_string = tool_input.get("old_string", "")
                    new_string = tool_input.get("new_string", "")
                    replace_all = tool_input.get("replace_all", False)
                    
                    if not os.path.exists(path):
                        result = {"error": f"File not found: {path}"}
                    else:
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        if old_string not in content:
                            result = {"error": f"String not found in file: {old_string[:100]}..."}
                        else:
                            if replace_all:
                                new_content = content.replace(old_string, new_string)
                                count = content.count(old_string)
                            else:
                                new_content = content.replace(old_string, new_string, 1)
                                count = 1
                            
                            with open(path, "w", encoding="utf-8") as f:
                                f.write(new_content)
                            result = {"success": True, "path": path, "replacements": count}

                elif tool_name == "execute_bash":
                    command = tool_input.get("command", "")
                    timeout_secs = tool_input.get("timeout", 120)

                    # Use subprocess.run in thread executor for Python 3.14 compatibility
                    # asyncio.timeout() has stricter task requirements in 3.14
                    import subprocess
                    import concurrent.futures

                    def run_command():
                        try:
                            completed = subprocess.run(
                                command,
                                shell=True,
                                capture_output=True,
                                timeout=timeout_secs,
                                text=False,  # Get bytes for consistent handling
                            )
                            return {
                                "stdout": completed.stdout.decode(errors="replace")[:10000],
                                "stderr": completed.stderr.decode(errors="replace")[:2000],
                                "returncode": completed.returncode,
                            }
                        except subprocess.TimeoutExpired:
                            return {"error": f"Command timed out after {timeout_secs}s"}
                        except Exception as e:
                            return {"error": f"Command failed: {str(e)}"}

                    # Run in thread executor to avoid blocking event loop
                    loop = asyncio.get_running_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        result = await loop.run_in_executor(executor, run_command)

                elif tool_name == "grep_files":
                    pattern = tool_input.get("pattern", "")
                    search_path = tool_input.get("path", ".")
                    file_pattern = tool_input.get("file_pattern", "*")
                    max_results = tool_input.get("max_results", 50)
                    
                    matches = []
                    try:
                        regex = re.compile(pattern, re.IGNORECASE)
                        search_glob = os.path.join(search_path, "**", file_pattern)
                        
                        for filepath in glob_module.glob(search_glob, recursive=True):
                            if os.path.isfile(filepath) and len(matches) < max_results:
                                try:
                                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                                        for line_num, line in enumerate(f, 1):
                                            if regex.search(line):
                                                matches.append({
                                                    "file": filepath,
                                                    "line": line_num,
                                                    "content": line.strip()[:200]
                                                })
                                                if len(matches) >= max_results:
                                                    break
                                except (IOError, OSError):
                                    continue
                        
                        result = {"matches": matches, "total": len(matches)}
                    except re.error as e:
                        result = {"error": f"Invalid regex: {str(e)}"}

                elif tool_name == "glob_files":
                    pattern = tool_input.get("pattern", "")
                    base_path = tool_input.get("path", ".")
                    
                    full_pattern = os.path.join(base_path, pattern)
                    files = glob_module.glob(full_pattern, recursive=True)
                    files = [f for f in files if os.path.isfile(f)][:100]  # Limit results
                    result = {"files": files, "total": len(files)}

                elif tool_name == "list_directory":
                    path = tool_input.get("path", ".")
                    recursive = tool_input.get("recursive", False)
                    max_depth = tool_input.get("max_depth", 3)
                    
                    def list_dir(dir_path: str, depth: int = 0) -> Dict[str, Any]:
                        entries = {"dirs": [], "files": []}
                        try:
                            for entry in os.scandir(dir_path):
                                if entry.is_file():
                                    entries["files"].append(entry.name)
                                elif entry.is_dir() and not entry.name.startswith("."):
                                    if recursive and depth < max_depth:
                                        entries["dirs"].append({
                                            "name": entry.name,
                                            "contents": list_dir(entry.path, depth + 1)
                                        })
                                    else:
                                        entries["dirs"].append(entry.name)
                        except PermissionError:
                            pass
                        return entries
                    
                    result = list_dir(path)

                elif tool_name == "web_fetch":
                    url = tool_input.get("url", "")
                    extract_text = tool_input.get("extract_text", True)
                    max_length = tool_input.get("max_length", 50000)
                    
                    try:
                        import httpx
                        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                            response = await client.get(url)
                            content = response.text[:max_length]
                            
                            if extract_text and "text/html" in response.headers.get("content-type", ""):
                                # Basic HTML text extraction
                                import html
                                # Remove script and style tags
                                content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
                                content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)
                                # Remove HTML tags
                                content = re.sub(r"<[^>]+>", " ", content)
                                # Unescape HTML entities
                                content = html.unescape(content)
                                # Clean up whitespace
                                content = re.sub(r"\s+", " ", content).strip()
                            
                            result = {
                                "content": content[:max_length],
                                "status": response.status_code,
                                "url": str(response.url)
                            }
                    except ImportError:
                        result = {"error": "httpx not available - install with: pip install httpx"}
                    except Exception as e:
                        result = {"error": f"Fetch failed: {str(e)}"}

                elif tool_name == "save_memory":
                    content = tool_input.get("content", "")
                    category = tool_input.get("category", "info")
                    tags = tool_input.get("tags", [])
                    
                    # Save to a simple JSON-lines memory file
                    import json
                    from datetime import datetime
                    
                    memory_file = os.path.join(os.path.expanduser("~"), ".unleash_memory.jsonl")
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        "content": content,
                        "category": category,
                        "tags": tags
                    }
                    with open(memory_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry) + "\n")
                    
                    result = {"success": True, "memory_file": memory_file}

                elif tool_name == "search_memory":
                    query = tool_input.get("query", "").lower()
                    limit = tool_input.get("limit", 10)
                    category_filter = tool_input.get("category")
                    
                    import json
                    memory_file = os.path.join(os.path.expanduser("~"), ".unleash_memory.jsonl")
                    matches = []
                    
                    if os.path.exists(memory_file):
                        with open(memory_file, "r", encoding="utf-8") as f:
                            for line in f:
                                try:
                                    entry = json.loads(line)
                                    if category_filter and entry.get("category") != category_filter:
                                        continue
                                    if query in entry.get("content", "").lower():
                                        matches.append(entry)
                                        if len(matches) >= limit:
                                            break
                                except json.JSONDecodeError:
                                    continue
                    
                    result = {"matches": matches, "total": len(matches)}

                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": str(result),
                })

            except Exception as e:
                logger.error("tool_execution_error", tool=tool_name, error=str(e))
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": f"Error: {str(e)}",
                    "is_error": True,
                })

        return results


# ============================================================================
# Public API Functions (used by CLI)
# ============================================================================

async def create_agent(
    name: str,
    model: str = "claude-sonnet-4-20250514",
    tools: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    **kwargs,
) -> Agent:
    """
    Create an agent with the specified configuration.

    Args:
        name: Agent name/identifier
        model: Model to use (default: claude-sonnet-4)
        tools: List of tool names to enable
        system_prompt: Optional system prompt
        **kwargs: Additional configuration options

    Returns:
        Configured Agent instance
    """
    config = AgentConfig(
        name=name,
        model=model,
        tools=tools or ["Read", "Write", "Bash"],
        system_prompt=system_prompt,
        **{k: v for k, v in kwargs.items() if hasattr(AgentConfig, k)},
    )

    agent = Agent(config)
    await agent.initialize()

    logger.info("agent_created", name=name, model=model, tools=config.tools)
    return agent


async def run_agent_loop(
    agent: Agent,
    prompt: str,
    max_turns: int = 10,
) -> Dict[str, Any]:
    """
    Run an agent loop with the given prompt.

    Args:
        agent: The agent to run
        prompt: Task/prompt for the agent
        max_turns: Maximum conversation turns

    Returns:
        Dictionary with output, tool_calls, and metadata
    """
    logger.info("agent_loop_starting", agent=agent.config.name, prompt=prompt[:50])

    result = await agent.run(prompt, max_turns=max_turns)

    logger.info(
        "agent_loop_completed",
        agent=agent.config.name,
        success=result.success,
        tool_calls=len(result.tool_calls),
    )

    return {
        "output": result.output,
        "tool_calls": result.tool_calls,
        "messages": result.messages,
        "metadata": result.metadata,
        "success": result.success,
        "error": result.error,
    }


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "create_agent",
    "run_agent_loop",
    "CLAUDE_AGENT_SDK_AVAILABLE",
    "ANTHROPIC_AVAILABLE",
]


if __name__ == "__main__":
    # Test the module
    async def test():
        agent = await create_agent(
            name="test_agent",
            model="claude-sonnet-4-20250514",
            tools=["Read", "Bash"],
        )
        result = await run_agent_loop(
            agent=agent,
            prompt="List the files in the current directory",
            max_turns=3,
        )
        print(f"Result: {result}")

    asyncio.run(test())
