#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "letta-client>=1.7.0",
#     "httpx>=0.26.0",
#     "structlog>=24.1.0",
# ]
# [tool.uv]
# exclude-newer = "2026-02-01"
# ///
"""
Letta Memory Sync Hook V2.1 - V10.2 Enhanced

Enhanced implementation using official Letta Python SDK v1.7.1+.
Handles SessionStart and SessionEnd hooks for cross-session memory persistence.

Based on official documentation (2026-01-17):
- Letta SDK: https://docs.letta.com/api-reference
- Letta Sleeptime: https://docs.letta.com/guides/agents/architectures/sleeptime
- Claude Code Hooks: https://code.claude.com/docs/en/hooks

V2.1 Enhancements (from Letta SDK v1.7.1 research):
- Tool Rules: RequiresApprovalToolRule, MaxCountPerStepToolRule for safety controls
- MCP Server Management: Create/refresh MCP servers via Letta API
- Compaction Settings: Sliding window summarization for context management
- Multi-Agent Tools: Inter-agent communication when enabled
- Enhanced agent configuration with all supported parameters

Usage:
    python letta_sync_v2.py start   # SessionStart hook
    python letta_sync_v2.py end     # SessionEnd hook
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Try to use hook_utils if available
try:
    from hook_utils import (
        HookConfig, HookInput, HookResponse, SessionState,
        PermissionDecision, get_logger, log_event
    )
    HAS_HOOK_UTILS = True
except ImportError:
    HAS_HOOK_UTILS = False

import structlog

# Fallback logging if hook_utils not available
if not HAS_HOOK_UTILS:
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

logger = structlog.get_logger(__name__)


class LettaSyncV2:
    """
    Enhanced Letta memory synchronization using official SDK v1.7.1+.

    V2.1 Features (based on Letta SDK research):
    - Automatic agent creation per project with sleeptime enabled
    - Memory block management (human, persona, project-context, learnings)
    - Tool rules for safety controls (RequiresApprovalToolRule, MaxCountPerStepToolRule)
    - MCP server management via Letta API
    - Compaction settings for sliding window summarization
    - Multi-agent tools for inter-agent communication
    - Graceful fallback when Letta unavailable
    """

    # Tool rule types from Letta SDK v1.7.1
    TOOL_RULE_TYPES = [
        "ChildToolRule",        # Tool can only be called by specific parent tools
        "InitToolRule",         # Tool must be called at start of conversation
        "TerminalToolRule",     # Tool ends the conversation
        "ConditionalToolRule",  # Tool enabled based on conditions
        "ContinueToolRule",     # Tool keeps conversation going
        "RequiredBeforeExitToolRule",  # Must call before ending
        "MaxCountPerStepToolRule",     # Limit calls per step
        "ParentToolRule",       # Tool can spawn child tools
        "RequiresApprovalToolRule",    # Requires human approval
    ]

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        enable_tool_rules: bool = True,
        enable_mcp_management: bool = False,
    ):
        self.base_url = base_url or os.environ.get("LETTA_URL", "http://localhost:8283")
        self.api_key = api_key or os.environ.get("LETTA_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_tool_rules = enable_tool_rules
        self.enable_mcp_management = enable_mcp_management
        self._client = None
        self._use_sdk = False
        self._sdk_types = None

        # V12 OPTIMIZATION: Agent discovery caching (3-5x speedup)
        self._agent_cache: Dict[str, str] = {}  # name -> id mapping
        self._agent_cache_time: float = 0.0  # Last cache update timestamp
        self._agent_cache_ttl: float = 300.0  # 5 minute TTL

        # Try to import Letta SDK
        try:
            from letta_client import Letta
            self._letta_class = Letta
            self._use_sdk = True

            # Import SDK types for tool rules
            try:
                from letta_client import types as letta_types
                self._sdk_types = letta_types
            except ImportError:
                pass

            logger.info("letta_sdk_available", version="1.7.1+")
        except ImportError:
            logger.warning("letta_sdk_unavailable", fallback="httpx")
            import httpx
            self._httpx = httpx

    def _get_client(self):
        """Get or create Letta client."""
        if self._client is None:
            if self._use_sdk:
                # Use api_key for Cloud, token deprecated (verified 2026-01-30)
                self._client = self._letta_class(
                    base_url=self.base_url,
                    api_key=self.api_key
                )
            else:
                # Fallback to httpx
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                self._client = self._httpx.Client(
                    base_url=self.base_url,
                    headers=headers,
                    timeout=self.timeout
                )
        return self._client

    def is_available(self) -> bool:
        """Check if Letta server is running and accessible."""
        try:
            if self._use_sdk:
                client = self._get_client()
                # SDK health check
                client.health()
                return True
            else:
                client = self._get_client()
                response = client.get("/v1/health")
                return response.status_code == 200
        except Exception as e:
            logger.debug("letta_health_check_failed", error=str(e))
            return False

    def get_or_create_agent(self, project_name: str) -> Optional[str]:
        """
        Get or create a Letta agent for the project.

        Uses official SDK patterns for agent creation with:
        - Sleeptime enabled for background consolidation
        - Four memory blocks: human, persona, project-context, learnings
        - Claude Sonnet 4.5 as the model (or fallback)
        """
        agent_name = f"claude-code-{project_name}"

        try:
            if self._use_sdk:
                return self._get_or_create_agent_sdk(agent_name, project_name)
            else:
                return self._get_or_create_agent_httpx(agent_name, project_name)
        except Exception as e:
            logger.error("agent_operation_failed", error=str(e))
            return None

    def _get_or_create_agent_sdk(self, agent_name: str, project_name: str) -> Optional[str]:
        """Create/get agent using official SDK with V2.1 enhancements.

        V12 OPTIMIZATION: Uses cached agent discovery to avoid full list
        fetch on every operation. Expected: 3-5x speedup for agent lookups.
        """
        client = self._get_client()

        # V12: Check cache first (avoids full list fetch)
        current_time = time.time()
        cache_valid = (current_time - self._agent_cache_time) < self._agent_cache_ttl

        if cache_valid and agent_name in self._agent_cache:
            agent_id = self._agent_cache[agent_name]
            logger.info("agent_found_in_cache", agent_id=agent_id, name=agent_name)
            return agent_id

        # Cache miss or expired - refresh cache from API
        try:
            agents = client.agents.list()
            # Rebuild entire cache for efficiency
            self._agent_cache = {agent.name: agent.id for agent in agents}
            self._agent_cache_time = current_time

            if agent_name in self._agent_cache:
                agent_id = self._agent_cache[agent_name]
                logger.info("existing_agent_found", agent_id=agent_id, name=agent_name)
                return agent_id
        except Exception as e:
            logger.warning("agent_list_failed", error=str(e))

        # Build tool rules if enabled (from Letta SDK v1.7.1)
        tool_rules = None
        if self.enable_tool_rules and self._sdk_types:
            try:
                tool_rules = [
                    # Limit dangerous tool calls per step
                    self._sdk_types.MaxCountPerStepToolRule(
                        tool_name="send_message",
                        max_count_limit=5
                    ),
                    # Core memory tools require approval for safety
                    self._sdk_types.RequiresApprovalToolRule(
                        tool_name="core_memory_replace"
                    ),
                ]
                logger.info("tool_rules_configured", count=len(tool_rules))
            except Exception as e:
                logger.warning("tool_rules_failed", error=str(e))
                tool_rules = None

        # Build compaction settings (from Letta SDK v1.7.1)
        compaction_settings = None
        if self._sdk_types:
            try:
                compaction_settings = self._sdk_types.CompactionSettings(
                    max_context_window_size=16000,  # Sliding window size
                    min_compaction_threshold=8000,   # Trigger compaction threshold
                    eviction_policy="lru",           # Least recently used
                    summarization_enabled=True,      # Enable summarization
                )
                logger.info("compaction_configured")
            except Exception as e:
                logger.debug("compaction_settings_not_available", error=str(e))

        # Create new agent with V2.1 enhanced configuration
        try:
            create_params = {
                "name": agent_name,
                "memory_blocks": [
                    {"label": "human", "value": "", "limit": 3000},
                    {
                        "label": "persona",
                        "value": f"I am the memory system for the '{project_name}' project. "
                                 "I remember user preferences, successful coding patterns, "
                                 "project conventions, and learnings from past sessions.",
                        "limit": 2000
                    },
                    {"label": "project-context", "value": "", "limit": 3000},
                    {"label": "learnings", "value": "", "limit": 4000}
                ],
                "model": "anthropic/claude-sonnet-4-5-20250929",
                "embedding": "openai/text-embedding-3-small",
                "enable_sleeptime": True,
                "include_multi_agent_tools": True,  # V2.1: Enable inter-agent communication
            }

            # Add optional V2.1 parameters if available
            if tool_rules:
                create_params["tool_rules"] = tool_rules
            if compaction_settings:
                create_params["compaction_settings"] = compaction_settings

            agent = client.agents.create(**create_params)

            # Update sleeptime frequency to 5 turns (V116: fixed to use groups.modify)
            if hasattr(agent, 'multi_agent_group') and agent.multi_agent_group:
                try:
                    from letta_client.types import SleeptimeManagerUpdate
                    client.groups.modify(
                        group_id=agent.multi_agent_group.id,
                        manager_config=SleeptimeManagerUpdate(
                            sleeptime_agent_frequency=5
                        )
                    )
                except Exception as e:
                    logger.warning("sleeptime_config_failed", error=str(e))

            logger.info("agent_created", agent_id=agent.id, name=agent_name,
                       tool_rules=bool(tool_rules), compaction=bool(compaction_settings))
            return agent.id

        except Exception as e:
            logger.error("agent_creation_failed", error=str(e))
            return None

    def create_mcp_server(
        self,
        server_name: str,
        server_type: str = "stdio",
        command: Optional[str] = None,
        url: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create an MCP server in Letta (V2.1 feature).

        Letta can manage MCP servers via API:
        - Stdio: Local command-based servers
        - SSE: Server-Sent Events servers
        - StreamableHTTP: HTTP streaming servers

        Returns the MCP server ID if successful.
        """
        if not self._use_sdk or not self.enable_mcp_management:
            logger.debug("mcp_management_disabled")
            return None

        try:
            client = self._get_client()

            if server_type == "stdio" and command:
                config = {"type": "stdio", "command": command}
            elif server_type == "sse" and url:
                config = {"type": "sse", "url": url}
            elif server_type == "streamable_http" and url:
                config = {"type": "streamable_http", "url": url}
            else:
                logger.error("invalid_mcp_config", server_type=server_type)
                return None

            result = client.mcp_servers.create(
                server_name=server_name,
                config=config
            )

            logger.info("mcp_server_created", server_name=server_name, server_id=result.id)
            return result.id

        except Exception as e:
            logger.error("mcp_server_creation_failed", error=str(e))
            return None

    def refresh_mcp_tools(self, mcp_server_id: str, agent_id: Optional[str] = None) -> bool:
        """
        Refresh tools from an MCP server (V2.1 feature).

        Updates the available tools from the MCP server.
        Optionally attach to a specific agent.
        """
        if not self._use_sdk or not self.enable_mcp_management:
            return False

        try:
            client = self._get_client()
            client.mcp_servers.refresh(mcp_server_id, agent_id=agent_id)
            logger.info("mcp_tools_refreshed", mcp_server_id=mcp_server_id)
            return True
        except Exception as e:
            logger.error("mcp_refresh_failed", error=str(e))
            return False

    def _get_or_create_agent_httpx(self, agent_name: str, project_name: str) -> Optional[str]:
        """Fallback: Create/get agent using httpx."""
        client = self._get_client()

        # List existing agents
        try:
            response = client.get("/v1/agents")
            if response.status_code == 200:
                agents = response.json()
                for agent in agents:
                    if agent.get("name") == agent_name:
                        return agent.get("id")
        except Exception:
            pass

        # Create new agent
        agent_config = {
            "name": agent_name,
            "model": "anthropic/claude-sonnet-4-5-20250929",
            "embedding_model": "openai/text-embedding-3-small",
            "enable_sleeptime": True,
            "sleeptime_agent_frequency": 5,
            "memory_blocks": [
                {"label": "human", "value": "", "limit": 3000},
                {
                    "label": "persona",
                    "value": f"I am the memory system for '{project_name}'.",
                    "limit": 2000
                },
                {"label": "project-context", "value": "", "limit": 3000},
                {"label": "learnings", "value": "", "limit": 4000}
            ]
        }

        try:
            response = client.post("/v1/agents", json=agent_config)
            if response.status_code in (200, 201):
                return response.json().get("id")
        except Exception as e:
            logger.error("agent_creation_failed", error=str(e))

        return None

    def load_memory_blocks(self, agent_id: str) -> Dict[str, str]:
        """Load memory blocks from Letta agent."""
        try:
            if self._use_sdk:
                client = self._get_client()
                blocks = client.agents.blocks.list(agent_id=agent_id)
                return {b.label: b.value for b in blocks if b.value}
            else:
                client = self._get_client()
                response = client.get(f"/v1/agents/{agent_id}/memory/blocks")
                if response.status_code == 200:
                    blocks = response.json()
                    return {b["label"]: b["value"] for b in blocks if b.get("value")}
        except Exception as e:
            logger.error("load_memory_failed", error=str(e))
        return {}

    def format_memory_as_markdown(self, blocks: Dict[str, str], agent_id: str) -> str:
        """Format memory blocks as markdown for CLAUDE.local.md."""
        lines = [
            "# Letta Memory (Auto-synced)",
            f"<!-- Agent: {agent_id} -->",
            f"<!-- Updated: {datetime.now(timezone.utc).isoformat()} -->",
            "",
        ]

        for label, value in blocks.items():
            if value.strip():
                title = label.replace("-", " ").replace("_", " ").title()
                lines.extend([f"## {title}", "", value.strip(), ""])

        return "\n".join(lines)

    def write_memory_to_file(self, agent_id: str, output_path: Path) -> bool:
        """Load memory and write to CLAUDE.local.md."""
        blocks = self.load_memory_blocks(agent_id)

        if not blocks:
            logger.info("no_memory_to_load", agent_id=agent_id)
            return False

        markdown = self.format_memory_as_markdown(blocks, agent_id)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")

        logger.info("memory_written", agent_id=agent_id, path=str(output_path), blocks=len(blocks))
        return True

    def save_session_summary(
        self,
        agent_id: str,
        session_id: str,
        summary: str
    ) -> bool:
        """Send session summary to Letta for memory consolidation."""
        try:
            message_content = (
                f"[SESSION_END] Session {session_id} completed.\n\n"
                f"{summary}\n\n"
                "Please update your memory blocks with any important learnings, "
                "patterns, or preferences observed during this session."
            )

            if self._use_sdk:
                client = self._get_client()
                # V2.1: Use correct API - messages.create() not send() (verified 2026-01-30)
                client.agents.messages.create(
                    agent_id=agent_id,
                    messages=[{"role": "user", "content": message_content}]
                )
            else:
                client = self._get_client()
                client.post(
                    f"/v1/agents/{agent_id}/messages",
                    json={"messages": [{"role": "user", "content": message_content}]}
                )

            logger.info("session_summary_saved", agent_id=agent_id, session_id=session_id)
            return True

        except Exception as e:
            logger.error("save_summary_failed", error=str(e))
            return False


def extract_session_summary(transcript_path: Optional[Path]) -> str:
    """Extract summary from session transcript."""
    if not transcript_path or not transcript_path.exists():
        return "No transcript available."

    try:
        transcript = json.loads(transcript_path.read_text())

        # Count metrics
        tool_uses: Dict[str, int] = {}
        files_modified: list = []
        errors: list = []

        for entry in transcript:
            entry_type = entry.get("type", "")
            if entry_type == "tool_use":
                tool_name = entry.get("name", "unknown")
                tool_uses[tool_name] = tool_uses.get(tool_name, 0) + 1
            elif entry_type in ("file_write", "file_edit"):
                files_modified.append(entry.get("path", ""))
            elif entry_type == "error":
                errors.append(entry.get("message", "")[:100])

        parts = []

        if tool_uses:
            top_tools = sorted(tool_uses.items(), key=lambda x: x[1], reverse=True)[:5]
            parts.append(f"Top tools: {', '.join(f'{t}({c})' for t, c in top_tools)}")

        if files_modified:
            unique_files = list(set(files_modified))[:10]
            parts.append(f"Files modified ({len(files_modified)}): {', '.join(unique_files)}")

        if errors:
            parts.append(f"Errors encountered: {len(errors)}")

        return " | ".join(parts) if parts else "Session completed normally."

    except Exception as e:
        return f"Could not parse transcript: {e}"


def handle_session_start() -> Dict[str, Any]:
    """
    Handle SessionStart hook with V2.1 enhanced output format.

    1. Get/create Letta agent for project (with tool rules & compaction)
    2. Load memory blocks
    3. Write to CLAUDE.local.md
    4. Return context for injection in official hookSpecificOutput format
    """
    # Get project info from environment
    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd()))
    project_name = project_dir.name
    session_id = os.environ.get("CLAUDE_SESSION_ID", "unknown")

    logger.info("session_start", project=project_name, session_id=session_id)

    # V2.1: Enable tool rules and MCP management based on environment
    enable_tool_rules = os.environ.get("LETTA_TOOL_RULES", "true").lower() == "true"
    enable_mcp = os.environ.get("LETTA_MCP_MANAGEMENT", "false").lower() == "true"

    sync = LettaSyncV2(
        enable_tool_rules=enable_tool_rules,
        enable_mcp_management=enable_mcp
    )

    # Check Letta availability
    if not sync.is_available():
        logger.info("letta_unavailable", message="Session will not persist memory")
        # V2.1: Return proper hookSpecificOutput format for SessionStart
        return {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": "Note: Cross-session memory is unavailable. Letta server not running."
            }
        }

    # Get or create agent with V2.1 features
    agent_id = sync.get_or_create_agent(project_name)
    if not agent_id:
        return {
            "status": "error",
            "message": "Failed to create Letta agent"
        }

    # Store session state
    state_file = Path.home() / ".claude" / "v10" / ".session_env"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        f"LETTA_AGENT_ID={agent_id}\n"
        f"SESSION_ID={session_id}\n"
        f"PROJECT={project_name}\n"
        f"TOOL_RULES={enable_tool_rules}\n"
        f"MCP_MANAGEMENT={enable_mcp}"
    )

    # Load and write memory
    local_md = project_dir / "CLAUDE.local.md"
    memory_loaded = sync.write_memory_to_file(agent_id, local_md)

    # Build context injection
    context_parts = []
    if memory_loaded:
        blocks = sync.load_memory_blocks(agent_id)
        if blocks.get("learnings"):
            context_parts.append(f"Previous learnings: {blocks['learnings'][:500]}...")
        if blocks.get("human"):
            context_parts.append(f"User preferences: {blocks['human'][:300]}...")

    additional_context = "\n".join(context_parts) if context_parts else None

    # V2.1: Return proper hookSpecificOutput format for SessionStart
    result: Dict[str, Any] = {
        "status": "ok",
        "agent_id": agent_id,
        "project": project_name,
        "memory_loaded": memory_loaded,
        "tool_rules_enabled": enable_tool_rules,
        "mcp_management_enabled": enable_mcp,
    }

    if additional_context:
        result["hookSpecificOutput"] = {
            "hookEventName": "SessionStart",
            "additionalContext": additional_context
        }

    return result


def handle_session_end() -> Dict[str, Any]:
    """
    Handle SessionEnd hook.

    1. Load session state
    2. Extract transcript summary
    3. Send to Letta for consolidation
    """
    # Read session state
    state_file = Path.home() / ".claude" / "v10" / ".session_env"
    if not state_file.exists():
        return {"status": "ok", "message": "No session state found"}

    state: Dict[str, str] = {}
    for line in state_file.read_text().split("\n"):
        if "=" in line:
            key, value = line.split("=", 1)
            state[key.strip()] = value.strip()

    agent_id = state.get("LETTA_AGENT_ID")
    session_id = state.get("SESSION_ID", "unknown")

    if not agent_id:
        return {"status": "ok", "message": "No agent ID in session state"}

    sync = LettaSyncV2()

    if not sync.is_available():
        return {"status": "ok", "message": "Letta unavailable, learnings not saved"}

    # Get transcript path from environment or hook input
    transcript_path = os.environ.get("CLAUDE_TRANSCRIPT_PATH")
    transcript = Path(transcript_path) if transcript_path else None

    # Extract and save summary
    summary = extract_session_summary(transcript)
    sync.save_session_summary(agent_id, session_id, summary)

    # Clean up session state
    state_file.unlink(missing_ok=True)

    return {
        "status": "ok",
        "agent_id": agent_id,
        "session_id": session_id,
        "learnings_saved": True
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: letta_sync_v2.py [start|end]", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "start":
        result = handle_session_start()
    elif command == "end":
        result = handle_session_end()
    else:
        result = {"status": "error", "message": f"Unknown command: {command}"}

    # Output result
    print(json.dumps(result))

    # Return additional context if present (for Claude Code injection)
    if result.get("additionalContext"):
        # This goes to a special output for context injection
        pass


if __name__ == "__main__":
    main()
