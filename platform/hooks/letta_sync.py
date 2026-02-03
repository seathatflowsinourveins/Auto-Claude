#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.26.0",
#     "structlog>=24.1.0",
# ]
# [tool.uv]
# exclude-newer = "2026-02-01"
# ///
"""
Letta Memory Sync Hook - V10 Optimized

⚠️  DEPRECATED: Use letta_sync_v2.py instead!
    This version uses raw httpx calls. V2 uses official letta-client SDK.
    Migration: Update settings.json to reference letta_sync_v2.py

Handles SessionStart and SessionEnd hooks to sync memory with Letta.
Falls back gracefully if Letta is not running.

Usage:
    python letta_sync.py start   # SessionStart hook
    python letta_sync.py end     # SessionEnd hook
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import structlog

# Configure structured logging
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


class LettaSync:
    """Handles synchronization between Claude Code sessions and Letta memory."""
    
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        timeout: float = 10.0
    ):
        self.base_url = base_url or os.environ.get("LETTA_URL", "http://localhost:8283")
        self.api_key = api_key or os.environ.get("LETTA_API_KEY")
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None
    
    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout
            )
        return self._client
    
    def is_letta_available(self) -> bool:
        """Check if Letta server is running."""
        try:
            response = self._get_client().get("/v1/health")
            return response.status_code == 200
        except Exception:
            return False
    
    def get_or_create_agent(self, project_name: str) -> Optional[str]:
        """Get or create a Letta agent for this project."""
        client = self._get_client()
        agent_name = f"claude-code-{project_name}"
        
        try:
            # List existing agents
            response = client.get("/v1/agents")
            if response.status_code == 200:
                agents = response.json()
                for agent in agents:
                    if agent.get("name") == agent_name:
                        return agent.get("id")
            
            # Create new agent
            agent_config = {
                "name": agent_name,
                "model": "anthropic/claude-sonnet-4-5-20250929",
                "embedding_model": "openai/text-embedding-3-small",
                "enable_sleeptime": True,
                "sleeptime_agent_frequency": 5,
                "memory_blocks": [
                    {"label": "human", "value": "", "limit": 3000},
                    {"label": "persona", "value": f"I am the memory system for the {project_name} project. I remember user preferences, successful patterns, and project conventions.", "limit": 2000},
                    {"label": "project-context", "value": "", "limit": 3000},
                    {"label": "learnings", "value": "", "limit": 4000}
                ]
            }
            
            response = client.post("/v1/agents", json=agent_config)
            if response.status_code in (200, 201):
                return response.json().get("id")
            
            logger.warning("agent_creation_failed", status=response.status_code)
            return None
            
        except Exception as e:
            logger.error("agent_error", error=str(e))
            return None
    
    def load_memory_to_file(self, agent_id: str, output_path: Path) -> bool:
        """Load memory blocks from Letta and write to CLAUDE.local.md."""
        client = self._get_client()
        
        try:
            response = client.get(f"/v1/agents/{agent_id}/memory/blocks")
            if response.status_code != 200:
                return False
            
            blocks = response.json()
            
            # Format as markdown
            lines = [
                "# Letta Memory (Auto-generated)",
                f"<!-- Agent ID: {agent_id} -->",
                f"<!-- Updated: {datetime.now(timezone.utc).isoformat()} -->",
                ""
            ]
            
            for block in blocks:
                label = block.get("label", "unknown")
                value = block.get("value", "")
                if value.strip():
                    lines.append(f"## {label.replace('-', ' ').title()}")
                    lines.append("")
                    lines.append(value)
                    lines.append("")
            
            # Write to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("\n".join(lines), encoding="utf-8")
            
            logger.info("memory_loaded", agent_id=agent_id, blocks=len(blocks))
            return True
            
        except Exception as e:
            logger.error("memory_load_error", error=str(e))
            return False
    
    def save_session_learnings(
        self,
        agent_id: str,
        transcript_path: Optional[Path],
        session_id: str
    ) -> bool:
        """Extract learnings from session and send to Letta."""
        client = self._get_client()
        
        try:
            # Read transcript if available
            summary = f"Session {session_id} completed."
            
            if transcript_path and transcript_path.exists():
                transcript = json.loads(transcript_path.read_text())
                
                # Extract key information
                tool_uses = []
                errors = []
                files_modified = []
                
                for entry in transcript:
                    if entry.get("type") == "tool_use":
                        tool_uses.append(entry.get("name", "unknown"))
                    if entry.get("type") == "error":
                        errors.append(entry.get("message", ""))
                    if entry.get("type") == "file_write":
                        files_modified.append(entry.get("path", ""))
                
                summary_parts = [f"Session {session_id} completed."]
                
                if tool_uses:
                    from collections import Counter
                    tool_counts = Counter(tool_uses)
                    top_tools = tool_counts.most_common(5)
                    summary_parts.append(f"Top tools: {', '.join(f'{t}({c})' for t, c in top_tools)}")
                
                if files_modified:
                    summary_parts.append(f"Files modified: {len(files_modified)}")
                
                if errors:
                    summary_parts.append(f"Errors encountered: {len(errors)}")
                
                summary = " ".join(summary_parts)
            
            # Send to Letta agent
            message = {
                "role": "user",
                "content": f"[SESSION_END] {summary}\n\nPlease update your memory with any important learnings from this session."
            }
            
            response = client.post(
                f"/v1/agents/{agent_id}/messages",
                json={"messages": [message]}
            )
            
            if response.status_code in (200, 201):
                logger.info("learnings_saved", agent_id=agent_id, summary=summary[:100])
                return True
            
            return False
            
        except Exception as e:
            logger.error("save_learnings_error", error=str(e))
            return False


def handle_session_start(hook_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle SessionStart hook."""
    session_id = hook_input.get("session_id", "unknown")
    cwd = hook_input.get("cwd", os.getcwd())
    project_name = Path(cwd).name
    
    sync = LettaSync()
    
    # Check if Letta is available
    if not sync.is_letta_available():
        logger.info("letta_unavailable", message="Continuing without Letta memory")
        return {
            "status": "ok",
            "letta": "unavailable",
            "message": "Letta not running, session will not persist memory"
        }
    
    # Get or create agent
    agent_id = sync.get_or_create_agent(project_name)
    if not agent_id:
        return {"status": "error", "message": "Failed to get/create Letta agent"}
    
    # Store agent ID for session end
    env_file = Path.home() / ".claude" / "v10" / ".session_env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(f"LETTA_AGENT_ID={agent_id}\nSESSION_ID={session_id}")
    
    # Load memory to CLAUDE.local.md
    local_md = Path(cwd) / "CLAUDE.local.md"
    sync.load_memory_to_file(agent_id, local_md)
    
    return {
        "status": "ok",
        "agent_id": agent_id,
        "project": project_name,
        "memory_loaded": local_md.exists()
    }


def handle_session_end(hook_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle SessionEnd hook."""
    session_id = hook_input.get("session_id", "unknown")
    transcript_path = hook_input.get("transcript_path")
    
    # Read stored agent ID
    env_file = Path.home() / ".claude" / "v10" / ".session_env"
    if not env_file.exists():
        return {"status": "ok", "message": "No session state found"}
    
    env_content = env_file.read_text()
    agent_id = None
    for line in env_content.split("\n"):
        if line.startswith("LETTA_AGENT_ID="):
            agent_id = line.split("=", 1)[1].strip()
            break
    
    if not agent_id:
        return {"status": "ok", "message": "No Letta agent ID found"}
    
    sync = LettaSync()
    
    if not sync.is_letta_available():
        return {"status": "ok", "message": "Letta unavailable, learnings not saved"}
    
    # Save session learnings
    transcript_file = Path(transcript_path) if transcript_path else None
    sync.save_session_learnings(agent_id, transcript_file, session_id)
    
    # Cleanup session env
    env_file.unlink(missing_ok=True)
    
    return {
        "status": "ok",
        "agent_id": agent_id,
        "session_id": session_id,
        "learnings_saved": True
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: letta_sync.py [start|end]", file=sys.stderr)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        hook_input = {}
    
    if command == "start":
        result = handle_session_start(hook_input)
    elif command == "end":
        result = handle_session_end(hook_input)
    else:
        result = {"status": "error", "message": f"Unknown command: {command}"}
    
    print(json.dumps(result))


if __name__ == "__main__":
    main()
