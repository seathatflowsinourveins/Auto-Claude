# Letta + Claude Code: Complete Deep-Dive Guide

> **Comprehensive reference covering Python hook scripts, advanced memory patterns, and production Kubernetes deployment with official documentation references.**

---

## Table of Contents

1. [Python Hook Scripts - Complete Reference](#part-1-python-hook-scripts---complete-reference)
2. [Advanced Memory Patterns](#part-2-advanced-memory-patterns)
3. [Production Kubernetes Deployment](#part-3-production-kubernetes-deployment)
4. [Official Documentation References](#part-4-official-documentation-references)

---

# Part 1: Python Hook Scripts - Complete Reference

## 1.1 Hook System Architecture

Claude Code's hooks system allows deterministic automation at specific lifecycle points. Hooks are configured in `.claude/settings.json` (project-level) or `~/.claude/settings.json` (user-level).

### Official Documentation Reference
- **Claude Code Hooks**: https://docs.anthropic.com/en/docs/claude-code/hooks
- **Claude Code Configuration**: https://docs.anthropic.com/en/docs/claude-code/settings

### Hook Event Types

| Event | Trigger Point | Input Data | Use Case |
|-------|--------------|------------|----------|
| `PreToolUse` | Before any tool executes | Tool name, arguments | Validation, logging |
| `PostToolUse` | After tool completes | Tool name, result, exit code | Formatting, tracking |
| `UserPromptSubmit` | When user submits prompt | Prompt text, session info | Context injection |
| `SessionStart` | Claude Code session begins | Session metadata | Load context |
| `SessionEnd` | Session terminates | Session summary, stats | Save learnings |
| `Stop` | Conversation stops | Conversation data | Cleanup, persistence |
| `SubagentStop` | Sub-agent completes | Sub-agent output | Log sub-agent activity |
| `Notification` | System notification | Notification data | Alert handling |

### Hook Configuration Schema

```json
{
  "hooks": {
    "EventName": [
      {
        "matcher": "RegexPattern",
        "hooks": [
          {
            "type": "command",
            "command": "script_or_command",
            "timeout": 30,
            "env": {
              "CUSTOM_VAR": "value"
            }
          }
        ]
      }
    ]
  }
}
```

### Exit Code Behavior

| Exit Code | Behavior |
|-----------|----------|
| `0` | Success - continue normally |
| `1` | Warning - continue but log warning |
| `2` | Block - prevent the action (PreToolUse only) |
| Non-zero | Error - show stderr to user |

---

## 1.2 Complete Hook Scripts Collection

### Session Start Hook - Full Implementation

**File: `~/.claude/hooks/letta-session-start.py`**

```python
#!/usr/bin/env python3
"""
Letta Session Start Hook
========================
Loads relevant context from Letta memory at Claude Code session start.

Features:
- Loads all core memory blocks
- Retrieves recent archival passages
- Searches for project-relevant context
- Handles connection failures gracefully

Environment Variables Required:
- LETTA_BASE_URL: Letta server URL (default: http://localhost:8283)
- LETTA_PASSWORD: Server authentication password
- LETTA_AGENT_ID: Agent identifier for memory operations

Official Letta Docs: https://docs.letta.com/api-reference/agents
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from letta_client import Letta
    from letta_client.types import Block, Passage
    LETTA_AVAILABLE = True
except ImportError:
    LETTA_AVAILABLE = False
    logger.warning("letta-client not installed. Run: pip install letta-client")


class LettaSessionLoader:
    """Manages loading Letta context at session start."""
    
    def __init__(self):
        self.base_url = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
        self.password = os.environ.get("LETTA_PASSWORD", "")
        self.agent_id = os.environ.get("LETTA_AGENT_ID", "")
        self.project_path = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
        
        if not LETTA_AVAILABLE:
            raise ImportError("letta-client package required")
        
        self.client = Letta(
            base_url=self.base_url,
            token=self.password
        )
    
    def load_core_memory_blocks(self) -> List[Dict[str, Any]]:
        """
        Load all core memory blocks for the agent.
        
        Core memory blocks contain always-present context like:
        - human: User profile and preferences
        - persona: Agent behavior configuration
        - project_context: Current project details
        - learnings: Accumulated insights
        
        Letta API Reference: https://docs.letta.com/api-reference/agents/list-agent-memory-blocks
        """
        try:
            blocks = self.client.agents.blocks.list(agent_id=self.agent_id)
            return [
                {
                    "id": block.id,
                    "label": block.label,
                    "value": block.value,
                    "limit": block.limit,
                    "updated_at": str(block.updated_at) if hasattr(block, 'updated_at') else None
                }
                for block in blocks
            ]
        except Exception as e:
            logger.error(f"Failed to load memory blocks: {e}")
            return []
    
    def load_recent_passages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Load recent archival memory passages.
        
        Archival memory stores long-term context that can be semantically searched.
        Recent passages often contain valuable session learnings.
        
        Letta API Reference: https://docs.letta.com/api-reference/agents/list-agent-archival-memory
        """
        try:
            passages = self.client.agents.passages.list(
                agent_id=self.agent_id,
                limit=limit
            )
            return [
                {
                    "id": p.id,
                    "text": p.text,
                    "metadata": p.metadata if hasattr(p, 'metadata') else {},
                    "created_at": str(p.created_at) if hasattr(p, 'created_at') else None
                }
                for p in passages
            ]
        except Exception as e:
            logger.error(f"Failed to load recent passages: {e}")
            return []
    
    def search_project_context(self, project_name: str = None) -> List[Dict[str, Any]]:
        """
        Search archival memory for project-specific context.
        
        Uses semantic search to find relevant past learnings for the current project.
        
        Letta API Reference: https://docs.letta.com/api-reference/agents/search-agent-archival-memory
        """
        if not project_name:
            project_name = Path(self.project_path).name
        
        try:
            passages = self.client.agents.passages.list(
                agent_id=self.agent_id,
                query=f"project {project_name}",
                limit=5
            )
            return [
                {
                    "text": p.text,
                    "relevance": "semantic match",
                    "metadata": p.metadata if hasattr(p, 'metadata') else {}
                }
                for p in passages
            ]
        except Exception as e:
            logger.error(f"Failed to search project context: {e}")
            return []
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """
        Get agent summary including message count and memory stats.
        
        Letta API Reference: https://docs.letta.com/api-reference/agents/get-agent
        """
        try:
            agent = self.client.agents.retrieve(agent_id=self.agent_id)
            return {
                "name": agent.name,
                "description": agent.description,
                "model": agent.model,
                "created_at": str(agent.created_at) if hasattr(agent, 'created_at') else None,
                "message_count": getattr(agent, 'message_count', 'N/A')
            }
        except Exception as e:
            logger.error(f"Failed to get agent summary: {e}")
            return {}
    
    def format_context_output(
        self,
        blocks: List[Dict],
        passages: List[Dict],
        project_context: List[Dict],
        agent_info: Dict
    ) -> str:
        """Format all loaded context for output to Claude Code."""
        
        output_parts = []
        
        # Header
        output_parts.append("# 📚 Letta Memory Loaded\n")
        output_parts.append(f"*Agent: {agent_info.get('name', 'Unknown')} | "
                          f"Model: {agent_info.get('model', 'Unknown')}*\n")
        
        # Core Memory Blocks
        output_parts.append("\n## Core Memory Blocks\n")
        for block in blocks:
            output_parts.append(f"### {block['label'].title()}\n")
            # Truncate very long blocks
            value = block['value']
            if len(value) > 500:
                value = value[:500] + "\n...[truncated]..."
            output_parts.append(f"{value}\n")
        
        # Recent Learnings
        if passages:
            output_parts.append("\n## Recent Session Learnings\n")
            for p in passages[:5]:
                text = p['text'][:200] + "..." if len(p['text']) > 200 else p['text']
                output_parts.append(f"- {text}\n")
        
        # Project-specific Context
        if project_context:
            output_parts.append("\n## Project-Relevant Context\n")
            for ctx in project_context:
                text = ctx['text'][:200] + "..." if len(ctx['text']) > 200 else ctx['text']
                output_parts.append(f"- {text}\n")
        
        return "\n".join(output_parts)


def main():
    """Main entry point for session start hook."""
    
    # Check for required environment variables
    agent_id = os.environ.get("LETTA_AGENT_ID")
    if not agent_id:
        logger.info("LETTA_AGENT_ID not set - skipping Letta memory load")
        return
    
    if not LETTA_AVAILABLE:
        print("⚠️ Letta client not installed. Memory features disabled.", file=sys.stderr)
        return
    
    try:
        loader = LettaSessionLoader()
        
        # Load all context
        agent_info = loader.get_agent_summary()
        blocks = loader.load_core_memory_blocks()
        passages = loader.load_recent_passages(limit=10)
        project_context = loader.search_project_context()
        
        # Format and output
        context = loader.format_context_output(
            blocks=blocks,
            passages=passages,
            project_context=project_context,
            agent_info=agent_info
        )
        
        print(context)
        logger.info(f"Loaded {len(blocks)} blocks, {len(passages)} passages")
        
    except Exception as e:
        logger.error(f"Session start hook failed: {e}")
        print(f"⚠️ Letta sync skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

---

### Session End Hook - Full Implementation

**File: `~/.claude/hooks/letta-session-end.py`**

```python
#!/usr/bin/env python3
"""
Letta Session End Hook
======================
Saves session learnings and context to Letta memory when Claude Code session ends.

Features:
- Extracts key learnings from session
- Saves to archival memory with metadata
- Updates relevant core memory blocks
- Handles conversation summarization

Environment Variables Required:
- LETTA_BASE_URL: Letta server URL
- LETTA_PASSWORD: Server authentication password
- LETTA_AGENT_ID: Agent identifier

Input (stdin): JSON with session data
- conversation_summary: AI-generated summary
- session_id: Unique session identifier
- tool_usage: List of tools used
- files_modified: List of files changed
- duration_seconds: Session length

Official Letta Docs: https://docs.letta.com/api-reference/agents/archival-memory
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from letta_client import Letta
    LETTA_AVAILABLE = True
except ImportError:
    LETTA_AVAILABLE = False


class LettaSessionSaver:
    """Manages saving session data to Letta memory."""
    
    def __init__(self):
        self.base_url = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
        self.password = os.environ.get("LETTA_PASSWORD", "")
        self.agent_id = os.environ.get("LETTA_AGENT_ID", "")
        self.project_path = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
        
        if LETTA_AVAILABLE:
            self.client = Letta(
                base_url=self.base_url,
                token=self.password
            )
        else:
            self.client = None
    
    def save_session_summary(self, session_data: Dict[str, Any]) -> Optional[str]:
        """
        Save session summary to archival memory.
        
        Creates a searchable passage with:
        - Session summary
        - Timestamp
        - Project context
        - Tools used
        - Files modified
        
        Letta API Reference: https://docs.letta.com/api-reference/agents/create-archival-memory
        """
        if not self.client:
            return None
        
        session_id = session_data.get("session_id", "unknown")
        summary = session_data.get("conversation_summary", "")
        tools_used = session_data.get("tool_usage", [])
        files_modified = session_data.get("files_modified", [])
        duration = session_data.get("duration_seconds", 0)
        project_name = Path(self.project_path).name
        
        # Format the passage text
        passage_text = f"""## Session Summary [{datetime.now().isoformat()}]
**Session ID**: {session_id}
**Project**: {project_name}
**Duration**: {duration // 60} minutes

### Summary
{summary if summary else "No summary available"}

### Tools Used
{', '.join(tools_used) if tools_used else 'None recorded'}

### Files Modified
{chr(10).join(f'- {f}' for f in files_modified) if files_modified else 'None recorded'}
"""
        
        # Create metadata for better searching
        metadata = {
            "type": "session_summary",
            "session_id": session_id,
            "project": project_name,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "tools_count": len(tools_used),
            "files_count": len(files_modified)
        }
        
        try:
            passage = self.client.agents.passages.create(
                agent_id=self.agent_id,
                text=passage_text,
                metadata=metadata
            )
            logger.info(f"Created passage: {passage.id}")
            return passage.id
        except Exception as e:
            logger.error(f"Failed to save session summary: {e}")
            return None
    
    def extract_learnings(self, session_data: Dict[str, Any]) -> List[str]:
        """
        Extract discrete learnings from session data.
        
        Looks for:
        - Bug fixes (what was wrong, how it was fixed)
        - New patterns discovered
        - Performance improvements
        - User preference discoveries
        """
        learnings = []
        
        # Extract from tools used
        tools = session_data.get("tool_usage", [])
        if "Edit" in tools or "Write" in tools:
            files = session_data.get("files_modified", [])
            if files:
                learnings.append(f"Modified files: {', '.join(files[:5])}")
        
        # Extract from summary using simple heuristics
        summary = session_data.get("conversation_summary", "")
        
        # Look for bug-related content
        if any(word in summary.lower() for word in ["fixed", "bug", "error", "issue"]):
            learnings.append(f"Bug fix identified in session")
        
        # Look for new implementations
        if any(word in summary.lower() for word in ["implemented", "created", "added"]):
            learnings.append(f"New implementation completed")
        
        return learnings
    
    def update_learnings_block(self, learnings: List[str]) -> bool:
        """
        Append new learnings to the 'learnings' memory block.
        
        Maintains a running log of session-by-session discoveries.
        
        Letta API Reference: https://docs.letta.com/api-reference/agents/update-memory-block
        """
        if not self.client or not learnings:
            return False
        
        try:
            # First, get the current learnings block
            blocks = self.client.agents.blocks.list(agent_id=self.agent_id)
            learnings_block = None
            
            for block in blocks:
                if block.label == "learnings":
                    learnings_block = block
                    break
            
            if not learnings_block:
                logger.warning("No 'learnings' block found")
                return False
            
            # Append new learnings
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            new_content = f"\n[{timestamp}]\n" + "\n".join(f"- {l}" for l in learnings)
            
            # Keep block under limit (default 5000 chars)
            current_value = learnings_block.value
            max_length = learnings_block.limit or 5000
            
            updated_value = current_value + new_content
            
            # Trim from beginning if too long
            if len(updated_value) > max_length:
                # Keep header and most recent content
                header_end = updated_value.find("---")
                if header_end > 0:
                    header = updated_value[:header_end + 3]
                    content = updated_value[header_end + 3:]
                    # Keep last N characters
                    keep_length = max_length - len(header) - 100
                    updated_value = header + "\n...[older entries truncated]...\n" + content[-keep_length:]
            
            # Update the block
            self.client.agents.blocks.update(
                agent_id=self.agent_id,
                block_id=learnings_block.id,
                value=updated_value
            )
            
            logger.info(f"Updated learnings block with {len(learnings)} new entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update learnings block: {e}")
            return False
    
    def save_file_change_context(self, files_modified: List[str]) -> int:
        """
        Save individual passages for significant file changes.
        
        Creates searchable context for each modified file.
        """
        if not self.client or not files_modified:
            return 0
        
        saved_count = 0
        project_name = Path(self.project_path).name
        
        for file_path in files_modified[:10]:  # Limit to 10 files
            try:
                passage_text = f"""File modified: {file_path}
Project: {project_name}
Timestamp: {datetime.now().isoformat()}
"""
                
                metadata = {
                    "type": "file_change",
                    "file_path": file_path,
                    "project": project_name,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.client.agents.passages.create(
                    agent_id=self.agent_id,
                    text=passage_text,
                    metadata=metadata
                )
                saved_count += 1
                
            except Exception as e:
                logger.error(f"Failed to save file change context: {e}")
        
        return saved_count


def read_session_data() -> Dict[str, Any]:
    """Read session data from stdin (provided by Claude Code)."""
    try:
        if sys.stdin.isatty():
            return {}
        return json.load(sys.stdin)
    except json.JSONDecodeError:
        return {}
    except Exception as e:
        logger.error(f"Failed to read stdin: {e}")
        return {}


def main():
    """Main entry point for session end hook."""
    
    agent_id = os.environ.get("LETTA_AGENT_ID")
    if not agent_id:
        logger.info("LETTA_AGENT_ID not set - skipping session save")
        return
    
    if not LETTA_AVAILABLE:
        print("⚠️ Letta client not installed. Session not saved.", file=sys.stderr)
        return
    
    # Read session data
    session_data = read_session_data()
    
    if not session_data:
        logger.info("No session data received - creating minimal summary")
        session_data = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "conversation_summary": "Session ended without detailed summary",
            "tool_usage": [],
            "files_modified": []
        }
    
    try:
        saver = LettaSessionSaver()
        
        # Save session summary
        passage_id = saver.save_session_summary(session_data)
        
        # Extract and save learnings
        learnings = saver.extract_learnings(session_data)
        if learnings:
            saver.update_learnings_block(learnings)
        
        # Save file change context
        files_modified = session_data.get("files_modified", [])
        if files_modified:
            saved = saver.save_file_change_context(files_modified)
            logger.info(f"Saved {saved} file change records")
        
        print(f"✅ Session saved to Letta memory (passage: {passage_id})")
        
    except Exception as e:
        logger.error(f"Session end hook failed: {e}")
        print(f"⚠️ Could not save session: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

---

### Prompt Context Injection Hook

**File: `~/.claude/hooks/letta-prompt-context.py`**

```python
#!/usr/bin/env python3
"""
Letta Prompt Context Hook
=========================
Injects relevant context from Letta memory based on user prompt.

Features:
- Semantic search of archival memory
- Relevance-based context injection
- Configurable retrieval limits
- Smart context truncation

This hook runs on UserPromptSubmit and enriches Claude's context
with relevant past knowledge.

Input (stdin): JSON with prompt data
- prompt: User's input text
- session_id: Current session identifier

Official Letta Docs: https://docs.letta.com/concepts/memory#archival-memory
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(
    level=logging.WARNING,  # Less verbose for per-prompt hook
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from letta_client import Letta
    LETTA_AVAILABLE = True
except ImportError:
    LETTA_AVAILABLE = False


class ContextInjector:
    """Searches Letta memory and formats relevant context."""
    
    # Minimum prompt length to trigger search
    MIN_PROMPT_LENGTH = 15
    
    # Maximum passages to retrieve
    MAX_PASSAGES = 5
    
    # Maximum characters per passage in output
    MAX_PASSAGE_LENGTH = 300
    
    def __init__(self):
        self.base_url = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
        self.password = os.environ.get("LETTA_PASSWORD", "")
        self.agent_id = os.environ.get("LETTA_AGENT_ID", "")
        
        if LETTA_AVAILABLE:
            self.client = Letta(
                base_url=self.base_url,
                token=self.password
            )
        else:
            self.client = None
    
    def should_search(self, prompt: str) -> bool:
        """
        Determine if this prompt warrants a memory search.
        
        Skip search for:
        - Very short prompts
        - Simple commands
        - Greetings
        """
        if len(prompt) < self.MIN_PROMPT_LENGTH:
            return False
        
        # Skip simple commands and greetings
        skip_patterns = [
            "hello", "hi", "hey", "thanks", "thank you",
            "yes", "no", "ok", "okay", "sure",
            "/", "?"  # Slash commands and simple questions
        ]
        
        prompt_lower = prompt.lower().strip()
        for pattern in skip_patterns:
            if prompt_lower.startswith(pattern) and len(prompt_lower) < 30:
                return False
        
        return True
    
    def search_relevant_context(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Search archival memory for relevant context.
        
        Uses Letta's semantic search with the prompt as query.
        
        Letta API Reference: https://docs.letta.com/api-reference/agents/search-agent-archival-memory
        """
        if not self.client:
            return []
        
        try:
            # Use first 500 chars of prompt as query
            query = prompt[:500]
            
            passages = self.client.agents.passages.list(
                agent_id=self.agent_id,
                query=query,
                limit=self.MAX_PASSAGES
            )
            
            results = []
            for p in passages:
                # Calculate a simple relevance indicator
                text = p.text
                metadata = p.metadata if hasattr(p, 'metadata') else {}
                
                results.append({
                    "text": text,
                    "metadata": metadata,
                    "type": metadata.get("type", "general")
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
    
    def check_core_memory_relevance(self, prompt: str) -> Optional[str]:
        """
        Check if any core memory blocks are especially relevant.
        
        Looks for explicit mentions of memory block topics.
        """
        if not self.client:
            return None
        
        try:
            blocks = self.client.agents.blocks.list(agent_id=self.agent_id)
            
            prompt_lower = prompt.lower()
            
            for block in blocks:
                label = block.label.lower()
                
                # Check if prompt mentions this block's domain
                if label == "project_context" and any(
                    word in prompt_lower for word in 
                    ["project", "architecture", "stack", "structure"]
                ):
                    return f"💡 Project context loaded from memory:\n{block.value[:500]}..."
                
                if label == "learnings" and any(
                    word in prompt_lower for word in 
                    ["learned", "previous", "before", "last time", "remember"]
                ):
                    return f"💡 Past learnings loaded:\n{block.value[-500:]}"
            
            return None
            
        except Exception as e:
            logger.error(f"Core memory check failed: {e}")
            return None
    
    def format_context_output(
        self,
        passages: List[Dict],
        core_memory_hint: Optional[str]
    ) -> str:
        """Format retrieved context for output."""
        
        if not passages and not core_memory_hint:
            return ""
        
        output_parts = []
        
        if core_memory_hint:
            output_parts.append(core_memory_hint)
        
        if passages:
            output_parts.append("\n📚 **Relevant memories found:**\n")
            
            for i, p in enumerate(passages[:self.MAX_PASSAGES], 1):
                text = p["text"]
                if len(text) > self.MAX_PASSAGE_LENGTH:
                    text = text[:self.MAX_PASSAGE_LENGTH] + "..."
                
                type_label = p.get("type", "")
                if type_label:
                    type_label = f" [{type_label}]"
                
                output_parts.append(f"{i}. {text}{type_label}\n")
        
        return "\n".join(output_parts)


def read_prompt_data() -> Dict[str, Any]:
    """Read prompt data from stdin."""
    try:
        if sys.stdin.isatty():
            return {}
        return json.load(sys.stdin)
    except:
        return {}


def main():
    """Main entry point for prompt context hook."""
    
    agent_id = os.environ.get("LETTA_AGENT_ID")
    if not agent_id:
        return  # Silent exit if not configured
    
    if not LETTA_AVAILABLE:
        return  # Silent exit if client not installed
    
    # Read prompt data
    input_data = read_prompt_data()
    prompt = input_data.get("prompt", "")
    
    if not prompt:
        return
    
    try:
        injector = ContextInjector()
        
        # Check if search is warranted
        if not injector.should_search(prompt):
            return
        
        # Search for relevant context
        passages = injector.search_relevant_context(prompt)
        
        # Check core memory for explicit relevance
        core_hint = injector.check_core_memory_relevance(prompt)
        
        # Format and output
        context = injector.format_context_output(passages, core_hint)
        
        if context:
            print(context)
        
    except Exception as e:
        # Silent failure - don't disrupt user experience
        logger.error(f"Context injection failed: {e}")


if __name__ == "__main__":
    main()
```

---

### File Change Tracking Hook

**File: `~/.claude/hooks/letta-track-changes.py`**

```python
#!/usr/bin/env python3
"""
Letta File Change Tracking Hook
================================
Tracks file modifications for future context retrieval.

Runs as PostToolUse hook on Edit and Write operations.

Features:
- Records file paths with timestamps
- Captures modification context
- Enables "what files did we change?" queries
- Maintains change history

Input (stdin): JSON with tool use data
- tool_name: Name of tool (Edit, Write)
- tool_input: Arguments passed to tool
- result: Tool execution result

Official Claude Code Hooks: https://docs.anthropic.com/en/docs/claude-code/hooks
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from letta_client import Letta
    LETTA_AVAILABLE = True
except ImportError:
    LETTA_AVAILABLE = False


class FileChangeTracker:
    """Tracks file changes and stores in Letta memory."""
    
    def __init__(self):
        self.base_url = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
        self.password = os.environ.get("LETTA_PASSWORD", "")
        self.agent_id = os.environ.get("LETTA_AGENT_ID", "")
        self.project_path = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
        
        if LETTA_AVAILABLE:
            self.client = Letta(
                base_url=self.base_url,
                token=self.password
            )
        else:
            self.client = None
    
    def track_file_change(
        self,
        file_path: str,
        operation: str,
        details: str = ""
    ) -> bool:
        """
        Record a file change in Letta archival memory.
        
        Creates a searchable passage for the file modification.
        """
        if not self.client:
            return False
        
        try:
            project_name = Path(self.project_path).name
            
            # Make path relative to project if possible
            try:
                rel_path = Path(file_path).relative_to(self.project_path)
                display_path = str(rel_path)
            except ValueError:
                display_path = file_path
            
            # Create passage text
            passage_text = f"""File Change Record
==================
**File**: {display_path}
**Operation**: {operation}
**Project**: {project_name}
**Timestamp**: {datetime.now().isoformat()}

{f"Details: {details}" if details else ""}
"""
            
            metadata = {
                "type": "file_change",
                "file_path": display_path,
                "operation": operation,
                "project": project_name,
                "timestamp": datetime.now().isoformat()
            }
            
            self.client.agents.passages.create(
                agent_id=self.agent_id,
                text=passage_text,
                metadata=metadata
            )
            
            logger.info(f"Tracked {operation} on {display_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to track file change: {e}")
            return False


def extract_file_path(tool_input: Dict) -> Optional[str]:
    """Extract file path from tool input."""
    # Handle different tool input formats
    if "file_path" in tool_input:
        return tool_input["file_path"]
    if "path" in tool_input:
        return tool_input["path"]
    if "file" in tool_input:
        return tool_input["file"]
    return None


def main():
    """Main entry point for file tracking hook."""
    
    agent_id = os.environ.get("LETTA_AGENT_ID")
    if not agent_id:
        return  # Silent exit
    
    if not LETTA_AVAILABLE:
        return
    
    # Read tool use data from stdin
    try:
        if sys.stdin.isatty():
            return
        tool_data = json.load(sys.stdin)
    except:
        return
    
    tool_name = tool_data.get("tool_name", "")
    tool_input = tool_data.get("tool_input", {})
    
    # Only track Edit and Write operations
    if tool_name not in ["Edit", "Write", "str_replace"]:
        return
    
    file_path = extract_file_path(tool_input)
    if not file_path:
        return
    
    try:
        tracker = FileChangeTracker()
        tracker.track_file_change(
            file_path=file_path,
            operation=tool_name,
            details=tool_input.get("description", "")
        )
    except Exception as e:
        logger.error(f"File tracking failed: {e}")


if __name__ == "__main__":
    main()
```

---

## 1.3 Complete Hooks Configuration

**File: `.claude/settings.json`**

```json
{
  "env": {
    "LETTA_BASE_URL": "http://localhost:8283/v1",
    "LETTA_PASSWORD": "${LETTA_PASSWORD}",
    "LETTA_AGENT_ID": "${LETTA_AGENT_ID}"
  },
  
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python ~/.claude/hooks/letta-session-start.py",
            "timeout": 30
          }
        ]
      }
    ],
    
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python ~/.claude/hooks/letta-prompt-context.py",
            "timeout": 10
          }
        ]
      }
    ],
    
    "PreToolUse": [
      {
        "matcher": "Bash\\(rm|Bash\\(sudo",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'Dangerous command blocked' && exit 2"
          }
        ]
      }
    ],
    
    "PostToolUse": [
      {
        "matcher": "Edit|Write|str_replace",
        "hooks": [
          {
            "type": "command",
            "command": "python ~/.claude/hooks/letta-track-changes.py",
            "timeout": 5
          }
        ]
      },
      {
        "matcher": "Write\\(.*\\.py$",
        "hooks": [
          {
            "type": "command",
            "command": "python -m black --quiet $CLAUDE_FILE_PATH 2>/dev/null || true"
          }
        ]
      }
    ],
    
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python ~/.claude/hooks/letta-session-end.py",
            "timeout": 30
          }
        ]
      }
    ]
  },
  
  "permissions": {
    "allow": [
      "Bash(python ~/.claude/hooks/*)",
      "Bash(python -m black:*)",
      "Read",
      "Glob",
      "Grep"
    ],
    "deny": [
      "Bash(rm -rf /)",
      "Bash(sudo:*)"
    ]
  }
}
```

---

# Part 2: Advanced Memory Patterns

## 2.1 Memory Architecture Deep Dive

### Letta Memory Tiers

```
┌─────────────────────────────────────────────────────────────────┐
│                      LETTA MEMORY SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              CORE MEMORY (Always In Context)             │   │
│  │                                                          │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │  human   │ │ persona  │ │ project  │ │learnings │   │   │
│  │  │          │ │          │ │ context  │ │          │   │   │
│  │  │  User    │ │  Agent   │ │ Current  │ │ Session  │   │   │
│  │  │ Profile  │ │ Behavior │ │ Project  │ │ History  │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  │                                                          │   │
│  │  Limit: ~5000 chars per block | Auto-loaded at start    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           ARCHIVAL MEMORY (Semantic Search)              │   │
│  │                                                          │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │  pgvector Embeddings (text-embedding-3-small)   │    │   │
│  │  │                                                  │    │   │
│  │  │  • Session summaries                            │    │   │
│  │  │  • File change records                          │    │   │
│  │  │  • Bug fixes and solutions                      │    │   │
│  │  │  • Architecture decisions                       │    │   │
│  │  │  • User preference discoveries                  │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  │                                                          │   │
│  │  Limit: Unlimited | Searched on demand via queries      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           RECALL MEMORY (Conversation History)           │   │
│  │                                                          │   │
│  │  PostgreSQL storage of past conversations                │   │
│  │  Searchable by content, date, session ID                 │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Official Letta Memory Documentation
- **Memory Concepts**: https://docs.letta.com/concepts/memory
- **Core Memory API**: https://docs.letta.com/api-reference/agents/memory
- **Archival Memory API**: https://docs.letta.com/api-reference/agents/archival-memory

---

## 2.2 Multi-Project Memory Isolation

### Project-Specific Agent Creation

```python
#!/usr/bin/env python3
"""
Multi-Project Memory Manager
============================
Creates and manages separate Letta agents per project.

This pattern ensures:
- Complete memory isolation between projects
- Project-specific context and learnings
- Easy agent switching when changing projects
- Shared team knowledge via block attachment

Official Letta Docs: https://docs.letta.com/api-reference/agents/create-agent
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from letta_client import Letta


class MultiProjectMemoryManager:
    """Manages Letta agents across multiple projects."""
    
    AGENT_CONFIG_FILE = ".claude/letta_agent.json"
    
    def __init__(self, base_url: str, password: str):
        self.client = Letta(base_url=base_url, token=password)
        self.base_url = base_url
    
    def get_or_create_project_agent(
        self,
        project_path: str,
        force_create: bool = False
    ) -> Dict[str, Any]:
        """
        Get existing agent for project or create new one.
        
        Stores agent configuration in .claude/letta_agent.json
        """
        project_path = Path(project_path)
        config_path = project_path / self.AGENT_CONFIG_FILE
        
        # Check for existing config
        if config_path.exists() and not force_create:
            with open(config_path) as f:
                config = json.load(f)
            
            # Verify agent still exists
            try:
                agent = self.client.agents.retrieve(agent_id=config["agent_id"])
                return {
                    "agent_id": agent.id,
                    "agent_name": agent.name,
                    "status": "existing"
                }
            except Exception:
                pass  # Agent deleted, create new one
        
        # Create new agent for this project
        project_name = project_path.name
        
        # Scan project for context
        project_context = self._scan_project_context(project_path)
        
        agent = self.client.agents.create(
            name=f"memory-{project_name}",
            description=f"Memory agent for {project_name}",
            model="openai/gpt-4.1",  # or preferred model
            embedding="openai/text-embedding-3-small",
            memory_blocks=[
                {
                    "label": "human",
                    "value": """# Developer Profile
Name: [Auto-detected or configured]
Preferences: [Discovered through interaction]
Current Focus: Active development"""
                },
                {
                    "label": "persona",
                    "value": f"""# Memory Agent for {project_name}
I maintain persistent memory for this project.
I remember past decisions, bugs, and solutions.
I proactively surface relevant context."""
                },
                {
                    "label": "project_context",
                    "value": project_context
                },
                {
                    "label": "learnings",
                    "value": f"""# Session Learnings
Project: {project_name}
Started: {datetime.now().isoformat()}

---
[Learnings will be added here]"""
                }
            ]
        )
        
        # Save configuration
        config = {
            "agent_id": agent.id,
            "agent_name": agent.name,
            "project_name": project_name,
            "created_at": datetime.now().isoformat(),
            "base_url": self.base_url
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return {
            "agent_id": agent.id,
            "agent_name": agent.name,
            "status": "created"
        }
    
    def _scan_project_context(self, project_path: Path) -> str:
        """Scan project files to build initial context."""
        context_parts = [f"# Project: {project_path.name}\n"]
        
        # Check for package files
        package_files = {
            "package.json": "Node.js/TypeScript",
            "pyproject.toml": "Python",
            "Cargo.toml": "Rust",
            "go.mod": "Go",
            "pom.xml": "Java/Maven",
            "build.gradle": "Java/Gradle"
        }
        
        detected_stacks = []
        for filename, stack in package_files.items():
            if (project_path / filename).exists():
                detected_stacks.append(stack)
        
        if detected_stacks:
            context_parts.append(f"## Tech Stack\n{', '.join(detected_stacks)}\n")
        
        # Check for README
        readme_files = ["README.md", "readme.md", "README.txt", "README"]
        for readme in readme_files:
            readme_path = project_path / readme
            if readme_path.exists():
                try:
                    content = readme_path.read_text()[:1000]
                    context_parts.append(f"## From README\n{content}\n")
                except:
                    pass
                break
        
        # Check for existing CLAUDE.md
        claude_md = project_path / "CLAUDE.md"
        if claude_md.exists():
            try:
                content = claude_md.read_text()[:1500]
                context_parts.append(f"## From CLAUDE.md\n{content}\n")
            except:
                pass
        
        return "\n".join(context_parts)
    
    def switch_project(self, project_path: str) -> Dict[str, str]:
        """
        Switch to a different project's agent.
        
        Returns environment variables to set.
        """
        result = self.get_or_create_project_agent(project_path)
        
        return {
            "LETTA_AGENT_ID": result["agent_id"],
            "LETTA_PROJECT": Path(project_path).name
        }
    
    def list_project_agents(self) -> list:
        """List all project-specific agents."""
        agents = self.client.agents.list()
        
        return [
            {
                "id": a.id,
                "name": a.name,
                "project": a.name.replace("memory-", "") if a.name.startswith("memory-") else "unknown",
                "created": str(a.created_at) if hasattr(a, 'created_at') else None
            }
            for a in agents
            if a.name.startswith("memory-")
        ]


# CLI interface
if __name__ == "__main__":
    import sys
    
    base_url = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
    password = os.environ.get("LETTA_PASSWORD", "")
    
    manager = MultiProjectMemoryManager(base_url, password)
    
    if len(sys.argv) < 2:
        print("Usage: python multi_project_memory.py <project_path>")
        sys.exit(1)
    
    project_path = sys.argv[1]
    result = manager.get_or_create_project_agent(project_path)
    
    print(f"Agent ID: {result['agent_id']}")
    print(f"Status: {result['status']}")
    print(f"\nSet environment variable:")
    print(f"export LETTA_AGENT_ID={result['agent_id']}")
```

---

## 2.3 Team Shared Memory

### Shared Knowledge Block Pattern

```python
#!/usr/bin/env python3
"""
Team Shared Memory System
=========================
Creates shared memory blocks that can be attached to multiple agents.

Use Cases:
- Team coding standards
- Architecture decisions
- Common patterns and anti-patterns
- Shared debugging knowledge

Official Letta Docs: https://docs.letta.com/api-reference/blocks
"""

from letta_client import Letta
from typing import List, Dict, Any
import json


class TeamSharedMemory:
    """Manages shared memory blocks across team agents."""
    
    def __init__(self, base_url: str, password: str):
        self.client = Letta(base_url=base_url, token=password)
    
    def create_shared_block(
        self,
        label: str,
        value: str,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Create a shared memory block.
        
        Shared blocks exist independently and can be attached
        to multiple agents.
        
        Letta API: https://docs.letta.com/api-reference/blocks/create-block
        """
        block = self.client.blocks.create(
            label=label,
            value=value,
            description=description
        )
        
        return {
            "block_id": block.id,
            "label": block.label,
            "status": "created"
        }
    
    def attach_to_agents(
        self,
        block_id: str,
        agent_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Attach a shared block to multiple agents.
        
        The block will appear in each agent's core memory.
        Changes to the block affect all attached agents.
        
        Letta API: https://docs.letta.com/api-reference/agents/attach-block
        """
        results = {"attached": [], "failed": []}
        
        for agent_id in agent_ids:
            try:
                self.client.agents.blocks.attach(
                    agent_id=agent_id,
                    block_id=block_id
                )
                results["attached"].append(agent_id)
            except Exception as e:
                results["failed"].append({
                    "agent_id": agent_id,
                    "error": str(e)
                })
        
        return results
    
    def update_shared_block(
        self,
        block_id: str,
        value: str
    ) -> bool:
        """
        Update a shared block.
        
        Changes propagate to all agents with this block attached.
        """
        try:
            self.client.blocks.update(
                block_id=block_id,
                value=value
            )
            return True
        except Exception as e:
            print(f"Failed to update block: {e}")
            return False
    
    def create_team_standards_block(self) -> Dict[str, Any]:
        """Create a standard team coding standards block."""
        
        standards_content = """# Team Coding Standards

## Code Style
- Use type hints for all function parameters and returns
- Maximum line length: 100 characters
- Use descriptive variable names (no single letters except loops)
- Document all public functions with docstrings

## Git Workflow
- Branch naming: feature/ticket-description, fix/ticket-description
- Commit messages: [type] Brief description (max 72 chars)
- Types: feat, fix, docs, refactor, test, chore
- Squash commits before merging

## Testing Requirements
- Minimum 80% code coverage for new code
- All bug fixes must include regression tests
- Integration tests for API endpoints
- Property-based testing for data transformations

## Architecture Patterns
- Use dependency injection for services
- Repository pattern for data access
- Event-driven for cross-service communication
- CQRS for complex domains

## Security
- Never log sensitive data
- Validate all external inputs
- Use parameterized queries
- Rotate secrets every 90 days
"""
        
        return self.create_shared_block(
            label="team_standards",
            value=standards_content,
            description="Team-wide coding standards and practices"
        )
    
    def create_architecture_decisions_block(self) -> Dict[str, Any]:
        """Create a block for tracking architecture decisions."""
        
        adr_content = """# Architecture Decision Records (ADR)

## ADR-001: Use PostgreSQL for Primary Database
**Status**: Accepted
**Date**: 2024-01-01
**Context**: Need reliable ACID-compliant database
**Decision**: PostgreSQL with pgvector for embeddings
**Consequences**: 
- Strong consistency guarantees
- Good ecosystem and tooling
- Vector search capability built-in

## ADR-002: Event Sourcing for Trading Domain
**Status**: Accepted
**Date**: 2024-02-15
**Context**: Need complete audit trail of trading decisions
**Decision**: Event sourcing with PostgreSQL event store
**Consequences**:
- Full history preservation
- Ability to replay and debug
- Increased storage requirements

## Template for New ADRs
```
## ADR-XXX: [Title]
**Status**: Proposed | Accepted | Deprecated | Superseded
**Date**: YYYY-MM-DD
**Context**: What is the issue?
**Decision**: What was decided?
**Consequences**: What are the results?
```
"""
        
        return self.create_shared_block(
            label="architecture_decisions",
            value=adr_content,
            description="Architecture Decision Records for the team"
        )


# Example usage
if __name__ == "__main__":
    import os
    
    manager = TeamSharedMemory(
        base_url=os.environ.get("LETTA_BASE_URL", "http://localhost:8283"),
        password=os.environ.get("LETTA_PASSWORD", "")
    )
    
    # Create shared blocks
    standards = manager.create_team_standards_block()
    print(f"Created standards block: {standards['block_id']}")
    
    decisions = manager.create_architecture_decisions_block()
    print(f"Created ADR block: {decisions['block_id']}")
    
    # Attach to team agents
    team_agents = [
        "agent-alice-xxxxx",
        "agent-bob-xxxxx",
        "agent-charlie-xxxxx"
    ]
    
    result = manager.attach_to_agents(standards['block_id'], team_agents)
    print(f"Attached to {len(result['attached'])} agents")
```

---

## 2.4 Automated Learning Pipeline

### CI/CD Integration Pattern

```python
#!/usr/bin/env python3
"""
Automated Learning Pipeline
===========================
Integrates Letta memory with CI/CD for automated knowledge capture.

Captures:
- Deployment outcomes
- Test failures and resolutions
- Performance metrics
- Incident learnings

Official Letta Docs: https://docs.letta.com/api-reference/agents/archival-memory
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional
from letta_client import Letta


class AutomatedLearningPipeline:
    """Captures learnings from CI/CD and operations."""
    
    def __init__(self):
        self.client = Letta(
            base_url=os.environ.get("LETTA_BASE_URL", "http://localhost:8283"),
            token=os.environ.get("LETTA_PASSWORD", "")
        )
        self.agent_id = os.environ.get("LETTA_AGENT_ID", "")
    
    def record_deployment(
        self,
        version: str,
        environment: str,
        duration_seconds: int,
        status: str,
        issues: Optional[list] = None,
        rollback: bool = False
    ) -> str:
        """
        Record a deployment event in archival memory.
        
        This creates searchable context for future deployments.
        """
        passage_text = f"""# Deployment Record

**Version**: {version}
**Environment**: {environment}
**Status**: {status}
**Duration**: {duration_seconds}s
**Timestamp**: {datetime.now().isoformat()}
**Rollback**: {'Yes' if rollback else 'No'}

## Issues
{chr(10).join(f'- {issue}' for issue in (issues or [])) or 'None'}

## Learnings
{self._extract_deployment_learnings(status, issues, rollback)}
"""
        
        metadata = {
            "type": "deployment",
            "version": version,
            "environment": environment,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        passage = self.client.agents.passages.create(
            agent_id=self.agent_id,
            text=passage_text,
            metadata=metadata
        )
        
        return passage.id
    
    def record_test_failure(
        self,
        test_name: str,
        error_message: str,
        stack_trace: str,
        file_path: str,
        resolution: Optional[str] = None
    ) -> str:
        """
        Record a test failure and its resolution.
        
        Enables "have we seen this error before?" queries.
        """
        passage_text = f"""# Test Failure Record

**Test**: {test_name}
**File**: {file_path}
**Timestamp**: {datetime.now().isoformat()}

## Error
```
{error_message}
```

## Stack Trace (truncated)
```
{stack_trace[:500]}
```

## Resolution
{resolution or 'Pending investigation'}
"""
        
        metadata = {
            "type": "test_failure",
            "test_name": test_name,
            "file_path": file_path,
            "resolved": resolution is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        passage = self.client.agents.passages.create(
            agent_id=self.agent_id,
            text=passage_text,
            metadata=metadata
        )
        
        return passage.id
    
    def record_incident(
        self,
        title: str,
        severity: str,
        description: str,
        root_cause: str,
        resolution: str,
        prevention: str
    ) -> str:
        """
        Record an incident with post-mortem learnings.
        
        Creates valuable context for preventing future incidents.
        """
        passage_text = f"""# Incident Record

**Title**: {title}
**Severity**: {severity}
**Timestamp**: {datetime.now().isoformat()}

## Description
{description}

## Root Cause
{root_cause}

## Resolution
{resolution}

## Prevention Measures
{prevention}

## Key Learnings
- Root cause identified: {root_cause[:100]}
- Resolution time documented
- Prevention measures established
"""
        
        metadata = {
            "type": "incident",
            "title": title,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        
        passage = self.client.agents.passages.create(
            agent_id=self.agent_id,
            text=passage_text,
            metadata=metadata
        )
        
        return passage.id
    
    def record_performance_regression(
        self,
        metric_name: str,
        baseline: float,
        current: float,
        change_percent: float,
        commit_sha: str
    ) -> str:
        """Record a performance regression for tracking."""
        passage_text = f"""# Performance Regression

**Metric**: {metric_name}
**Baseline**: {baseline}
**Current**: {current}
**Change**: {change_percent:.1f}%
**Commit**: {commit_sha}
**Timestamp**: {datetime.now().isoformat()}

## Analysis
Performance degraded by {abs(change_percent):.1f}%.
Investigate changes in commit {commit_sha}.
"""
        
        metadata = {
            "type": "performance_regression",
            "metric": metric_name,
            "change_percent": change_percent,
            "commit": commit_sha,
            "timestamp": datetime.now().isoformat()
        }
        
        passage = self.client.agents.passages.create(
            agent_id=self.agent_id,
            text=passage_text,
            metadata=metadata
        )
        
        return passage.id
    
    def _extract_deployment_learnings(
        self,
        status: str,
        issues: Optional[list],
        rollback: bool
    ) -> str:
        """Generate learnings from deployment outcome."""
        learnings = []
        
        if status == "success" and not rollback:
            learnings.append("Successful deployment - process working well")
        
        if rollback:
            learnings.append("Rollback required - investigate root cause")
            learnings.append("Consider adding pre-deployment checks")
        
        if issues:
            learnings.append(f"{len(issues)} issues encountered")
            learnings.append("Review deployment checklist")
        
        return "\n".join(f"- {l}" for l in learnings) or "Standard deployment"


# GitHub Actions integration example
def github_action_handler():
    """Handler for GitHub Actions workflow."""
    import json
    
    # Read GitHub event payload
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if event_path and os.path.exists(event_path):
        with open(event_path) as f:
            event = json.load(f)
    else:
        event = {}
    
    pipeline = AutomatedLearningPipeline()
    
    # Determine event type and record appropriately
    github_event = os.environ.get("GITHUB_EVENT_NAME", "")
    
    if github_event == "deployment_status":
        pipeline.record_deployment(
            version=event.get("deployment", {}).get("ref", "unknown"),
            environment=event.get("deployment", {}).get("environment", "unknown"),
            duration_seconds=0,  # Calculate from timestamps
            status=event.get("deployment_status", {}).get("state", "unknown")
        )
    
    elif github_event == "workflow_run" and event.get("workflow_run", {}).get("conclusion") == "failure":
        pipeline.record_test_failure(
            test_name=event.get("workflow_run", {}).get("name", "unknown"),
            error_message="Workflow failed",
            stack_trace="See GitHub Actions logs",
            file_path=event.get("workflow_run", {}).get("path", "unknown")
        )


if __name__ == "__main__":
    github_action_handler()
```

---

## 2.5 Semantic Memory Search Patterns

### Advanced Search Implementations

```python
#!/usr/bin/env python3
"""
Semantic Memory Search Patterns
===============================
Advanced patterns for searching and retrieving Letta memories.

Official Letta Docs: https://docs.letta.com/api-reference/agents/search-agent-archival-memory
"""

from letta_client import Letta
from typing import List, Dict, Any, Optional
import os


class SemanticMemorySearch:
    """Advanced semantic search over Letta memory."""
    
    def __init__(self):
        self.client = Letta(
            base_url=os.environ.get("LETTA_BASE_URL", "http://localhost:8283"),
            token=os.environ.get("LETTA_PASSWORD", "")
        )
        self.agent_id = os.environ.get("LETTA_AGENT_ID", "")
    
    def search_by_topic(
        self,
        topic: str,
        limit: int = 10,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search memories by topic with optional type filtering.
        
        The query is converted to an embedding and matched
        against stored passage embeddings using cosine similarity.
        """
        passages = self.client.agents.passages.list(
            agent_id=self.agent_id,
            query=topic,
            limit=limit
        )
        
        results = []
        for p in passages:
            metadata = p.metadata if hasattr(p, 'metadata') else {}
            
            # Apply type filter if specified
            if filter_type and metadata.get("type") != filter_type:
                continue
            
            results.append({
                "id": p.id,
                "text": p.text,
                "metadata": metadata,
                "type": metadata.get("type", "general")
            })
        
        return results
    
    def search_similar_bugs(self, error_message: str, limit: int = 5) -> List[Dict]:
        """
        Find similar bugs from past sessions.
        
        Useful for "have we seen this error before?" queries.
        """
        return self.search_by_topic(
            topic=f"error bug {error_message}",
            limit=limit,
            filter_type="test_failure"
        )
    
    def search_architecture_decisions(self, topic: str, limit: int = 5) -> List[Dict]:
        """
        Search for relevant architecture decisions.
        
        Useful when making design choices.
        """
        results = self.search_by_topic(
            topic=f"architecture decision {topic}",
            limit=limit
        )
        
        # Also check ADR block
        try:
            blocks = self.client.agents.blocks.list(agent_id=self.agent_id)
            for block in blocks:
                if block.label == "architecture_decisions":
                    # Check if topic is mentioned in ADRs
                    if topic.lower() in block.value.lower():
                        results.insert(0, {
                            "id": block.id,
                            "text": block.value,
                            "metadata": {"type": "adr_block"},
                            "type": "adr_block"
                        })
        except Exception:
            pass
        
        return results
    
    def search_file_history(self, file_path: str, limit: int = 10) -> List[Dict]:
        """
        Get modification history for a specific file.
        
        Returns all recorded changes to the file.
        """
        return self.search_by_topic(
            topic=f"file {file_path}",
            limit=limit,
            filter_type="file_change"
        )
    
    def search_deployment_history(
        self,
        environment: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search deployment history, optionally filtered by environment.
        """
        query = "deployment"
        if environment:
            query += f" {environment}"
        
        return self.search_by_topic(
            topic=query,
            limit=limit,
            filter_type="deployment"
        )
    
    def get_recent_learnings(self, days: int = 7, limit: int = 20) -> List[Dict]:
        """
        Get recent learnings from the past N days.
        
        Note: Letta doesn't have direct date filtering,
        so we retrieve more and filter client-side.
        """
        from datetime import datetime, timedelta
        
        results = self.search_by_topic(
            topic="learning session",
            limit=limit * 2  # Fetch extra for filtering
        )
        
        cutoff = datetime.now() - timedelta(days=days)
        
        filtered = []
        for r in results:
            timestamp = r["metadata"].get("timestamp")
            if timestamp:
                try:
                    ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    if ts.replace(tzinfo=None) > cutoff:
                        filtered.append(r)
                except:
                    filtered.append(r)  # Include if can't parse
            else:
                filtered.append(r)  # Include if no timestamp
        
        return filtered[:limit]
    
    def search_by_metadata(
        self,
        metadata_key: str,
        metadata_value: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search passages by metadata field.
        
        Note: This requires client-side filtering as Letta
        doesn't support metadata-based search directly.
        """
        # Fetch more results and filter
        all_passages = self.client.agents.passages.list(
            agent_id=self.agent_id,
            limit=100  # Fetch batch
        )
        
        filtered = []
        for p in all_passages:
            metadata = p.metadata if hasattr(p, 'metadata') else {}
            if metadata.get(metadata_key) == metadata_value:
                filtered.append({
                    "id": p.id,
                    "text": p.text,
                    "metadata": metadata
                })
                if len(filtered) >= limit:
                    break
        
        return filtered


# Example usage
if __name__ == "__main__":
    search = SemanticMemorySearch()
    
    # Find similar bugs
    bugs = search.search_similar_bugs("NullPointerException in UserService")
    print(f"Found {len(bugs)} similar bugs")
    
    # Get architecture decisions about databases
    decisions = search.search_architecture_decisions("database")
    print(f"Found {len(decisions)} relevant architecture decisions")
    
    # Get recent learnings
    learnings = search.get_recent_learnings(days=7)
    print(f"Found {len(learnings)} learnings from the past week")
```

---

# Part 3: Production Kubernetes Deployment

## 3.1 Complete Kubernetes Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        KUBERNETES CLUSTER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        INGRESS (NGINX/Traefik)                       │   │
│  │                    SSL Termination / Rate Limiting                    │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │                                          │
│           ┌──────────────────────┼──────────────────────┐                  │
│           │                      │                      │                   │
│           ▼                      ▼                      ▼                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │  letta-server   │  │  letta-server   │  │  letta-server   │            │
│  │   (replica 1)   │  │   (replica 2)   │  │   (replica 3)   │            │
│  │                 │  │                 │  │                 │            │
│  │  Port: 8283     │  │  Port: 8283     │  │  Port: 8283     │            │
│  │  CPU: 500m-1000m│  │  CPU: 500m-1000m│  │  CPU: 500m-1000m│            │
│  │  Mem: 1Gi-2Gi   │  │  Mem: 1Gi-2Gi   │  │  Mem: 1Gi-2Gi   │            │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘            │
│           │                    │                    │                      │
│           └────────────────────┼────────────────────┘                      │
│                                │                                           │
│                                ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    ClusterIP Service: letta-server                   │  │
│  │                         Port: 8283 → 8283                            │  │
│  └───────────────────────────────┬─────────────────────────────────────┘  │
│                                  │                                         │
│           ┌──────────────────────┼──────────────────────┐                 │
│           │                      │                      │                  │
│           ▼                      ▼                      ▼                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │   letta-mcp     │  │   letta-mcp     │  │   postgresql    │           │
│  │   (replica 1)   │  │   (replica 2)   │  │   + pgvector    │           │
│  │                 │  │                 │  │                 │           │
│  │  Port: 3001     │  │  Port: 3001     │  │  Port: 5432     │           │
│  └─────────────────┘  └─────────────────┘  │  StatefulSet    │           │
│                                            │  PVC: 50Gi      │           │
│                                            └─────────────────┘           │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                           ConfigMaps & Secrets                       │ │
│  │  - letta-config (environment variables)                              │ │
│  │  - letta-secrets (passwords, API keys)                               │ │
│  │  - tls-secrets (SSL certificates)                                    │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Official Kubernetes Documentation References
- **Deployments**: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
- **StatefulSets**: https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/
- **Services**: https://kubernetes.io/docs/concepts/services-networking/service/
- **Secrets Management**: https://kubernetes.io/docs/concepts/configuration/secret/
- **Resource Management**: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/

### Official Letta Documentation References
- **Letta Server Deployment**: https://docs.letta.com/server/docker
- **Letta Configuration**: https://docs.letta.com/server/configuration
- **Letta MCP Server**: https://github.com/oculairmedia/Letta-MCP-server

---

## 3.2 Complete Kubernetes Manifests

### Namespace and RBAC

**File: `k8s/00-namespace.yaml`**

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: letta
  labels:
    name: letta
    app.kubernetes.io/name: letta
    app.kubernetes.io/part-of: letta-system
---
# Service Account for Letta components
apiVersion: v1
kind: ServiceAccount
metadata:
  name: letta-service-account
  namespace: letta
  labels:
    app.kubernetes.io/name: letta
---
# Role for Letta (minimal permissions)
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: letta-role
  namespace: letta
rules:
  - apiGroups: [""]
    resources: ["configmaps", "secrets"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: letta-role-binding
  namespace: letta
subjects:
  - kind: ServiceAccount
    name: letta-service-account
    namespace: letta
roleRef:
  kind: Role
  name: letta-role
  apiGroup: rbac.authorization.k8s.io
```

---

### Secrets Management

**File: `k8s/01-secrets.yaml`**

```yaml
# External Secrets Operator integration (recommended for production)
# Reference: https://external-secrets.io/
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: letta-secrets
  namespace: letta
spec:
  refreshInterval: 1h
  secretStoreRef:
    kind: ClusterSecretStore
    name: vault-backend  # or aws-secretsmanager, gcp-secretmanager
  target:
    name: letta-secrets
    creationPolicy: Owner
  data:
    - secretKey: pg-password
      remoteRef:
        key: letta/database
        property: password
    - secretKey: server-password
      remoteRef:
        key: letta/api
        property: password
    - secretKey: openai-key
      remoteRef:
        key: letta/llm-providers
        property: openai-api-key
    - secretKey: anthropic-key
      remoteRef:
        key: letta/llm-providers
        property: anthropic-api-key
---
# Alternative: Direct Kubernetes Secret (for development/testing)
# IMPORTANT: Use External Secrets or sealed-secrets in production
apiVersion: v1
kind: Secret
metadata:
  name: letta-secrets-dev
  namespace: letta
  labels:
    app.kubernetes.io/name: letta
type: Opaque
stringData:
  pg-password: "CHANGE_ME_secure_database_password_123"
  server-password: "CHANGE_ME_secure_api_password_456"
  openai-key: "sk-..."
  anthropic-key: "sk-ant-..."
```

---

### ConfigMap

**File: `k8s/02-configmap.yaml`**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: letta-config
  namespace: letta
  labels:
    app.kubernetes.io/name: letta
data:
  # Database Configuration
  LETTA_PG_USER: "letta"
  LETTA_PG_DB: "letta"
  LETTA_PG_HOST: "postgresql-letta.letta.svc.cluster.local"
  LETTA_PG_PORT: "5432"
  
  # Server Configuration
  SECURE: "true"
  LETTA_UVICORN_WORKERS: "5"
  LETTA_LOG_LEVEL: "INFO"
  
  # Model Defaults
  DEFAULT_LLM_MODEL: "openai/gpt-4.1"
  DEFAULT_EMBEDDING_MODEL: "openai/text-embedding-3-small"
  
  # Memory Limits
  MAX_CORE_MEMORY_SIZE: "5000"
  MAX_ARCHIVAL_RESULTS: "100"
  
  # MCP Server Configuration
  MCP_PORT: "3001"
  NODE_ENV: "production"
---
# PostgreSQL ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgresql-config
  namespace: letta
data:
  postgresql.conf: |
    # Performance tuning for Letta workloads
    shared_buffers = 256MB
    effective_cache_size = 768MB
    maintenance_work_mem = 128MB
    checkpoint_completion_target = 0.9
    wal_buffers = 16MB
    default_statistics_target = 100
    random_page_cost = 1.1
    effective_io_concurrency = 200
    min_wal_size = 1GB
    max_wal_size = 4GB
    max_worker_processes = 4
    max_parallel_workers_per_gather = 2
    max_parallel_workers = 4
    max_parallel_maintenance_workers = 2
    
    # pgvector settings
    shared_preload_libraries = 'vector'
```

---

### PostgreSQL StatefulSet

**File: `k8s/03-postgresql.yaml`**

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgresql-letta
  namespace: letta
  labels:
    app: postgresql
    app.kubernetes.io/name: postgresql
    app.kubernetes.io/component: database
spec:
  serviceName: postgresql-letta
  replicas: 1
  selector:
    matchLabels:
      app: postgresql
  template:
    metadata:
      labels:
        app: postgresql
        app.kubernetes.io/name: postgresql
    spec:
      serviceAccountName: letta-service-account
      
      # Security context
      securityContext:
        runAsNonRoot: true
        runAsUser: 999
        fsGroup: 999
      
      containers:
        - name: postgresql
          image: ankane/pgvector:v0.5.1
          imagePullPolicy: IfNotPresent
          
          ports:
            - containerPort: 5432
              name: postgresql
          
          env:
            - name: POSTGRES_USER
              valueFrom:
                configMapKeyRef:
                  name: letta-config
                  key: LETTA_PG_USER
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: letta-secrets
                  key: pg-password
            - name: POSTGRES_DB
              valueFrom:
                configMapKeyRef:
                  name: letta-config
                  key: LETTA_PG_DB
            - name: PGDATA
              value: /var/lib/postgresql/data/pgdata
          
          volumeMounts:
            - name: data
              mountPath: /var/lib/postgresql/data
            - name: config
              mountPath: /etc/postgresql/postgresql.conf
              subPath: postgresql.conf
          
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
          
          livenessProbe:
            exec:
              command:
                - pg_isready
                - -U
                - letta
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 6
          
          readinessProbe:
            exec:
              command:
                - pg_isready
                - -U
                - letta
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
      
      volumes:
        - name: config
          configMap:
            name: postgresql-config
  
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: standard  # Change based on cloud provider
        resources:
          requests:
            storage: 50Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgresql-letta
  namespace: letta
  labels:
    app: postgresql
spec:
  type: ClusterIP
  ports:
    - port: 5432
      targetPort: 5432
      name: postgresql
  selector:
    app: postgresql
```

---

### Letta Server Deployment

**File: `k8s/04-letta-server.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: letta-server
  namespace: letta
  labels:
    app: letta-server
    app.kubernetes.io/name: letta-server
    app.kubernetes.io/component: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: letta-server
  
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  
  template:
    metadata:
      labels:
        app: letta-server
        app.kubernetes.io/name: letta-server
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8283"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: letta-service-account
      
      # Anti-affinity to spread across nodes
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app
                      operator: In
                      values:
                        - letta-server
                topologyKey: kubernetes.io/hostname
      
      # Init container to wait for PostgreSQL
      initContainers:
        - name: wait-for-postgres
          image: busybox:1.36
          command:
            - sh
            - -c
            - |
              until nc -z postgresql-letta.letta.svc.cluster.local 5432; do
                echo "Waiting for PostgreSQL..."
                sleep 2
              done
              echo "PostgreSQL is ready"
      
      containers:
        - name: letta-server
          image: letta/letta:latest
          imagePullPolicy: Always
          
          ports:
            - containerPort: 8283
              name: http
          
          env:
            - name: LETTA_PG_URI
              value: "postgresql://$(LETTA_PG_USER):$(LETTA_PG_PASSWORD)@$(LETTA_PG_HOST):$(LETTA_PG_PORT)/$(LETTA_PG_DB)"
            - name: LETTA_PG_USER
              valueFrom:
                configMapKeyRef:
                  name: letta-config
                  key: LETTA_PG_USER
            - name: LETTA_PG_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: letta-secrets
                  key: pg-password
            - name: LETTA_PG_HOST
              valueFrom:
                configMapKeyRef:
                  name: letta-config
                  key: LETTA_PG_HOST
            - name: LETTA_PG_PORT
              valueFrom:
                configMapKeyRef:
                  name: letta-config
                  key: LETTA_PG_PORT
            - name: LETTA_PG_DB
              valueFrom:
                configMapKeyRef:
                  name: letta-config
                  key: LETTA_PG_DB
            - name: SECURE
              valueFrom:
                configMapKeyRef:
                  name: letta-config
                  key: SECURE
            - name: LETTA_SERVER_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: letta-secrets
                  key: server-password
            - name: LETTA_UVICORN_WORKERS
              valueFrom:
                configMapKeyRef:
                  name: letta-config
                  key: LETTA_UVICORN_WORKERS
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: letta-secrets
                  key: openai-key
            - name: ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: letta-secrets
                  key: anthropic-key
          
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
          
          livenessProbe:
            httpGet:
              path: /health
              port: 8283
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          
          readinessProbe:
            httpGet:
              path: /health
              port: 8283
            initialDelaySeconds: 10
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
          
          # Security context
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: false
            runAsNonRoot: true
            runAsUser: 1000
---
apiVersion: v1
kind: Service
metadata:
  name: letta-server
  namespace: letta
  labels:
    app: letta-server
spec:
  type: ClusterIP
  ports:
    - port: 8283
      targetPort: 8283
      name: http
  selector:
    app: letta-server
---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: letta-server-hpa
  namespace: letta
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: letta-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Pods
          value: 1
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
```

---

### Letta MCP Server Deployment

**File: `k8s/05-letta-mcp.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: letta-mcp
  namespace: letta
  labels:
    app: letta-mcp
    app.kubernetes.io/name: letta-mcp
    app.kubernetes.io/component: mcp-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: letta-mcp
  
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  
  template:
    metadata:
      labels:
        app: letta-mcp
        app.kubernetes.io/name: letta-mcp
    spec:
      serviceAccountName: letta-service-account
      
      # Wait for Letta server
      initContainers:
        - name: wait-for-letta
          image: curlimages/curl:8.1.2
          command:
            - sh
            - -c
            - |
              until curl -sf http://letta-server.letta.svc.cluster.local:8283/health; do
                echo "Waiting for Letta server..."
                sleep 5
              done
              echo "Letta server is ready"
      
      containers:
        - name: letta-mcp
          image: ghcr.io/oculairmedia/letta-mcp-server:latest
          imagePullPolicy: Always
          
          ports:
            - containerPort: 3001
              name: mcp
          
          env:
            - name: LETTA_BASE_URL
              value: "http://letta-server.letta.svc.cluster.local:8283/v1"
            - name: LETTA_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: letta-secrets
                  key: server-password
            - name: PORT
              value: "3001"
            - name: NODE_ENV
              value: "production"
          
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          
          livenessProbe:
            httpGet:
              path: /health
              port: 3001
            initialDelaySeconds: 15
            periodSeconds: 10
          
          readinessProbe:
            httpGet:
              path: /health
              port: 3001
            initialDelaySeconds: 5
            periodSeconds: 5
          
          securityContext:
            allowPrivilegeEscalation: false
            runAsNonRoot: true
            runAsUser: 1000
---
apiVersion: v1
kind: Service
metadata:
  name: letta-mcp
  namespace: letta
  labels:
    app: letta-mcp
spec:
  type: ClusterIP
  ports:
    - port: 3001
      targetPort: 3001
      name: mcp
  selector:
    app: letta-mcp
```

---

### Ingress Configuration

**File: `k8s/06-ingress.yaml`**

```yaml
# NGINX Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: letta-ingress
  namespace: letta
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    # Rate limiting
    nginx.ingress.kubernetes.io/limit-rps: "100"
    nginx.ingress.kubernetes.io/limit-connections: "50"
    # CORS
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "*"
    # Certificate management with cert-manager
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
    - hosts:
        - letta.yourdomain.com
        - mcp.yourdomain.com
      secretName: letta-tls-secret
  rules:
    - host: letta.yourdomain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: letta-server
                port:
                  number: 8283
    - host: mcp.yourdomain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: letta-mcp
                port:
                  number: 3001
---
# Network Policy for security
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: letta-network-policy
  namespace: letta
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow from ingress controller
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - port: 8283
        - port: 3001
    # Allow internal communication
    - from:
        - podSelector: {}
  egress:
    # Allow DNS
    - to:
        - namespaceSelector: {}
      ports:
        - port: 53
          protocol: UDP
    # Allow HTTPS (for LLM APIs)
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - port: 443
```

---

### Pod Disruption Budget

**File: `k8s/07-pdb.yaml`**

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: letta-server-pdb
  namespace: letta
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: letta-server
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: letta-mcp-pdb
  namespace: letta
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: letta-mcp
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: postgresql-pdb
  namespace: letta
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: postgresql
```

---

## 3.3 Backup and Disaster Recovery

### Automated Backup CronJob

**File: `k8s/08-backup.yaml`**

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: letta-backup
  namespace: letta
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: letta-service-account
          containers:
            - name: backup
              image: postgres:15-alpine
              command:
                - /bin/sh
                - -c
                - |
                  set -e
                  
                  # Create backup directory
                  DATE=$(date +%Y%m%d_%H%M%S)
                  BACKUP_DIR="/backups/$DATE"
                  mkdir -p $BACKUP_DIR
                  
                  # Database backup
                  echo "Starting database backup..."
                  PGPASSWORD=$POSTGRES_PASSWORD pg_dump \
                    -h $POSTGRES_HOST \
                    -U $POSTGRES_USER \
                    -d $POSTGRES_DB \
                    -F c \
                    -f "$BACKUP_DIR/database.dump"
                  
                  # Compress backup
                  gzip "$BACKUP_DIR/database.dump"
                  
                  # Export agents (via Letta API)
                  echo "Exporting agents..."
                  for agent_id in $(curl -s -H "Authorization: Bearer $LETTA_PASSWORD" \
                    http://letta-server.letta.svc.cluster.local:8283/v1/agents | \
                    jq -r '.[].id'); do
                    
                    curl -s -H "Authorization: Bearer $LETTA_PASSWORD" \
                      "http://letta-server.letta.svc.cluster.local:8283/v1/agents/$agent_id/export" \
                      > "$BACKUP_DIR/agent_${agent_id}.json"
                  done
                  
                  # Upload to S3 (or other storage)
                  echo "Uploading to S3..."
                  aws s3 sync $BACKUP_DIR s3://$S3_BUCKET/letta-backups/$DATE/
                  
                  # Cleanup local
                  rm -rf $BACKUP_DIR
                  
                  echo "Backup completed: $DATE"
              
              env:
                - name: POSTGRES_HOST
                  value: "postgresql-letta.letta.svc.cluster.local"
                - name: POSTGRES_USER
                  valueFrom:
                    configMapKeyRef:
                      name: letta-config
                      key: LETTA_PG_USER
                - name: POSTGRES_PASSWORD
                  valueFrom:
                    secretKeyRef:
                      name: letta-secrets
                      key: pg-password
                - name: POSTGRES_DB
                  valueFrom:
                    configMapKeyRef:
                      name: letta-config
                      key: LETTA_PG_DB
                - name: LETTA_PASSWORD
                  valueFrom:
                    secretKeyRef:
                      name: letta-secrets
                      key: server-password
                - name: S3_BUCKET
                  value: "your-backup-bucket"
                - name: AWS_ACCESS_KEY_ID
                  valueFrom:
                    secretKeyRef:
                      name: aws-credentials
                      key: access-key
                - name: AWS_SECRET_ACCESS_KEY
                  valueFrom:
                    secretKeyRef:
                      name: aws-credentials
                      key: secret-key
              
              volumeMounts:
                - name: backup-volume
                  mountPath: /backups
          
          volumes:
            - name: backup-volume
              emptyDir: {}
          
          restartPolicy: OnFailure
```

---

### Restore Script

**File: `scripts/restore-letta.sh`**

```bash
#!/bin/bash
# Letta Backup Restore Script
# Usage: ./restore-letta.sh <backup_date>
# Example: ./restore-letta.sh 20240115_020000

set -e

BACKUP_DATE=${1:-$(ls -1 /backups | tail -1)}
BACKUP_DIR="/backups/$BACKUP_DATE"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "Backup not found: $BACKUP_DIR"
    echo "Available backups:"
    ls -1 /backups
    exit 1
fi

echo "=== Letta Restore ==="
echo "Backup: $BACKUP_DATE"
echo ""

# Confirm restoration
read -p "This will overwrite existing data. Continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Restore cancelled"
    exit 0
fi

# Scale down Letta server to prevent writes
echo "Scaling down Letta server..."
kubectl scale deployment letta-server -n letta --replicas=0
kubectl scale deployment letta-mcp -n letta --replicas=0

# Wait for pods to terminate
kubectl wait --for=delete pod -l app=letta-server -n letta --timeout=120s || true
kubectl wait --for=delete pod -l app=letta-mcp -n letta --timeout=120s || true

# Restore database
echo "Restoring database..."
gunzip -c "$BACKUP_DIR/database.dump.gz" | \
    PGPASSWORD=$POSTGRES_PASSWORD pg_restore \
    -h $POSTGRES_HOST \
    -U $POSTGRES_USER \
    -d $POSTGRES_DB \
    --clean \
    --if-exists

# Scale up Letta server
echo "Scaling up Letta server..."
kubectl scale deployment letta-server -n letta --replicas=3
kubectl scale deployment letta-mcp -n letta --replicas=2

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=letta-server -n letta --timeout=300s

# Verify agents
echo "Verifying agents..."
AGENT_COUNT=$(curl -s -H "Authorization: Bearer $LETTA_PASSWORD" \
    http://letta-server.letta.svc.cluster.local:8283/v1/agents | jq length)
echo "Found $AGENT_COUNT agents"

echo ""
echo "=== Restore Complete ==="
echo "Backup: $BACKUP_DATE"
echo "Agents: $AGENT_COUNT"
```

---

## 3.4 Monitoring and Observability

### Prometheus ServiceMonitor

**File: `k8s/09-monitoring.yaml`**

```yaml
# Prometheus ServiceMonitor for Letta
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: letta-server-monitor
  namespace: letta
  labels:
    release: prometheus  # Match your Prometheus operator selector
spec:
  selector:
    matchLabels:
      app: letta-server
  endpoints:
    - port: http
      path: /metrics
      interval: 30s
      scrapeTimeout: 10s
---
# PrometheusRule for alerts
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: letta-alerts
  namespace: letta
spec:
  groups:
    - name: letta.rules
      rules:
        - alert: LettaServerDown
          expr: up{job="letta-server"} == 0
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "Letta server is down"
            description: "Letta server {{ $labels.instance }} has been down for more than 5 minutes."
        
        - alert: LettaHighLatency
          expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="letta-server"}[5m])) > 2
          for: 10m
          labels:
            severity: warning
          annotations:
            summary: "Letta server high latency"
            description: "95th percentile latency is above 2 seconds."
        
        - alert: LettaDatabaseConnectionError
          expr: increase(letta_database_errors_total[5m]) > 10
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "Letta database connection errors"
            description: "Database connection errors increasing."
        
        - alert: LettaHighMemoryUsage
          expr: container_memory_usage_bytes{container="letta-server"} / container_spec_memory_limit_bytes{container="letta-server"} > 0.9
          for: 10m
          labels:
            severity: warning
          annotations:
            summary: "Letta server high memory usage"
            description: "Memory usage is above 90% of limit."
---
# Grafana Dashboard ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: letta-grafana-dashboard
  namespace: letta
  labels:
    grafana_dashboard: "1"
data:
  letta-dashboard.json: |
    {
      "dashboard": {
        "title": "Letta Memory System",
        "panels": [
          {
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(http_requests_total{job=\"letta-server\"}[5m])",
                "legendFormat": "{{method}} {{path}}"
              }
            ]
          },
          {
            "title": "Memory Operations",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(letta_memory_operations_total[5m])",
                "legendFormat": "{{operation}}"
              }
            ]
          },
          {
            "title": "Agent Count",
            "type": "stat",
            "targets": [
              {
                "expr": "letta_agents_total"
              }
            ]
          },
          {
            "title": "Archival Passages",
            "type": "stat",
            "targets": [
              {
                "expr": "letta_archival_passages_total"
              }
            ]
          }
        ]
      }
    }
```

---

## 3.5 Deployment Scripts

### Deploy Script

**File: `scripts/deploy-letta.sh`**

```bash
#!/bin/bash
# Letta Kubernetes Deployment Script
# Usage: ./deploy-letta.sh [environment]
# Example: ./deploy-letta.sh production

set -e

ENVIRONMENT=${1:-production}
NAMESPACE="letta"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$SCRIPT_DIR/../k8s"

echo "=== Letta Kubernetes Deployment ==="
echo "Environment: $ENVIRONMENT"
echo "Namespace: $NAMESPACE"
echo ""

# Check prerequisites
command -v kubectl >/dev/null 2>&1 || { echo "kubectl required but not installed"; exit 1; }
command -v helm >/dev/null 2>&1 || { echo "helm required but not installed"; exit 1; }

# Verify cluster connection
kubectl cluster-info || { echo "Cannot connect to Kubernetes cluster"; exit 1; }

# Create namespace if not exists
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply manifests in order
echo "Applying Kubernetes manifests..."

kubectl apply -f "$K8S_DIR/00-namespace.yaml"
echo "✓ Namespace and RBAC"

kubectl apply -f "$K8S_DIR/01-secrets.yaml"
echo "✓ Secrets (ensure you've configured these properly)"

kubectl apply -f "$K8S_DIR/02-configmap.yaml"
echo "✓ ConfigMaps"

kubectl apply -f "$K8S_DIR/03-postgresql.yaml"
echo "✓ PostgreSQL StatefulSet"

# Wait for PostgreSQL
echo "Waiting for PostgreSQL..."
kubectl rollout status statefulset/postgresql-letta -n $NAMESPACE --timeout=300s

kubectl apply -f "$K8S_DIR/04-letta-server.yaml"
echo "✓ Letta Server Deployment"

# Wait for Letta server
echo "Waiting for Letta server..."
kubectl rollout status deployment/letta-server -n $NAMESPACE --timeout=300s

kubectl apply -f "$K8S_DIR/05-letta-mcp.yaml"
echo "✓ Letta MCP Server"

kubectl apply -f "$K8S_DIR/06-ingress.yaml"
echo "✓ Ingress"

kubectl apply -f "$K8S_DIR/07-pdb.yaml"
echo "✓ Pod Disruption Budgets"

kubectl apply -f "$K8S_DIR/08-backup.yaml"
echo "✓ Backup CronJob"

kubectl apply -f "$K8S_DIR/09-monitoring.yaml"
echo "✓ Monitoring"

# Verify deployment
echo ""
echo "=== Deployment Status ==="
kubectl get pods -n $NAMESPACE
echo ""
kubectl get services -n $NAMESPACE
echo ""
kubectl get ingress -n $NAMESPACE

# Health check
echo ""
echo "=== Health Check ==="
kubectl exec -n $NAMESPACE deployment/letta-server -- curl -sf http://localhost:8283/health || echo "Health check pending..."

echo ""
echo "=== Deployment Complete ==="
echo "Letta Server: http://letta-server.$NAMESPACE.svc.cluster.local:8283"
echo "Letta MCP: http://letta-mcp.$NAMESPACE.svc.cluster.local:3001"
```

---

# Part 4: Official Documentation References

## Letta Documentation

| Topic | URL |
|-------|-----|
| **Getting Started** | https://docs.letta.com/introduction |
| **Memory Concepts** | https://docs.letta.com/concepts/memory |
| **Server Deployment** | https://docs.letta.com/server/docker |
| **Python SDK** | https://docs.letta.com/clients/python |
| **API Reference** | https://docs.letta.com/api-reference |
| **Agent Management API** | https://docs.letta.com/api-reference/agents |
| **Memory Blocks API** | https://docs.letta.com/api-reference/agents/memory |
| **Archival Memory API** | https://docs.letta.com/api-reference/agents/archival-memory |
| **Blocks API** | https://docs.letta.com/api-reference/blocks |

## Claude Code Documentation

| Topic | URL |
|-------|-----|
| **Introduction** | https://docs.anthropic.com/en/docs/claude-code/introduction |
| **Configuration** | https://docs.anthropic.com/en/docs/claude-code/settings |
| **Hooks System** | https://docs.anthropic.com/en/docs/claude-code/hooks |
| **Custom Commands** | https://docs.anthropic.com/en/docs/claude-code/custom-commands |
| **Skills** | https://docs.anthropic.com/en/docs/claude-code/skills |
| **MCP Integration** | https://docs.anthropic.com/en/docs/claude-code/mcp |
| **CLAUDE.md** | https://docs.anthropic.com/en/docs/claude-code/claude-md |

## Kubernetes Documentation

| Topic | URL |
|-------|-----|
| **Deployments** | https://kubernetes.io/docs/concepts/workloads/controllers/deployment/ |
| **StatefulSets** | https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/ |
| **Services** | https://kubernetes.io/docs/concepts/services-networking/service/ |
| **Ingress** | https://kubernetes.io/docs/concepts/services-networking/ingress/ |
| **ConfigMaps & Secrets** | https://kubernetes.io/docs/concepts/configuration/ |
| **Resource Management** | https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/ |
| **HPA** | https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/ |
| **Network Policies** | https://kubernetes.io/docs/concepts/services-networking/network-policies/ |

## Related Technologies

| Technology | Documentation |
|------------|---------------|
| **PostgreSQL** | https://www.postgresql.org/docs/ |
| **pgvector** | https://github.com/pgvector/pgvector |
| **Model Context Protocol** | https://modelcontextprotocol.io/ |
| **External Secrets Operator** | https://external-secrets.io/ |
| **Prometheus** | https://prometheus.io/docs/ |
| **Grafana** | https://grafana.com/docs/ |

---

## Quick Reference Commands

```bash
# Deploy everything
./scripts/deploy-letta.sh production

# Check status
kubectl get all -n letta

# View logs
kubectl logs -f deployment/letta-server -n letta

# Scale up/down
kubectl scale deployment letta-server -n letta --replicas=5

# Force restart
kubectl rollout restart deployment/letta-server -n letta

# Port forward for local access
kubectl port-forward svc/letta-server 8283:8283 -n letta

# Backup manually
kubectl create job --from=cronjob/letta-backup manual-backup-$(date +%s) -n letta

# Restore
./scripts/restore-letta.sh 20240115_020000
```

---

*This comprehensive guide provides production-ready patterns for integrating Letta memory with Claude Code, complete with detailed Python hook scripts, advanced memory patterns, and full Kubernetes deployment manifests.*
