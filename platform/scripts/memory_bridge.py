#!/usr/bin/env python3
"""
Memory Bridge - Unified Memory Access Across Systems
=====================================================

Connects Auto-Claude's Graphiti memory, Letta persistence, and Claude Code's
local memory (CLAUDE.local.md) into a seamless integration.

This bridge enables:
- Cross-session knowledge sharing
- Pattern and gotcha retrieval
- Insights injection into Claude Code context

Usage:
    python memory_bridge.py --query "task description" --project-id "project-name"
    python memory_bridge.py --sync --source graphiti --target letta
    python memory_bridge.py --inject-context --output ~/.claude/CLAUDE.local.md
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add Auto-Claude paths for imports
AUTO_CLAUDE_PATH = Path("Z:/insider/AUTO CLAUDE/unleash/auto-claude/apps/backend")
if AUTO_CLAUDE_PATH.exists():
    sys.path.insert(0, str(AUTO_CLAUDE_PATH))
    sys.path.insert(0, str(AUTO_CLAUDE_PATH / "integrations" / "graphiti"))

# Try importing memory systems
GRAPHITI_AVAILABLE = False
LETTA_AVAILABLE = False

try:
    from memory import GraphitiMemory, get_graphiti_memory, is_graphiti_enabled
    from config import GraphitiConfig
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False

try:
    from letta_client import Letta
    LETTA_AVAILABLE = True
except ImportError:
    try:
        import httpx
        LETTA_AVAILABLE = True  # Can use HTTP API
    except ImportError:
        LETTA_AVAILABLE = False


class MemoryBridge:
    """Unified memory access across Graphiti, Letta, and file-based systems."""

    def __init__(
        self,
        project_dir: Path | None = None,
        letta_url: str = "http://localhost:8283",
    ):
        self.project_dir = project_dir or Path.cwd()
        self.letta_url = letta_url
        self._graphiti: GraphitiMemory | None = None
        self._letta_client: Any = None

    @property
    def graphiti_enabled(self) -> bool:
        """Check if Graphiti is available and enabled."""
        if not GRAPHITI_AVAILABLE:
            return False
        try:
            return is_graphiti_enabled()
        except Exception:
            return False

    @property
    def letta_enabled(self) -> bool:
        """Check if Letta server is reachable."""
        if not LETTA_AVAILABLE:
            return False
        try:
            import httpx
            response = httpx.get(f"{self.letta_url}/", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def get_graphiti(self, spec_dir: Path | None = None) -> "GraphitiMemory | None":
        """Get or create Graphiti memory instance."""
        if not self.graphiti_enabled:
            return None

        if self._graphiti is None:
            spec = spec_dir or self.project_dir / ".auto-claude"
            self._graphiti = get_graphiti_memory(
                spec_dir=spec,
                project_dir=self.project_dir,
                group_id_mode="project"  # Cross-spec learning
            )
        return self._graphiti

    async def query_graphiti(
        self,
        query: str,
        max_results: int = 10,
        include_patterns: bool = True,
        include_gotchas: bool = True,
    ) -> list[dict]:
        """Query Graphiti for relevant context."""
        if not self.graphiti_enabled:
            return []

        graphiti = self.get_graphiti()
        if not graphiti:
            return []

        try:
            # Get general context
            results = await graphiti.get_context(
                query=query,
                num_results=max_results,
            )

            # Optionally fetch patterns and gotchas
            if include_patterns or include_gotchas:
                episode_types = []
                if include_patterns:
                    episode_types.append("PATTERN")
                if include_gotchas:
                    episode_types.append("GOTCHA")

                for episode_type in episode_types:
                    episodes = await graphiti.search_episodes(
                        query=query,
                        episode_type=episode_type,
                        num_results=max_results // 2,
                    )
                    results.extend(episodes)

            return results
        except Exception as e:
            print(f"[WARNING] Graphiti query failed: {e}", file=sys.stderr)
            return []

    async def query_letta(
        self,
        agent_id: str | None = None,
        query: str | None = None,
    ) -> list[dict]:
        """Query Letta for agent memory."""
        if not self.letta_enabled:
            return []

        try:
            import httpx

            # List agents if no specific agent
            if not agent_id:
                response = httpx.get(f"{self.letta_url}/v1/agents/", timeout=10.0)
                if response.status_code == 200:
                    return response.json()
                return []

            # Get specific agent memory
            response = httpx.get(
                f"{self.letta_url}/v1/agents/{agent_id}/memory",
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            return []

        except Exception as e:
            print(f"[WARNING] Letta query failed: {e}", file=sys.stderr)
            return []

    def query_file_memory(self, memory_dir: Path | None = None) -> list[dict]:
        """Query file-based memory (fallback)."""
        mem_dir = memory_dir or self.project_dir / ".auto-claude" / "memory"

        if not mem_dir.exists():
            return []

        results = []
        for json_file in mem_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        results.extend(data)
                    else:
                        results.append(data)
            except Exception:
                continue

        return results

    async def unified_query(
        self,
        query: str,
        max_results: int = 20,
    ) -> dict[str, list[dict]]:
        """Query all available memory systems."""
        results = {
            "graphiti": [],
            "letta": [],
            "file": [],
            "sources": [],
        }

        # Query Graphiti (primary)
        if self.graphiti_enabled:
            results["graphiti"] = await self.query_graphiti(query, max_results // 2)
            if results["graphiti"]:
                results["sources"].append("graphiti")

        # Query Letta (secondary)
        if self.letta_enabled:
            results["letta"] = await self.query_letta(query=query)
            if results["letta"]:
                results["sources"].append("letta")

        # Query file-based (fallback)
        results["file"] = self.query_file_memory()
        if results["file"]:
            results["sources"].append("file")

        return results

    def format_for_claude_local_md(
        self,
        results: dict[str, list[dict]],
        max_items: int = 10,
    ) -> str:
        """Format memory results for CLAUDE.local.md injection."""
        lines = [
            "# Cross-Session Memory Context",
            f"> Generated: {datetime.now().isoformat()}",
            f"> Sources: {', '.join(results.get('sources', ['none']))}",
            "",
        ]

        # Graphiti patterns and gotchas
        if results.get("graphiti"):
            lines.append("## Patterns & Insights")
            for i, item in enumerate(results["graphiti"][:max_items]):
                if isinstance(item, dict):
                    name = item.get("name", item.get("title", f"Item {i+1}"))
                    content = item.get("content", item.get("summary", ""))
                    lines.append(f"- **{name}**: {content[:200]}")
            lines.append("")

        # Letta memory blocks
        if results.get("letta"):
            lines.append("## Agent Memory")
            for item in results["letta"][:max_items // 2]:
                if isinstance(item, dict):
                    name = item.get("name", "Memory")
                    lines.append(f"- {name}")
            lines.append("")

        # File-based context
        if results.get("file"):
            lines.append("## Historical Context")
            for item in results["file"][:max_items // 2]:
                if isinstance(item, dict):
                    summary = item.get("summary", item.get("content", ""))[:150]
                    lines.append(f"- {summary}")
            lines.append("")

        return "\n".join(lines)

    async def inject_context(
        self,
        query: str,
        output_path: Path | None = None,
    ) -> Path:
        """Query memory and inject into CLAUDE.local.md."""
        results = await self.unified_query(query)
        content = self.format_for_claude_local_md(results)

        output = output_path or self.project_dir / "CLAUDE.local.md"

        # Append to existing or create new
        if output.exists():
            existing = output.read_text()
            # Remove old memory section if present
            if "# Cross-Session Memory Context" in existing:
                parts = existing.split("# Cross-Session Memory Context")
                existing = parts[0].strip()

            content = f"{existing}\n\n{content}"

        output.write_text(content)
        return output

    def status(self) -> dict:
        """Get status of all memory systems."""
        return {
            "graphiti": {
                "available": GRAPHITI_AVAILABLE,
                "enabled": self.graphiti_enabled,
            },
            "letta": {
                "available": LETTA_AVAILABLE,
                "enabled": self.letta_enabled,
                "url": self.letta_url,
            },
            "file": {
                "available": True,
                "enabled": True,
                "path": str(self.project_dir / ".auto-claude" / "memory"),
            },
            "project_dir": str(self.project_dir),
        }


async def main():
    parser = argparse.ArgumentParser(description="Memory Bridge - Unified Memory Access")
    parser.add_argument("--query", type=str, help="Query to search across memory systems")
    parser.add_argument("--project-dir", type=str, help="Project directory")
    parser.add_argument("--inject-context", action="store_true", help="Inject context to CLAUDE.local.md")
    parser.add_argument("--output", type=str, help="Output path for context injection")
    parser.add_argument("--status", action="store_true", help="Show memory systems status")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    project_dir = Path(args.project_dir) if args.project_dir else Path.cwd()
    bridge = MemoryBridge(project_dir=project_dir)

    if args.status:
        status = bridge.status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("Memory Bridge Status")
            print("=" * 40)
            for system, info in status.items():
                if isinstance(info, dict):
                    print(f"\n{system.upper()}:")
                    for key, value in info.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"{system}: {info}")
        return

    if args.inject_context:
        query = args.query or "recent patterns and insights"
        output = Path(args.output) if args.output else None
        result_path = await bridge.inject_context(query, output)
        print(f"Context injected to: {result_path}")
        return

    if args.query:
        results = await bridge.unified_query(args.query)
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            formatted = bridge.format_for_claude_local_md(results)
            print(formatted)
        return

    parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
