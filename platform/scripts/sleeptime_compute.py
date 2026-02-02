#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.26.0",
#     "pydantic>=2.0.0",
#     "structlog>=24.1.0",
# ]
# [tool.uv]
# exclude-newer = "2026-02-01"
# ///
"""
Sleep-Time Compute Module - V10 Ultimate Platform

Implements Letta's sleep-time compute paradigm:
- Background memory consolidation during idle periods
- Warm start context pre-computation
- Proactive insight generation
- Session continuity optimization

Based on: https://www.letta.com/blog/sleep-time-compute

Usage:
    uv run sleeptime_compute.py status           # Check sleep-time agent status
    uv run sleeptime_compute.py consolidate      # Trigger memory consolidation
    uv run sleeptime_compute.py warmstart        # Generate warm start context
    uv run sleeptime_compute.py insights         # Generate proactive insights
    uv run sleeptime_compute.py daemon           # Run as background service

Platform: Windows 11 + Python 3.11+
Architecture: V10 Optimized (Verified, Minimal, Seamless)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
V10_DIR = SCRIPT_DIR.parent
UNLEASH_DIR = V10_DIR.parent
DATA_DIR = V10_DIR / "data"
MEMORY_DIR = DATA_DIR / "memory"
INSIGHTS_DIR = DATA_DIR / "insights"

# Ensure directories exist
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Letta configuration
LETTA_URL = os.environ.get("LETTA_URL", "http://localhost:8500")
LETTA_AGENT_ID = os.environ.get("LETTA_AGENT_ID", "uap-sleeptime-agent")

# Windows compatibility
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class SleepPhase(str, Enum):
    """Sleep-time compute phases."""
    IDLE = "idle"               # Waiting for work
    CONSOLIDATING = "consolidating"  # Processing memories
    GENERATING = "generating"   # Creating insights
    WARMING = "warming"         # Pre-computing context


class MemoryType(str, Enum):
    """Types of memory blocks."""
    CONVERSATION = "conversation"   # Chat history
    ARCHIVAL = "archival"           # Long-term storage
    WORKING = "working"             # Current session context
    LEARNED = "learned"             # Derived insights


@dataclass
class MemoryBlock:
    """A block of agent memory."""
    id: str
    type: MemoryType
    content: str
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "embedding_hash": self.embedding_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryBlock":
        return cls(
            id=data["id"],
            type=MemoryType(data["type"]),
            content=data["content"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata=data.get("metadata", {}),
            embedding_hash=data.get("embedding_hash"),
        )


@dataclass
class Insight:
    """A proactively generated insight."""
    id: str
    topic: str
    content: str
    confidence: float
    source_memories: List[str]
    created_at: str
    expires_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "content": self.content,
            "confidence": self.confidence,
            "source_memories": self.source_memories,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "tags": self.tags,
        }


@dataclass
class WarmStartContext:
    """Pre-computed context for session warm start."""
    session_id: str
    project_context: str
    recent_decisions: List[str]
    active_tasks: List[str]
    learned_patterns: List[str]
    generated_at: str
    ttl_seconds: int = 3600  # 1 hour default

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "project_context": self.project_context,
            "recent_decisions": self.recent_decisions,
            "active_tasks": self.active_tasks,
            "learned_patterns": self.learned_patterns,
            "generated_at": self.generated_at,
            "ttl_seconds": self.ttl_seconds,
        }


@dataclass
class SleepTimeStatus:
    """Status of the sleep-time compute system."""
    phase: SleepPhase
    letta_connected: bool
    memory_blocks: int
    insights_generated: int
    last_consolidation: Optional[str]
    last_warmstart: Optional[str]
    uptime_seconds: float


# =============================================================================
# Memory Manager
# =============================================================================

class MemoryManager:
    """Manages memory blocks for sleep-time processing."""

    def __init__(self, memory_dir: Path = MEMORY_DIR):
        self.memory_dir = memory_dir
        self.blocks: Dict[str, MemoryBlock] = {}
        self._load_blocks()

    def _load_blocks(self):
        """Load memory blocks from disk."""
        for file in self.memory_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                block = MemoryBlock.from_dict(data)
                self.blocks[block.id] = block
            except Exception as e:
                logger.warning("Failed to load memory block", file=str(file), error=str(e))

    def save_block(self, block: MemoryBlock):
        """Save a memory block to disk."""
        self.blocks[block.id] = block
        file_path = self.memory_dir / f"{block.id}.json"
        file_path.write_text(json.dumps(block.to_dict(), indent=2), encoding="utf-8")

    def get_blocks_by_type(self, memory_type: MemoryType) -> List[MemoryBlock]:
        """Get all blocks of a specific type."""
        return [b for b in self.blocks.values() if b.type == memory_type]

    def create_block(
        self,
        memory_type: MemoryType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryBlock:
        """Create a new memory block."""
        now = datetime.now(timezone.utc).isoformat()
        block_id = hashlib.sha256(f"{now}:{content[:100]}".encode()).hexdigest()[:16]

        block = MemoryBlock(
            id=block_id,
            type=memory_type,
            content=content,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            embedding_hash=hashlib.md5(content.encode()).hexdigest(),
        )

        self.save_block(block)
        return block

    def consolidate(self) -> List[MemoryBlock]:
        """Consolidate working memory into learned context."""
        working = self.get_blocks_by_type(MemoryType.WORKING)
        if not working:
            return []

        # Group by topic/project
        topics: Dict[str, List[str]] = {}
        for block in working:
            topic = block.metadata.get("topic", "general")
            if topic not in topics:
                topics[topic] = []
            topics[topic].append(block.content)

        # Create consolidated learned blocks
        consolidated = []
        for topic, contents in topics.items():
            combined = "\n---\n".join(contents)
            summary = self._summarize(combined)

            learned_block = self.create_block(
                memory_type=MemoryType.LEARNED,
                content=summary,
                metadata={"topic": topic, "source_count": len(contents)},
            )
            consolidated.append(learned_block)

        return consolidated

    def _summarize(self, content: str) -> str:
        """Summarize content (placeholder for LLM call)."""
        # In production, this would call Letta or Claude for summarization
        lines = content.split("\n")
        if len(lines) > 10:
            return "\n".join(lines[:5]) + f"\n... ({len(lines) - 5} more lines consolidated)"
        return content


# =============================================================================
# Insight Generator
# =============================================================================

class InsightGenerator:
    """Generates proactive insights during sleep time."""

    def __init__(self, memory_manager: MemoryManager, insights_dir: Path = INSIGHTS_DIR):
        self.memory = memory_manager
        self.insights_dir = insights_dir
        self.insights: List[Insight] = []
        self._load_insights()

    def _load_insights(self):
        """Load existing insights from disk."""
        for file in self.insights_dir.glob("insight_*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                self.insights.append(Insight(**data))
            except Exception as e:
                logger.warning("Failed to load insight", file=str(file), error=str(e))

    def save_insight(self, insight: Insight):
        """Save an insight to disk."""
        self.insights.append(insight)
        file_path = self.insights_dir / f"insight_{insight.id}.json"
        file_path.write_text(json.dumps(insight.to_dict(), indent=2), encoding="utf-8")

    def generate_insights(self) -> List[Insight]:
        """Generate insights from learned memory blocks."""
        learned = self.memory.get_blocks_by_type(MemoryType.LEARNED)
        if not learned:
            return []

        generated = []
        now = datetime.now(timezone.utc).isoformat()

        # Pattern detection: Find recurring themes
        all_content = " ".join(b.content for b in learned)
        patterns = self._detect_patterns(all_content)

        for pattern, confidence in patterns:
            insight_id = hashlib.sha256(f"{now}:{pattern}".encode()).hexdigest()[:12]
            insight = Insight(
                id=insight_id,
                topic="pattern_detection",
                content=pattern,
                confidence=confidence,
                source_memories=[b.id for b in learned],
                created_at=now,
                tags=["auto-generated", "pattern"],
            )
            self.save_insight(insight)
            generated.append(insight)

        return generated

    def _detect_patterns(self, content: str) -> List[tuple[str, float]]:
        """Detect patterns in content (placeholder for ML analysis)."""
        # In production, this would use embeddings and clustering
        patterns = []

        # Simple keyword frequency analysis
        keywords = ["error", "fix", "improve", "pattern", "architecture", "memory"]
        words = content.lower().split()
        word_count = len(words)

        for keyword in keywords:
            count = words.count(keyword)
            if count > 0:
                frequency = count / max(word_count, 1)
                if frequency > 0.01:  # At least 1% occurrence
                    patterns.append((
                        f"Recurring theme: '{keyword}' appears {count} times",
                        min(frequency * 10, 0.95)  # Scale to confidence
                    ))

        return patterns[:5]  # Top 5 patterns


# =============================================================================
# Warm Start Engine
# =============================================================================

class WarmStartEngine:
    """Pre-computes context for fast session starts."""

    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager
        self.cache_file = DATA_DIR / "warmstart_cache.json"

    def generate_warmstart(self, project_name: str = "default") -> WarmStartContext:
        """Generate warm start context for a project."""
        now = datetime.now(timezone.utc).isoformat()
        session_id = hashlib.sha256(f"{now}:{project_name}".encode()).hexdigest()[:16]

        # Gather recent decisions from learned memory
        learned = self.memory.get_blocks_by_type(MemoryType.LEARNED)
        recent_decisions = [
            b.content[:200] for b in sorted(
                learned,
                key=lambda x: x.updated_at,
                reverse=True
            )[:5]
        ]

        # Extract active tasks from working memory
        working = self.memory.get_blocks_by_type(MemoryType.WORKING)
        active_tasks = [
            b.metadata.get("task", "Unknown task")
            for b in working
            if b.metadata.get("status") == "active"
        ]

        # Generate project context summary
        all_content = " ".join(b.content for b in learned[-10:])
        project_context = self._generate_context_summary(all_content, project_name)

        # Extract learned patterns
        learned_patterns = [
            f"Pattern: {b.metadata.get('topic', 'general')}"
            for b in learned
            if b.metadata.get("source_count", 0) > 1
        ]

        context = WarmStartContext(
            session_id=session_id,
            project_context=project_context,
            recent_decisions=recent_decisions,
            active_tasks=active_tasks,
            learned_patterns=learned_patterns,
            generated_at=now,
        )

        # Cache the context
        self._cache_context(context)

        return context

    def _generate_context_summary(self, content: str, project_name: str) -> str:
        """Generate a context summary (placeholder for LLM)."""
        word_count = len(content.split())
        return (
            f"Project: {project_name}\n"
            f"Context size: {word_count} words from {len(self.memory.blocks)} memory blocks\n"
            f"Ready for warm start with pre-computed context."
        )

    def _cache_context(self, context: WarmStartContext):
        """Cache warm start context to disk."""
        self.cache_file.write_text(
            json.dumps(context.to_dict(), indent=2),
            encoding="utf-8"
        )

    def get_cached_context(self) -> Optional[WarmStartContext]:
        """Retrieve cached warm start context if valid."""
        if not self.cache_file.exists():
            return None

        try:
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            context = WarmStartContext(**data)

            # Check TTL
            generated = datetime.fromisoformat(context.generated_at.replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - generated).total_seconds()

            if age > context.ttl_seconds:
                return None  # Expired

            return context
        except Exception:
            return None


# =============================================================================
# Sleep-Time Daemon
# =============================================================================

class SleepTimeDaemon:
    """Background service for sleep-time compute."""

    def __init__(self):
        self.memory = MemoryManager()
        self.insights = InsightGenerator(self.memory)
        self.warmstart_engine = WarmStartEngine(self.memory)
        self.phase = SleepPhase.IDLE
        self.start_time = time.time()
        self.last_consolidation: Optional[str] = None
        self.last_warmstart: Optional[str] = None

    async def check_letta_connection(self) -> bool:
        """Check if Letta server is reachable."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{LETTA_URL}/v1/health")
                return response.status_code == 200
        except Exception:
            return False

    async def get_status(self) -> SleepTimeStatus:
        """Get current status of sleep-time system."""
        letta_connected = await self.check_letta_connection()
        return SleepTimeStatus(
            phase=self.phase,
            letta_connected=letta_connected,
            memory_blocks=len(self.memory.blocks),
            insights_generated=len(self.insights.insights),
            last_consolidation=self.last_consolidation,
            last_warmstart=self.last_warmstart,
            uptime_seconds=time.time() - self.start_time,
        )

    async def consolidate(self) -> List[MemoryBlock]:
        """Run memory consolidation."""
        self.phase = SleepPhase.CONSOLIDATING
        try:
            consolidated = self.memory.consolidate()
            self.last_consolidation = datetime.now(timezone.utc).isoformat()
            return consolidated
        finally:
            self.phase = SleepPhase.IDLE

    async def generate_insights(self) -> List[Insight]:
        """Generate proactive insights."""
        self.phase = SleepPhase.GENERATING
        try:
            return self.insights.generate_insights()
        finally:
            self.phase = SleepPhase.IDLE

    async def generate_warmstart(self, project: str = "default") -> WarmStartContext:
        """Generate warm start context."""
        self.phase = SleepPhase.WARMING
        try:
            context = self.warmstart_engine.generate_warmstart(project)
            self.last_warmstart = datetime.now(timezone.utc).isoformat()
            return context
        finally:
            self.phase = SleepPhase.IDLE

    async def run_daemon(self, interval_seconds: int = 300):
        """Run as background daemon."""
        print("=" * 60)
        print("SLEEP-TIME COMPUTE DAEMON - V10 Ultimate Platform")
        print("=" * 60)
        print(f"Memory dir: {MEMORY_DIR}")
        print(f"Insights dir: {INSIGHTS_DIR}")
        print(f"Letta URL: {LETTA_URL}")
        print(f"Check interval: {interval_seconds}s")
        print("-" * 60)

        cycle = 0
        while True:
            cycle += 1
            print(f"\n[Cycle {cycle}] {datetime.now().strftime('%H:%M:%S')}")

            # Step 1: Consolidate memories
            print("  [1/3] Consolidating memories...", end=" ")
            consolidated = await self.consolidate()
            print(f"{len(consolidated)} blocks")

            # Step 2: Generate insights
            print("  [2/3] Generating insights...", end=" ")
            insights = await self.generate_insights()
            print(f"{len(insights)} insights")

            # Step 3: Update warm start cache
            print("  [3/3] Updating warm start cache...", end=" ")
            context = await self.generate_warmstart()
            print(f"session {context.session_id[:8]}...")

            # Status summary
            status = await self.get_status()
            print(f"  Status: {status.memory_blocks} memories, {status.insights_generated} insights")

            # Wait for next cycle
            await asyncio.sleep(interval_seconds)


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sleep-Time Compute Module - V10 Ultimate Platform",
    )
    parser.add_argument(
        "command",
        choices=["status", "consolidate", "warmstart", "insights", "daemon"],
        help="Command to execute",
    )
    parser.add_argument(
        "--project",
        default="default",
        help="Project name for warm start",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Daemon check interval in seconds",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()
    daemon = SleepTimeDaemon()

    if args.command == "status":
        status = await daemon.get_status()
        if args.json:
            print(json.dumps({
                "phase": status.phase.value,
                "letta_connected": status.letta_connected,
                "memory_blocks": status.memory_blocks,
                "insights_generated": status.insights_generated,
                "last_consolidation": status.last_consolidation,
                "last_warmstart": status.last_warmstart,
                "uptime_seconds": status.uptime_seconds,
            }, indent=2))
        else:
            print("=" * 50)
            print("SLEEP-TIME COMPUTE STATUS")
            print("=" * 50)
            print(f"Phase:              {status.phase.value}")
            print(f"Letta Connected:    {'Yes' if status.letta_connected else 'No'}")
            print(f"Memory Blocks:      {status.memory_blocks}")
            print(f"Insights Generated: {status.insights_generated}")
            print(f"Last Consolidation: {status.last_consolidation or 'Never'}")
            print(f"Last Warm Start:    {status.last_warmstart or 'Never'}")
            print("=" * 50)

    elif args.command == "consolidate":
        print("Consolidating memories...")
        consolidated = await daemon.consolidate()
        print(f"Consolidated {len(consolidated)} memory blocks")
        if args.json:
            print(json.dumps([b.to_dict() for b in consolidated], indent=2))

    elif args.command == "warmstart":
        print(f"Generating warm start context for '{args.project}'...")
        context = await daemon.generate_warmstart(args.project)
        if args.json:
            print(json.dumps(context.to_dict(), indent=2))
        else:
            print(f"Session ID: {context.session_id}")
            print(f"Recent decisions: {len(context.recent_decisions)}")
            print(f"Active tasks: {len(context.active_tasks)}")
            print(f"Learned patterns: {len(context.learned_patterns)}")

    elif args.command == "insights":
        print("Generating insights...")
        insights = await daemon.generate_insights()
        print(f"Generated {len(insights)} insights")
        if args.json:
            print(json.dumps([i.to_dict() for i in insights], indent=2))

    elif args.command == "daemon":
        await daemon.run_daemon(args.interval)


if __name__ == "__main__":
    asyncio.run(main())
