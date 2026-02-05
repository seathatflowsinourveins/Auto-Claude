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
Sleep-Time Compute Module - V65 Ultimate Platform

Implements Letta's sleep-time compute paradigm:
- Background memory consolidation during idle periods
- Warm start context pre-computation
- Proactive insight generation
- Session continuity optimization
- Integration with Letta's native sleeptime API (V65)

Benefits (from Letta research):
- 91% latency reduction via async consolidation
- 90% token savings through intelligent summarization
- Zero latency impact during active sessions

Based on: https://www.letta.com/blog/sleep-time-compute

Usage:
    uv run sleeptime_compute.py status           # Check sleep-time agent status
    uv run sleeptime_compute.py consolidate      # Trigger memory consolidation
    uv run sleeptime_compute.py warmstart        # Generate warm start context
    uv run sleeptime_compute.py insights         # Generate proactive insights
    uv run sleeptime_compute.py daemon           # Run as background service
    uv run sleeptime_compute.py cleanup          # Clean up low-scoring memory blocks
    uv run sleeptime_compute.py cleanup-preview  # Preview cleanup without deleting
    uv run sleeptime_compute.py letta-sync       # Sync with Letta native sleeptime (V65)

Platform: Windows 11 + Python 3.11+
Architecture: V65 Optimized (Sleeptime Integration)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
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
LETTA_API_KEY = os.environ.get("LETTA_API_KEY", "")
LETTA_AGENT_ID = os.environ.get("LETTA_AGENT_ID", "uap-sleeptime-agent")
LETTA_SLEEPTIME_ENABLED = os.environ.get("LETTA_SLEEPTIME_ENABLED", "false").lower() == "true"
LETTA_SLEEPTIME_FREQUENCY = int(os.environ.get("LETTA_SLEEPTIME_FREQUENCY", "5"))

# Importance scoring threshold for consolidation (V60)
# Blocks below this score are filtered out during WORKING -> LEARNED promotion
IMPORTANCE_THRESHOLD = 0.3

# Memory cleanup configuration (V65)
# Maximum number of memory blocks to retain - prevents unbounded growth
MAX_MEMORY_BLOCKS = 500
# Minimum importance score to retain during cleanup
MIN_RETENTION_SCORE = 0.2

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
    consolidation_count: int = 0


# =============================================================================
# Memory Manager
# =============================================================================

class MemoryManager:
    """Manages memory blocks for sleep-time processing."""

    def __init__(self, memory_dir: Path = MEMORY_DIR):
        self.memory_dir = memory_dir
        self.blocks: Dict[str, MemoryBlock] = {}
        self.consolidation_count: int = 0
        self._load_blocks()

    def _load_blocks(self):
        """Load memory blocks from disk."""
        for file in self.memory_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                block = MemoryBlock.from_dict(data)
                self.blocks[block.id] = block
            except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
                logger.warning("Failed to load memory block", file=str(file), error=str(e))

    def save_block(self, block: MemoryBlock):
        """Save a memory block to disk."""
        self.blocks[block.id] = block
        file_path = self.memory_dir / f"{block.id}.json"
        file_path.write_text(json.dumps(block.to_dict(), indent=2), encoding="utf-8")

    def get_blocks_by_type(self, memory_type: MemoryType) -> List[MemoryBlock]:
        """Get all blocks of a specific type."""
        return [b for b in self.blocks.values() if b.type == memory_type]

    def _content_hash(self, content: str) -> str:
        """Compute MD5 hash for content deduplication."""
        return hashlib.md5(content.encode()).hexdigest()

    def _is_duplicate_content(self, content: str, memory_type: MemoryType) -> bool:
        """Check if a block with the same content hash already exists for the given type."""
        new_hash = self._content_hash(content)
        for block in self.blocks.values():
            if block.type == memory_type and block.embedding_hash == new_hash:
                return True
        return False

    def compute_importance_score(self, block: MemoryBlock) -> float:
        """Compute an importance score for a memory block (V60).

        The score combines three signals with fixed weights:
          - recency (weight 0.3): exponential decay at 5% per day
          - frequency (weight 0.4): normalized access count on log scale
          - confidence (weight 0.3): from block metadata (default 0.5)

        Args:
            block: The memory block to score.

        Returns:
            A float in [0, 1] representing the block's importance.
        """
        now = datetime.now(timezone.utc)

        # --- Recency: 0.95 ** age_days (5% daily decay) ---
        try:
            created = datetime.fromisoformat(block.created_at.replace("Z", "+00:00"))
            age_days = max((now - created).total_seconds() / 86400.0, 0.0)
        except (ValueError, TypeError):
            age_days = 0.0
        recency = 0.95 ** age_days

        # --- Frequency: normalized access count (log scale) ---
        # If no access_count in metadata, default to 1.0 (backward compat)
        access_count = block.metadata.get("access_count", 1)
        try:
            access_count = max(float(access_count), 1.0)
        except (ValueError, TypeError):
            access_count = 1.0
        # log1p normalizes: 1 access -> 0.69, 10 -> 2.4, 100 -> 4.6
        # Normalize to [0, 1] by capping at log1p(100) ~= 4.615
        frequency = min(math.log1p(access_count) / math.log1p(100), 1.0)

        # --- Confidence: from metadata (default 0.5) ---
        confidence_raw = block.metadata.get("confidence", 0.5)
        try:
            confidence = max(0.0, min(float(confidence_raw), 1.0))
        except (ValueError, TypeError):
            confidence = 0.5

        # --- Weighted combination ---
        score = 0.3 * recency + 0.4 * frequency + 0.3 * confidence

        return round(score, 4)

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

    def promote_to_learned(self, block: MemoryBlock) -> None:
        """Promote a WORKING block to LEARNED status in place.

        Updates the block type and updated_at timestamp, then persists
        the change to disk.
        """
        if block.type != MemoryType.WORKING:
            return
        block.type = MemoryType.LEARNED
        block.updated_at = datetime.now(timezone.utc).isoformat()
        block.metadata["promoted_from"] = "working"
        self.save_block(block)

    def consolidate(self) -> List[MemoryBlock]:
        """Consolidate working memory into learned context.

        After creating consolidated LEARNED blocks, promotes the original
        WORKING blocks to LEARNED so they are not re-processed on the
        next consolidation cycle.

        Returns:
            List of newly created consolidated LEARNED blocks.
        """
        working = self.get_blocks_by_type(MemoryType.WORKING)
        if not working:
            return []

        # Score all WORKING blocks and filter by importance threshold (V60)
        scored_working: List[tuple[MemoryBlock, float]] = []
        filtered_out: List[tuple[MemoryBlock, float]] = []
        for block in working:
            score = self.compute_importance_score(block)
            if score >= IMPORTANCE_THRESHOLD:
                scored_working.append((block, score))
            else:
                filtered_out.append((block, score))

        if filtered_out:
            logger.info(
                "Importance filter: blocks below threshold",
                threshold=IMPORTANCE_THRESHOLD,
                filtered_count=len(filtered_out),
                filtered_ids=[b.id for b, _ in filtered_out],
                filtered_scores=[s for _, s in filtered_out],
            )

        if not scored_working:
            logger.info(
                "No WORKING blocks passed importance threshold",
                threshold=IMPORTANCE_THRESHOLD,
                total_working=len(working),
                filtered_out=len(filtered_out),
            )
            return []

        # Group by topic/project
        topics: Dict[str, List[tuple[MemoryBlock, float]]] = {}
        for block, score in scored_working:
            topic = block.metadata.get("topic", "general")
            if topic not in topics:
                topics[topic] = []
            topics[topic].append((block, score))

        # Create consolidated learned blocks
        consolidated = []
        promoted_count = 0
        for topic, block_scores in topics.items():
            blocks = [b for b, _ in block_scores]
            avg_importance = sum(s for _, s in block_scores) / len(block_scores)
            contents = [b.content for b in blocks]
            combined = "\n---\n".join(contents)
            summary = self._summarize(combined)

            # Dedup check: skip if identical learned content already exists
            if self._is_duplicate_content(summary, MemoryType.LEARNED):
                logger.info(
                    "Skipping duplicate consolidated block",
                    topic=topic,
                    source_count=len(contents),
                )
            else:
                learned_block = self.create_block(
                    memory_type=MemoryType.LEARNED,
                    content=summary,
                    metadata={
                        "topic": topic,
                        "source_count": len(contents),
                        "source_ids": [b.id for b in blocks],
                        "importance_score": round(avg_importance, 4),
                    },
                )
                consolidated.append(learned_block)

            # Promote original WORKING blocks to LEARNED
            for block in blocks:
                self.promote_to_learned(block)
                promoted_count += 1

        self.consolidation_count += 1
        logger.info(
            "Consolidation complete",
            new_learned=len(consolidated),
            promoted=promoted_count,
            consolidation_count=self.consolidation_count,
        )

        # Run cleanup after consolidation to prevent unbounded growth (V65)
        deleted_count = self.cleanup()
        if deleted_count > 0:
            logger.info("Post-consolidation cleanup", deleted=deleted_count)

        return consolidated

    def cleanup(self, max_blocks: int = MAX_MEMORY_BLOCKS, min_score: float = MIN_RETENTION_SCORE) -> int:
        """Clean up old/low-scoring memory blocks to prevent unbounded growth (V65).

        Strategy:
        1. If total blocks <= max_blocks, only remove blocks with score < min_score
        2. If total blocks > max_blocks, remove lowest scoring blocks until at max_blocks

        Args:
            max_blocks: Maximum number of blocks to retain (default: 500)
            min_score: Minimum importance score to retain (default: 0.2)

        Returns:
            Number of blocks deleted
        """
        if not self.blocks:
            return 0

        # Score all blocks
        scored_blocks: list[tuple[str, float]] = []
        for block_id, block in self.blocks.items():
            score = self.compute_importance_score(block)
            scored_blocks.append((block_id, score))

        # Sort by score ascending (lowest first for deletion)
        scored_blocks.sort(key=lambda x: x[1])

        blocks_to_delete: list[str] = []
        total_blocks = len(scored_blocks)

        # First pass: mark blocks below min_score for deletion
        for block_id, score in scored_blocks:
            if score < min_score:
                blocks_to_delete.append(block_id)

        # Second pass: if still over max_blocks, delete more lowest-scoring blocks
        remaining_after_first_pass = total_blocks - len(blocks_to_delete)
        if remaining_after_first_pass > max_blocks:
            excess = remaining_after_first_pass - max_blocks
            # Find lowest scoring blocks not already marked
            for block_id, score in scored_blocks:
                if block_id not in blocks_to_delete:
                    blocks_to_delete.append(block_id)
                    excess -= 1
                    if excess <= 0:
                        break

        # Delete the blocks
        deleted_count = 0
        for block_id in blocks_to_delete:
            try:
                file_path = self.memory_dir / f"{block_id}.json"
                if file_path.exists():
                    file_path.unlink()
                del self.blocks[block_id]
                deleted_count += 1
            except OSError as e:
                logger.warning("Failed to delete memory block", block_id=block_id, error=str(e))

        if deleted_count > 0:
            logger.info(
                "Memory cleanup complete",
                deleted=deleted_count,
                remaining=len(self.blocks),
                max_blocks=max_blocks,
                min_score=min_score,
            )

        return deleted_count

    def get_cleanup_preview(self, max_blocks: int = MAX_MEMORY_BLOCKS, min_score: float = MIN_RETENTION_SCORE) -> dict:
        """Preview what cleanup would delete without actually deleting (V65).

        Returns:
            Dict with counts and details of what would be deleted
        """
        if not self.blocks:
            return {"would_delete": 0, "current_count": 0, "blocks": []}

        scored_blocks: list[tuple[str, float, MemoryBlock]] = []
        for block_id, block in self.blocks.items():
            score = self.compute_importance_score(block)
            scored_blocks.append((block_id, score, block))

        scored_blocks.sort(key=lambda x: x[1])

        blocks_to_delete: list[dict] = []
        total_blocks = len(scored_blocks)

        # Mark blocks below min_score
        for block_id, score, block in scored_blocks:
            if score < min_score:
                blocks_to_delete.append({
                    "id": block_id,
                    "score": round(score, 4),
                    "reason": "below_min_score",
                    "type": block.type.value,
                    "topic": block.metadata.get("topic", "unknown"),
                })

        # If still over max_blocks
        remaining = total_blocks - len(blocks_to_delete)
        if remaining > max_blocks:
            excess = remaining - max_blocks
            marked_ids = {b["id"] for b in blocks_to_delete}
            for block_id, score, block in scored_blocks:
                if block_id not in marked_ids:
                    blocks_to_delete.append({
                        "id": block_id,
                        "score": round(score, 4),
                        "reason": "over_limit",
                        "type": block.type.value,
                        "topic": block.metadata.get("topic", "unknown"),
                    })
                    excess -= 1
                    if excess <= 0:
                        break

        return {
            "would_delete": len(blocks_to_delete),
            "current_count": total_blocks,
            "remaining_after": total_blocks - len(blocks_to_delete),
            "max_blocks": max_blocks,
            "min_score": min_score,
            "blocks": blocks_to_delete,
        }

    def _summarize(self, content: str) -> str:
        """Summarize consolidated content by extracting structured insights.

        Parses iteration result lines to produce a meaningful summary
        with success/failure counts, date ranges, and trend information
        rather than naive truncation.
        """
        sections = [s.strip() for s in content.split("---") if s.strip()]
        if not sections:
            return content

        # Parse iteration results from each section
        iterations: list[dict[str, Any]] = []
        for section in sections:
            parsed = self._parse_iteration_section(section)
            if parsed:
                iterations.append(parsed)

        if not iterations:
            # Fallback: structured excerpt for non-iteration content
            lines = content.split("\n")
            unique_lines = list(dict.fromkeys(line.strip() for line in lines if line.strip()))
            if len(unique_lines) <= 10:
                return "\n".join(unique_lines)
            return (
                f"Consolidated {len(unique_lines)} entries:\n"
                + "\n".join(f"  - {line}" for line in unique_lines[:8])
                + f"\n  ... and {len(unique_lines) - 8} more"
            )

        # Build structured summary from parsed iterations
        total = len(iterations)
        successes = sum(1 for it in iterations if it.get("status") == "success")
        warnings = sum(1 for it in iterations if it.get("status") == "warning")
        failures = sum(1 for it in iterations if it.get("status") not in ("success", "warning"))
        iter_numbers = sorted(it.get("number", 0) for it in iterations if it.get("number"))

        range_str = ""
        if iter_numbers:
            range_str = f" (iterations {iter_numbers[0]}-{iter_numbers[-1]})"

        summary_parts = [
            f"Consolidated {total} iterations{range_str}:",
            f"  Successes: {successes}/{total}",
        ]
        if warnings:
            summary_parts.append(f"  Warnings: {warnings}/{total}")
        if failures:
            summary_parts.append(f"  Failures: {failures}/{total}")

        success_rate = (successes / total * 100) if total else 0
        summary_parts.append(f"  Success rate: {success_rate:.0f}%")

        # Add trend: is performance improving or degrading?
        if len(iterations) >= 3:
            recent = iterations[-3:]
            recent_successes = sum(1 for it in recent if it.get("status") == "success")
            if recent_successes == 3:
                summary_parts.append("  Trend: Stable (last 3 all succeeded)")
            elif recent_successes >= 2:
                summary_parts.append("  Trend: Mostly stable (2/3 recent succeeded)")
            else:
                summary_parts.append("  Trend: Degrading (recent failures detected)")

        return "\n".join(summary_parts)

    @staticmethod
    def _parse_iteration_section(section: str) -> Optional[Dict[str, Any]]:
        """Parse an iteration result line into structured data."""
        # Match patterns like "Iteration #170 [success]: 4 success, 0 warnings, 0 failed."
        m = re.match(
            r"Iteration\s+#(\d+)\s+\[(\w+)\]:\s*(\d+)\s+success,\s*(\d+)\s+warning\w*,\s*(\d+)\s+fail",
            section.strip(),
        )
        if m:
            return {
                "number": int(m.group(1)),
                "status": m.group(2),
                "success_count": int(m.group(3)),
                "warning_count": int(m.group(4)),
                "failure_count": int(m.group(5)),
            }
        return None


# =============================================================================
# Iteration Report Analyzer
# =============================================================================

REPORTS_DIR = DATA_DIR / "reports"


class IterationInsightAnalyzer:
    """Analyzes iteration reports to extract cross-session insights.

    Reads JSON reports from platform/data/reports/ and computes:
    - Consecutive success streaks
    - Recurring warning patterns
    - Performance improvement trends (duration, health counts)
    - Phase-level reliability statistics
    """

    def __init__(self, reports_dir: Path = REPORTS_DIR):
        self.reports_dir = reports_dir
        self._reports: List[Dict[str, Any]] = []

    def load_reports(self, limit: int = 200) -> int:
        """Load iteration reports from disk, most recent first.

        Args:
            limit: Maximum number of reports to load.

        Returns:
            Number of reports loaded.
        """
        if not self.reports_dir.exists():
            return 0

        files = sorted(self.reports_dir.glob("iteration_*.json"), reverse=True)
        self._reports = []
        for f in files[:limit]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                self._reports.append(data)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load report", file=str(f), error=str(e))

        # Reverse so index 0 is oldest within loaded window
        self._reports.reverse()
        return len(self._reports)

    def consecutive_success_streak(self) -> int:
        """Count how many consecutive recent iterations succeeded."""
        streak = 0
        for report in reversed(self._reports):
            if report.get("overall_status") == "success":
                streak += 1
            else:
                break
        return streak

    def recurring_warnings(self, min_occurrences: int = 3) -> List[Dict[str, Any]]:
        """Find warning messages that recur across iterations.

        Returns a list of dicts with 'message', 'count', 'phase', and
        'first_seen'/'last_seen' iteration numbers.
        """
        warning_tracker: Dict[str, Dict[str, Any]] = {}

        for report in self._reports:
            iter_num = report.get("iteration_number", 0)
            for phase in report.get("phases", []):
                if phase.get("status") == "warning":
                    msg = phase.get("message", "unknown")
                    phase_name = phase.get("phase", "unknown")
                    key = f"{phase_name}:{msg}"
                    if key not in warning_tracker:
                        warning_tracker[key] = {
                            "phase": phase_name,
                            "message": msg,
                            "count": 0,
                            "first_seen": iter_num,
                            "last_seen": iter_num,
                        }
                    warning_tracker[key]["count"] += 1
                    warning_tracker[key]["last_seen"] = iter_num

        return [
            w for w in warning_tracker.values()
            if w["count"] >= min_occurrences
        ]

    def improvement_trends(self, window: int = 20) -> Dict[str, Any]:
        """Analyze improvement trends over a sliding window.

        Compares the first half of the window to the second half to detect
        whether duration, success rate, or health counts are improving.
        """
        if len(self._reports) < 4:
            return {"sufficient_data": False}

        recent = self._reports[-window:]
        mid = len(recent) // 2
        first_half = recent[:mid]
        second_half = recent[mid:]

        def _avg_duration(reports: List[Dict[str, Any]]) -> float:
            durations = [r.get("total_duration_ms", 0) for r in reports]
            return sum(durations) / max(len(durations), 1)

        def _success_rate(reports: List[Dict[str, Any]]) -> float:
            if not reports:
                return 0.0
            ok = sum(1 for r in reports if r.get("overall_status") == "success")
            return ok / len(reports)

        def _avg_healthy(reports: List[Dict[str, Any]]) -> float:
            counts: list[int] = []
            for r in reports:
                for p in r.get("phases", []):
                    details = p.get("details", {})
                    summary = details.get("summary", {})
                    if "healthy" in summary:
                        counts.append(summary["healthy"])
            return sum(counts) / max(len(counts), 1)

        first_dur = _avg_duration(first_half)
        second_dur = _avg_duration(second_half)
        dur_change = ((second_dur - first_dur) / first_dur * 100) if first_dur else 0

        first_sr = _success_rate(first_half)
        second_sr = _success_rate(second_half)

        first_health = _avg_healthy(first_half)
        second_health = _avg_healthy(second_half)

        return {
            "sufficient_data": True,
            "window_size": len(recent),
            "duration_trend_pct": round(dur_change, 1),
            "duration_improving": dur_change < -5,
            "success_rate_first_half": round(first_sr * 100, 1),
            "success_rate_second_half": round(second_sr * 100, 1),
            "success_rate_improving": second_sr > first_sr,
            "avg_healthy_first_half": round(first_health, 1),
            "avg_healthy_second_half": round(second_health, 1),
        }

    def phase_reliability(self) -> Dict[str, Dict[str, Any]]:
        """Compute reliability statistics per phase across all loaded reports."""
        phase_stats: Dict[str, Dict[str, int]] = {}

        for report in self._reports:
            for phase in report.get("phases", []):
                name = phase.get("phase", "unknown")
                if name not in phase_stats:
                    phase_stats[name] = {"success": 0, "warning": 0, "failure": 0, "total": 0}
                phase_stats[name]["total"] += 1
                status = phase.get("status", "unknown")
                if status == "success":
                    phase_stats[name]["success"] += 1
                elif status == "warning":
                    phase_stats[name]["warning"] += 1
                else:
                    phase_stats[name]["failure"] += 1

        result: Dict[str, Dict[str, Any]] = {}
        for name, stats in phase_stats.items():
            total = stats["total"]
            result[name] = {
                **stats,
                "success_rate_pct": round(stats["success"] / total * 100, 1) if total else 0,
            }
        return result

    def generate_all_insights(self) -> List[Dict[str, Any]]:
        """Produce a list of structured insight dicts from iteration analysis."""
        if not self._reports:
            self.load_reports()
        if not self._reports:
            return []

        insights: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc).isoformat()
        total_reports = len(self._reports)

        # 1. Consecutive success streak
        streak = self.consecutive_success_streak()
        if streak >= 3:
            insights.append({
                "type": "success_streak",
                "title": f"Consecutive success streak: {streak} iterations",
                "content": (
                    f"The last {streak} iterations completed successfully without "
                    f"warnings or failures, demonstrating system stability."
                ),
                "confidence": min(0.5 + streak * 0.05, 0.98),
                "tags": ["stability", "streak", "positive"],
                "generated_at": now,
            })

        # 2. Recurring warnings
        warnings = self.recurring_warnings(min_occurrences=3)
        for w in warnings:
            insights.append({
                "type": "recurring_warning",
                "title": f"Recurring warning in {w['phase']}: {w['message'][:60]}",
                "content": (
                    f"Warning '{w['message']}' in phase '{w['phase']}' has occurred "
                    f"{w['count']} times (iterations {w['first_seen']}-{w['last_seen']}). "
                    f"This is a persistent issue that should be investigated."
                ),
                "confidence": min(0.6 + w["count"] * 0.03, 0.95),
                "tags": ["warning", "recurring", w["phase"]],
                "generated_at": now,
            })

        # 3. Improvement trends
        trends = self.improvement_trends()
        if trends.get("sufficient_data"):
            if trends["success_rate_improving"]:
                insights.append({
                    "type": "trend_improvement",
                    "title": "Success rate is improving",
                    "content": (
                        f"Success rate improved from {trends['success_rate_first_half']}% "
                        f"to {trends['success_rate_second_half']}% over the last "
                        f"{trends['window_size']} iterations."
                    ),
                    "confidence": 0.80,
                    "tags": ["trend", "improvement", "positive"],
                    "generated_at": now,
                })
            if trends["duration_improving"]:
                insights.append({
                    "type": "trend_duration",
                    "title": "Iteration duration is decreasing",
                    "content": (
                        f"Average iteration duration decreased by "
                        f"{abs(trends['duration_trend_pct'])}% in the recent window, "
                        f"indicating performance improvements."
                    ),
                    "confidence": 0.75,
                    "tags": ["trend", "performance", "positive"],
                    "generated_at": now,
                })

        # 4. Phase reliability
        phase_rel = self.phase_reliability()
        for phase_name, stats in phase_rel.items():
            if stats["success_rate_pct"] < 90 and stats["total"] >= 10:
                insights.append({
                    "type": "phase_reliability",
                    "title": f"Phase '{phase_name}' reliability below 90%",
                    "content": (
                        f"Phase '{phase_name}' has a {stats['success_rate_pct']}% success rate "
                        f"across {stats['total']} runs ({stats['warning']} warnings, "
                        f"{stats['failure']} failures). Consider investigating root causes."
                    ),
                    "confidence": 0.85,
                    "tags": ["reliability", phase_name, "action-needed"],
                    "generated_at": now,
                })

        # 5. Overall health summary
        if total_reports >= 5:
            overall_success = sum(
                1 for r in self._reports if r.get("overall_status") == "success"
            )
            overall_rate = overall_success / total_reports * 100
            insights.append({
                "type": "overall_health",
                "title": f"Overall system health: {overall_rate:.0f}% success across {total_reports} iterations",
                "content": (
                    f"Analyzed {total_reports} iteration reports. "
                    f"{overall_success} succeeded, "
                    f"{total_reports - overall_success} had warnings or failures. "
                    f"Current streak: {streak} consecutive successes."
                ),
                "confidence": 0.90,
                "tags": ["summary", "health"],
                "generated_at": now,
            })

        return insights


# =============================================================================
# Insight Generator
# =============================================================================

class InsightGenerator:
    """Generates proactive insights during sleep time."""

    def __init__(self, memory_manager: MemoryManager, insights_dir: Path = INSIGHTS_DIR):
        self.memory = memory_manager
        self.insights_dir = insights_dir
        self.insights: List[Insight] = []
        self.iteration_analyzer = IterationInsightAnalyzer()
        self._existing_content_hashes: set[str] = set()
        self._load_insights()

    def _load_insights(self):
        """Load existing insights from disk."""
        for file in self.insights_dir.glob("insight_*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                insight = Insight(**data)
                self.insights.append(insight)
                # Track content hash for dedup
                content_hash = hashlib.md5(insight.content.encode()).hexdigest()
                self._existing_content_hashes.add(content_hash)
            except (json.JSONDecodeError, KeyError, TypeError, OSError) as e:
                logger.warning("Failed to load insight", file=str(file), error=str(e))

    def _is_duplicate_insight(self, content: str) -> bool:
        """Check if an insight with the same content hash already exists."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return content_hash in self._existing_content_hashes

    def save_insight(self, insight: Insight):
        """Save an insight to disk if not a duplicate."""
        content_hash = hashlib.md5(insight.content.encode()).hexdigest()
        if content_hash in self._existing_content_hashes:
            logger.info("Skipping duplicate insight", insight_id=insight.id, topic=insight.topic)
            return
        self._existing_content_hashes.add(content_hash)
        self.insights.append(insight)
        file_path = self.insights_dir / f"insight_{insight.id}.json"
        file_path.write_text(json.dumps(insight.to_dict(), indent=2), encoding="utf-8")

    def generate_insights(self) -> List[Insight]:
        """Generate insights from learned memory blocks and iteration reports.

        Combines two sources:
        1. Learned memory blocks (keyword pattern detection)
        2. Iteration reports (streak, trend, and reliability analysis)

        Deduplicates against existing insights by content hash before saving.
        """
        generated: List[Insight] = []
        now = datetime.now(timezone.utc).isoformat()

        # --- Source 1: Learned memory block patterns ---
        learned = self.memory.get_blocks_by_type(MemoryType.LEARNED)
        if learned:
            all_content = " ".join(b.content for b in learned)
            patterns = self._detect_patterns(all_content)

            for pattern, confidence in patterns:
                if self._is_duplicate_insight(pattern):
                    logger.info("Skipping duplicate pattern insight", pattern=pattern[:60])
                    continue
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

        # --- Source 2: Iteration report analysis ---
        report_count = self.iteration_analyzer.load_reports()
        if report_count > 0:
            iteration_insights = self.iteration_analyzer.generate_all_insights()
            for idata in iteration_insights:
                content = idata["content"]
                if self._is_duplicate_insight(content):
                    logger.info("Skipping duplicate iteration insight", title=idata.get("title", "")[:60])
                    continue
                insight_id = hashlib.sha256(
                    f"{now}:{idata['title']}".encode()
                ).hexdigest()[:12]
                insight = Insight(
                    id=insight_id,
                    topic=idata.get("type", "iteration_analysis"),
                    content=content,
                    confidence=idata.get("confidence", 0.7),
                    source_memories=[],
                    created_at=now,
                    tags=idata.get("tags", ["iteration"]),
                )
                self.save_insight(insight)
                generated.append(insight)

        return generated

    def _detect_patterns(self, content: str) -> List[tuple[str, float]]:
        """Detect patterns in learned memory content."""
        patterns = []

        # Keyword frequency analysis
        keywords = ["error", "fix", "improve", "pattern", "architecture", "memory",
                     "success", "warning", "failure", "stable", "degrading"]
        words = content.lower().split()
        word_count = len(words)

        for keyword in keywords:
            count = words.count(keyword)
            if count > 0:
                frequency = count / max(word_count, 1)
                if frequency > 0.01:  # At least 1% occurrence
                    patterns.append((
                        f"Recurring theme: '{keyword}' appears {count} times "
                        f"({frequency * 100:.1f}% of content)",
                        min(frequency * 10, 0.95)
                    ))

        return sorted(patterns, key=lambda p: p[1], reverse=True)[:5]


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
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
            logger.warning("Failed to load warmstart cache", error=str(e))
            return None


# =============================================================================
# Sleep-Time Daemon
# =============================================================================

class LettaSleeptimeClient:
    """Client for Letta's native sleeptime API (V65).

    Provides integration with Letta's server-side sleeptime compute when available,
    enabling 91% latency reduction and 90% token savings through async consolidation.
    """

    def __init__(
        self,
        api_key: str = LETTA_API_KEY,
        base_url: str = LETTA_URL,
        agent_id: str = LETTA_AGENT_ID,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.agent_id = agent_id
        self._client = None
        self._available = False
        self._sleeptime_enabled = False
        self._last_sync: Optional[str] = None
        self._sync_count = 0

    async def initialize(self) -> bool:
        """Initialize connection to Letta API."""
        if not self.api_key:
            logger.debug("Letta API key not configured, native sleeptime unavailable")
            return False

        try:
            from letta_client import Letta  # type: ignore
            self._client = Letta(api_key=self.api_key, base_url=self.base_url)
            # Verify connection
            self._client.agents.list(limit=1)
            self._available = True
            logger.info("Letta native sleeptime client initialized")
            return True
        except ImportError:
            logger.debug("letta-client not installed, native sleeptime unavailable")
            return False
        except Exception as e:
            logger.warning("Failed to initialize Letta client: %s", e)
            return False

    async def check_sleeptime_status(self) -> Dict[str, Any]:
        """Check if sleeptime is enabled for the configured agent."""
        if not self._available or not self._client:
            return {"available": False, "reason": "Client not initialized"}

        try:
            agent = self._client.agents.get(agent_id=self.agent_id)
            self._sleeptime_enabled = getattr(agent, 'enable_sleeptime', False)
            group_id = getattr(agent, 'group_id', None)

            result = {
                "available": True,
                "agent_id": self.agent_id,
                "sleeptime_enabled": self._sleeptime_enabled,
                "group_id": group_id,
            }

            if group_id:
                try:
                    group = self._client.groups.get(group_id=group_id)
                    manager_config = getattr(group, 'manager_config', {}) or {}
                    result["sleeptime_frequency"] = manager_config.get(
                        "sleeptime_agent_frequency", LETTA_SLEEPTIME_FREQUENCY
                    )
                except (AttributeError, TypeError):
                    pass

            return result
        except Exception as e:
            return {"available": False, "reason": str(e)}

    async def enable_sleeptime(self, frequency: int = LETTA_SLEEPTIME_FREQUENCY) -> Dict[str, Any]:
        """Enable sleeptime compute for the configured agent."""
        if not self._available or not self._client:
            return {"success": False, "error": "Client not initialized"}

        try:
            # Update agent to enable sleeptime
            self._client.agents.update(self.agent_id, enable_sleeptime=True)
            self._sleeptime_enabled = True

            result = {
                "success": True,
                "agent_id": self.agent_id,
                "sleeptime_enabled": True,
            }

            # Try to set frequency
            agent = self._client.agents.get(agent_id=self.agent_id)
            group_id = getattr(agent, 'group_id', None)
            if group_id:
                try:
                    self._client.groups.update(
                        group_id,
                        manager_config={"sleeptime_agent_frequency": frequency}
                    )
                    result["sleeptime_frequency"] = frequency
                    result["group_id"] = group_id
                except (AttributeError, TypeError):
                    pass

            logger.info("Enabled Letta native sleeptime", agent_id=self.agent_id, frequency=frequency)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def trigger_consolidation(self, context: str = "") -> Dict[str, Any]:
        """Trigger immediate sleeptime consolidation."""
        if not self._available or not self._client:
            return {"success": False, "error": "Client not initialized"}

        if not self._sleeptime_enabled:
            return {"success": False, "error": "Sleeptime not enabled for this agent"}

        try:
            trigger_message = (
                f"[SLEEPTIME_CONSOLIDATION] {context}" if context else
                "[SLEEPTIME_CONSOLIDATION] Please consolidate recent memories."
            )

            response = self._client.agents.messages.create(
                agent_id=self.agent_id,
                messages=[{"role": "user", "content": trigger_message}]
            )

            self._sync_count += 1
            self._last_sync = datetime.now(timezone.utc).isoformat()

            messages = []
            for msg in getattr(response, 'messages', []):
                if hasattr(msg, 'assistant_message') and msg.assistant_message:
                    messages.append(msg.assistant_message)
                elif hasattr(msg, 'content'):
                    messages.append(msg.content)

            return {
                "success": True,
                "triggered": True,
                "response_count": len(messages),
                "sync_count": self._sync_count,
                "last_sync": self._last_sync,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_metrics(self) -> Dict[str, Any]:
        """Get sleeptime consolidation metrics."""
        return {
            "available": self._available,
            "sleeptime_enabled": self._sleeptime_enabled,
            "sync_count": self._sync_count,
            "last_sync": self._last_sync,
            "agent_id": self.agent_id if self._available else None,
        }


class SleepTimeDaemon:
    """Background service for sleep-time compute with Letta integration (V65)."""

    def __init__(self):
        self.memory = MemoryManager()
        self.insights = InsightGenerator(self.memory)
        self.warmstart_engine = WarmStartEngine(self.memory)
        self.letta_client = LettaSleeptimeClient()  # V65: Native Letta integration
        self.phase = SleepPhase.IDLE
        self.start_time = time.time()
        self.last_consolidation: Optional[str] = None
        self.last_warmstart: Optional[str] = None
        self._letta_initialized = False

    async def check_letta_connection(self) -> bool:
        """Check if Letta server is reachable."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{LETTA_URL}/v1/health")
                return response.status_code == 200
        except (ImportError, OSError, httpx.HTTPError) as e:
            logger.debug("Letta connection check failed", error=str(e))
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
            consolidation_count=self.memory.consolidation_count,
        )

    async def initialize_letta(self) -> bool:
        """Initialize Letta native sleeptime integration (V65)."""
        if self._letta_initialized:
            return self.letta_client._available
        self._letta_initialized = True
        return await self.letta_client.initialize()

    async def get_letta_sleeptime_status(self) -> Dict[str, Any]:
        """Get Letta native sleeptime status (V65)."""
        if not self._letta_initialized:
            await self.initialize_letta()
        return await self.letta_client.check_sleeptime_status()

    async def enable_letta_sleeptime(self, frequency: int = LETTA_SLEEPTIME_FREQUENCY) -> Dict[str, Any]:
        """Enable Letta native sleeptime for the configured agent (V65)."""
        if not self._letta_initialized:
            await self.initialize_letta()
        return await self.letta_client.enable_sleeptime(frequency)

    async def sync_with_letta(self, context: str = "") -> Dict[str, Any]:
        """Sync local memory with Letta's native sleeptime (V65).

        This triggers Letta's server-side sleeptime agent to consolidate
        memories, providing 91% latency reduction and 90% token savings.

        Args:
            context: Optional context to guide consolidation

        Returns:
            Dict with sync results and metrics
        """
        if not self._letta_initialized:
            await self.initialize_letta()

        if not self.letta_client._available:
            return {
                "success": False,
                "error": "Letta client not available",
                "fallback": "Using local consolidation",
            }

        # Check if sleeptime is enabled
        status = await self.letta_client.check_sleeptime_status()
        if not status.get("sleeptime_enabled"):
            # Try to enable it
            enable_result = await self.letta_client.enable_sleeptime()
            if not enable_result.get("success"):
                return {
                    "success": False,
                    "error": "Could not enable Letta sleeptime",
                    "details": enable_result,
                }

        # Trigger consolidation
        result = await self.letta_client.trigger_consolidation(context)

        if result.get("success"):
            # Also run local consolidation to keep in sync
            local_consolidated = self.memory.consolidate()
            result["local_consolidated"] = len(local_consolidated)
            self.last_consolidation = datetime.now(timezone.utc).isoformat()

        return result

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

    async def run_daemon(self, interval_seconds: int = 300, enable_letta_sync: bool = True):
        """Run as background daemon with optional Letta integration (V65)."""
        print("=" * 60)
        print("SLEEP-TIME COMPUTE DAEMON - V65 Ultimate Platform")
        print("=" * 60)
        print(f"Memory dir: {MEMORY_DIR}")
        print(f"Insights dir: {INSIGHTS_DIR}")
        print(f"Letta URL: {LETTA_URL}")
        print(f"Check interval: {interval_seconds}s")
        print(f"Letta sync enabled: {enable_letta_sync}")
        print("-" * 60)

        # Initialize Letta if enabled
        letta_available = False
        if enable_letta_sync:
            print("Initializing Letta native sleeptime...")
            letta_available = await self.initialize_letta()
            if letta_available:
                letta_status = await self.get_letta_sleeptime_status()
                print(f"  Letta connected: Yes")
                print(f"  Sleeptime enabled: {letta_status.get('sleeptime_enabled', False)}")
                print(f"  Frequency: {letta_status.get('sleeptime_frequency', 'N/A')} steps")
            else:
                print("  Letta not available, using local consolidation only")
        print("-" * 60)

        cycle = 0
        while True:
            cycle += 1
            print(f"\n[Cycle {cycle}] {datetime.now().strftime('%H:%M:%S')}")

            # Step 1: Consolidate memories (with optional Letta sync)
            if letta_available and enable_letta_sync:
                print("  [1/4] Syncing with Letta native sleeptime...", end=" ")
                sync_result = await self.sync_with_letta(
                    context=f"Daemon cycle {cycle}"
                )
                if sync_result.get("success"):
                    print(f"synced (local: {sync_result.get('local_consolidated', 0)} blocks)")
                else:
                    print(f"failed ({sync_result.get('error', 'unknown')}), using local")
                    consolidated = await self.consolidate()
                    print(f"    Local fallback: {len(consolidated)} blocks")
            else:
                print("  [1/4] Consolidating memories (local)...", end=" ")
                consolidated = await self.consolidate()
                print(f"{len(consolidated)} blocks")

            # Step 2: Generate insights
            print("  [2/4] Generating insights...", end=" ")
            insights = await self.generate_insights()
            print(f"{len(insights)} insights")

            # Step 3: Update warm start cache
            print("  [3/4] Updating warm start cache...", end=" ")
            context = await self.generate_warmstart()
            print(f"session {context.session_id[:8]}...")

            # Step 4: Get metrics
            print("  [4/4] Collecting metrics...", end=" ")
            status = await self.get_status()
            letta_metrics = await self.letta_client.get_metrics() if letta_available else {}
            print("done")

            # Status summary
            print(f"  Status: {status.memory_blocks} memories, "
                  f"{status.insights_generated} insights, "
                  f"{status.consolidation_count} consolidations")
            if letta_available:
                print(f"  Letta: {letta_metrics.get('sync_count', 0)} syncs, "
                      f"last: {letta_metrics.get('last_sync', 'never')[:19] if letta_metrics.get('last_sync') else 'never'}")

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
        choices=[
            "status", "consolidate", "warmstart", "insights", "daemon",
            "cleanup", "cleanup-preview", "letta-sync", "letta-status", "letta-enable"
        ],
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
    parser.add_argument(
        "--no-letta-sync",
        action="store_true",
        help="Disable Letta native sleeptime sync in daemon mode",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=LETTA_SLEEPTIME_FREQUENCY,
        help="Sleeptime frequency (steps between updates)",
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
                "consolidation_count": status.consolidation_count,
            }, indent=2))
        else:
            print("=" * 50)
            print("SLEEP-TIME COMPUTE STATUS")
            print("=" * 50)
            print(f"Phase:              {status.phase.value}")
            print(f"Letta Connected:    {'Yes' if status.letta_connected else 'No'}")
            print(f"Memory Blocks:      {status.memory_blocks}")
            print(f"Insights Generated: {status.insights_generated}")
            print(f"Consolidations:     {status.consolidation_count}")
            print(f"Last Consolidation: {status.last_consolidation or 'Never'}")
            print(f"Last Warm Start:    {status.last_warmstart or 'Never'}")
            print("=" * 50)

    elif args.command == "consolidate":
        print("Consolidating memories...")
        consolidated = await daemon.consolidate()
        print(f"Consolidated {len(consolidated)} memory blocks "
              f"(total consolidations: {daemon.memory.consolidation_count})")
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
        await daemon.run_daemon(args.interval, enable_letta_sync=not args.no_letta_sync)

    elif args.command == "cleanup":
        print(f"Running memory cleanup (max={MAX_MEMORY_BLOCKS}, min_score={MIN_RETENTION_SCORE})...")
        deleted = daemon.memory.cleanup()
        remaining = len(daemon.memory.blocks)
        print(f"Deleted {deleted} memory blocks. Remaining: {remaining}")
        if args.json:
            print(json.dumps({
                "deleted": deleted,
                "remaining": remaining,
                "max_blocks": MAX_MEMORY_BLOCKS,
                "min_score": MIN_RETENTION_SCORE,
            }, indent=2))

    elif args.command == "cleanup-preview":
        print(f"Previewing cleanup (max={MAX_MEMORY_BLOCKS}, min_score={MIN_RETENTION_SCORE})...")
        preview = daemon.memory.get_cleanup_preview()
        if args.json:
            print(json.dumps(preview, indent=2))
        else:
            print(f"Current blocks: {preview['current_count']}")
            print(f"Would delete: {preview['would_delete']}")
            print(f"Remaining after: {preview['remaining_after']}")
            if preview['blocks']:
                print("\nBlocks to delete:")
                for b in preview['blocks'][:20]:  # Show first 20
                    print(f"  {b['id']}: {b['type']} ({b['topic']}) score={b['score']} reason={b['reason']}")
                if len(preview['blocks']) > 20:
                    print(f"  ... and {len(preview['blocks']) - 20} more")

    elif args.command == "letta-status":
        print("Checking Letta native sleeptime status...")
        await daemon.initialize_letta()
        letta_status = await daemon.get_letta_sleeptime_status()
        if args.json:
            print(json.dumps(letta_status, indent=2))
        else:
            print("=" * 50)
            print("LETTA NATIVE SLEEPTIME STATUS")
            print("=" * 50)
            print(f"Available:          {letta_status.get('available', False)}")
            if letta_status.get('available'):
                print(f"Agent ID:           {letta_status.get('agent_id', 'N/A')}")
                print(f"Sleeptime Enabled:  {letta_status.get('sleeptime_enabled', False)}")
                print(f"Frequency:          {letta_status.get('sleeptime_frequency', 'N/A')} steps")
                print(f"Group ID:           {letta_status.get('group_id', 'N/A')}")
            else:
                print(f"Reason:             {letta_status.get('reason', 'Unknown')}")
            print("=" * 50)

    elif args.command == "letta-enable":
        print(f"Enabling Letta native sleeptime (frequency={args.frequency})...")
        await daemon.initialize_letta()
        result = await daemon.enable_letta_sleeptime(frequency=args.frequency)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result.get("success"):
                print("Successfully enabled Letta native sleeptime!")
                print(f"  Agent ID: {result.get('agent_id')}")
                print(f"  Frequency: {result.get('sleeptime_frequency', args.frequency)} steps")
                if result.get('group_id'):
                    print(f"  Group ID: {result.get('group_id')}")
            else:
                print(f"Failed to enable: {result.get('error', 'Unknown error')}")

    elif args.command == "letta-sync":
        print("Syncing with Letta native sleeptime...")
        result = await daemon.sync_with_letta(context="Manual sync via CLI")
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result.get("success"):
                print("Successfully synced with Letta!")
                print(f"  Triggered: {result.get('triggered', False)}")
                print(f"  Local consolidated: {result.get('local_consolidated', 0)} blocks")
                print(f"  Sync count: {result.get('sync_count', 0)}")
            else:
                print(f"Sync failed: {result.get('error', 'Unknown error')}")
                if result.get('fallback'):
                    print(f"  Fallback: {result.get('fallback')}")


if __name__ == "__main__":
    asyncio.run(main())
