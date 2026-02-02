#!/usr/bin/env python3
"""
Thinking Patterns Module - Sequential and Enhanced Thinking

This module contains thinking patterns for structured reasoning.
Extracted from hook_utils.py for modular architecture.

Exports:
- ThoughtData: Basic thought data structure
- ThinkingSession: Sequential thinking session
- ThoughtType: Types of sequential thoughts
- ThoughtBranch: Branching thought paths
- EnhancedThought: Thought with branching and revision
- BranchingThinkingSession: Advanced thinking with branches

Version: V1.0.0 (2026-01-30) - Extracted from hook_utils.py V10.11
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ThoughtData:
    """
    Sequential thought data structure.

    Based on MCP sequential-thinking server pattern.
    Supports thought chaining, revision, and branching.
    """
    thought: str
    thought_number: int
    total_thoughts: int
    next_thought_needed: bool = True
    is_revision: bool = False
    revises_thought: Optional[int] = None
    branch_from_thought: Optional[int] = None
    branch_id: Optional[str] = None
    needs_more_thoughts: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-compatible dict."""
        data: Dict[str, Any] = {
            "thought": self.thought,
            "thoughtNumber": self.thought_number,
            "totalThoughts": self.total_thoughts,
            "nextThoughtNeeded": self.next_thought_needed
        }
        if self.is_revision:
            data["isRevision"] = True
            if self.revises_thought:
                data["revisesThought"] = self.revises_thought
        if self.branch_from_thought:
            data["branchFromThought"] = self.branch_from_thought
        if self.branch_id:
            data["branchId"] = self.branch_id
        if self.needs_more_thoughts:
            data["needsMoreThoughts"] = True
        return data


class ThinkingSession:
    """
    Session for sequential thinking with history.

    Manages a chain of thoughts with optional JSONL persistence.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.thoughts: List[ThoughtData] = []
        self.created_at = datetime.now(timezone.utc).isoformat()

    def add_thought(self, thought: ThoughtData) -> None:
        """Add a thought to the session."""
        self.thoughts.append(thought)

    def get_last_thought(self) -> Optional[ThoughtData]:
        """Get the most recent thought."""
        return self.thoughts[-1] if self.thoughts else None

    def is_complete(self) -> bool:
        """Check if thinking session is complete."""
        last = self.get_last_thought()
        return last is not None and not last.next_thought_needed

    def save_to_jsonl(self, path: Path) -> None:
        """Save session history to JSONL file."""
        with open(path, "a") as f:
            for thought in self.thoughts:
                entry = {
                    "sessionId": self.session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **thought.to_dict()
                }
                f.write(json.dumps(entry) + "\n")


class ThoughtType(str, Enum):
    """Types of sequential thoughts."""
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    ANALYSIS = "analysis"
    CONCLUSION = "conclusion"
    QUESTION = "question"
    REVISION = "revision"
    BRANCH = "branch"


@dataclass
class ThoughtBranch:
    """
    A branch in sequential thinking.

    Represents an alternative line of reasoning.
    """
    id: str
    parent_thought_id: str
    name: str
    created_at: str
    merged: bool = False
    merged_into: Optional[str] = None


@dataclass
class EnhancedThought:
    """
    Enhanced sequential thought with branching and revision.

    Extends ThoughtData with branching and revision tracking.
    """
    id: str
    content: str
    thought_type: ThoughtType
    thought_number: int
    total_thoughts: int
    next_thought_needed: bool = True
    is_revision: bool = False
    revises_thought_id: Optional[str] = None
    branch_id: Optional[str] = None
    branch_from_thought_id: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON format."""
        return {
            "id": self.id,
            "content": self.content,
            "thoughtType": self.thought_type.value,
            "thoughtNumber": self.thought_number,
            "totalThoughts": self.total_thoughts,
            "nextThoughtNeeded": self.next_thought_needed,
            "isRevision": self.is_revision,
            "revisesThoughtId": self.revises_thought_id,
            "branchId": self.branch_id,
            "branchFromThoughtId": self.branch_from_thought_id,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class BranchingThinkingSession:
    """
    Thinking session with branching and revision support.

    Extends beyond simple sequential thinking to support
    branching thought paths and revision tracking.
    """
    session_id: str
    thoughts: List[EnhancedThought] = field(default_factory=list)
    branches: List[ThoughtBranch] = field(default_factory=list)
    current_branch_id: Optional[str] = None
    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def add_thought(self, thought: EnhancedThought) -> None:
        """Add a thought to the session."""
        thought.branch_id = self.current_branch_id
        self.thoughts.append(thought)

    def create_branch(
        self,
        name: str,
        from_thought_id: str
    ) -> ThoughtBranch:
        """Create a new branch from a thought."""
        branch = ThoughtBranch(
            id=str(uuid.uuid4()),
            parent_thought_id=from_thought_id,
            name=name,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        self.branches.append(branch)
        return branch

    def switch_branch(self, branch_id: str) -> None:
        """Switch to a different branch."""
        if any(b.id == branch_id for b in self.branches):
            self.current_branch_id = branch_id

    def get_branch_thoughts(self, branch_id: Optional[str] = None) -> List[EnhancedThought]:
        """Get thoughts for a specific branch."""
        target_branch = branch_id or self.current_branch_id
        return [t for t in self.thoughts if t.branch_id == target_branch]

    def revise_thought(
        self,
        thought_id: str,
        new_content: str,
        reason: str = ""
    ) -> EnhancedThought:
        """Create a revision of an existing thought."""
        original = next((t for t in self.thoughts if t.id == thought_id), None)
        if not original:
            raise ValueError(f"Thought {thought_id} not found")

        revision = EnhancedThought(
            id=str(uuid.uuid4()),
            content=new_content,
            thought_type=ThoughtType.REVISION,
            thought_number=len(self.thoughts) + 1,
            total_thoughts=original.total_thoughts + 1,
            is_revision=True,
            revises_thought_id=thought_id,
            branch_id=self.current_branch_id,
            metadata={"revision_reason": reason}
        )
        self.add_thought(revision)
        return revision

    def get_revision_chain(self, thought_id: str) -> List[EnhancedThought]:
        """Get all revisions of a thought."""
        chain = []
        current_id: Optional[str] = thought_id

        while current_id:
            thought = next((t for t in self.thoughts if t.id == current_id), None)
            if thought:
                chain.append(thought)
                # Find revision of this thought
                revision = next(
                    (t for t in self.thoughts if t.revises_thought_id == current_id),
                    None
                )
                current_id = revision.id if revision else None
            else:
                break

        return chain

    def merge_branches(self, source_branch_id: str, target_branch_id: Optional[str] = None) -> None:
        """Merge a branch into another (default: main branch)."""
        source_branch = next((b for b in self.branches if b.id == source_branch_id), None)
        if source_branch:
            source_branch.merged = True
            source_branch.merged_into = target_branch_id

    def get_main_chain(self) -> List[EnhancedThought]:
        """Get thoughts from the main (non-branched) chain."""
        return [t for t in self.thoughts if t.branch_id is None]

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "sessionId": self.session_id,
            "thoughts": [t.to_dict() for t in self.thoughts],
            "branches": [
                {
                    "id": b.id,
                    "parentThoughtId": b.parent_thought_id,
                    "name": b.name,
                    "createdAt": b.created_at,
                    "merged": b.merged,
                    "mergedInto": b.merged_into
                }
                for b in self.branches
            ],
            "currentBranchId": self.current_branch_id,
            "createdAt": self.created_at
        }


# Export all symbols
__all__ = [
    "ThoughtData",
    "ThinkingSession",
    "ThoughtType",
    "ThoughtBranch",
    "EnhancedThought",
    "BranchingThinkingSession",
]
