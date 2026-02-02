#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0.0",
# ]
# ///
"""
Agent Harness - Context Engineering for Long-Running Agents

Implements the harness pattern from Anthropic's research on effective
long-running agents. Key insight: agents must bridge context windows
like engineers working in shifts - each shift handoff preserves context.

Core Concepts:
- Context Window: Limited token budget requiring careful management
- Shift Handoff: Transfer essential context between sessions
- Context Engineering: Four pillars for token optimization
- Checkpointing: Save/restore agent state for recovery

Reference: https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents

Usage:
    from harness import AgentHarness, ContextWindow

    harness = AgentHarness(max_tokens=100000)

    # Start task
    harness.begin_task("Implement authentication system")

    # Work in context
    harness.add_to_context("requirement", "OAuth2 + JWT tokens")
    harness.record_action("Created user model")

    # When context fills up, create handoff
    handoff = harness.create_shift_handoff()

    # New session picks up with handoff
    new_harness = AgentHarness.from_handoff(handoff)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ContextPillar(str, Enum):
    """Four pillars of context engineering (from Anthropic)."""
    INSTRUCTIONS = "instructions"  # System prompts, guidelines
    KNOWLEDGE = "knowledge"        # Domain-specific information
    TOOLS = "tools"               # Available capabilities
    HISTORY = "history"           # Past actions and outcomes


class ContextPriority(str, Enum):
    """Priority levels for context items."""
    CRITICAL = "critical"     # Must preserve across all handoffs
    HIGH = "high"            # Preserve if space allows
    MEDIUM = "medium"        # Summarize if needed
    LOW = "low"              # Can be dropped


@dataclass
class ContextItem:
    """A single item in the context window."""
    pillar: ContextPillar
    key: str
    content: str
    priority: ContextPriority = ContextPriority.MEDIUM
    token_estimate: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def estimate_tokens(self) -> int:
        """Estimate token count (rough: ~4 chars per token)."""
        if self.token_estimate == 0:
            self.token_estimate = len(self.content) // 4
        return self.token_estimate

    def access(self) -> None:
        """Record an access to this item."""
        self.last_accessed = time.time()
        self.access_count += 1


class ActionRecord(BaseModel):
    """Record of an action taken by the agent."""
    action: str
    result: str
    success: bool
    timestamp: float = Field(default_factory=time.time)
    context_used: List[str] = Field(default_factory=list)
    tokens_consumed: int = 0


class Checkpoint(BaseModel):
    """Checkpoint for saving agent state."""
    checkpoint_id: str
    task_id: str
    timestamp: float
    context_snapshot: Dict[str, Any]
    action_history: List[ActionRecord]
    current_step: str
    remaining_steps: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ShiftHandoff(BaseModel):
    """
    Handoff document for transferring context between sessions.

    Inspired by Anthropic's observation that agents should work like
    engineers in shifts - each handoff preserves critical context.
    """
    task_id: str
    task_description: str

    # What was accomplished
    completed_steps: List[str]
    key_decisions: List[Dict[str, str]]
    discovered_facts: List[str]

    # Current state
    current_step: str
    blockers: List[str]
    open_questions: List[str]

    # What remains
    remaining_steps: List[str]
    next_actions: List[str]

    # Context to preserve (compressed)
    critical_context: Dict[str, str]
    file_modifications: List[str]

    # Metadata
    created_at: float = Field(default_factory=time.time)
    session_tokens_used: int = 0
    handoff_number: int = 1


class ContextWindow:
    """
    Manages the agent's context window with token budgeting.

    Implements the four pillars of context engineering:
    1. Instructions: System prompts and guidelines
    2. Knowledge: Domain-specific information
    3. Tools: Available capabilities and their schemas
    4. History: Past actions and their outcomes
    """

    def __init__(self, max_tokens: int = 100000):
        self.max_tokens = max_tokens
        self._items: Dict[str, ContextItem] = {}
        self._token_budgets: Dict[ContextPillar, int] = {
            ContextPillar.INSTRUCTIONS: int(max_tokens * 0.15),  # 15%
            ContextPillar.KNOWLEDGE: int(max_tokens * 0.35),     # 35%
            ContextPillar.TOOLS: int(max_tokens * 0.10),         # 10%
            ContextPillar.HISTORY: int(max_tokens * 0.40),       # 40%
        }

    def add(
        self,
        pillar: ContextPillar,
        key: str,
        content: str,
        priority: ContextPriority = ContextPriority.MEDIUM
    ) -> bool:
        """Add an item to the context window."""
        item = ContextItem(
            pillar=pillar,
            key=key,
            content=content,
            priority=priority
        )

        # Check if we have budget
        pillar_usage = self.get_pillar_usage(pillar)
        if pillar_usage + item.estimate_tokens() > self._token_budgets[pillar]:
            # Try to make room
            if not self._evict_for_space(pillar, item.estimate_tokens()):
                return False

        self._items[f"{pillar.value}:{key}"] = item
        return True

    def get(self, pillar: ContextPillar, key: str) -> Optional[str]:
        """Get content from context, recording access."""
        full_key = f"{pillar.value}:{key}"
        if full_key in self._items:
            self._items[full_key].access()
            return self._items[full_key].content
        return None

    def get_pillar_usage(self, pillar: ContextPillar) -> int:
        """Get current token usage for a pillar."""
        total = 0
        for key, item in self._items.items():
            if item.pillar == pillar:
                total += item.estimate_tokens()
        return total

    def get_total_usage(self) -> int:
        """Get total token usage."""
        return sum(item.estimate_tokens() for item in self._items.values())

    def get_utilization(self) -> Dict[str, float]:
        """Get utilization percentages."""
        result = {"total": self.get_total_usage() / self.max_tokens}
        for pillar in ContextPillar:
            budget = self._token_budgets[pillar]
            usage = self.get_pillar_usage(pillar)
            result[pillar.value] = usage / budget if budget > 0 else 0
        return result

    def _evict_for_space(self, pillar: ContextPillar, needed: int) -> bool:
        """Evict low-priority items to make space."""
        # Get items in this pillar, sorted by priority (low first) then by access
        pillar_items = [
            (k, v) for k, v in self._items.items()
            if v.pillar == pillar
        ]
        pillar_items.sort(
            key=lambda x: (
                x[1].priority.value,
                -x[1].last_accessed,
                -x[1].access_count
            )
        )

        freed = 0
        for key, item in pillar_items:
            if item.priority in (ContextPriority.CRITICAL, ContextPriority.HIGH):
                break  # Don't evict high priority
            del self._items[key]
            freed += item.estimate_tokens()
            if freed >= needed:
                return True

        return False

    def get_critical_context(self) -> Dict[str, str]:
        """Extract all critical priority items."""
        return {
            item.key: item.content
            for item in self._items.values()
            if item.priority == ContextPriority.CRITICAL
        }

    def get_context_summary(self) -> str:
        """Get a summary of all context items organized by pillar."""
        summary_parts = []
        for pillar in ContextPillar:
            pillar_items = [
                item for item in self._items.values()
                if item.pillar == pillar
            ]
            if pillar_items:
                summary_parts.append(f"[{pillar.value.upper()}]")
                for item in pillar_items:
                    summary_parts.append(f"  - {item.key}: {item.content[:100]}...")
        return "\n".join(summary_parts) if summary_parts else "(empty context)"

    def summarize_history(self, max_items: int = 10) -> List[str]:
        """Get summarized history of recent actions."""
        history_items = [
            item for item in self._items.values()
            if item.pillar == ContextPillar.HISTORY
        ]
        history_items.sort(key=lambda x: x.created_at, reverse=True)
        return [item.content for item in history_items[:max_items]]


class AgentHarness:
    """
    Agent Harness for managing long-running agent tasks.

    Implements context engineering patterns from Anthropic's research
    to enable agents to work effectively across multiple context windows.

    Key Features:
    - Context window management with token budgeting
    - Shift handoff protocol for session transfers
    - Checkpoint/restore for recovery
    - Action recording and summarization
    """

    def __init__(
        self,
        max_tokens: int = 100000,
        storage_path: Optional[Path] = None
    ):
        self.max_tokens = max_tokens
        self.storage_path = storage_path or Path(".harness")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.context = ContextWindow(max_tokens)
        self.actions: List[ActionRecord] = []
        self.checkpoints: List[str] = []

        self._task_id: Optional[str] = None
        self._task_description: Optional[str] = None
        self._current_step: str = "initialization"
        self._remaining_steps: List[str] = []
        self._handoff_count: int = 0
        self._session_start: float = time.time()

    def begin_task(
        self,
        description: str,
        task_id: Optional[str] = None,
        steps: Optional[List[str]] = None
    ) -> str:
        """Begin a new task."""
        self._task_id = task_id or f"task_{int(time.time())}"
        self._task_description = description
        self._remaining_steps = steps or []
        self._current_step = steps[0] if steps else "working"

        # Add task to instructions context
        self.context.add(
            ContextPillar.INSTRUCTIONS,
            "task",
            f"Task: {description}",
            ContextPriority.CRITICAL
        )

        return self._task_id

    def add_to_context(
        self,
        key: str,
        content: str,
        pillar: ContextPillar = ContextPillar.KNOWLEDGE,
        priority: ContextPriority = ContextPriority.MEDIUM
    ) -> bool:
        """Add information to the agent's context."""
        return self.context.add(pillar, key, content, priority)

    def record_action(
        self,
        action: str,
        result: str = "success",
        success: bool = True,
        context_keys: Optional[List[str]] = None
    ) -> None:
        """Record an action taken by the agent."""
        record = ActionRecord(
            action=action,
            result=result,
            success=success,
            context_used=context_keys or []
        )
        self.actions.append(record)

        # Add to history context
        status = "[OK]" if success else "[FAIL]"
        self.context.add(
            ContextPillar.HISTORY,
            f"action_{len(self.actions)}",
            f"{status} {action}: {result}",
            ContextPriority.MEDIUM
        )

    def record_decision(
        self,
        decision: str,
        rationale: str,
        alternatives: Optional[List[str]] = None
    ) -> None:
        """Record a key decision."""
        content = f"Decision: {decision}\nRationale: {rationale}"
        if alternatives:
            content += f"\nAlternatives considered: {', '.join(alternatives)}"
        self.context.add(
            ContextPillar.KNOWLEDGE,
            f"decision_{len(self.actions)}",
            content,
            ContextPriority.HIGH
        )

    def advance_step(self) -> Optional[str]:
        """Advance to the next step."""
        if self._remaining_steps:
            self._current_step = self._remaining_steps.pop(0)
            return self._current_step
        return None

    def create_checkpoint(self, label: str = "") -> str:
        """Create a checkpoint for recovery."""
        checkpoint_id = f"ckpt_{int(time.time())}_{label}"

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            task_id=self._task_id or "unknown",
            timestamp=time.time(),
            context_snapshot=self.context.get_critical_context(),
            action_history=self.actions.copy(),
            current_step=self._current_step,
            remaining_steps=self._remaining_steps.copy()
        )

        # Save to disk
        checkpoint_file = self.storage_path / f"{checkpoint_id}.json"
        checkpoint_file.write_text(checkpoint.model_dump_json(indent=2))
        self.checkpoints.append(checkpoint_id)

        return checkpoint_id

    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore from a checkpoint."""
        checkpoint_file = self.storage_path / f"{checkpoint_id}.json"
        if not checkpoint_file.exists():
            return False

        checkpoint = Checkpoint.model_validate_json(checkpoint_file.read_text())

        # Restore state
        self._task_id = checkpoint.task_id
        self._current_step = checkpoint.current_step
        self._remaining_steps = checkpoint.remaining_steps
        self.actions = checkpoint.action_history

        # Restore critical context
        for key, content in checkpoint.context_snapshot.items():
            self.context.add(
                ContextPillar.KNOWLEDGE,
                key,
                content,
                ContextPriority.CRITICAL
            )

        return True

    def create_shift_handoff(self) -> ShiftHandoff:
        """
        Create a shift handoff document for session transfer.

        This is the core mechanism for bridging context windows,
        inspired by how human engineers hand off work between shifts.
        """
        self._handoff_count += 1

        # Extract completed steps from action history
        completed = []
        for action in self.actions:
            if action.success:
                completed.append(action.action)

        # Extract key decisions
        decisions = []
        for item_key, item in self.context._items.items():
            if "decision" in item_key:
                decisions.append({
                    "decision": item.key,
                    "content": item.content
                })

        handoff = ShiftHandoff(
            task_id=self._task_id or "unknown",
            task_description=self._task_description or "",
            completed_steps=completed[-20:],  # Last 20 completed steps
            key_decisions=decisions[-10:],     # Last 10 decisions
            discovered_facts=list(self.context.get_critical_context().values())[:10],
            current_step=self._current_step,
            blockers=[],
            open_questions=[],
            remaining_steps=self._remaining_steps,
            next_actions=self._remaining_steps[:3] if self._remaining_steps else [],
            critical_context=self.context.get_critical_context(),
            file_modifications=[],
            session_tokens_used=self.context.get_total_usage(),
            handoff_number=self._handoff_count
        )

        # Save handoff
        handoff_file = self.storage_path / f"handoff_{self._task_id}_{self._handoff_count}.json"
        handoff_file.write_text(handoff.model_dump_json(indent=2))

        return handoff

    @classmethod
    def from_handoff(
        cls,
        handoff: ShiftHandoff,
        max_tokens: int = 100000,
        storage_path: Optional[Path] = None
    ) -> "AgentHarness":
        """Create a new harness from a shift handoff."""
        harness = cls(max_tokens=max_tokens, storage_path=storage_path)

        harness._task_id = handoff.task_id
        harness._task_description = handoff.task_description
        harness._current_step = handoff.current_step
        harness._remaining_steps = handoff.remaining_steps.copy()
        harness._handoff_count = handoff.handoff_number

        # Restore critical context
        for key, content in handoff.critical_context.items():
            harness.context.add(
                ContextPillar.KNOWLEDGE,
                key,
                content,
                ContextPriority.CRITICAL
            )

        # Add handoff summary to context
        summary = f"""
SHIFT HANDOFF #{handoff.handoff_number}
Task: {handoff.task_description}
Completed: {len(handoff.completed_steps)} steps
Current: {handoff.current_step}
Remaining: {len(handoff.remaining_steps)} steps
"""
        harness.context.add(
            ContextPillar.INSTRUCTIONS,
            "handoff_summary",
            summary,
            ContextPriority.CRITICAL
        )

        return harness

    def get_status(self) -> Dict[str, Any]:
        """Get current harness status."""
        return {
            "task_id": self._task_id,
            "task_description": self._task_description,
            "current_step": self._current_step,
            "remaining_steps": len(self._remaining_steps),
            "actions_recorded": len(self.actions),
            "checkpoints": len(self.checkpoints),
            "handoff_count": self._handoff_count,
            "context_utilization": self.context.get_utilization(),
            "session_duration_sec": time.time() - self._session_start
        }


def main():
    """Demo the agent harness."""
    print("=" * 60)
    print("AGENT HARNESS DEMO")
    print("=" * 60)
    print()

    # Create harness
    harness = AgentHarness(max_tokens=50000)

    # Begin a task
    task_id = harness.begin_task(
        description="Implement user authentication system",
        steps=[
            "Design database schema",
            "Create user model",
            "Implement JWT tokens",
            "Add OAuth2 provider",
            "Write tests"
        ]
    )
    print(f"[OK] Started task: {task_id}")

    # Add context
    harness.add_to_context(
        "tech_stack",
        "Python 3.11, FastAPI, PostgreSQL, Redis",
        priority=ContextPriority.HIGH
    )
    harness.add_to_context(
        "auth_requirement",
        "Support OAuth2 (Google, GitHub) and email/password",
        priority=ContextPriority.CRITICAL
    )

    # Record some actions
    harness.record_action(
        "Created User model with email, password_hash, created_at",
        "models/user.py created"
    )
    harness.record_decision(
        "Use bcrypt for password hashing",
        "Industry standard, resistant to brute force",
        ["argon2", "scrypt"]
    )
    harness.record_action(
        "Implemented JWT token generation",
        "auth/jwt.py with 15min access, 7d refresh"
    )
    harness.advance_step()
    harness.advance_step()

    # Get status
    print("\n[>>] Harness Status:")
    status = harness.get_status()
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v:.1%}" if isinstance(v, float) else f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    # Create checkpoint
    ckpt_id = harness.create_checkpoint("after_jwt")
    print(f"\n[OK] Checkpoint created: {ckpt_id}")

    # Create shift handoff (simulating context window filling up)
    print("\n[>>] Creating shift handoff...")
    handoff = harness.create_shift_handoff()
    print(f"  Task: {handoff.task_id}")
    print(f"  Handoff #: {handoff.handoff_number}")
    print(f"  Completed steps: {len(handoff.completed_steps)}")
    print(f"  Remaining steps: {len(handoff.remaining_steps)}")
    print(f"  Critical context items: {len(handoff.critical_context)}")

    # Simulate new session picking up handoff
    print("\n[>>] New session from handoff...")
    new_harness = AgentHarness.from_handoff(handoff)
    new_status = new_harness.get_status()
    print(f"  Current step: {new_status['current_step']}")
    print(f"  Remaining: {new_status['remaining_steps']} steps")
    print(f"  Context loaded: {new_status['context_utilization']['total']:.1%}")

    print("\n[OK] Agent harness demo complete")


if __name__ == "__main__":
    main()
