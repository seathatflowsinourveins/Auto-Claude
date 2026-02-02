#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0.0",
#     "structlog>=24.1.0",
# ]
# [tool.uv]
# exclude-newer = "2026-02-01"
# ///
"""
Session Continuity Module - V10 Ultimate Platform

Implements the Trinity Pattern (Skills + Hooks + Agents) for seamless
session management and knowledge base coordination.

Features:
1. Session state management and persistence
2. Knowledge base structure (9-file pattern)
3. Trinity Pattern coordination
4. Session teleportation preparation
5. Context continuity across sessions

Based on research:
- Claude Code 2.1.0 session teleportation
- 9-file knowledge base pattern
- Trinity Pattern: Skills + Hooks + Agents

Usage:
    uv run session_continuity.py init              # Initialize knowledge base
    uv run session_continuity.py status            # Show session status
    uv run session_continuity.py export            # Export session for teleport
    uv run session_continuity.py trinity           # Show Trinity Pattern status
    uv run session_continuity.py knowledge         # Show knowledge base

Platform: Windows 11 + Python 3.11+
Architecture: V10 Optimized (Verified, Minimal, Seamless)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
HOOKS_DIR = V10_DIR / "hooks"
DATA_DIR = V10_DIR / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge_base"
SESSION_DIR = DATA_DIR / "sessions"

# Ensure directories exist
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# Windows compatibility
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

logger = structlog.get_logger(__name__)


# =============================================================================
# Knowledge Base Structure (9-File Pattern)
# =============================================================================

KNOWLEDGE_BASE_FILES = {
    "01_style.md": {
        "title": "Coding Style & Conventions",
        "description": "Your preferred coding style, naming conventions, formatting rules",
        "template": """# Coding Style & Conventions

## Language Preferences
- Primary: Python 3.11+, TypeScript
- Type hints: Always required
- Async: Preferred for I/O

## Naming Conventions
- Functions: snake_case
- Classes: PascalCase
- Constants: UPPER_SNAKE_CASE
- Private: _leading_underscore

## Formatting
- Line length: 100 characters
- Indentation: 4 spaces (Python), 2 spaces (TypeScript)
- Quotes: Double quotes for strings

## Documentation
- Docstrings: Google style
- Comments: Only when logic isn't self-evident
- Type annotations: Inline preferred
""",
    },
    "02_principles.md": {
        "title": "Development Principles",
        "description": "Core principles guiding your development approach",
        "template": """# Development Principles

## V10 Philosophy
1. **Verified**: Everything tested, nothing assumed
2. **Minimal**: No unnecessary complexity
3. **Seamless**: Invisible infrastructure

## Design Principles
- SOLID principles
- Composition over inheritance
- Explicit over implicit
- Fail fast, fail loud

## Code Quality
- Tests before features
- Refactor continuously
- Document decisions (ADRs)
- Review before commit
""",
    },
    "03_architecture.md": {
        "title": "System Architecture",
        "description": "High-level architecture patterns and decisions",
        "template": """# System Architecture

## Layer Structure
1. Presentation (CLI/API)
2. Application (Use Cases)
3. Domain (Business Logic)
4. Infrastructure (External Services)

## Patterns Used
- Repository pattern for data access
- Circuit breaker for resilience
- Event sourcing for audit
- CQRS for complex queries

## Integration Points
- MCP servers for external tools
- Letta for memory persistence
- Qdrant for vector search
- Neo4j for knowledge graph
""",
    },
    "04_domain.md": {
        "title": "Domain Language",
        "description": "Project-specific terminology and concepts",
        "template": """# Domain Language

## Core Concepts
- **Ralph Loop**: Autonomous improvement iteration
- **Sleep-time compute**: Background processing during idle
- **Warm start**: Pre-computed context loading
- **Trinity Pattern**: Skills + Hooks + Agents

## Entity Definitions
- **Session**: A single Claude Code interaction
- **Memory Block**: Persistent knowledge unit
- **Insight**: Auto-generated observation
- **Workflow**: Multi-step automation

## Abbreviations
- UAP: Ultimate Autonomous Platform
- MCP: Model Context Protocol
- V10: Version 10 (current architecture)
""",
    },
    "05_workflows.md": {
        "title": "Common Workflows",
        "description": "Step-by-step procedures for recurring tasks",
        "template": """# Common Workflows

## Feature Development
1. Research existing patterns
2. Create plan in PLAN.md
3. Implement with TDD
4. Update documentation
5. Create PR with summary

## Bug Fixing
1. Reproduce the issue
2. Write failing test
3. Fix the bug
4. Verify fix doesn't break others
5. Document root cause

## Ralph Loop Iteration
1. Check ecosystem status
2. Research improvements
3. Implement changes
4. Validate with tests
5. Update GOALS_TRACKING.md
""",
    },
    "06_decisions.md": {
        "title": "Key Decisions",
        "description": "Important technical decisions and rationale",
        "template": """# Key Decisions (ADRs)

## ADR-001: Use Pydantic for Data Models
- **Status**: Accepted
- **Context**: Need structured data with validation
- **Decision**: Use Pydantic v2 for all data classes
- **Rationale**: Type safety, serialization, IDE support

## ADR-002: Async-First Design
- **Status**: Accepted
- **Context**: Many I/O operations (MCP, HTTP, DB)
- **Decision**: Use async/await throughout
- **Rationale**: Better resource utilization, concurrency

## ADR-003: Windows Compatibility
- **Status**: Accepted
- **Context**: Development on Windows 11
- **Decision**: ASCII-safe output, Path handling
- **Rationale**: Consistent cross-platform behavior
""",
    },
    "07_patterns.md": {
        "title": "Code Patterns",
        "description": "Reusable code patterns and snippets",
        "template": """# Code Patterns

## Async Context Manager
```python
async def __aenter__(self):
    await self.connect()
    return self

async def __aexit__(self, *args):
    await self.disconnect()
```

## Circuit Breaker
```python
@circuit_breaker(failure_threshold=5, reset_timeout=60)
async def call_external_service():
    ...
```

## Retry with Backoff
```python
@with_retry(max_attempts=3, backoff_factor=2.0)
async def flaky_operation():
    ...
```

## Structured Logging
```python
logger.info("operation_completed",
    operation="create_session",
    duration_ms=elapsed,
    result="success")
```
""",
    },
    "08_testing.md": {
        "title": "Testing Strategy",
        "description": "Testing approach and common patterns",
        "template": """# Testing Strategy

## Test Pyramid
- Unit tests: 70% (fast, isolated)
- Integration tests: 20% (component interaction)
- E2E tests: 10% (full workflows)

## Naming Convention
- test_<function>_<scenario>_<expected>
- Example: test_create_session_valid_input_returns_id

## Fixtures
- Use pytest fixtures for setup
- Scope: function (default), module, session
- Async fixtures with pytest-asyncio

## Mocking
- Mock external services
- Use responses for HTTP
- Use fakeredis for Redis
""",
    },
    "09_context.md": {
        "title": "Project Context",
        "description": "Current project state and active work",
        "template": """# Project Context

## Current Focus
- Ralph Loop iterations (target: 20+)
- Sleep-time compute integration
- Trinity Pattern implementation

## Active Tasks
- [ ] Session continuity module
- [ ] Performance optimization
- [ ] Documentation updates

## Recent Changes
- Created sleeptime_compute.py
- Created auto_validate.py
- Fixed Windows compatibility issues

## Blockers
- Letta server not running (Docker)
- Need to test with real MCP servers
""",
    },
}


# =============================================================================
# Data Models
# =============================================================================

class TrinityComponent(str, Enum):
    """Components of the Trinity Pattern."""
    SKILLS = "skills"
    HOOKS = "hooks"
    AGENTS = "agents"


class SessionState(str, Enum):
    """States of a session."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    EXPORTING = "exporting"
    COMPLETED = "completed"


@dataclass
class KnowledgeFile:
    """A file in the knowledge base."""
    name: str
    title: str
    description: str
    path: Path
    exists: bool
    last_modified: Optional[str] = None
    word_count: int = 0


@dataclass
class TrinityStatus:
    """Status of the Trinity Pattern components."""
    skills_count: int
    hooks_count: int
    agents_count: int
    skills_active: List[str]
    hooks_active: List[str]
    agents_available: List[str]


@dataclass
class SessionContext:
    """Context for a Claude Code session."""
    session_id: str
    state: SessionState
    started_at: str
    project_name: str
    knowledge_loaded: List[str]
    active_workflows: List[str]
    memory_blocks: int
    insights_available: int
    trinity_status: Optional[TrinityStatus] = None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "started_at": self.started_at,
            "project_name": self.project_name,
            "knowledge_loaded": self.knowledge_loaded,
            "active_workflows": self.active_workflows,
            "memory_blocks": self.memory_blocks,
            "insights_available": self.insights_available,
            "trinity_status": {
                "skills_count": self.trinity_status.skills_count,
                "hooks_count": self.trinity_status.hooks_count,
                "agents_count": self.trinity_status.agents_count,
            } if self.trinity_status else None,
        }


@dataclass
class TeleportPackage:
    """Package for session teleportation."""
    session_id: str
    exported_at: str
    knowledge_base: Dict[str, str]
    session_context: Dict[str, Any]
    memory_snapshot: Dict[str, Any]
    active_tasks: List[str]
    environment: Dict[str, str]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "exported_at": self.exported_at,
            "knowledge_base": self.knowledge_base,
            "session_context": self.session_context,
            "memory_snapshot": self.memory_snapshot,
            "active_tasks": self.active_tasks,
            "environment": self.environment,
        }


# =============================================================================
# Knowledge Base Manager
# =============================================================================

class KnowledgeBaseManager:
    """Manages the 9-file knowledge base."""

    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR):
        self.knowledge_dir = knowledge_dir

    def initialize(self, force: bool = False) -> List[KnowledgeFile]:
        """Initialize the knowledge base with template files."""
        created = []

        for filename, config in KNOWLEDGE_BASE_FILES.items():
            file_path = self.knowledge_dir / filename

            if file_path.exists() and not force:
                continue

            file_path.write_text(config["template"], encoding="utf-8")
            created.append(KnowledgeFile(
                name=filename,
                title=config["title"],
                description=config["description"],
                path=file_path,
                exists=True,
                last_modified=datetime.now(timezone.utc).isoformat(),
                word_count=len(config["template"].split()),
            ))

        return created

    def get_status(self) -> List[KnowledgeFile]:
        """Get status of all knowledge base files."""
        files = []

        for filename, config in KNOWLEDGE_BASE_FILES.items():
            file_path = self.knowledge_dir / filename
            exists = file_path.exists()

            word_count = 0
            last_modified = None

            if exists:
                content = file_path.read_text(encoding="utf-8")
                word_count = len(content.split())
                stat = file_path.stat()
                last_modified = datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat()

            files.append(KnowledgeFile(
                name=filename,
                title=config["title"],
                description=config["description"],
                path=file_path,
                exists=exists,
                last_modified=last_modified,
                word_count=word_count,
            ))

        return files

    def load_all(self) -> Dict[str, str]:
        """Load all knowledge base files."""
        knowledge = {}

        for filename in KNOWLEDGE_BASE_FILES.keys():
            file_path = self.knowledge_dir / filename
            if file_path.exists():
                knowledge[filename] = file_path.read_text(encoding="utf-8")

        return knowledge


# =============================================================================
# Trinity Pattern Manager
# =============================================================================

class TrinityManager:
    """Manages the Trinity Pattern (Skills + Hooks + Agents)."""

    def __init__(self, v10_dir: Path = V10_DIR):
        self.v10_dir = v10_dir
        self.hooks_dir = v10_dir / "hooks"
        self.skills_dir = v10_dir / "skills"
        self.agents_dir = v10_dir / "agents"

    def get_hooks(self) -> List[str]:
        """Get list of available hooks."""
        if not self.hooks_dir.exists():
            return []
        return [f.stem for f in self.hooks_dir.glob("*.py") if not f.name.startswith("_")]

    def get_skills(self) -> List[str]:
        """Get list of available skills."""
        # Skills can be in .md files or skill directories
        skills = []

        if self.skills_dir.exists():
            for item in self.skills_dir.iterdir():
                if item.is_file() and item.suffix == ".md":
                    skills.append(item.stem)
                elif item.is_dir() and (item / "SKILL.md").exists():
                    skills.append(item.name)

        # Also check for skills in the main directory
        skill_md = self.v10_dir / "SKILLS.md"
        if skill_md.exists():
            skills.append("main")

        return skills

    def get_agents(self) -> List[str]:
        """Get list of available agents."""
        if not self.agents_dir.exists():
            return []

        agents = []
        for item in self.agents_dir.iterdir():
            if item.is_file() and item.suffix == ".md":
                agents.append(item.stem)
            elif item.is_dir() and (item / "AGENT.md").exists():
                agents.append(item.name)

        return agents

    def get_status(self) -> TrinityStatus:
        """Get complete Trinity Pattern status."""
        hooks = self.get_hooks()
        skills = self.get_skills()
        agents = self.get_agents()

        return TrinityStatus(
            skills_count=len(skills),
            hooks_count=len(hooks),
            agents_count=len(agents),
            skills_active=skills,
            hooks_active=hooks,
            agents_available=agents,
        )


# =============================================================================
# Session Manager
# =============================================================================

class SessionManager:
    """Manages session state and continuity."""

    def __init__(self):
        self.knowledge = KnowledgeBaseManager()
        self.trinity = TrinityManager()
        self.session_file = SESSION_DIR / "current_session.json"
        self.current_session: Optional[SessionContext] = None

    def create_session(self, project_name: str = "default") -> SessionContext:
        """Create a new session."""
        import hashlib

        now = datetime.now(timezone.utc).isoformat()
        session_id = hashlib.sha256(f"{now}:{project_name}".encode()).hexdigest()[:16]

        # Load knowledge base
        knowledge_files = self.knowledge.get_status()
        knowledge_loaded = [f.name for f in knowledge_files if f.exists]

        # Get Trinity status
        trinity_status = self.trinity.get_status()

        # Count memory blocks (from sleeptime module data)
        memory_dir = DATA_DIR / "memory"
        memory_blocks = len(list(memory_dir.glob("*.json"))) if memory_dir.exists() else 0

        # Count insights
        insights_dir = DATA_DIR / "insights"
        insights = len(list(insights_dir.glob("insight_*.json"))) if insights_dir.exists() else 0

        session = SessionContext(
            session_id=session_id,
            state=SessionState.ACTIVE,
            started_at=now,
            project_name=project_name,
            knowledge_loaded=knowledge_loaded,
            active_workflows=[],
            memory_blocks=memory_blocks,
            insights_available=insights,
            trinity_status=trinity_status,
        )

        self.current_session = session
        self._save_session(session)

        return session

    def _save_session(self, session: SessionContext):
        """Save session to disk."""
        self.session_file.write_text(
            json.dumps(session.to_dict(), indent=2),
            encoding="utf-8"
        )

    def load_session(self) -> Optional[SessionContext]:
        """Load the current session from disk."""
        if not self.session_file.exists():
            return None

        try:
            data = json.loads(self.session_file.read_text(encoding="utf-8"))
            trinity_data = data.get("trinity_status")
            trinity_status = TrinityStatus(
                skills_count=trinity_data.get("skills_count", 0) if trinity_data else 0,
                hooks_count=trinity_data.get("hooks_count", 0) if trinity_data else 0,
                agents_count=trinity_data.get("agents_count", 0) if trinity_data else 0,
                skills_active=[],
                hooks_active=[],
                agents_available=[],
            ) if trinity_data else None

            return SessionContext(
                session_id=data["session_id"],
                state=SessionState(data["state"]),
                started_at=data["started_at"],
                project_name=data["project_name"],
                knowledge_loaded=data["knowledge_loaded"],
                active_workflows=data["active_workflows"],
                memory_blocks=data["memory_blocks"],
                insights_available=data["insights_available"],
                trinity_status=trinity_status,
            )
        except Exception as e:
            logger.warning("Failed to load session", error=str(e))
            return None

    def export_for_teleport(self) -> TeleportPackage:
        """Export current session for teleportation."""
        session = self.current_session or self.load_session()
        if not session:
            session = self.create_session()

        now = datetime.now(timezone.utc).isoformat()

        # Load knowledge base
        knowledge = self.knowledge.load_all()

        # Get memory snapshot
        memory_snapshot = {}
        memory_dir = DATA_DIR / "memory"
        if memory_dir.exists():
            for f in list(memory_dir.glob("*.json"))[:10]:  # Limit to 10
                try:
                    memory_snapshot[f.stem] = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    pass

        # Get active tasks from goals tracking
        active_tasks = []
        goals_file = UNLEASH_DIR / "GOALS_TRACKING.md"
        if goals_file.exists():
            content = goals_file.read_text(encoding="utf-8")
            # Extract TODO items
            for line in content.split("\n"):
                if "📋 TODO" in line or "[ ]" in line:
                    active_tasks.append(line.strip()[:100])

        # Environment info
        environment = {
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "v10_dir": str(V10_DIR),
            "project_name": session.project_name,
        }

        package = TeleportPackage(
            session_id=session.session_id,
            exported_at=now,
            knowledge_base=knowledge,
            session_context=session.to_dict(),
            memory_snapshot=memory_snapshot,
            active_tasks=active_tasks[:20],  # Limit to 20
            environment=environment,
        )

        # Save teleport package
        teleport_file = SESSION_DIR / f"teleport_{session.session_id}.json"
        teleport_file.write_text(
            json.dumps(package.to_dict(), indent=2),
            encoding="utf-8"
        )

        return package


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Session Continuity Module - V10 Ultimate Platform",
    )
    parser.add_argument(
        "command",
        choices=["init", "status", "export", "trinity", "knowledge"],
        help="Command to execute",
    )
    parser.add_argument(
        "--project",
        default="UltimateAutonomousPlatform",
        help="Project name",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()
    manager = SessionManager()

    if args.command == "init":
        print("Initializing knowledge base...")
        created = manager.knowledge.initialize(force=args.force)
        print(f"Created {len(created)} knowledge base files")

        if created:
            for f in created:
                print(f"  [+] {f.name}: {f.title}")
        else:
            print("  (All files already exist. Use --force to overwrite)")

        print(f"\nKnowledge base location: {KNOWLEDGE_DIR}")

    elif args.command == "status":
        session = manager.load_session()
        if not session:
            print("No active session. Creating new session...")
            session = manager.create_session(args.project)

        if args.json:
            print(json.dumps(session.to_dict(), indent=2))
        else:
            print("=" * 55)
            print("SESSION CONTINUITY STATUS")
            print("=" * 55)
            print(f"Session ID:       {session.session_id}")
            print(f"State:            {session.state.value}")
            print(f"Project:          {session.project_name}")
            print(f"Started:          {session.started_at[:19]}")
            print("-" * 55)
            print(f"Knowledge Files:  {len(session.knowledge_loaded)}/9")
            print(f"Memory Blocks:    {session.memory_blocks}")
            print(f"Insights:         {session.insights_available}")
            print(f"Active Workflows: {len(session.active_workflows)}")
            if session.trinity_status:
                print("-" * 55)
                print("TRINITY PATTERN:")
                print(f"  Skills:  {session.trinity_status.skills_count}")
                print(f"  Hooks:   {session.trinity_status.hooks_count}")
                print(f"  Agents:  {session.trinity_status.agents_count}")
            print("=" * 55)

    elif args.command == "export":
        print("Exporting session for teleportation...")
        package = manager.export_for_teleport()

        if args.json:
            print(json.dumps(package.to_dict(), indent=2))
        else:
            print(f"Session ID:      {package.session_id}")
            print(f"Exported at:     {package.exported_at[:19]}")
            print(f"Knowledge files: {len(package.knowledge_base)}")
            print(f"Memory blocks:   {len(package.memory_snapshot)}")
            print(f"Active tasks:    {len(package.active_tasks)}")
            print(f"\nTeleport package saved to:")
            print(f"  {SESSION_DIR / f'teleport_{package.session_id}.json'}")

    elif args.command == "trinity":
        status = manager.trinity.get_status()

        if args.json:
            print(json.dumps({
                "skills": {"count": status.skills_count, "active": status.skills_active},
                "hooks": {"count": status.hooks_count, "active": status.hooks_active},
                "agents": {"count": status.agents_count, "available": status.agents_available},
            }, indent=2))
        else:
            print("=" * 55)
            print("TRINITY PATTERN STATUS")
            print("=" * 55)
            print(f"\nSKILLS ({status.skills_count}):")
            if status.skills_active:
                for s in status.skills_active:
                    print(f"  - {s}")
            else:
                print("  (none found)")

            print(f"\nHOOKS ({status.hooks_count}):")
            if status.hooks_active:
                for h in status.hooks_active:
                    print(f"  - {h}")
            else:
                print("  (none found)")

            print(f"\nAGENTS ({status.agents_count}):")
            if status.agents_available:
                for a in status.agents_available:
                    print(f"  - {a}")
            else:
                print("  (none found)")
            print("=" * 55)

    elif args.command == "knowledge":
        files = manager.knowledge.get_status()

        if args.json:
            print(json.dumps([{
                "name": f.name,
                "title": f.title,
                "exists": f.exists,
                "word_count": f.word_count,
            } for f in files], indent=2))
        else:
            print("=" * 65)
            print("KNOWLEDGE BASE (9-File Pattern)")
            print("=" * 65)
            print(f"{'File':<20} {'Title':<30} {'Status':<10} {'Words'}")
            print("-" * 65)

            for f in files:
                status_str = "[OK]" if f.exists else "[MISSING]"
                print(f"{f.name:<20} {f.title[:28]:<30} {status_str:<10} {f.word_count}")

            existing = sum(1 for f in files if f.exists)
            print("-" * 65)
            print(f"Total: {existing}/9 files present")
            print(f"Location: {KNOWLEDGE_DIR}")
            print("=" * 65)


if __name__ == "__main__":
    main()
