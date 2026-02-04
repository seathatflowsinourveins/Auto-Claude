"""
Memory Hooks - Session Start/End Integration for Cross-Session Memory

This module provides hooks that integrate with Claude Code's session lifecycle
to enable automatic memory persistence and restoration.

Hooks:
- session_start: Load previous context, restore memory state
- session_end: Persist learnings, consolidate memories

Integration:
    These hooks are called via .claude/settings.json SessionStart/SessionEnd hooks:
    - npx @claude-flow/cli hooks session-restore --session-id $SESSION_ID
    - npx @claude-flow/cli hooks session-save --session-id $SESSION_ID

Usage (Python):
    from core.memory.hooks import session_start_hook, session_end_hook

    # On session start
    context = await session_start_hook(session_id="abc123", project_path="/path/to/project")

    # On session end
    await session_end_hook(session_id="abc123", summary="Completed feature X")

Usage (CLI):
    python -m platform.core.memory.hooks start --session-id abc123
    python -m platform.core.memory.hooks end --session-id abc123 --summary "Done"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Embedding Providers
# =============================================================================

def get_embedding_provider() -> Optional[Callable[[str], List[float]]]:
    """Get the best available embedding provider.

    Checks in order:
    1. OpenAI API (OPENAI_API_KEY)
    2. Jina Embeddings (JINA_API_KEY)
    3. Local sentence-transformers (if installed)

    Returns:
        A function that takes text and returns embeddings, or None if unavailable.
    """

    # Try OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)

            def openai_embed(text: str) -> List[float]:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text[:8000]  # Token limit
                )
                return response.data[0].embedding

            logger.info("Using OpenAI embeddings")
            return openai_embed
        except ImportError:
            logger.debug("OpenAI package not installed")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI embeddings: {e}")

    # Try Jina
    jina_key = os.environ.get("JINA_API_KEY")
    if jina_key:
        try:
            import httpx

            def jina_embed(text: str) -> List[float]:
                response = httpx.post(
                    "https://api.jina.ai/v1/embeddings",
                    headers={"Authorization": f"Bearer {jina_key}"},
                    json={
                        "input": [text[:8000]],
                        "model": "jina-embeddings-v3"
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return data["data"][0]["embedding"]

            logger.info("Using Jina embeddings")
            return jina_embed
        except ImportError:
            logger.debug("httpx not installed for Jina")
        except Exception as e:
            logger.warning(f"Failed to initialize Jina embeddings: {e}")

    # Try local sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        def local_embed(text: str) -> List[float]:
            return model.encode(text[:5000]).tolist()

        logger.info("Using local sentence-transformers")
        return local_embed
    except ImportError:
        logger.debug("sentence-transformers not installed")

    logger.warning("No embedding provider available - semantic search disabled")
    return None


# =============================================================================
# Session Hooks
# =============================================================================

async def session_start_hook(
    session_id: Optional[str] = None,
    project_path: Optional[str] = None,
    load_context: bool = True
) -> Dict[str, Any]:
    """Hook called on session start.

    This hook:
    1. Creates a new session record
    2. Loads previous context from memory
    3. Returns context summary for warm start

    Args:
        session_id: Optional session ID (generated if not provided)
        project_path: Project path for session tracking
        load_context: Whether to load previous context

    Returns:
        Dict with session info and loaded context
    """
    from .backends.sqlite import get_sqlite_backend

    embedding_provider = get_embedding_provider()
    backend = get_sqlite_backend(embedding_provider)

    # Start new session
    if session_id is None:
        session_id = await backend.start_session(
            task_summary="Session started",
            project_path=project_path or os.getcwd()
        )
    else:
        await backend.start_session(
            task_summary="Session started",
            project_path=project_path or os.getcwd()
        )

    result = {
        "session_id": session_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "project_path": project_path or os.getcwd(),
        "context_loaded": False,
        "context": "",
        "memories_available": 0,
        "recent_decisions": [],
        "recent_learnings": [],
    }

    if load_context:
        try:
            # Load context summary
            context = await backend.get_session_context(max_tokens=2000)
            result["context"] = context
            result["context_loaded"] = bool(context)

            # Get stats
            stats = await backend.get_stats()
            result["memories_available"] = stats.get("total_memories", 0)

            # Get recent decisions and learnings
            decisions = await backend.get_decisions(5)
            learnings = await backend.get_learnings(5)

            result["recent_decisions"] = [d.content[:100] for d in decisions]
            result["recent_learnings"] = [l.content[:100] for l in learnings]

        except Exception as e:
            logger.error(f"Failed to load context: {e}")
            result["error"] = str(e)

    logger.info(f"Session started: {session_id} ({result['memories_available']} memories available)")
    return result


async def session_end_hook(
    session_id: str,
    summary: Optional[str] = None,
    learnings: Optional[List[str]] = None,
    decisions: Optional[List[str]] = None,
    consolidate: bool = True
) -> Dict[str, Any]:
    """Hook called on session end.

    This hook:
    1. Records any final learnings/decisions
    2. Updates session summary
    3. Optionally consolidates memory

    Args:
        session_id: Session ID to end
        summary: Optional session summary
        learnings: List of learnings from this session
        decisions: List of decisions made this session
        consolidate: Whether to run memory consolidation

    Returns:
        Dict with session end status
    """
    from .backends.sqlite import get_sqlite_backend

    backend = get_sqlite_backend()

    result = {
        "session_id": session_id,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "memories_stored": 0,
        "consolidated": False,
    }

    try:
        # Store learnings
        if learnings:
            for learning in learnings:
                await backend.store_memory(
                    content=learning,
                    memory_type="learning",
                    importance=0.7,
                    session_id=session_id
                )
                result["memories_stored"] += 1

        # Store decisions
        if decisions:
            for decision in decisions:
                await backend.store_memory(
                    content=decision,
                    memory_type="decision",
                    importance=0.8,
                    session_id=session_id
                )
                result["memories_stored"] += 1

        # End the session
        await backend.end_session(session_id, summary)

        # Consolidate memory if requested
        if consolidate:
            consolidated = await _consolidate_memory(backend)
            result["consolidated"] = True
            result["consolidation_stats"] = consolidated

    except Exception as e:
        logger.error(f"Session end hook failed: {e}")
        result["error"] = str(e)

    logger.info(f"Session ended: {session_id} ({result['memories_stored']} memories stored)")
    return result


async def _consolidate_memory(backend) -> Dict[str, Any]:
    """Consolidate memory by removing duplicates and expired entries.

    Returns:
        Consolidation statistics
    """
    stats = {
        "duplicates_removed": 0,
        "expired_removed": 0,
        "total_remaining": 0,
    }

    # Note: Advanced consolidation would include:
    # - Semantic deduplication (similar embeddings)
    # - Temporal decay (reduce importance of old memories)
    # - Summarization of related memories
    # For now, we just get the count

    all_stats = await backend.get_stats()
    stats["total_remaining"] = all_stats.get("total_memories", 0)

    return stats


# =============================================================================
# Convenience Functions
# =============================================================================

async def remember_decision(
    content: str,
    importance: float = 0.8,
    tags: Optional[List[str]] = None,
    session_id: Optional[str] = None
) -> str:
    """Store a decision in cross-session memory.

    Args:
        content: The decision content
        importance: Importance score (0.0-1.0)
        tags: Optional tags
        session_id: Optional session ID

    Returns:
        Memory ID
    """
    from .backends.sqlite import get_sqlite_backend
    backend = get_sqlite_backend()
    return await backend.store_memory(
        content=content,
        memory_type="decision",
        importance=importance,
        tags=tags,
        session_id=session_id
    )


async def remember_learning(
    content: str,
    importance: float = 0.7,
    tags: Optional[List[str]] = None,
    session_id: Optional[str] = None
) -> str:
    """Store a learning in cross-session memory.

    Args:
        content: The learning content
        importance: Importance score (0.0-1.0)
        tags: Optional tags
        session_id: Optional session ID

    Returns:
        Memory ID
    """
    from .backends.sqlite import get_sqlite_backend
    backend = get_sqlite_backend()
    return await backend.store_memory(
        content=content,
        memory_type="learning",
        importance=importance,
        tags=tags,
        session_id=session_id
    )


async def remember_fact(
    content: str,
    importance: float = 0.6,
    tags: Optional[List[str]] = None,
    session_id: Optional[str] = None
) -> str:
    """Store a fact in cross-session memory.

    Args:
        content: The fact content
        importance: Importance score (0.0-1.0)
        tags: Optional tags
        session_id: Optional session ID

    Returns:
        Memory ID
    """
    from .backends.sqlite import get_sqlite_backend
    backend = get_sqlite_backend()
    return await backend.store_memory(
        content=content,
        memory_type="fact",
        importance=importance,
        tags=tags,
        session_id=session_id
    )


async def recall(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search cross-session memory.

    Args:
        query: Search query
        limit: Maximum results

    Returns:
        List of matching memories as dicts
    """
    from .backends.sqlite import get_sqlite_backend
    backend = get_sqlite_backend()
    results = await backend.search(query, limit)
    return [
        {
            "id": m.id,
            "content": m.content,
            "type": m.content_type,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "tags": m.tags,
        }
        for m in results
    ]


async def get_context() -> str:
    """Get the full context summary for a new session.

    Returns:
        Formatted context string
    """
    from .backends.sqlite import get_sqlite_backend
    backend = get_sqlite_backend()
    return await backend.get_session_context()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for memory hooks."""
    parser = argparse.ArgumentParser(
        description="Cross-Session Memory Hooks",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start session hook")
    start_parser.add_argument("--session-id", help="Session ID")
    start_parser.add_argument("--project-path", help="Project path")
    start_parser.add_argument("--no-context", action="store_true", help="Don't load context")

    # End command
    end_parser = subparsers.add_parser("end", help="End session hook")
    end_parser.add_argument("--session-id", required=True, help="Session ID")
    end_parser.add_argument("--summary", help="Session summary")
    end_parser.add_argument("--no-consolidate", action="store_true", help="Skip consolidation")

    # Store command
    store_parser = subparsers.add_parser("store", help="Store a memory")
    store_parser.add_argument("content", help="Memory content")
    store_parser.add_argument("--type", choices=["decision", "learning", "fact", "context"], default="context")
    store_parser.add_argument("--importance", type=float, default=0.5)
    store_parser.add_argument("--tags", nargs="*", help="Tags for the memory")
    store_parser.add_argument("--session-id", help="Session ID")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=10)

    # Context command
    context_parser = subparsers.add_parser("context", help="Get context summary")
    context_parser.add_argument("--max-tokens", type=int, default=2000)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get memory statistics")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export memories to JSON")
    export_parser.add_argument("--output", help="Output file path")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test cross-session persistence")
    test_parser.add_argument("--unique-id", help="Unique test ID")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Run async commands
    async def run():
        if args.command == "start":
            result = await session_start_hook(
                session_id=args.session_id,
                project_path=args.project_path,
                load_context=not args.no_context
            )
            print(json.dumps(result, indent=2))

        elif args.command == "end":
            result = await session_end_hook(
                session_id=args.session_id,
                summary=args.summary,
                consolidate=not args.no_consolidate
            )
            print(json.dumps(result, indent=2))

        elif args.command == "store":
            from .backends.sqlite import get_sqlite_backend
            backend = get_sqlite_backend()
            memory_id = await backend.store_memory(
                content=args.content,
                memory_type=args.type,
                importance=args.importance,
                tags=args.tags,
                session_id=args.session_id
            )
            print(json.dumps({"memory_id": memory_id, "status": "stored"}))

        elif args.command == "search":
            results = await recall(args.query, args.limit)
            print(json.dumps(results, indent=2))

        elif args.command == "context":
            from .backends.sqlite import get_sqlite_backend
            backend = get_sqlite_backend()
            context = await backend.get_session_context(args.max_tokens)
            print(context)

        elif args.command == "stats":
            from .backends.sqlite import get_sqlite_backend
            backend = get_sqlite_backend()
            stats = await backend.get_stats()
            print(json.dumps(stats, indent=2))

        elif args.command == "export":
            from .backends.sqlite import get_sqlite_backend
            backend = get_sqlite_backend()
            output_path = await backend.export_to_json(
                Path(args.output) if args.output else None
            )
            print(json.dumps({"exported_to": str(output_path)}))

        elif args.command == "test":
            # Test cross-session persistence
            unique_id = args.unique_id or f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"Testing cross-session memory with ID: {unique_id}")

            # Store
            from .backends.sqlite import get_sqlite_backend
            backend = get_sqlite_backend()
            memory_id = await backend.store_memory(
                content=f"Test memory {unique_id}: This is a test of cross-session persistence",
                memory_type="fact",
                importance=0.9,
                tags=["test", unique_id]
            )
            print(f"Stored memory: {memory_id}")

            # Read back
            results = await backend.search(unique_id, limit=5)
            print(f"Found {len(results)} results")

            for r in results:
                print(f"  - {r.id}: {r.content[:50]}...")

            # Check persistence path
            print(f"\nPersistence path: {backend.db_path}")
            print(f"Database exists: {backend.db_path.exists()}")
            if backend.db_path.exists():
                print(f"Database size: {backend.db_path.stat().st_size} bytes")

    asyncio.run(run())


if __name__ == "__main__":
    main()
