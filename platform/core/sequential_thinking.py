# -*- coding: utf-8 -*-
"""
Sequential Thinking Engine for Unleash Platform

A Python implementation of dynamic, reflective problem-solving through structured thought chains.
Ported from Anthropic's MCP Sequential Thinking server with enhanced features for research integration.

Key Capabilities:
- Break down complex research questions into manageable thought steps
- Support revision and branching for exploring alternative reasoning paths
- Dynamic thought count adjustment based on problem complexity
- Integration points for research execution and knowledge persistence

Based on: modelcontextprotocol/servers - sequential thinking implementation
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Awaitable
from enum import Enum
from datetime import datetime, timezone

# Handle import for both standalone and package usage
try:
    from ..utils import get_logger, log_operation, UnleashError
except ImportError:
    import logging
    def get_logger(name: str):
        return logging.getLogger(name)
    def log_operation(name: Optional[str] = None, log_args: bool = False):  # noqa
        def decorator(func):
            return func
        return decorator
    class UnleashError(Exception):
        pass

logger = get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class ThoughtStatus(Enum):
    """Status of a thought in the chain."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REVISED = "revised"
    BRANCHED = "branched"


class ThoughtType(Enum):
    """Type classification for thoughts."""
    ANALYSIS = "analysis"           # Breaking down the problem
    HYPOTHESIS = "hypothesis"       # Generating potential solutions
    VERIFICATION = "verification"   # Testing hypotheses
    SYNTHESIS = "synthesis"         # Combining findings
    REVISION = "revision"           # Correcting previous thinking
    CONCLUSION = "conclusion"       # Final answer/output
    RESEARCH_QUERY = "research_query"  # Triggering research
    BRANCH_POINT = "branch_point"   # Decision to explore alternatives


@dataclass
class ThoughtData:
    """
    Represents a single thought in the sequential thinking process.

    Attributes:
        thought: The actual thinking content
        thought_number: Current position in the sequence
        total_thoughts: Estimated total thoughts needed (can be adjusted)
        next_thought_needed: Whether more thinking is required
        is_revision: Whether this revises previous thinking
        revises_thought: Which thought number is being reconsidered
        branch_from_thought: If branching, the point of divergence
        branch_id: Unique identifier for this branch
        needs_more_thoughts: Signal that estimate was too low
        thought_type: Classification of the thought
        metadata: Additional context (research results, etc.)
    """
    thought: str
    thought_number: int
    total_thoughts: int
    next_thought_needed: bool

    # Revision support
    is_revision: bool = False
    revises_thought: Optional[int] = None

    # Branching support
    branch_from_thought: Optional[int] = None
    branch_id: Optional[str] = None

    # Dynamic expansion
    needs_more_thoughts: bool = False

    # Enhanced fields for integration
    thought_type: ThoughtType = ThoughtType.ANALYSIS
    status: ThoughtStatus = ThoughtStatus.COMPLETED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Research integration
    research_queries: List[str] = field(default_factory=list)
    research_results: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "thought": self.thought,
            "thought_number": self.thought_number,
            "total_thoughts": self.total_thoughts,
            "next_thought_needed": self.next_thought_needed,
            "is_revision": self.is_revision,
            "revises_thought": self.revises_thought,
            "branch_from_thought": self.branch_from_thought,
            "branch_id": self.branch_id,
            "needs_more_thoughts": self.needs_more_thoughts,
            "thought_type": self.thought_type.value,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "research_queries": self.research_queries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThoughtData":
        """Create from dictionary."""
        return cls(
            thought=data["thought"],
            thought_number=data["thought_number"],
            total_thoughts=data["total_thoughts"],
            next_thought_needed=data["next_thought_needed"],
            is_revision=data.get("is_revision", False),
            revises_thought=data.get("revises_thought"),
            branch_from_thought=data.get("branch_from_thought"),
            branch_id=data.get("branch_id"),
            needs_more_thoughts=data.get("needs_more_thoughts", False),
            thought_type=ThoughtType(data.get("thought_type", "analysis")),
            status=ThoughtStatus(data.get("status", "completed")),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
            duration_ms=data.get("duration_ms"),
            metadata=data.get("metadata", {}),
            research_queries=data.get("research_queries", []),
        )


@dataclass
class ThinkingSession:
    """
    A complete sequential thinking session.

    Tracks the full history of thoughts, branches, and research integration.
    """
    session_id: str
    question: str
    thought_history: List[ThoughtData] = field(default_factory=list)
    branches: Dict[str, List[ThoughtData]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    final_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if thinking is complete."""
        if not self.thought_history:
            return False
        return not self.thought_history[-1].next_thought_needed

    @property
    def current_thought_number(self) -> int:
        """Get the current thought number."""
        return len(self.thought_history)

    @property
    def total_branches(self) -> int:
        """Count total branches."""
        return len(self.branches)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "session_id": self.session_id,
            "question": self.question,
            "thought_history": [t.to_dict() for t in self.thought_history],
            "branches": {
                k: [t.to_dict() for t in v]
                for k, v in self.branches.items()
            },
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "final_answer": self.final_answer,
            "metadata": self.metadata,
        }


# =============================================================================
# Sequential Thinking Engine
# =============================================================================

class SequentialThinkingEngine:
    """
    Engine for structured, sequential problem-solving.

    Implements Anthropic's Sequential Thinking patterns with:
    - Dynamic thought expansion/contraction
    - Revision support for correcting previous thinking
    - Branching for exploring alternative paths
    - Research integration hooks
    - Persistence callbacks for Graphiti/Letta

    Example:
        engine = SequentialThinkingEngine()
        session = engine.create_session("How does async/await work in Python?")

        # Add thoughts
        thought = engine.add_thought(
            session_id=session.session_id,
            thought="First, I need to understand the event loop concept...",
            thought_type=ThoughtType.ANALYSIS,
            total_thoughts=5,
        )

        # The engine tracks history, supports revisions, and integrates with research
    """

    def __init__(
        self,
        on_thought_added: Optional[Callable[[ThinkingSession, ThoughtData], Awaitable[None]]] = None,
        on_session_complete: Optional[Callable[[ThinkingSession], Awaitable[None]]] = None,
        on_research_needed: Optional[Callable[[ThinkingSession, ThoughtData, List[str]], Awaitable[Dict[str, Any]]]] = None,
    ):
        """
        Initialize the Sequential Thinking Engine.

        Args:
            on_thought_added: Async callback when a thought is added (for persistence)
            on_session_complete: Async callback when session completes (for final storage)
            on_research_needed: Async callback when research queries are detected
        """
        self.sessions: Dict[str, ThinkingSession] = {}
        self._on_thought_added = on_thought_added
        self._on_session_complete = on_session_complete
        self._on_research_needed = on_research_needed
        self._thought_start_time: Optional[float] = None

    def create_session(
        self,
        question: str,
        initial_total_thoughts: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ThinkingSession:
        """
        Create a new thinking session.

        Args:
            question: The research question or problem to solve
            initial_total_thoughts: Initial estimate of thoughts needed
            metadata: Additional context for the session

        Returns:
            New ThinkingSession instance
        """
        session_id = str(uuid.uuid4())
        session = ThinkingSession(
            session_id=session_id,
            question=question,
            metadata=metadata or {"initial_total_thoughts": initial_total_thoughts},
        )
        self.sessions[session_id] = session

        logger.info(f"Created thinking session {session_id[:8]}... for: {question[:50]}...")
        return session

    def get_session(self, session_id: str) -> Optional[ThinkingSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    @log_operation("add_thought")
    async def add_thought(
        self,
        session_id: str,
        thought: str,
        thought_type: ThoughtType = ThoughtType.ANALYSIS,
        total_thoughts: Optional[int] = None,
        is_revision: bool = False,
        revises_thought: Optional[int] = None,
        branch_from_thought: Optional[int] = None,
        needs_more_thoughts: bool = False,
        research_queries: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ThoughtData:
        """
        Add a thought to a session.

        Args:
            session_id: The session to add to
            thought: The thinking content
            thought_type: Classification of the thought
            total_thoughts: Updated estimate (or None to keep previous)
            is_revision: Whether this revises previous thinking
            revises_thought: Which thought is being revised
            branch_from_thought: If branching, the divergence point
            needs_more_thoughts: Signal that more thinking is needed
            research_queries: Queries to execute for research
            metadata: Additional context

        Returns:
            The created ThoughtData
        """
        session = self.sessions.get(session_id)
        if not session:
            raise UnleashError(f"Session not found: {session_id}")

        # Calculate thought number
        thought_number = len(session.thought_history) + 1

        # Handle total thoughts - adjust if exceeded or explicitly set
        effective_total: int
        if total_thoughts is not None:
            effective_total = total_thoughts
        elif session.thought_history:
            effective_total = session.thought_history[-1].total_thoughts
        else:
            effective_total = int(session.metadata.get("initial_total_thoughts", 5))

        if thought_number > effective_total:
            effective_total = thought_number

        total_thoughts = effective_total

        # Generate branch ID if branching
        branch_id = None
        if branch_from_thought is not None:
            branch_id = f"branch_{branch_from_thought}_{str(uuid.uuid4())[:8]}"

        # Determine if next thought is needed
        # Only complete if this is a conclusion type and no more thoughts flagged
        next_thought_needed = (
            thought_type != ThoughtType.CONCLUSION
            or needs_more_thoughts
            or thought_number < total_thoughts
        )

        # Create thought data
        thought_data = ThoughtData(
            thought=thought,
            thought_number=thought_number,
            total_thoughts=total_thoughts,
            next_thought_needed=next_thought_needed,
            is_revision=is_revision,
            revises_thought=revises_thought,
            branch_from_thought=branch_from_thought,
            branch_id=branch_id,
            needs_more_thoughts=needs_more_thoughts,
            thought_type=thought_type,
            status=ThoughtStatus.COMPLETED,
            metadata=metadata or {},
            research_queries=research_queries or [],
        )

        # Add to history
        session.thought_history.append(thought_data)

        # Add to branch if applicable
        if branch_id:
            if branch_id not in session.branches:
                session.branches[branch_id] = []
            session.branches[branch_id].append(thought_data)

        # Mark revised thought if applicable
        if is_revision and revises_thought:
            for t in session.thought_history:
                if t.thought_number == revises_thought:
                    t.status = ThoughtStatus.REVISED
                    break

        # Execute research if queries provided
        if research_queries and self._on_research_needed:
            try:
                results = await self._on_research_needed(session, thought_data, research_queries)
                thought_data.research_results = results
            except Exception as e:
                logger.error(f"Research callback failed: {e}")
                thought_data.metadata["research_error"] = str(e)

        # Trigger callback
        if self._on_thought_added:
            try:
                await self._on_thought_added(session, thought_data)
            except Exception as e:
                logger.error(f"Thought added callback failed: {e}")

        # Check for session completion
        if not next_thought_needed:
            session.completed_at = datetime.now(timezone.utc)
            session.final_answer = thought

            if self._on_session_complete:
                try:
                    await self._on_session_complete(session)
                except Exception as e:
                    logger.error(f"Session complete callback failed: {e}")

        logger.debug(
            f"Added thought {thought_number}/{total_thoughts} "
            f"[{thought_type.value}] to session {session_id[:8]}..."
        )

        return thought_data

    async def revise_thought(
        self,
        session_id: str,
        revises_thought: int,
        new_thought: str,
        reason: Optional[str] = None,
    ) -> ThoughtData:
        """
        Revise a previous thought.

        Args:
            session_id: The session
            revises_thought: Which thought number to revise
            new_thought: The revised thinking
            reason: Why the revision is needed

        Returns:
            The revision ThoughtData
        """
        return await self.add_thought(
            session_id=session_id,
            thought=new_thought,
            thought_type=ThoughtType.REVISION,
            is_revision=True,
            revises_thought=revises_thought,
            metadata={"revision_reason": reason} if reason else None,
        )

    async def branch_thought(
        self,
        session_id: str,
        branch_from: int,
        branch_thought: str,
        branch_reason: Optional[str] = None,
    ) -> ThoughtData:
        """
        Create a branch from a previous thought.

        Args:
            session_id: The session
            branch_from: Which thought to branch from
            branch_thought: The branching thought
            branch_reason: Why branching is needed

        Returns:
            The branch ThoughtData
        """
        return await self.add_thought(
            session_id=session_id,
            thought=branch_thought,
            thought_type=ThoughtType.BRANCH_POINT,
            branch_from_thought=branch_from,
            metadata={"branch_reason": branch_reason} if branch_reason else None,
        )

    async def conclude_session(
        self,
        session_id: str,
        conclusion: str,
        confidence: Optional[float] = None,
    ) -> ThoughtData:
        """
        Add a concluding thought and complete the session.

        Args:
            session_id: The session
            conclusion: The final answer/conclusion
            confidence: Confidence level (0-1)

        Returns:
            The conclusion ThoughtData
        """
        return await self.add_thought(
            session_id=session_id,
            thought=conclusion,
            thought_type=ThoughtType.CONCLUSION,
            needs_more_thoughts=False,
            metadata={"confidence": confidence} if confidence is not None else None,
        )

    def get_thought_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of the thinking session.

        Args:
            session_id: The session

        Returns:
            Summary dictionary with statistics and key thoughts
        """
        session = self.sessions.get(session_id)
        if not session:
            return {"error": f"Session not found: {session_id}"}

        # Collect statistics
        type_counts = {}
        for t in session.thought_history:
            type_name = t.thought_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        revision_count = sum(1 for t in session.thought_history if t.is_revision)
        research_count = sum(1 for t in session.thought_history if t.research_queries)

        return {
            "session_id": session_id,
            "question": session.question,
            "total_thoughts": len(session.thought_history),
            "branches": list(session.branches.keys()),
            "branch_count": len(session.branches),
            "revision_count": revision_count,
            "research_queries_count": research_count,
            "thought_types": type_counts,
            "is_complete": session.is_complete,
            "final_answer": session.final_answer,
            "duration_seconds": (
                (session.completed_at - session.created_at).total_seconds()
                if session.completed_at else None
            ),
        }

    def format_thought_chain(
        self,
        session_id: str,
        include_metadata: bool = False,
    ) -> str:
        """
        Format the thought chain as readable text.

        Args:
            session_id: The session
            include_metadata: Whether to include metadata

        Returns:
            Formatted string representation
        """
        session = self.sessions.get(session_id)
        if not session:
            return f"Session not found: {session_id}"

        lines = [
            f"Question: {session.question}",
            f"Session: {session.session_id[:8]}...",
            "-" * 60,
        ]

        for thought in session.thought_history:
            prefix = self._get_thought_prefix(thought)
            lines.append(
                f"\n{prefix} Thought {thought.thought_number}/{thought.total_thoughts}"
            )
            lines.append(f"Type: {thought.thought_type.value}")
            lines.append(f"Content: {thought.thought}")

            if thought.is_revision:
                lines.append(f"(Revises thought #{thought.revises_thought})")
            if thought.branch_id:
                lines.append(f"(Branch: {thought.branch_id})")
            if thought.research_queries:
                lines.append(f"Research: {thought.research_queries}")

            if include_metadata and thought.metadata:
                lines.append(f"Metadata: {thought.metadata}")

        if session.final_answer:
            lines.extend([
                "",
                "=" * 60,
                "CONCLUSION:",
                session.final_answer,
            ])

        return "\n".join(lines)

    def _get_thought_prefix(self, thought: ThoughtData) -> str:
        """Get emoji prefix for thought type."""
        prefixes = {
            ThoughtType.ANALYSIS: "🔍",
            ThoughtType.HYPOTHESIS: "💡",
            ThoughtType.VERIFICATION: "✅",
            ThoughtType.SYNTHESIS: "🔄",
            ThoughtType.REVISION: "🔄",
            ThoughtType.CONCLUSION: "🎯",
            ThoughtType.RESEARCH_QUERY: "📚",
            ThoughtType.BRANCH_POINT: "🌿",
        }
        return prefixes.get(thought.thought_type, "💭")


# =============================================================================
# Factory Functions
# =============================================================================

def create_thinking_engine(
    on_thought_added: Optional[Callable] = None,
    on_session_complete: Optional[Callable] = None,
    on_research_needed: Optional[Callable] = None,
) -> SequentialThinkingEngine:
    """
    Factory function to create a configured Sequential Thinking Engine.

    Args:
        on_thought_added: Callback for persistence (e.g., Graphiti)
        on_session_complete: Callback for final storage (e.g., Letta)
        on_research_needed: Callback for research execution

    Returns:
        Configured SequentialThinkingEngine
    """
    return SequentialThinkingEngine(
        on_thought_added=on_thought_added,
        on_session_complete=on_session_complete,
        on_research_needed=on_research_needed,
    )


# =============================================================================
# Research-Enhanced Sequential Thinking Engine
# =============================================================================

# Import deep research module if available
try:
    from .deep_research import (
        DeepResearchEngine,
        ResearchState,
        ResearchDepth,
        ResearchPhase,
        SearchType,
        SearchQuery,
        create_deep_research_engine,
    )
    DEEP_RESEARCH_AVAILABLE = True
except ImportError:
    DEEP_RESEARCH_AVAILABLE = False
    DeepResearchEngine = None
    ResearchState = None
    ResearchDepth = None


class ResearchEnhancedThinkingEngine(SequentialThinkingEngine):
    """
    Sequential Thinking Engine with integrated Deep Research capabilities.

    Extends the base engine with:
    - Automatic research query detection from thought content
    - Deep research execution via Tavily/Exa integration
    - Research results synthesis into thinking chain
    - Knowledge persistence via Graphiti/Letta hooks

    The research-enhanced engine follows the PLAN → RESEARCH → ANALYZE → SYNTHESIZE → STORE
    workflow, integrating each phase into the sequential thinking process.

    Example:
        engine = ResearchEnhancedThinkingEngine()
        session = engine.create_session("What are the latest advances in quantum computing?")

        # Engine automatically detects research needs and executes them
        thought = await engine.add_thought_with_research(
            session_id=session.session_id,
            thought="I need to research recent quantum computing breakthroughs...",
            auto_research=True,
        )
        # thought.research_results now contains deep research data
    """

    def __init__(
        self,
        research_engine: Optional["DeepResearchEngine"] = None,
        default_research_depth: "ResearchDepth" = None,
        auto_research_keywords: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the Research-Enhanced Thinking Engine.

        Args:
            research_engine: Pre-configured DeepResearchEngine (auto-creates if None)
            default_research_depth: Default depth for research (QUICK, STANDARD, DEEP)
            auto_research_keywords: Keywords that trigger automatic research
            **kwargs: Additional arguments for base SequentialThinkingEngine
        """
        super().__init__(**kwargs)

        # Initialize research engine
        if DEEP_RESEARCH_AVAILABLE:
            self.research_engine = research_engine or create_deep_research_engine()
            self.default_depth = default_research_depth or ResearchDepth.STANDARD
        else:
            self.research_engine = None
            self.default_depth = None
            logger.warning("Deep research not available - running in basic mode")

        # Keywords that indicate research is needed
        self.auto_research_keywords = auto_research_keywords or [
            "research", "investigate", "find out", "look up", "search for",
            "what is", "how does", "why does", "when did", "who is",
            "latest", "recent", "current", "up-to-date", "2024", "2025", "2026",
        ]

        # Track research results per session
        self._research_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _detect_research_need(self, thought: str) -> Optional[str]:
        """
        Detect if a thought requires research.

        Args:
            thought: The thought content to analyze

        Returns:
            Extracted research query if research is needed, None otherwise
        """
        thought_lower = thought.lower()

        # Check for research keywords
        for keyword in self.auto_research_keywords:
            if keyword in thought_lower:
                # Extract a research query from the thought
                # Simple heuristic: use the sentence containing the keyword
                sentences = thought.replace("?", ".").replace("!", ".").split(".")
                for sentence in sentences:
                    if keyword in sentence.lower() and len(sentence.strip()) > 10:
                        return sentence.strip()
                # Fallback: use the whole thought
                return thought[:200]

        return None

    async def execute_research(
        self,
        query: str,
        depth: Optional["ResearchDepth"] = None,
        search_type: "SearchType" = None,
    ) -> Dict[str, Any]:
        """
        Execute deep research for a query.

        Args:
            query: The research query
            depth: Research depth (defaults to instance default)
            search_type: Type of search (WEB, NEWS, etc.)

        Returns:
            Research results dictionary
        """
        if not self.research_engine or not DEEP_RESEARCH_AVAILABLE:
            return {
                "error": "Deep research not available",
                "query": query,
                "results": [],
            }

        depth = depth or self.default_depth
        search_type = search_type or SearchType.WEB

        try:
            # Use the deep research engine
            result = await self.research_engine.research(
                query=query,
                depth=depth,
            )

            return {
                "query": query,
                "depth": depth.value if depth else "standard",
                "phase": result.phase.value if result.phase else "unknown",
                "summary": result.summary,
                "key_insights": result.key_insights,
                "confidence": result.confidence,
                "entities": result.entities,
                "relations": result.relations,
                "search_results_count": len(result.search_results),
                "raw_results": [
                    {
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.snippet[:200] if r.snippet else None,
                        "score": r.score,
                    }
                    for r in result.search_results[:5]  # Top 5 results
                ],
            }
        except Exception as e:
            logger.error(f"Research execution failed: {e}")
            return {
                "error": str(e),
                "query": query,
                "results": [],
            }

    async def add_thought_with_research(
        self,
        session_id: str,
        thought: str,
        auto_research: bool = True,
        research_query: Optional[str] = None,
        research_depth: Optional["ResearchDepth"] = None,
        **kwargs,
    ) -> ThoughtData:
        """
        Add a thought with optional automatic research execution.

        Args:
            session_id: The session ID
            thought: The thought content
            auto_research: Whether to auto-detect and execute research
            research_query: Explicit research query (overrides auto-detection)
            research_depth: Depth for research execution
            **kwargs: Additional arguments for add_thought

        Returns:
            ThoughtData with research results if applicable
        """
        # Detect research need if auto_research enabled
        query = research_query
        if auto_research and not query:
            query = self._detect_research_need(thought)

        # Execute research if query detected
        research_results = None
        if query and DEEP_RESEARCH_AVAILABLE:
            logger.info(f"Executing research for: {query[:50]}...")

            research_results = await self.execute_research(
                query=query,
                depth=research_depth,
            )

            # Cache results for the session
            if session_id not in self._research_cache:
                self._research_cache[session_id] = []
            self._research_cache[session_id].append(research_results)

            # Set thought type to RESEARCH_QUERY if research was performed
            if "thought_type" not in kwargs:
                kwargs["thought_type"] = ThoughtType.RESEARCH_QUERY

            # Add research query to the list
            kwargs.setdefault("research_queries", [])
            kwargs["research_queries"].append(query)

        # Add the thought
        thought_data = await self.add_thought(
            session_id=session_id,
            thought=thought,
            **kwargs,
        )

        # Attach research results
        if research_results:
            thought_data.research_results = research_results
            thought_data.metadata["research_executed"] = True
            thought_data.metadata["research_confidence"] = research_results.get("confidence", 0)

        return thought_data

    async def research_and_think(
        self,
        session_id: str,
        research_query: str,
        research_depth: Optional["ResearchDepth"] = None,
        synthesis_prompt: Optional[str] = None,
    ) -> List[ThoughtData]:
        """
        Execute a full research-to-thinking cycle.

        This method:
        1. Executes deep research on the query
        2. Creates RESEARCH_QUERY thought with results
        3. Creates ANALYSIS thought synthesizing findings
        4. Creates SYNTHESIS thought with key insights

        Args:
            session_id: The session ID
            research_query: The query to research
            research_depth: Depth of research
            synthesis_prompt: Optional prompt for synthesis

        Returns:
            List of ThoughtData created during the cycle
        """
        thoughts = []

        # Step 1: Execute research
        research_results = await self.execute_research(
            query=research_query,
            depth=research_depth or self.default_depth,
        )

        # Step 2: Create research thought
        research_thought = await self.add_thought(
            session_id=session_id,
            thought=f"Researching: {research_query}",
            thought_type=ThoughtType.RESEARCH_QUERY,
            research_queries=[research_query],
            metadata={"research_phase": "query"},
        )
        research_thought.research_results = research_results
        thoughts.append(research_thought)

        # Step 3: Create analysis thought
        insights = research_results.get("key_insights", [])
        summary = research_results.get("summary", "No summary available")

        analysis_content = f"Analysis of research findings:\n\n"
        analysis_content += f"Summary: {summary}\n\n"
        if insights:
            analysis_content += "Key Insights:\n"
            for i, insight in enumerate(insights[:5], 1):
                analysis_content += f"  {i}. {insight}\n"

        analysis_thought = await self.add_thought(
            session_id=session_id,
            thought=analysis_content,
            thought_type=ThoughtType.ANALYSIS,
            metadata={"research_phase": "analysis"},
        )
        thoughts.append(analysis_thought)

        # Step 4: Create synthesis thought
        entities = research_results.get("entities", [])
        confidence = research_results.get("confidence", 0)

        synthesis_content = synthesis_prompt or f"Synthesis of research on '{research_query}':\n\n"
        synthesis_content += f"Confidence: {confidence:.1%}\n"
        if entities:
            synthesis_content += f"Key entities identified: {len(entities)}\n"
        synthesis_content += f"\nBased on the research, the main findings indicate:\n{summary}"

        synthesis_thought = await self.add_thought(
            session_id=session_id,
            thought=synthesis_content,
            thought_type=ThoughtType.SYNTHESIS,
            metadata={
                "research_phase": "synthesis",
                "confidence": confidence,
            },
        )
        thoughts.append(synthesis_thought)

        return thoughts

    def get_session_research(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all research results for a session.

        Args:
            session_id: The session ID

        Returns:
            List of research result dictionaries
        """
        return self._research_cache.get(session_id, [])

    def format_research_summary(self, session_id: str) -> str:
        """
        Format a summary of all research performed in a session.

        Args:
            session_id: The session ID

        Returns:
            Formatted research summary string
        """
        research = self.get_session_research(session_id)
        if not research:
            return "No research performed in this session."

        lines = [
            f"Research Summary for Session {session_id[:8]}...",
            "=" * 60,
        ]

        for i, r in enumerate(research, 1):
            lines.append(f"\n[{i}] Query: {r.get('query', 'N/A')}")
            lines.append(f"    Depth: {r.get('depth', 'N/A')}")
            lines.append(f"    Confidence: {r.get('confidence', 0):.1%}")
            lines.append(f"    Results: {r.get('search_results_count', 0)} sources")

            insights = r.get("key_insights", [])
            if insights:
                lines.append("    Insights:")
                for insight in insights[:3]:
                    lines.append(f"      • {insight[:80]}...")

        return "\n".join(lines)


def create_research_enhanced_engine(
    research_depth: str = "standard",
    **kwargs,
) -> ResearchEnhancedThinkingEngine:
    """
    Factory function to create a Research-Enhanced Thinking Engine.

    Args:
        research_depth: Default research depth ("quick", "standard", "deep")
        **kwargs: Additional arguments for the engine

    Returns:
        Configured ResearchEnhancedThinkingEngine
    """
    depth = None
    if DEEP_RESEARCH_AVAILABLE and ResearchDepth:
        depth_map = {
            "quick": ResearchDepth.QUICK,
            "standard": ResearchDepth.STANDARD,
            "deep": ResearchDepth.DEEP,
        }
        depth = depth_map.get(research_depth.lower())

    return ResearchEnhancedThinkingEngine(
        default_research_depth=depth,
        **kwargs,
    )
