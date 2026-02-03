"""
Claude Research Bridge - Auto-Research Integration
===================================================

This module enables Claude to automatically trigger deep research
when it detects complex questions that would benefit from multi-source
investigation.

Usage:
    from core.claude_research_bridge import should_deep_dive, auto_research

    # Check if a query warrants deep research
    if should_deep_dive(user_query):
        result = await auto_research(user_query)

    # Or use the decorator
    @with_research_context
    async def answer_query(query: str) -> str:
        # Research results are automatically injected
        pass
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

# Import the swarm
try:
    from .ultimate_research_swarm import (
        UltimateResearchSwarm,
        UltimateResearchResult,
        ResearchDepth,
        get_ultimate_swarm,
        quick_research,
        comprehensive_research,
        deep_research,
        deep_dive_research,
    )
except ImportError:
    from ultimate_research_swarm import (
        UltimateResearchSwarm,
        UltimateResearchResult,
        ResearchDepth,
        get_ultimate_swarm,
        quick_research,
        comprehensive_research,
        deep_research,
        deep_dive_research,
    )


# =============================================================================
# Query Classification
# =============================================================================

# Patterns that indicate complex research is needed
RESEARCH_TRIGGERS = [
    # Comparison patterns
    r"\bvs\b",
    r"\bversus\b",
    r"compare\w*",
    r"comparison",
    r"difference\s+between",
    r"which\s+is\s+better",
    r"pros\s+and\s+cons",

    # Deep analysis patterns
    r"how\s+does\s+\w+\s+work",
    r"explain\s+in\s+detail",
    r"comprehensive",
    r"in-depth",
    r"architecture",
    r"best\s+practices",

    # Technical patterns
    r"implement\w*",
    r"production",
    r"scalab\w+",
    r"performance",
    r"optimization",

    # Research patterns
    r"latest\s+research",
    r"state\s+of\s+the\s+art",
    r"current\s+trends",
    r"what\s+are\s+the\s+options",
    r"alternatives\s+to",

    # Framework/tool patterns
    r"langchain|langgraph|langsmith",
    r"crewai|autogen",
    r"rag|retrieval|vector",
    r"embedding\w*",
    r"consensus|distributed",
    r"fine-?tun\w*",
    r"lora|qlora",
]

# Quick response patterns (don't need deep research)
QUICK_RESPONSE_PATTERNS = [
    r"^(hi|hello|hey|thanks|thank you)",
    r"^what\s+time",
    r"^(yes|no|ok|okay)$",
    r"^(can you|could you)\s+(help|assist)",
]

# SDK/documentation patterns (use Context7)
SDK_PATTERNS = [
    r"(api|sdk)\s+(docs?|documentation|reference)",
    r"how\s+to\s+(use|install|import)",
    r"(function|method|class)\s+signature",
    r"(langchain|langgraph|anthropic|openai|react|fastapi|pytorch)",
]


@dataclass
class QueryAnalysis:
    """Analysis of a user query for research routing."""
    query: str
    needs_research: bool
    research_depth: ResearchDepth
    is_sdk_query: bool
    detected_patterns: list[str]
    confidence: float


def analyze_query(query: str) -> QueryAnalysis:
    """
    Analyze a query to determine if deep research is needed.

    Returns QueryAnalysis with routing recommendations.
    """
    query_lower = query.lower().strip()
    detected_patterns = []

    # Check for quick response patterns
    for pattern in QUICK_RESPONSE_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return QueryAnalysis(
                query=query,
                needs_research=False,
                research_depth=ResearchDepth.QUICK,
                is_sdk_query=False,
                detected_patterns=["quick_response"],
                confidence=0.95,
            )

    # Check for research triggers
    trigger_count = 0
    for pattern in RESEARCH_TRIGGERS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            trigger_count += 1
            detected_patterns.append(pattern)

    # Check for SDK patterns
    is_sdk_query = any(
        re.search(p, query_lower, re.IGNORECASE)
        for p in SDK_PATTERNS
    )
    if is_sdk_query:
        detected_patterns.append("sdk_query")

    # Determine depth based on triggers
    if trigger_count == 0:
        needs_research = False
        depth = ResearchDepth.QUICK
        confidence = 0.7
    elif trigger_count == 1:
        needs_research = True
        depth = ResearchDepth.STANDARD
        confidence = 0.75
    elif trigger_count == 2:
        needs_research = True
        depth = ResearchDepth.COMPREHENSIVE
        confidence = 0.85
    else:  # 3+
        needs_research = True
        depth = ResearchDepth.DEEP
        confidence = 0.95

    # Long queries often need more research
    word_count = len(query.split())
    if word_count > 20:
        if depth == ResearchDepth.STANDARD:
            depth = ResearchDepth.COMPREHENSIVE
        elif depth == ResearchDepth.COMPREHENSIVE:
            depth = ResearchDepth.DEEP

    return QueryAnalysis(
        query=query,
        needs_research=needs_research,
        research_depth=depth,
        is_sdk_query=is_sdk_query,
        detected_patterns=detected_patterns[:5],
        confidence=confidence,
    )


def should_deep_dive(query: str) -> bool:
    """Quick check if a query warrants deep research."""
    analysis = analyze_query(query)
    return analysis.needs_research and analysis.research_depth in [
        ResearchDepth.COMPREHENSIVE,
        ResearchDepth.DEEP
    ]


# =============================================================================
# Auto-Research Functions
# =============================================================================

async def auto_research(
    query: str,
    memory_key: Optional[str] = None,
) -> Optional[UltimateResearchResult]:
    """
    Automatically research a query based on its complexity.

    Returns None if research is not needed, otherwise returns
    the research result.
    """
    analysis = analyze_query(query)

    if not analysis.needs_research:
        return None

    swarm = get_ultimate_swarm()
    await swarm.initialize()

    if analysis.research_depth == ResearchDepth.QUICK:
        return await swarm.research(query, ResearchDepth.QUICK)
    elif analysis.research_depth == ResearchDepth.STANDARD:
        return await swarm.research(query, ResearchDepth.STANDARD, memory_key=memory_key)
    elif analysis.research_depth == ResearchDepth.COMPREHENSIVE:
        return await swarm.research(query, ResearchDepth.COMPREHENSIVE, memory_key=memory_key)
    else:  # DEEP
        return await swarm.deep_dive(query, memory_key=memory_key)


async def research_for_response(
    query: str,
    max_context_chars: int = 4000,
) -> str:
    """
    Research a query and format results for Claude's context.

    Returns a formatted string that can be prepended to Claude's
    response context.
    """
    result = await auto_research(query)

    if not result:
        return ""

    # Format research for context injection
    lines = [
        "---",
        f"**Research Results** ({len(result.sources)} sources, {result.confidence:.0%} confidence)",
        "",
    ]

    # Add key findings
    if result.key_findings:
        lines.append("**Key Findings:**")
        for finding in result.key_findings[:3]:
            lines.append(f"- {finding[:200]}")
        lines.append("")

    # Add summary (truncated)
    if result.summary:
        summary = result.summary[:max_context_chars - sum(len(l) for l in lines)]
        lines.append("**Summary:**")
        lines.append(summary)

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Decorator for Auto-Research
# =============================================================================

def with_research_context(func: Callable) -> Callable:
    """
    Decorator that automatically injects research context.

    Usage:
        @with_research_context
        async def answer(query: str, context: str = "") -> str:
            # context will contain research results if needed
            return f"Based on my research: {context}..."
    """
    async def wrapper(query: str, *args, **kwargs) -> Any:
        research_context = await research_for_response(query)
        kwargs["research_context"] = research_context
        return await func(query, *args, **kwargs)

    return wrapper


# =============================================================================
# CLI
# =============================================================================

async def main():
    """Test the research bridge."""
    import sys

    test_queries = [
        "Hello, how are you?",
        "What is LangGraph?",
        "Compare RAG vs fine-tuning for production LLM applications",
        "What are the best practices for implementing distributed consensus in microservices?",
        "How does Claude's MCP architecture work and what are the available tool patterns?",
    ]

    if len(sys.argv) > 1:
        test_queries = [" ".join(sys.argv[1:])]

    print("=" * 60)
    print("CLAUDE RESEARCH BRIDGE TEST")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: {query[:50]}...")
        analysis = analyze_query(query)

        print(f"  Needs Research: {analysis.needs_research}")
        print(f"  Depth: {analysis.research_depth.value}")
        print(f"  SDK Query: {analysis.is_sdk_query}")
        print(f"  Patterns: {analysis.detected_patterns[:3]}")
        print(f"  Confidence: {analysis.confidence:.0%}")

        if should_deep_dive(query):
            print("  -> Would trigger DEEP DIVE")


if __name__ == "__main__":
    asyncio.run(main())
