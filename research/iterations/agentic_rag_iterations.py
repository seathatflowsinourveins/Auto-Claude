"""
AGENTIC RAG ITERATIONS - Agent-Enhanced Retrieval
==================================================
Agentic RAG, agent retrieval, multi-step

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


AGENTIC_RAG_TOPICS = [
    # Patterns
    {"topic": "Agentic RAG: agent-driven retrieval", "area": "patterns"},
    {"topic": "Self-RAG: self-reflective retrieval", "area": "patterns"},
    {"topic": "Corrective RAG: iterative refinement", "area": "patterns"},
    {"topic": "Adaptive RAG: dynamic strategy", "area": "patterns"},

    # Multi-step
    {"topic": "Multi-hop reasoning: chain retrieval", "area": "multi_step"},
    {"topic": "Query decomposition: sub-questions", "area": "multi_step"},
    {"topic": "Iterative retrieval: progressive", "area": "multi_step"},
    {"topic": "HyDE: hypothetical documents", "area": "multi_step"},

    # Agents
    {"topic": "RAG agents: LangChain agents", "area": "agents"},
    {"topic": "Tool-augmented retrieval: functions", "area": "agents"},
    {"topic": "Routing agents: query routing", "area": "agents"},
    {"topic": "Research agents: deep search", "area": "agents"},

    # Evaluation
    {"topic": "RAG evaluation: RAGAS metrics", "area": "evaluation"},
    {"topic": "Faithfulness: grounding check", "area": "evaluation"},
    {"topic": "Answer relevancy: quality metrics", "area": "evaluation"},
    {"topic": "Context precision: retrieval quality", "area": "evaluation"},

    # Production
    {"topic": "RAG pipelines: production patterns", "area": "production"},
    {"topic": "Caching: retrieval optimization", "area": "production"},
    {"topic": "Streaming RAG: real-time", "area": "production"},
    {"topic": "RAG monitoring: observability", "area": "production"},
]


class AgenticRAGExecutor(BaseResearchExecutor):
    """Custom executor with Agentic RAG-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"Agentic RAG agent-driven retrieval augmented generation: {topic}"


if __name__ == "__main__":
    run_research(
        "agentic_rag",
        "AGENTIC RAG ITERATIONS",
        AGENTIC_RAG_TOPICS,
        executor_class=AgenticRAGExecutor,
    )
