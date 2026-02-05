"""
BATTLE-TESTED ITERATIONS - Comprehensive SDK, Memory, Gap Resolution
=====================================================================
Full iteration across all aspects: SDKs, Memory, Gaps, Integration, Beyond

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


BATTLE_TESTED_TOPICS = [
    # SDK Patterns
    {"topic": "Exa neural search: auto vs keyword vs neural modes production patterns", "area": "sdk_patterns"},
    {"topic": "Tavily AI search: search_depth advanced vs basic, include_answer optimization", "area": "sdk_patterns"},
    {"topic": "Jina embeddings v3: task parameter optimization for retrieval vs classification", "area": "sdk_patterns"},
    {"topic": "Perplexity sonar-pro: citations, streaming, multi-turn research patterns", "area": "sdk_patterns"},
    {"topic": "Qdrant vector DB: HNSW indexing, quantization, filtering production patterns", "area": "sdk_patterns"},

    # Memory Systems
    {"topic": "Hierarchical memory architecture: working memory vs episodic vs semantic", "area": "memory_patterns"},
    {"topic": "Memory consolidation: summarization, compression, importance scoring", "area": "memory_patterns"},
    {"topic": "Cross-session persistence: state serialization, context restoration", "area": "memory_patterns"},
    {"topic": "Memory retrieval: similarity search vs recency vs importance weighting", "area": "memory_patterns"},
    {"topic": "Forgetting mechanisms: decay functions, capacity limits, pruning strategies", "area": "memory_patterns"},

    # Gap Resolution
    {"topic": "Rate limiting and backoff: exponential backoff, jitter, circuit breakers", "area": "gap_resolution"},
    {"topic": "Error handling patterns: retry logic, fallbacks, graceful degradation", "area": "gap_resolution"},
    {"topic": "Caching strategies: semantic cache, TTL, invalidation, cache warming", "area": "gap_resolution"},
    {"topic": "Query optimization: intent classification, routing, query rewriting", "area": "gap_resolution"},
    {"topic": "Cost optimization: model routing, token reduction, batching strategies", "area": "gap_resolution"},

    # Integration Patterns
    {"topic": "LangChain LCEL: RunnableSequence, RunnableParallel, RunnableBranch patterns", "area": "integration"},
    {"topic": "LangGraph StateGraph: nodes, edges, conditional routing, persistence", "area": "integration"},
    {"topic": "DSPy signatures: ChainOfThought, ReAct, ProgramOfThought optimization", "area": "integration"},
    {"topic": "CrewAI agents: roles, goals, backstory, task delegation patterns", "area": "integration"},
    {"topic": "AutoGen conversable agents: group chat, function calling, code execution", "area": "integration"},

    # Beyond - Advanced Patterns
    {"topic": "Agentic RAG: iterative retrieval, self-correction, tool-augmented generation", "area": "beyond"},
    {"topic": "Multi-agent debate: adversarial verification, consensus mechanisms", "area": "beyond"},
    {"topic": "Reflection patterns: self-critique, iterative refinement, meta-cognition", "area": "beyond"},
    {"topic": "Planning algorithms: ReAct, Plan-and-Execute, Tree of Thoughts implementation", "area": "beyond"},
    {"topic": "Tool use optimization: function calling, tool selection, parallel execution", "area": "beyond"},
]


class BattleTestedExecutor(BaseResearchExecutor):
    """Custom executor with battle-tested pattern-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        area_prompts = {
            "sdk_patterns": "SDK production patterns and best practices",
            "memory_patterns": "Memory system architecture and implementation",
            "gap_resolution": "Gap resolution patterns and implementation",
            "integration": "Framework integration patterns and best practices",
            "beyond": "Advanced AI patterns and cutting-edge implementations",
        }
        prefix = area_prompts.get(area, "Technical deep-dive")
        return f"{prefix}: {topic}"


if __name__ == "__main__":
    run_research(
        "battle_tested",
        "BATTLE-TESTED ITERATIONS",
        BATTLE_TESTED_TOPICS,
        executor_class=BattleTestedExecutor,
    )
