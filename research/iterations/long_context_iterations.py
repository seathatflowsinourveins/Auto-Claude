"""
LONG CONTEXT ITERATIONS - Extended Context
==========================================
Long context, million tokens, RAG vs context

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


LONG_CONTEXT_TOPICS = [
    # Architectures
    {"topic": "Long context transformers: 1M+ tokens", "area": "architectures"},
    {"topic": "Claude 200K context: extended window", "area": "architectures"},
    {"topic": "Gemini long context: 2M tokens", "area": "architectures"},
    {"topic": "GPT-4 Turbo: 128K context", "area": "architectures"},

    # Techniques
    {"topic": "RoPE scaling: position embedding extension", "area": "techniques"},
    {"topic": "ALiBi: attention with linear biases", "area": "techniques"},
    {"topic": "Ring attention: distributed context", "area": "techniques"},
    {"topic": "Landmark attention: efficient retrieval", "area": "techniques"},

    # Challenges
    {"topic": "Lost in the middle: retrieval degradation", "area": "challenges"},
    {"topic": "Context utilization: effective usage", "area": "challenges"},
    {"topic": "Needle in haystack: long retrieval", "area": "challenges"},
    {"topic": "Memory constraints: inference cost", "area": "challenges"},

    # Applications
    {"topic": "Document QA: long document understanding", "area": "applications"},
    {"topic": "Code understanding: repository context", "area": "applications"},
    {"topic": "Book summarization: entire book input", "area": "applications"},
    {"topic": "Multi-document: cross-reference", "area": "applications"},

    # RAG vs Context
    {"topic": "RAG vs long context: tradeoffs", "area": "comparison"},
    {"topic": "Hybrid approaches: RAG + long context", "area": "comparison"},
    {"topic": "Cost comparison: tokens vs retrieval", "area": "comparison"},
    {"topic": "Quality comparison: accuracy analysis", "area": "comparison"},
]


class LongContextExecutor(BaseResearchExecutor):
    """Custom executor with long context-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"Long context LLM: {topic}"


if __name__ == "__main__":
    run_research(
        "long_context",
        "LONG CONTEXT ITERATIONS",
        LONG_CONTEXT_TOPICS,
        executor_class=LongContextExecutor,
    )
