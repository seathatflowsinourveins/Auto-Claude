"""
LLMOPS ITERATIONS - LLM Operations & Deployment
================================================
LLMOps, LLM deployment, LLM infrastructure

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


LLMOPS_TOPICS = [
    # Platforms
    {"topic": "LangSmith: LLM observability", "area": "platforms"},
    {"topic": "Langfuse: LLM analytics", "area": "platforms"},
    {"topic": "Helicone: LLM monitoring", "area": "platforms"},
    {"topic": "Portkey: LLM gateway", "area": "platforms"},

    # Deployment
    {"topic": "LLM deployment: production serving", "area": "deployment"},
    {"topic": "vLLM deployment: inference", "area": "deployment"},
    {"topic": "OpenLLM: model serving", "area": "deployment"},
    {"topic": "LiteLLM: unified API", "area": "deployment"},

    # Cost
    {"topic": "LLM cost optimization: spending", "area": "cost"},
    {"topic": "Token optimization: efficiency", "area": "cost"},
    {"topic": "Model routing: cost-based", "area": "cost"},
    {"topic": "Prompt caching: cost reduction", "area": "cost"},

    # Quality
    {"topic": "Prompt testing: evaluation", "area": "quality"},
    {"topic": "Output validation: guardrails", "area": "quality"},
    {"topic": "LLM regression: testing", "area": "quality"},
    {"topic": "Human feedback: RLHF ops", "area": "quality"},

    # Infrastructure
    {"topic": "LLM gateway: API management", "area": "infrastructure"},
    {"topic": "Rate limiting: API throttling", "area": "infrastructure"},
    {"topic": "Fallback routing: reliability", "area": "infrastructure"},
    {"topic": "Multi-provider: vendor diversity", "area": "infrastructure"},
]


class LLMOpsExecutor(BaseResearchExecutor):
    """Custom executor with LLMOps-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"LLMOps LLM operations best practices: {topic}"


if __name__ == "__main__":
    run_research(
        "llmops",
        "LLMOPS ITERATIONS",
        LLMOPS_TOPICS,
        executor_class=LLMOpsExecutor,
    )
