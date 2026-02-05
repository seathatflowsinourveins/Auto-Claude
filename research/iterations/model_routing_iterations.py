"""
MODEL ROUTING ITERATIONS - Multi-Model Orchestration
=====================================================
Router patterns, cost optimization, performance routing

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


ROUTING_TOPICS = [
    # Routing Fundamentals
    {"topic": "LLM router architecture: query classification, model selection", "area": "fundamentals"},
    {"topic": "Intent-based routing: matching queries to specialized models", "area": "fundamentals"},
    {"topic": "Complexity estimation: predicting task difficulty for routing", "area": "fundamentals"},
    {"topic": "Semantic router: embedding-based query classification", "area": "fundamentals"},

    # Cost Optimization
    {"topic": "Cost-aware routing: balancing quality vs API costs", "area": "cost"},
    {"topic": "Tiered model strategy: cheap fast models vs expensive capable", "area": "cost"},
    {"topic": "Token budget management: staying within cost limits", "area": "cost"},
    {"topic": "Batch optimization: grouping requests for efficiency", "area": "cost"},

    # Performance Routing
    {"topic": "Latency-based routing: choosing fastest model per task", "area": "performance"},
    {"topic": "Load balancing across providers: OpenAI, Anthropic, Google", "area": "performance"},
    {"topic": "Fallback chains: handling rate limits and failures", "area": "performance"},
    {"topic": "Speculative execution: parallel model calls with early termination", "area": "performance"},

    # Advanced Patterns
    {"topic": "Mixture of experts routing: specialized model ensemble", "area": "advanced"},
    {"topic": "Cascading models: try cheap first, escalate on failure", "area": "advanced"},
    {"topic": "Quality-aware routing: matching task to model capability", "area": "advanced"},
    {"topic": "Dynamic routing: learning optimal paths from feedback", "area": "advanced"},

    # Implementation
    {"topic": "LiteLLM for model routing: unified API across providers", "area": "implementation"},
    {"topic": "OpenRouter integration: single API for 100+ models", "area": "implementation"},
    {"topic": "Custom router implementation: building routing logic", "area": "implementation"},
    {"topic": "Router evaluation: measuring routing decision quality", "area": "implementation"},
]


class RoutingExecutor(BaseResearchExecutor):
    """Custom executor with model routing-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"LLM model routing: {topic}"


if __name__ == "__main__":
    run_research(
        "routing",
        "MODEL ROUTING ITERATIONS",
        ROUTING_TOPICS,
        executor_class=RoutingExecutor,
    )
