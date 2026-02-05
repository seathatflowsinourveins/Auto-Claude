"""
REASONING LLM ITERATIONS - Chain of Thought
============================================
Reasoning, CoT, structured thinking, o1

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


REASONING_LLM_TOPICS = [
    # Chain of Thought
    {"topic": "Chain-of-thought prompting: step-by-step", "area": "cot"},
    {"topic": "Zero-shot CoT: let's think step by step", "area": "cot"},
    {"topic": "Few-shot CoT: exemplar reasoning", "area": "cot"},
    {"topic": "Self-consistency: multiple reasoning paths", "area": "cot"},

    # Reasoning Models
    {"topic": "OpenAI o1: reasoning models, thinking", "area": "models"},
    {"topic": "Claude reasoning: extended thinking", "area": "models"},
    {"topic": "DeepSeek reasoning: R1 model", "area": "models"},
    {"topic": "Reasoning fine-tuning: training for logic", "area": "models"},

    # Techniques
    {"topic": "Tree of thoughts: branching reasoning", "area": "techniques"},
    {"topic": "ReAct: reasoning and acting", "area": "techniques"},
    {"topic": "Scratchpad reasoning: working memory", "area": "techniques"},
    {"topic": "Deliberation: iterative refinement", "area": "techniques"},

    # Evaluation
    {"topic": "Math reasoning: GSM8K, MATH", "area": "evaluation"},
    {"topic": "Logical reasoning: deduction, induction", "area": "evaluation"},
    {"topic": "Commonsense reasoning: world knowledge", "area": "evaluation"},
    {"topic": "Multi-step reasoning: complex problems", "area": "evaluation"},

    # Applications
    {"topic": "Code reasoning: program synthesis", "area": "applications"},
    {"topic": "Scientific reasoning: hypothesis generation", "area": "applications"},
    {"topic": "Legal reasoning: case analysis", "area": "applications"},
    {"topic": "Medical reasoning: diagnosis chains", "area": "applications"},
]


class ReasoningLLMExecutor(BaseResearchExecutor):
    """Custom executor with reasoning LLM-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"LLM reasoning: {topic}"


if __name__ == "__main__":
    run_research(
        "reasoning_llm",
        "REASONING LLM ITERATIONS",
        REASONING_LLM_TOPICS,
        executor_class=ReasoningLLMExecutor,
    )
