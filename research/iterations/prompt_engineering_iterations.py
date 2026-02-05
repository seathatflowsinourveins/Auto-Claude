"""
PROMPT ENGINEERING ITERATIONS - Advanced Prompting Techniques
=============================================================
System prompts, few-shot, chain-of-thought, structured outputs

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


PROMPT_TOPICS = [
    # System Prompts
    {"topic": "System prompt design: persona, constraints, output format", "area": "system"},
    {"topic": "Role-playing prompts: expert personas, domain knowledge injection", "area": "system"},
    {"topic": "Instruction hierarchy: system vs user prompt precedence", "area": "system"},
    {"topic": "Meta-prompting: prompts that generate prompts", "area": "system"},

    # Few-Shot Learning
    {"topic": "Few-shot example selection: similarity-based, diverse sampling", "area": "fewshot"},
    {"topic": "Dynamic few-shot: runtime example retrieval from vector DB", "area": "fewshot"},
    {"topic": "Example formatting: input-output pairs, chain-of-thought examples", "area": "fewshot"},
    {"topic": "Zero-shot vs few-shot: task complexity thresholds", "area": "fewshot"},

    # Reasoning Techniques
    {"topic": "Chain-of-thought prompting: step-by-step reasoning elicitation", "area": "reasoning"},
    {"topic": "Self-consistency: multiple reasoning paths with voting", "area": "reasoning"},
    {"topic": "Tree of thoughts: branching exploration, backtracking", "area": "reasoning"},
    {"topic": "ReAct: reasoning and acting interleaved", "area": "reasoning"},

    # Structured Outputs
    {"topic": "JSON mode: schema enforcement, nested structures", "area": "structured"},
    {"topic": "Function calling: tool definitions, parameter extraction", "area": "structured"},
    {"topic": "Pydantic output parsing: type validation, error handling", "area": "structured"},
    {"topic": "XML and markdown structured outputs: parsing strategies", "area": "structured"},

    # Advanced Techniques
    {"topic": "Prompt compression: LLMLingua, context distillation", "area": "advanced"},
    {"topic": "Prompt injection defense: delimiters, input validation", "area": "advanced"},
    {"topic": "Multi-turn prompt design: context management, summarization", "area": "advanced"},
    {"topic": "Prompt versioning and testing: A/B testing, regression", "area": "advanced"},
]


class PromptEngineeringExecutor(BaseResearchExecutor):
    """Custom executor with prompt engineering-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"Prompt engineering best practices and techniques: {topic}"


if __name__ == "__main__":
    run_research(
        "prompts",
        "PROMPT ENGINEERING ITERATIONS",
        PROMPT_TOPICS,
        executor_class=PromptEngineeringExecutor,
    )
