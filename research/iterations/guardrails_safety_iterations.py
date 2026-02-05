"""
GUARDRAILS & SAFETY ITERATIONS - AI Security & Filtering
=========================================================
Input validation, output filtering, content moderation

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


GUARDRAILS_TOPICS = [
    # Input Guardrails
    {"topic": "Prompt injection defense: detecting and blocking attacks", "area": "input"},
    {"topic": "Input validation: sanitizing user prompts, length limits", "area": "input"},
    {"topic": "Jailbreak detection: identifying bypass attempts", "area": "input"},
    {"topic": "PII detection in prompts: protecting sensitive data", "area": "input"},

    # Output Guardrails
    {"topic": "Output filtering: blocking harmful content generation", "area": "output"},
    {"topic": "Hallucination detection: fact-checking LLM outputs", "area": "output"},
    {"topic": "Response validation: schema compliance, format checking", "area": "output"},
    {"topic": "Toxicity filtering: detecting offensive content", "area": "output"},

    # Frameworks
    {"topic": "NeMo Guardrails: NVIDIA's programmable rails", "area": "frameworks"},
    {"topic": "Guardrails AI: Python framework for LLM validation", "area": "frameworks"},
    {"topic": "LlamaGuard: Meta's content safety classifier", "area": "frameworks"},
    {"topic": "Rebuff: self-hardening prompt injection detector", "area": "frameworks"},

    # Safety Patterns
    {"topic": "Constitutional AI: self-correcting outputs", "area": "safety"},
    {"topic": "Safety classifiers: multi-label content moderation", "area": "safety"},
    {"topic": "Red team testing: adversarial safety evaluation", "area": "safety"},
    {"topic": "Responsible AI guidelines: ethical deployment", "area": "safety"},

    # Production
    {"topic": "Guardrail latency: balancing safety and speed", "area": "production"},
    {"topic": "Guardrail monitoring: tracking block rates, false positives", "area": "production"},
    {"topic": "Cascading guardrails: multiple layers of protection", "area": "production"},
    {"topic": "Guardrail bypass logging: audit trails for security", "area": "production"},
]


class GuardrailsSafetyExecutor(BaseResearchExecutor):
    """Custom executor with guardrails-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"LLM guardrails, safety, and content moderation: {topic}"


if __name__ == "__main__":
    run_research(
        "guardrails",
        "GUARDRAILS & SAFETY ITERATIONS",
        GUARDRAILS_TOPICS,
        executor_class=GuardrailsSafetyExecutor,
    )
