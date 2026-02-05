"""
STRUCTURED OUTPUT ITERATIONS - Schema-Compliant Generation
==========================================================
JSON mode, function calling, structured extraction

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


STRUCTURED_TOPICS = [
    # JSON Mode
    {"topic": "JSON mode in LLMs: guaranteed valid JSON output", "area": "json"},
    {"topic": "Schema-constrained generation: Pydantic models for LLMs", "area": "json"},
    {"topic": "Nested JSON structures: complex object generation", "area": "json"},
    {"topic": "JSON repair: fixing malformed LLM outputs", "area": "json"},

    # Function Calling
    {"topic": "Function calling APIs: OpenAI, Anthropic, Gemini formats", "area": "functions"},
    {"topic": "Tool use patterns: when to use function calling", "area": "functions"},
    {"topic": "Parallel function calls: multiple tools in one response", "area": "functions"},
    {"topic": "Function call validation: type checking arguments", "area": "functions"},

    # Extraction
    {"topic": "Named entity extraction: people, places, organizations", "area": "extraction"},
    {"topic": "Relation extraction: identifying connections between entities", "area": "extraction"},
    {"topic": "Form filling: extracting structured data from text", "area": "extraction"},
    {"topic": "Table extraction: converting text to tabular data", "area": "extraction"},

    # Frameworks
    {"topic": "Instructor library: structured outputs with Pydantic", "area": "frameworks"},
    {"topic": "Outlines: constrained generation with grammars", "area": "frameworks"},
    {"topic": "LMQL: query language for LLM outputs", "area": "frameworks"},
    {"topic": "Guidance: template-based structured generation", "area": "frameworks"},

    # Advanced
    {"topic": "Constrained decoding: token-level schema enforcement", "area": "advanced"},
    {"topic": "Grammar-guided generation: BNF, regex constraints", "area": "advanced"},
    {"topic": "Multi-step structured outputs: chains of schema objects", "area": "advanced"},
    {"topic": "Streaming structured outputs: incremental JSON parsing", "area": "advanced"},
]


class StructuredExecutor(BaseResearchExecutor):
    """Custom executor with structured output-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"LLM structured outputs: {topic}"


if __name__ == "__main__":
    run_research(
        "structured",
        "STRUCTURED OUTPUT ITERATIONS",
        STRUCTURED_TOPICS,
        executor_class=StructuredExecutor,
    )
