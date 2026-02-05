"""
TOOL USE ITERATIONS - Function Calling & Tool Patterns
=======================================================
Tool definitions, execution, error handling, composition

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


TOOL_TOPICS = [
    # Tool Definition
    {"topic": "OpenAI function calling: JSON schema, parameter types", "area": "definition"},
    {"topic": "Anthropic tool use: tool_choice, parallel tool calls", "area": "definition"},
    {"topic": "Tool description best practices: clarity, examples, constraints", "area": "definition"},
    {"topic": "Dynamic tool generation: runtime tool creation from APIs", "area": "definition"},

    # Tool Execution
    {"topic": "Tool call parsing: structured extraction, validation", "area": "execution"},
    {"topic": "Tool result formatting: success, error, partial results", "area": "execution"},
    {"topic": "Parallel tool execution: concurrent calls, result aggregation", "area": "execution"},
    {"topic": "Tool timeouts and retries: deadline handling, fallbacks", "area": "execution"},

    # Tool Composition
    {"topic": "Tool chaining: sequential execution, data flow", "area": "composition"},
    {"topic": "Tool routing: selecting appropriate tool based on query", "area": "composition"},
    {"topic": "Multi-tool orchestration: complex workflows, dependencies", "area": "composition"},
    {"topic": "Tool augmented generation: interleaved reasoning and tools", "area": "composition"},

    # Safety & Validation
    {"topic": "Tool input validation: type checking, sanitization", "area": "safety"},
    {"topic": "Tool permission systems: scopes, user consent", "area": "safety"},
    {"topic": "Tool output verification: result validation, hallucination check", "area": "safety"},
    {"topic": "Sandboxed tool execution: isolation, resource limits", "area": "safety"},

    # Advanced Patterns
    {"topic": "Computer use tools: browser, file system, GUI automation", "area": "advanced"},
    {"topic": "Code interpreter patterns: execution, state management", "area": "advanced"},
    {"topic": "API wrapper tools: REST, GraphQL, database queries", "area": "advanced"},
    {"topic": "Human-in-the-loop tools: approval workflows, escalation", "area": "advanced"},
]


class ToolExecutor(BaseResearchExecutor):
    """Custom executor with tool use-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"LLM tool use implementation: {topic}"


if __name__ == "__main__":
    run_research(
        "tools",
        "TOOL USE ITERATIONS",
        TOOL_TOPICS,
        executor_class=ToolExecutor,
    )
