"""
CODE GENERATION ITERATIONS - AI-Assisted Development
=====================================================
Code completion, refactoring, debugging, documentation

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


CODE_TOPICS = [
    # Code Completion
    {"topic": "Code completion: context-aware suggestions, fill-in-middle", "area": "completion"},
    {"topic": "Multi-file context: cross-file references, project understanding", "area": "completion"},
    {"topic": "Language-specific completion: syntax awareness, idioms", "area": "completion"},
    {"topic": "Copilot alternatives: Codeium, Cursor, Continue, TabNine", "area": "completion"},

    # Code Generation
    {"topic": "Natural language to code: specification parsing, implementation", "area": "generation"},
    {"topic": "Test generation: unit tests, edge cases, mocking", "area": "generation"},
    {"topic": "Documentation generation: docstrings, README, API docs", "area": "generation"},
    {"topic": "SQL generation: natural language to SQL, schema awareness", "area": "generation"},

    # Code Analysis
    {"topic": "Code review automation: style, bugs, security issues", "area": "analysis"},
    {"topic": "Bug detection: pattern matching, static analysis with LLM", "area": "analysis"},
    {"topic": "Code explanation: function summaries, complexity analysis", "area": "analysis"},
    {"topic": "Dependency analysis: import graphs, vulnerability detection", "area": "analysis"},

    # Refactoring
    {"topic": "AI-assisted refactoring: rename, extract, inline", "area": "refactoring"},
    {"topic": "Code modernization: legacy code updates, pattern migration", "area": "refactoring"},
    {"topic": "Performance optimization suggestions: bottleneck identification", "area": "refactoring"},
    {"topic": "Type inference and annotation: dynamic to static typing", "area": "refactoring"},

    # Development Workflow
    {"topic": "Commit message generation: conventional commits, context-aware", "area": "workflow"},
    {"topic": "PR description generation: summary, changes, impact", "area": "workflow"},
    {"topic": "Issue triage: labeling, assignment, priority estimation", "area": "workflow"},
    {"topic": "Code search: semantic search across repositories", "area": "workflow"},
]


class CodeExecutor(BaseResearchExecutor):
    """Custom executor with code generation-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"AI code generation: {topic}"


if __name__ == "__main__":
    run_research(
        "code",
        "CODE GENERATION ITERATIONS",
        CODE_TOPICS,
        executor_class=CodeExecutor,
    )
