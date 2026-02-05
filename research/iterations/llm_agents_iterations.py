"""
LLM AGENTS ITERATIONS - Autonomous Agents
==========================================
LLM agents, autonomous systems, planning

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


LLM_AGENTS_TOPICS = [
    # Frameworks
    {"topic": "LangChain agents: tool-using LLMs", "area": "frameworks"},
    {"topic": "AutoGPT: autonomous agents", "area": "frameworks"},
    {"topic": "CrewAI: multi-agent collaboration", "area": "frameworks"},
    {"topic": "OpenAI Assistants: API agents", "area": "frameworks"},

    # Planning
    {"topic": "ReAct: reasoning acting loop", "area": "planning"},
    {"topic": "Plan-and-execute: hierarchical", "area": "planning"},
    {"topic": "Tree of Thoughts: branching search", "area": "planning"},
    {"topic": "Reflexion: self-reflection", "area": "planning"},

    # Tools
    {"topic": "Function calling: tool use", "area": "tools"},
    {"topic": "MCP: model context protocol", "area": "tools"},
    {"topic": "Code execution: sandbox", "area": "tools"},
    {"topic": "Browser automation: web agents", "area": "tools"},

    # Memory
    {"topic": "Agent memory: long-term context", "area": "memory"},
    {"topic": "Working memory: scratch pad", "area": "memory"},
    {"topic": "Episodic memory: experience", "area": "memory"},
    {"topic": "Retrieval memory: RAG agents", "area": "memory"},

    # Applications
    {"topic": "Coding agents: software development", "area": "applications"},
    {"topic": "Research agents: information gathering", "area": "applications"},
    {"topic": "Customer service: support agents", "area": "applications"},
    {"topic": "Data analysis: analytics agents", "area": "applications"},
]


class LLMAgentsExecutor(BaseResearchExecutor):
    """Custom executor with LLM agents-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"LLM autonomous agents and agentic systems: {topic}"


if __name__ == "__main__":
    run_research(
        "llm_agents",
        "LLM AGENTS ITERATIONS",
        LLM_AGENTS_TOPICS,
        executor_class=LLMAgentsExecutor,
    )
