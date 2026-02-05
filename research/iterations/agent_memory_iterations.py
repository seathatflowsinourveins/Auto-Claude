"""
AGENT MEMORY ITERATIONS - Long-Term Agent Memory Systems
========================================================
Persistent memory, learning, knowledge accumulation

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


AGENT_MEMORY_TOPICS = [
    # Memory Architectures
    {"topic": "Agent memory taxonomy: working, episodic, semantic, procedural", "area": "architecture"},
    {"topic": "MemGPT: operating system for LLM agents with memory", "area": "architecture"},
    {"topic": "Cognitive architectures: ACT-R, SOAR principles for agents", "area": "architecture"},
    {"topic": "Memory hierarchies: short-term to long-term consolidation", "area": "architecture"},

    # Memory Systems
    {"topic": "Letta agent memory: core, archival, recall systems", "area": "systems"},
    {"topic": "Mem0 for agents: universal memory layer patterns", "area": "systems"},
    {"topic": "Vector-based agent memory: embedding storage and retrieval", "area": "systems"},
    {"topic": "Graph-based memory: knowledge graphs for agents", "area": "systems"},

    # Learning & Adaptation
    {"topic": "Agent self-improvement: learning from experiences", "area": "learning"},
    {"topic": "Skill acquisition: agents learning new capabilities", "area": "learning"},
    {"topic": "Preference learning: adapting to user preferences", "area": "learning"},
    {"topic": "Meta-learning for agents: learning to learn", "area": "learning"},

    # Implementation
    {"topic": "Memory indexing strategies: fast retrieval patterns", "area": "implementation"},
    {"topic": "Memory compression: summarizing and compacting", "area": "implementation"},
    {"topic": "Cross-session persistence: maintaining state across runs", "area": "implementation"},
    {"topic": "Memory synchronization: multi-agent shared memory", "area": "implementation"},

    # Advanced
    {"topic": "Reflection mechanisms: agents reasoning about memory", "area": "advanced"},
    {"topic": "Memory-augmented planning: using history for decisions", "area": "advanced"},
    {"topic": "Forgetting mechanisms: relevance-based memory pruning", "area": "advanced"},
    {"topic": "Episodic future thinking: using memory for prediction", "area": "advanced"},
]


class AgentMemoryExecutor(BaseResearchExecutor):
    """Custom executor with agent memory-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"AI agent memory systems and implementations: {topic}"


if __name__ == "__main__":
    run_research(
        "agent_memory",
        "AGENT MEMORY ITERATIONS",
        AGENT_MEMORY_TOPICS,
        executor_class=AgentMemoryExecutor,
    )
