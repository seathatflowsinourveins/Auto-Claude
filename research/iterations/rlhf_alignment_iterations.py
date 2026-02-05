"""
RLHF & ALIGNMENT ITERATIONS - Human Feedback
============================================
RLHF, DPO, alignment, preference learning

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


RLHF_ALIGNMENT_TOPICS = [
    # RLHF
    {"topic": "RLHF: reinforcement learning human feedback", "area": "rlhf"},
    {"topic": "Reward modeling: learning human preferences", "area": "rlhf"},
    {"topic": "PPO: proximal policy optimization for LLMs", "area": "rlhf"},
    {"topic": "InstructGPT: following instructions", "area": "rlhf"},

    # Alternative Methods
    {"topic": "DPO: direct preference optimization", "area": "alternatives"},
    {"topic": "ORPO: odds ratio preference optimization", "area": "alternatives"},
    {"topic": "IPO: identity preference optimization", "area": "alternatives"},
    {"topic": "KTO: Kahneman-Tversky optimization", "area": "alternatives"},

    # Data Collection
    {"topic": "Human preference data: collection methods", "area": "data"},
    {"topic": "Comparison data: pairwise preferences", "area": "data"},
    {"topic": "Red teaming: adversarial testing", "area": "data"},
    {"topic": "Synthetic preferences: AI-generated", "area": "data"},

    # Constitutional AI
    {"topic": "Constitutional AI: self-improvement", "area": "constitutional"},
    {"topic": "RLAIF: RL from AI feedback", "area": "constitutional"},
    {"topic": "Principle-based training: rules", "area": "constitutional"},
    {"topic": "Self-critique: model reflection", "area": "constitutional"},

    # Challenges
    {"topic": "Reward hacking: gaming metrics", "area": "challenges"},
    {"topic": "Goodhart's law: optimization problems", "area": "challenges"},
    {"topic": "Preference inconsistency: human disagreement", "area": "challenges"},
    {"topic": "Scalable oversight: supervising AI", "area": "challenges"},
]


class RLHFAlignmentExecutor(BaseResearchExecutor):
    """Custom executor with RLHF alignment-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"RLHF and AI alignment: {topic}"


if __name__ == "__main__":
    run_research(
        "rlhf_alignment",
        "RLHF & ALIGNMENT ITERATIONS",
        RLHF_ALIGNMENT_TOPICS,
        executor_class=RLHFAlignmentExecutor,
    )
