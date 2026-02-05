"""
EVALUATION & BENCHMARKING ITERATIONS - Quality Assessment Patterns
===================================================================
LLM evaluation, RAG metrics, benchmark suites, automated testing

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


EVAL_TOPICS = [
    # LLM Evaluation
    {"topic": "RAGAS: RAG evaluation metrics, faithfulness, relevancy, context recall", "area": "rag_eval"},
    {"topic": "DeepEval: LLM evaluation framework, metrics, test cases", "area": "rag_eval"},
    {"topic": "TruLens: feedback functions, groundedness, answer relevance", "area": "rag_eval"},
    {"topic": "LangSmith evaluation: datasets, evaluators, experiments", "area": "rag_eval"},

    # Benchmark Suites
    {"topic": "MMLU benchmark: multitask language understanding evaluation", "area": "benchmarks"},
    {"topic": "HumanEval: code generation evaluation, pass@k metrics", "area": "benchmarks"},
    {"topic": "MT-Bench: multi-turn conversation evaluation, GPT-4 judge", "area": "benchmarks"},
    {"topic": "HELM benchmark: holistic evaluation of language models", "area": "benchmarks"},

    # Quality Metrics
    {"topic": "Semantic similarity metrics: BERT score, embedding cosine", "area": "metrics"},
    {"topic": "Factual consistency: NLI-based verification, claim extraction", "area": "metrics"},
    {"topic": "Hallucination detection: cross-reference validation, citation checking", "area": "metrics"},
    {"topic": "Response quality: coherence, fluency, helpfulness scoring", "area": "metrics"},

    # Automated Testing
    {"topic": "LLM unit testing: deterministic tests, snapshot testing", "area": "testing"},
    {"topic": "Regression testing for LLMs: golden datasets, drift detection", "area": "testing"},
    {"topic": "A/B testing for prompts: statistical significance, effect size", "area": "testing"},
    {"topic": "Continuous evaluation: CI/CD integration, automated scoring", "area": "testing"},

    # Human Evaluation
    {"topic": "Human preference collection: Likert scales, pairwise comparison", "area": "human"},
    {"topic": "Inter-annotator agreement: Cohen's kappa, Fleiss' kappa", "area": "human"},
    {"topic": "Crowdsourcing evaluation: quality control, spam detection", "area": "human"},
    {"topic": "Expert evaluation protocols: rubrics, calibration sessions", "area": "human"},
]


class EvaluationBenchmarkingExecutor(BaseResearchExecutor):
    """Custom executor with evaluation-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"LLM evaluation and benchmarking implementation: {topic}"


if __name__ == "__main__":
    run_research(
        "evaluation",
        "EVALUATION & BENCHMARKING ITERATIONS",
        EVAL_TOPICS,
        executor_class=EvaluationBenchmarkingExecutor,
    )
