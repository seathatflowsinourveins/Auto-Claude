"""
RERANKING ITERATIONS - Retrieval Reranking
===========================================
Reranking models, cross-encoders, retrieval

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


RERANKING_TOPICS = [
    # Models
    {"topic": "Cohere Rerank: reranking API", "area": "models"},
    {"topic": "BGE Reranker: BAAI cross-encoder", "area": "models"},
    {"topic": "Jina Reranker: jina-reranker-v2", "area": "models"},
    {"topic": "ColBERT: late interaction", "area": "models"},

    # Architecture
    {"topic": "Cross-encoder: bi-encoder comparison", "area": "architecture"},
    {"topic": "Late interaction: efficient reranking", "area": "architecture"},
    {"topic": "Multi-vector retrieval: ColBERT", "area": "architecture"},
    {"topic": "Hybrid retrieval: dense sparse", "area": "architecture"},

    # Training
    {"topic": "Reranker training: contrastive", "area": "training"},
    {"topic": "Distillation: cross-encoder to bi", "area": "training"},
    {"topic": "Hard negative mining: reranking", "area": "training"},
    {"topic": "Multi-task reranking: generalization", "area": "training"},

    # Integration
    {"topic": "RAG reranking: two-stage retrieval", "area": "integration"},
    {"topic": "Reciprocal rank fusion: RRF", "area": "integration"},
    {"topic": "Score calibration: normalization", "area": "integration"},
    {"topic": "Cascade ranking: efficiency", "area": "integration"},

    # Evaluation
    {"topic": "Reranking metrics: NDCG MRR", "area": "evaluation"},
    {"topic": "Latency tradeoffs: speed quality", "area": "evaluation"},
    {"topic": "A/B testing: reranking impact", "area": "evaluation"},
    {"topic": "BEIR benchmark: zero-shot", "area": "evaluation"},
]


class RerankingExecutor(BaseResearchExecutor):
    """Custom executor with reranking-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"Reranking retrieval models: {topic}"


if __name__ == "__main__":
    run_research(
        "reranking",
        "RERANKING ITERATIONS",
        RERANKING_TOPICS,
        executor_class=RerankingExecutor,
    )
