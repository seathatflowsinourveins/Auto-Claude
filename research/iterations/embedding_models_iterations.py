"""
EMBEDDING MODELS ITERATIONS - Vector Representation Deep Dive
==============================================================
Models, fine-tuning, evaluation, specialized embeddings

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


EMBEDDING_TOPICS = [
    # Embedding Models
    {"topic": "OpenAI text-embedding-3: small, large, dimensions", "area": "models"},
    {"topic": "Cohere embed-v3: multilingual, clustering, search tasks", "area": "models"},
    {"topic": "Jina embeddings v3: task-specific, matryoshka", "area": "models"},
    {"topic": "Voyage AI embeddings: code, legal, finance specialized", "area": "models"},

    # Open Source Models
    {"topic": "Sentence transformers: all-MiniLM, all-mpnet, BGE", "area": "opensource"},
    {"topic": "Nomic embed: open weights, long context", "area": "opensource"},
    {"topic": "GTE embeddings: General Text Embeddings from Alibaba", "area": "opensource"},
    {"topic": "E5 embeddings: text embedding from Microsoft", "area": "opensource"},

    # Fine-tuning
    {"topic": "Embedding fine-tuning: contrastive learning, triplet loss", "area": "finetuning"},
    {"topic": "Domain adaptation: synthetic data, hard negatives", "area": "finetuning"},
    {"topic": "Matryoshka representation learning: flexible dimensions", "area": "finetuning"},
    {"topic": "Late chunking: context-aware chunk embeddings", "area": "finetuning"},

    # Evaluation
    {"topic": "MTEB benchmark: massive text embedding benchmark", "area": "evaluation"},
    {"topic": "Retrieval evaluation: recall@k, MRR, NDCG", "area": "evaluation"},
    {"topic": "Clustering evaluation: silhouette score, NMI", "area": "evaluation"},
    {"topic": "Semantic similarity: STS benchmark, correlation", "area": "evaluation"},

    # Advanced Patterns
    {"topic": "Multi-vector embeddings: ColBERT, late interaction", "area": "advanced"},
    {"topic": "Sparse embeddings: SPLADE, learned sparse retrieval", "area": "advanced"},
    {"topic": "Cross-encoder reranking: relevance scoring", "area": "advanced"},
    {"topic": "Multimodal embeddings: CLIP, ImageBind, unified space", "area": "advanced"},
]


class EmbeddingModelsExecutor(BaseResearchExecutor):
    """Custom executor with embedding-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"Text embedding models and implementation: {topic}"


if __name__ == "__main__":
    run_research(
        "embeddings",
        "EMBEDDING MODELS ITERATIONS",
        EMBEDDING_TOPICS,
        executor_class=EmbeddingModelsExecutor,
    )
