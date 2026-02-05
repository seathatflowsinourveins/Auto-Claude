"""
ADVANCED RAG ITERATIONS - Cutting-Edge Retrieval Patterns
==========================================================
GraphRAG, RAPTOR, ColBERT, hybrid search, multi-modal RAG

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


RAG_TOPICS = [
    # Graph-Enhanced RAG
    {"topic": "Microsoft GraphRAG: community detection, entity extraction, summarization", "area": "graph"},
    {"topic": "Knowledge graph construction: entity linking, relation extraction", "area": "graph"},
    {"topic": "LightRAG: lightweight graph-based retrieval augmentation", "area": "graph"},
    {"topic": "Graph neural networks for RAG: node embeddings, message passing", "area": "graph"},

    # Hierarchical RAG
    {"topic": "RAPTOR: recursive abstractive processing for tree retrieval", "area": "hierarchical"},
    {"topic": "Multi-level summarization: document, section, chunk hierarchies", "area": "hierarchical"},
    {"topic": "Tree-based indexing: clustering, parent-child retrieval", "area": "hierarchical"},
    {"topic": "Long-context RAG: 128K+ token handling, sliding windows", "area": "hierarchical"},

    # Dense Retrieval
    {"topic": "ColBERT: late interaction, maxsim, passage retrieval", "area": "dense"},
    {"topic": "Dense passage retrieval (DPR): bi-encoder training, hard negatives", "area": "dense"},
    {"topic": "Sentence transformers: fine-tuning, domain adaptation", "area": "dense"},
    {"topic": "Matryoshka embeddings: flexible dimensionality, truncation", "area": "dense"},

    # Hybrid Search
    {"topic": "Hybrid search: BM25 + dense fusion, reciprocal rank fusion", "area": "hybrid"},
    {"topic": "Sparse-dense hybrid: SPLADE, learned sparse retrieval", "area": "hybrid"},
    {"topic": "Re-ranking: cross-encoder, ColBERT, listwise ranking", "area": "hybrid"},
    {"topic": "Query expansion: HyDE, pseudo-relevance feedback", "area": "hybrid"},

    # Multi-modal RAG
    {"topic": "Vision-language RAG: CLIP retrieval, image understanding", "area": "multimodal"},
    {"topic": "Table RAG: table parsing, structured data retrieval", "area": "multimodal"},
    {"topic": "Code RAG: syntax-aware chunking, AST-based retrieval", "area": "multimodal"},
    {"topic": "Audio RAG: transcription retrieval, speaker diarization", "area": "multimodal"},
]


class AdvancedRAGExecutor(BaseResearchExecutor):
    """Custom executor with RAG-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"Advanced RAG implementation patterns and best practices: {topic}"


if __name__ == "__main__":
    run_research(
        "rag",
        "ADVANCED RAG ITERATIONS",
        RAG_TOPICS,
        executor_class=AdvancedRAGExecutor,
    )
