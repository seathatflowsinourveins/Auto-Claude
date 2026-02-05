"""
KNOWLEDGE GRAPHS ITERATIONS - Structured Knowledge
===================================================
Graph construction, querying, reasoning, integration

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


KG_TOPICS = [
    # Graph Construction
    {"topic": "Entity extraction: NER, relation extraction, coreference", "area": "construction"},
    {"topic": "Knowledge graph from documents: schema design, population", "area": "construction"},
    {"topic": "LLM for KG construction: prompting strategies, validation", "area": "construction"},
    {"topic": "Incremental graph updates: change detection, consistency", "area": "construction"},

    # Graph Databases
    {"topic": "Neo4j for knowledge graphs: Cypher queries, graph algorithms", "area": "databases"},
    {"topic": "Amazon Neptune: RDF, property graphs, SPARQL", "area": "databases"},
    {"topic": "Graph databases comparison: Neo4j vs Neptune vs TigerGraph", "area": "databases"},
    {"topic": "Graph embeddings: node2vec, TransE, knowledge graph embeddings", "area": "databases"},

    # Querying & Reasoning
    {"topic": "Natural language to graph query: text-to-Cypher, text-to-SPARQL", "area": "querying"},
    {"topic": "Graph reasoning: path finding, inference, rule-based", "area": "querying"},
    {"topic": "Multi-hop reasoning: complex queries, chain of relations", "area": "querying"},
    {"topic": "Graph analytics: centrality, community detection, similarity", "area": "querying"},

    # GraphRAG
    {"topic": "Microsoft GraphRAG: community summaries, entity resolution", "area": "graphrag"},
    {"topic": "LightRAG: efficient graph-based retrieval", "area": "graphrag"},
    {"topic": "Graph + vector hybrid: combining structured and unstructured", "area": "graphrag"},
    {"topic": "GraphRAG evaluation: faithfulness, coverage metrics", "area": "graphrag"},

    # Applications
    {"topic": "Knowledge graphs for question answering: KGQA systems", "area": "applications"},
    {"topic": "Enterprise knowledge graphs: organizational knowledge", "area": "applications"},
    {"topic": "Biomedical knowledge graphs: drug discovery, literature", "area": "applications"},
    {"topic": "Product knowledge graphs: e-commerce, recommendations", "area": "applications"},
]


class KGExecutor(BaseResearchExecutor):
    """Custom executor with knowledge graph-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"Knowledge graph implementation: {topic}"


if __name__ == "__main__":
    run_research(
        "kg",
        "KNOWLEDGE GRAPHS ITERATIONS",
        KG_TOPICS,
        executor_class=KGExecutor,
    )
