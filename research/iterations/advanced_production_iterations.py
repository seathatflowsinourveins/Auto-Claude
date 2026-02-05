"""
ADVANCED PRODUCTION ITERATIONS - Observability, Security, Deployment
=====================================================================
Production-grade patterns for deployment, monitoring, and security

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


ADVANCED_PRODUCTION_TOPICS = [
    # Observability & Monitoring
    {"topic": "LLM observability: Langfuse vs Langsmith vs Phoenix tracing comparison", "area": "observability"},
    {"topic": "Token usage monitoring: cost tracking, budget alerts, usage analytics", "area": "observability"},
    {"topic": "Latency profiling: p50/p95/p99 tracking, bottleneck identification", "area": "observability"},
    {"topic": "Quality metrics: relevance scoring, hallucination detection, factuality", "area": "observability"},
    {"topic": "Distributed tracing: OpenTelemetry for LLM applications, span correlation", "area": "observability"},

    # Security Patterns
    {"topic": "Prompt injection defense: input sanitization, output filtering, guardrails", "area": "security"},
    {"topic": "PII detection and redaction: NER-based filtering, regex patterns, anonymization", "area": "security"},
    {"topic": "Rate limiting strategies: token bucket, sliding window, per-user quotas", "area": "security"},
    {"topic": "API key rotation: secret management, key versioning, zero-downtime rotation", "area": "security"},
    {"topic": "Content moderation: toxicity detection, NSFW filtering, compliance checks", "area": "security"},

    # Deployment Patterns
    {"topic": "LLM deployment: vLLM vs TGI vs Triton inference servers comparison", "area": "deployment"},
    {"topic": "Kubernetes for LLM: GPU scheduling, autoscaling, resource quotas", "area": "deployment"},
    {"topic": "Edge deployment: ONNX runtime, quantization, mobile inference", "area": "deployment"},
    {"topic": "A/B testing LLMs: experiment tracking, statistical significance, rollout", "area": "deployment"},
    {"topic": "Blue-green deployment: zero-downtime updates, traffic shifting, rollback", "area": "deployment"},

    # Evaluation & Testing
    {"topic": "LLM evaluation frameworks: RAGAS vs DeepEval vs TruLens comparison", "area": "evaluation"},
    {"topic": "Retrieval evaluation: MRR, NDCG, precision@k, recall metrics", "area": "evaluation"},
    {"topic": "Generation quality: BLEU, ROUGE, BERTScore, human evaluation", "area": "evaluation"},
    {"topic": "Regression testing: prompt versioning, output diff, golden datasets", "area": "evaluation"},
    {"topic": "Load testing LLMs: concurrent users, throughput, stress testing", "area": "evaluation"},

    # Data Pipeline Patterns
    {"topic": "Document processing: chunking strategies, overlap, semantic splitting", "area": "data_pipeline"},
    {"topic": "Embedding pipeline: batch processing, incremental updates, deduplication", "area": "data_pipeline"},
    {"topic": "Data versioning: DVC, LakeFS, MLflow for dataset management", "area": "data_pipeline"},
    {"topic": "ETL for RAG: document ingestion, metadata extraction, indexing", "area": "data_pipeline"},
    {"topic": "Data quality: validation, schema enforcement, anomaly detection", "area": "data_pipeline"},

    # Scaling Patterns
    {"topic": "Horizontal scaling: load balancing, sharding, distributed inference", "area": "scaling"},
    {"topic": "Caching at scale: Redis clustering, CDN for embeddings, cache warming", "area": "scaling"},
    {"topic": "Queue-based processing: Celery, RabbitMQ, async task orchestration", "area": "scaling"},
    {"topic": "Microservices for AI: service mesh, gRPC, protocol buffers", "area": "scaling"},
    {"topic": "Multi-region deployment: latency optimization, data residency, failover", "area": "scaling"},
]


class AdvancedProductionExecutor(BaseResearchExecutor):
    """Custom executor with production-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        area_prompts = {
            "observability": "Production LLM observability and monitoring",
            "security": "LLM security patterns and implementation",
            "deployment": "Production LLM deployment and infrastructure",
            "evaluation": "LLM evaluation and testing frameworks",
            "data_pipeline": "Data pipeline patterns for LLM applications",
            "scaling": "Scaling patterns for production LLM systems",
        }
        prefix = area_prompts.get(area, "Production-grade LLM patterns")
        return f"{prefix}: {topic}. Focus on best practices, configuration, and monitoring."


if __name__ == "__main__":
    run_research(
        "advanced_production",
        "ADVANCED PRODUCTION ITERATIONS",
        ADVANCED_PRODUCTION_TOPICS,
        executor_class=AdvancedProductionExecutor,
    )
