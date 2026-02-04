"""
RAG Retrieval Metrics Collection for Optimization

Comprehensive metrics collection for RAG pipeline optimization, providing:
1. Retrieval Metrics - Latency (p50/p95/p99), documents retrieved, source distribution, cache hit rate
2. Quality Metrics - Relevance score distribution, diversity (unique sources), coverage (query terms)
3. Usage Metrics - Queries per minute, token usage by model, cost tracking, error rates
4. Exporters - Prometheus format, JSON for dashboards, structured logging
5. Analysis - Identify slow queries, detect degradation, recommendations

Integration with production_loop.py patterns for seamless monitoring.

Created: 2026-02-04
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RAGMetricsConfig:
    """Configuration for RAG metrics collection."""
    # Sampling
    sample_rate: float = 1.0  # 100% by default
    max_samples: int = 10000

    # Thresholds for degradation detection
    latency_p95_warning_ms: float = 500.0
    latency_p95_critical_ms: float = 2000.0
    cache_hit_rate_warning: float = 0.3
    error_rate_warning: float = 0.05

    # Rolling window
    window_minutes: int = 15

    # Export intervals
    export_interval_seconds: int = 60


class RetrievalSource(Enum):
    """Source types for retrieved documents."""
    VECTOR_DB = "vector_db"
    EXA = "exa"
    TAVILY = "tavily"
    PERPLEXITY = "perplexity"
    MEMORY = "memory"
    WEB_SEARCH = "web_search"
    CACHE = "cache"
    UNKNOWN = "unknown"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LatencyPercentiles:
    """Latency percentile statistics."""
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    avg: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    count: int = 0

    @classmethod
    def from_samples(cls, samples: List[float]) -> "LatencyPercentiles":
        if not samples:
            return cls()
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        return cls(
            p50=sorted_samples[int(n * 0.5)] if n > 0 else 0,
            p95=sorted_samples[int(n * 0.95)] if n > 0 else 0,
            p99=sorted_samples[int(n * 0.99)] if n > 0 else 0,
            avg=statistics.mean(sorted_samples),
            min_val=min(sorted_samples),
            max_val=max(sorted_samples),
            count=n,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "p50_ms": round(self.p50, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
            "avg_ms": round(self.avg, 2),
            "min_ms": round(self.min_val, 2),
            "max_ms": round(self.max_val, 2),
            "count": self.count,
        }


@dataclass
class RetrievalMetrics:
    """Core retrieval performance metrics."""
    latency_samples: List[float] = field(default_factory=list)
    documents_retrieved: List[int] = field(default_factory=list)
    source_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def latency(self) -> LatencyPercentiles:
        return LatencyPercentiles.from_samples(self.latency_samples)

    @property
    def avg_docs_retrieved(self) -> float:
        return statistics.mean(self.documents_retrieved) if self.documents_retrieved else 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latency": self.latency.to_dict(),
            "avg_documents_retrieved": round(self.avg_docs_retrieved, 2),
            "source_distribution": dict(self.source_counts),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }


@dataclass
class QualityMetrics:
    """Retrieval quality metrics."""
    relevance_scores: List[float] = field(default_factory=list)
    unique_sources_per_query: List[int] = field(default_factory=list)
    query_term_coverage: List[float] = field(default_factory=list)

    @property
    def relevance_distribution(self) -> Dict[str, float]:
        if not self.relevance_scores:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        return {
            "avg": round(statistics.mean(self.relevance_scores), 4),
            "min": round(min(self.relevance_scores), 4),
            "max": round(max(self.relevance_scores), 4),
            "std": round(statistics.stdev(self.relevance_scores), 4) if len(self.relevance_scores) > 1 else 0.0,
        }

    @property
    def diversity_score(self) -> float:
        return statistics.mean(self.unique_sources_per_query) if self.unique_sources_per_query else 0.0

    @property
    def coverage_score(self) -> float:
        return statistics.mean(self.query_term_coverage) if self.query_term_coverage else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relevance_distribution": self.relevance_distribution,
            "diversity_score": round(self.diversity_score, 4),
            "coverage_score": round(self.coverage_score, 4),
            "samples": len(self.relevance_scores),
        }


@dataclass
class UsageMetrics:
    """Usage and cost metrics."""
    query_timestamps: List[float] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cost_by_model: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_queries: int = 0
    total_errors: int = 0

    def queries_per_minute(self, window_seconds: int = 60) -> float:
        now = time.time()
        recent = [t for t in self.query_timestamps if now - t < window_seconds]
        return len(recent) * (60 / window_seconds) if recent else 0.0

    @property
    def error_rate(self) -> float:
        return self.total_errors / self.total_queries if self.total_queries > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queries_per_minute": round(self.queries_per_minute(), 2),
            "total_queries": self.total_queries,
            "token_usage_by_model": dict(self.token_usage),
            "cost_by_model_usd": {k: round(v, 6) for k, v in self.cost_by_model.items()},
            "total_cost_usd": round(sum(self.cost_by_model.values()), 6),
            "error_rate": round(self.error_rate, 4),
            "errors_by_type": dict(self.error_counts),
        }


@dataclass
class SlowQueryRecord:
    """Record of a slow query for analysis."""
    query: str
    latency_ms: float
    timestamp: datetime
    source: str
    documents_retrieved: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# METRICS COLLECTOR
# =============================================================================

class RAGMetricsCollector:
    """
    Central collector for RAG retrieval metrics.

    Thread-safe metrics collection with rolling windows and efficient storage.

    Example:
        collector = RAGMetricsCollector()

        async with collector.track_retrieval("exa") as tracker:
            results = await exa_search(query)
            tracker.record_docs(len(results))
            tracker.record_relevance([r.score for r in results])

        metrics = collector.get_summary()
    """

    def __init__(self, config: Optional[RAGMetricsConfig] = None):
        self.config = config or RAGMetricsConfig()
        self._lock = threading.Lock()

        self.retrieval = RetrievalMetrics()
        self.quality = QualityMetrics()
        self.usage = UsageMetrics()

        self._slow_queries: List[SlowQueryRecord] = []
        self._window_start = datetime.now(timezone.utc)

    def record_retrieval(
        self,
        latency_ms: float,
        docs_retrieved: int,
        source: RetrievalSource = RetrievalSource.UNKNOWN,
        cache_hit: bool = False,
    ) -> None:
        """Record a retrieval operation."""
        with self._lock:
            self.retrieval.latency_samples.append(latency_ms)
            self.retrieval.documents_retrieved.append(docs_retrieved)
            self.retrieval.source_counts[source.value] += 1

            if cache_hit:
                self.retrieval.cache_hits += 1
            else:
                self.retrieval.cache_misses += 1

            self.usage.total_queries += 1
            self.usage.query_timestamps.append(time.time())

            self._trim_samples()

    def record_quality(
        self,
        relevance_scores: List[float],
        unique_sources: int,
        query_coverage: float,
    ) -> None:
        """Record quality metrics for a retrieval."""
        with self._lock:
            self.quality.relevance_scores.extend(relevance_scores)
            self.quality.unique_sources_per_query.append(unique_sources)
            self.quality.query_term_coverage.append(query_coverage)
            self._trim_samples()

    def record_tokens(self, model: str, tokens: int, cost_usd: float = 0.0) -> None:
        """Record token usage and cost."""
        with self._lock:
            self.usage.token_usage[model] += tokens
            self.usage.cost_by_model[model] += cost_usd

    def record_error(self, error_type: str) -> None:
        """Record an error."""
        with self._lock:
            self.usage.error_counts[error_type] += 1
            self.usage.total_errors += 1

    def record_slow_query(
        self,
        query: str,
        latency_ms: float,
        source: str,
        docs_retrieved: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a slow query for analysis."""
        with self._lock:
            self._slow_queries.append(SlowQueryRecord(
                query=query[:500],
                latency_ms=latency_ms,
                timestamp=datetime.now(timezone.utc),
                source=source,
                documents_retrieved=docs_retrieved,
                metadata=metadata or {},
            ))
            if len(self._slow_queries) > 100:
                self._slow_queries = self._slow_queries[-100:]

    def _trim_samples(self) -> None:
        """Trim samples to max size."""
        max_samples = self.config.max_samples
        if len(self.retrieval.latency_samples) > max_samples:
            self.retrieval.latency_samples = self.retrieval.latency_samples[-max_samples:]
        if len(self.retrieval.documents_retrieved) > max_samples:
            self.retrieval.documents_retrieved = self.retrieval.documents_retrieved[-max_samples:]
        if len(self.quality.relevance_scores) > max_samples:
            self.quality.relevance_scores = self.quality.relevance_scores[-max_samples:]
        window_seconds = self.config.window_minutes * 60
        cutoff = time.time() - window_seconds
        self.usage.query_timestamps = [t for t in self.usage.query_timestamps if t > cutoff]

    @asynccontextmanager
    async def track_retrieval(self, source: str = "unknown"):
        """Async context manager for tracking a retrieval operation."""
        tracker = RetrievalTracker(self, source)
        start = time.perf_counter()
        try:
            yield tracker
        except Exception as e:
            tracker.error_type = type(e).__name__
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            tracker.finalize(latency_ms)

    def get_summary(self) -> Dict[str, Any]:
        """Get complete metrics summary."""
        with self._lock:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "window_minutes": self.config.window_minutes,
                "retrieval": self.retrieval.to_dict(),
                "quality": self.quality.to_dict(),
                "usage": self.usage.to_dict(),
                "slow_queries_count": len(self._slow_queries),
            }

    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent slow queries."""
        with self._lock:
            queries = sorted(self._slow_queries, key=lambda x: x.latency_ms, reverse=True)
            return [
                {
                    "query": q.query,
                    "latency_ms": q.latency_ms,
                    "timestamp": q.timestamp.isoformat(),
                    "source": q.source,
                    "documents_retrieved": q.documents_retrieved,
                }
                for q in queries[:limit]
            ]

    def reset(self) -> Dict[str, Any]:
        """Reset metrics and return previous window data."""
        with self._lock:
            summary = self.get_summary()
            self.retrieval = RetrievalMetrics()
            self.quality = QualityMetrics()
            self.usage = UsageMetrics()
            self._slow_queries = []
            self._window_start = datetime.now(timezone.utc)
            return summary


class RetrievalTracker:
    """Helper for tracking individual retrieval operations."""

    def __init__(self, collector: RAGMetricsCollector, source: str):
        self.collector = collector
        self.source = source
        self.docs_retrieved = 0
        self.relevance_scores: List[float] = []
        self.unique_sources = 0
        self.query_coverage = 0.0
        self.cache_hit = False
        self.query = ""
        self.error_type: Optional[str] = None

    def record_docs(self, count: int) -> None:
        self.docs_retrieved = count

    def record_relevance(self, scores: List[float]) -> None:
        self.relevance_scores = scores

    def record_diversity(self, unique_sources: int, coverage: float) -> None:
        self.unique_sources = unique_sources
        self.query_coverage = coverage

    def set_cache_hit(self, hit: bool) -> None:
        self.cache_hit = hit

    def set_query(self, query: str) -> None:
        self.query = query

    def finalize(self, latency_ms: float) -> None:
        source_enum = RetrievalSource(self.source) if self.source in [s.value for s in RetrievalSource] else RetrievalSource.UNKNOWN
        self.collector.record_retrieval(latency_ms, self.docs_retrieved, source_enum, self.cache_hit)

        if self.relevance_scores:
            self.collector.record_quality(self.relevance_scores, self.unique_sources, self.query_coverage)

        if self.error_type:
            self.collector.record_error(self.error_type)

        if latency_ms > self.collector.config.latency_p95_warning_ms:
            self.collector.record_slow_query(self.query, latency_ms, self.source, self.docs_retrieved)


# =============================================================================
# EXPORTERS
# =============================================================================

class PrometheusExporter:
    """Export metrics in Prometheus text format."""

    def export(self, collector: RAGMetricsCollector) -> str:
        summary = collector.get_summary()
        lines = []

        # Retrieval latency
        latency = summary["retrieval"]["latency"]
        lines.append("# TYPE rag_retrieval_latency_ms summary")
        lines.append(f'rag_retrieval_latency_ms{{quantile="0.5"}} {latency["p50_ms"]}')
        lines.append(f'rag_retrieval_latency_ms{{quantile="0.95"}} {latency["p95_ms"]}')
        lines.append(f'rag_retrieval_latency_ms{{quantile="0.99"}} {latency["p99_ms"]}')
        lines.append(f"rag_retrieval_latency_ms_count {latency['count']}")

        # Documents retrieved
        lines.append("# TYPE rag_documents_retrieved gauge")
        lines.append(f"rag_documents_retrieved {summary['retrieval']['avg_documents_retrieved']}")

        # Cache hit rate
        lines.append("# TYPE rag_cache_hit_rate gauge")
        lines.append(f"rag_cache_hit_rate {summary['retrieval']['cache_hit_rate']}")

        # Source distribution
        lines.append("# TYPE rag_source_requests_total counter")
        for source, count in summary["retrieval"]["source_distribution"].items():
            lines.append(f'rag_source_requests_total{{source="{source}"}} {count}')

        # Quality metrics
        quality = summary["quality"]
        lines.append("# TYPE rag_relevance_score gauge")
        lines.append(f"rag_relevance_score {quality['relevance_distribution']['avg']}")
        lines.append("# TYPE rag_diversity_score gauge")
        lines.append(f"rag_diversity_score {quality['diversity_score']}")
        lines.append("# TYPE rag_coverage_score gauge")
        lines.append(f"rag_coverage_score {quality['coverage_score']}")

        # Usage metrics
        usage = summary["usage"]
        lines.append("# TYPE rag_queries_per_minute gauge")
        lines.append(f"rag_queries_per_minute {usage['queries_per_minute']}")
        lines.append("# TYPE rag_error_rate gauge")
        lines.append(f"rag_error_rate {usage['error_rate']}")
        lines.append("# TYPE rag_total_cost_usd counter")
        lines.append(f"rag_total_cost_usd {usage['total_cost_usd']}")

        return "\n".join(lines)


class JSONExporter:
    """Export metrics as JSON for dashboards."""

    def export(self, collector: RAGMetricsCollector) -> Dict[str, Any]:
        return {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "metrics": collector.get_summary(),
            "slow_queries": collector.get_slow_queries(10),
        }

    def export_string(self, collector: RAGMetricsCollector) -> str:
        return json.dumps(self.export(collector), indent=2)


class StructuredLogger:
    """Structured logging for metrics."""

    def __init__(self, logger_name: str = "rag.metrics"):
        self.logger = logging.getLogger(logger_name)

    def log_summary(self, collector: RAGMetricsCollector) -> None:
        summary = collector.get_summary()
        self.logger.info("RAG metrics summary", extra={"metrics": summary})

    def log_slow_query(self, query: SlowQueryRecord) -> None:
        self.logger.warning(
            f"Slow RAG query: {query.latency_ms:.0f}ms",
            extra={
                "query": query.query[:100],
                "latency_ms": query.latency_ms,
                "source": query.source,
                "documents": query.documents_retrieved,
            }
        )


# =============================================================================
# ANALYSIS
# =============================================================================

@dataclass
class DegradationAlert:
    """Alert for detected degradation."""
    metric: str
    current_value: float
    threshold: float
    severity: str  # "warning" or "critical"
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RAGAnalyzer:
    """Analyze RAG metrics for issues and recommendations."""

    def __init__(self, config: Optional[RAGMetricsConfig] = None):
        self.config = config or RAGMetricsConfig()

    def detect_degradation(self, collector: RAGMetricsCollector) -> List[DegradationAlert]:
        """Detect performance degradation."""
        alerts = []
        summary = collector.get_summary()

        # Check latency
        p95 = summary["retrieval"]["latency"]["p95_ms"]
        if p95 > self.config.latency_p95_critical_ms:
            alerts.append(DegradationAlert(
                metric="latency_p95",
                current_value=p95,
                threshold=self.config.latency_p95_critical_ms,
                severity="critical",
                message=f"P95 latency {p95:.0f}ms exceeds critical threshold",
            ))
        elif p95 > self.config.latency_p95_warning_ms:
            alerts.append(DegradationAlert(
                metric="latency_p95",
                current_value=p95,
                threshold=self.config.latency_p95_warning_ms,
                severity="warning",
                message=f"P95 latency {p95:.0f}ms exceeds warning threshold",
            ))

        # Check cache hit rate
        cache_rate = summary["retrieval"]["cache_hit_rate"]
        if cache_rate < self.config.cache_hit_rate_warning:
            alerts.append(DegradationAlert(
                metric="cache_hit_rate",
                current_value=cache_rate,
                threshold=self.config.cache_hit_rate_warning,
                severity="warning",
                message=f"Cache hit rate {cache_rate:.1%} below threshold",
            ))

        # Check error rate
        error_rate = summary["usage"]["error_rate"]
        if error_rate > self.config.error_rate_warning:
            alerts.append(DegradationAlert(
                metric="error_rate",
                current_value=error_rate,
                threshold=self.config.error_rate_warning,
                severity="warning",
                message=f"Error rate {error_rate:.1%} above threshold",
            ))

        return alerts

    def generate_recommendations(self, collector: RAGMetricsCollector) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        summary = collector.get_summary()

        # Latency recommendations
        p95 = summary["retrieval"]["latency"]["p95_ms"]
        if p95 > 1000:
            recommendations.append(
                "HIGH LATENCY: Consider implementing query caching, "
                "reducing top_k, or using faster embedding models."
            )

        # Cache recommendations
        cache_rate = summary["retrieval"]["cache_hit_rate"]
        if cache_rate < 0.3 and summary["usage"]["total_queries"] > 100:
            recommendations.append(
                "LOW CACHE HIT RATE: Enable semantic caching or increase cache TTL. "
                "Consider query normalization for better cache utilization."
            )

        # Quality recommendations
        relevance_avg = summary["quality"]["relevance_distribution"]["avg"]
        if relevance_avg < 0.5 and relevance_avg > 0:
            recommendations.append(
                "LOW RELEVANCE SCORES: Consider implementing reranking (e.g., cross-encoder), "
                "hybrid search (dense + sparse), or query expansion."
            )

        diversity = summary["quality"]["diversity_score"]
        if diversity < 2 and diversity > 0:
            recommendations.append(
                "LOW DIVERSITY: Retrieved documents from few sources. "
                "Consider MMR diversity reranking or multi-source retrieval."
            )

        # Source distribution
        sources = summary["retrieval"]["source_distribution"]
        if len(sources) == 1:
            recommendations.append(
                "SINGLE SOURCE: All retrievals from one source. "
                "Consider adding fallback sources for resilience."
            )

        if not recommendations:
            recommendations.append("All metrics within acceptable ranges.")

        return recommendations

    def identify_slow_query_patterns(self, collector: RAGMetricsCollector) -> Dict[str, Any]:
        """Identify patterns in slow queries."""
        slow_queries = collector.get_slow_queries(50)

        if not slow_queries:
            return {"patterns": [], "total_slow_queries": 0}

        # Analyze by source
        by_source: Dict[str, List[float]] = defaultdict(list)
        for q in slow_queries:
            by_source[q["source"]].append(q["latency_ms"])

        source_analysis = {
            source: {
                "count": len(latencies),
                "avg_latency_ms": statistics.mean(latencies),
            }
            for source, latencies in by_source.items()
        }

        return {
            "total_slow_queries": len(slow_queries),
            "by_source": source_analysis,
            "slowest_query_ms": max(q["latency_ms"] for q in slow_queries),
        }


# =============================================================================
# INTEGRATION
# =============================================================================

class RAGMetricsIntegration:
    """Integration with ProductionMonitoringLoop."""

    def __init__(self, collector: RAGMetricsCollector, monitoring_loop: Optional[Any] = None):
        self.collector = collector
        self.monitoring_loop = monitoring_loop
        self.prometheus_exporter = PrometheusExporter()
        self.json_exporter = JSONExporter()
        self.analyzer = RAGAnalyzer(collector.config)
        self.structured_logger = StructuredLogger()

    async def export_to_monitoring_loop(self) -> None:
        """Export metrics to the main monitoring loop."""
        if self.monitoring_loop is None:
            return

        summary = self.collector.get_summary()

        # Record to monitoring loop if available
        if hasattr(self.monitoring_loop, "record_request"):
            latency = summary["retrieval"]["latency"]
            self.monitoring_loop.record_request(
                adapter="rag",
                method="retrieval",
                latency_ms=latency["avg_ms"],
            )

    async def run_analysis(self) -> Dict[str, Any]:
        """Run full analysis and return results."""
        alerts = self.analyzer.detect_degradation(self.collector)
        recommendations = self.analyzer.generate_recommendations(self.collector)
        slow_patterns = self.analyzer.identify_slow_query_patterns(self.collector)

        return {
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.current_value,
                    "threshold": a.threshold,
                    "severity": a.severity,
                    "message": a.message,
                }
                for a in alerts
            ],
            "recommendations": recommendations,
            "slow_query_patterns": slow_patterns,
        }

    def get_prometheus_metrics(self) -> str:
        return self.prometheus_exporter.export(self.collector)

    def get_json_metrics(self) -> Dict[str, Any]:
        return self.json_exporter.export(self.collector)


# =============================================================================
# FACTORY
# =============================================================================

_global_collector: Optional[RAGMetricsCollector] = None


def get_rag_metrics_collector(config: Optional[RAGMetricsConfig] = None) -> RAGMetricsCollector:
    """Get or create the global RAG metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = RAGMetricsCollector(config)
    return _global_collector


def create_rag_metrics_integration(
    monitoring_loop: Optional[Any] = None,
    config: Optional[RAGMetricsConfig] = None,
) -> RAGMetricsIntegration:
    """Factory to create RAG metrics integration."""
    collector = get_rag_metrics_collector(config)
    return RAGMetricsIntegration(collector, monitoring_loop)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "RAGMetricsConfig",
    "RetrievalSource",
    # Data structures
    "LatencyPercentiles",
    "RetrievalMetrics",
    "QualityMetrics",
    "UsageMetrics",
    "SlowQueryRecord",
    # Collector
    "RAGMetricsCollector",
    "RetrievalTracker",
    # Exporters
    "PrometheusExporter",
    "JSONExporter",
    "StructuredLogger",
    # Analysis
    "DegradationAlert",
    "RAGAnalyzer",
    # Integration
    "RAGMetricsIntegration",
    # Factory
    "get_rag_metrics_collector",
    "create_rag_metrics_integration",
]
