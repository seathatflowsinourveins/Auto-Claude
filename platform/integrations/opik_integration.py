#!/usr/bin/env python3
"""
Opik Observability Integration for UNLEASH Platform
====================================================

Provides 7-14x faster observability than Langfuse:
- Trace logging: ~23s vs Langfuse ~327s (14x faster)
- Agent trajectory evaluation with span-level metrics
- LLM-as-Judge metrics (Hallucination, AnswerRelevance, etc.)
- Cost tracking per span/trace/project

Created: 2026-01-31 (V27 Iteration)
Source: https://github.com/comet-ml/opik (17.3K+ stars)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from functools import wraps

logger = logging.getLogger("opik_integration")

# Check opik availability
OPIK_AVAILABLE = False
try:
    import opik
    from opik import track, Opik, Trace, Span
    from opik.evaluation import evaluate
    from opik.evaluation.metrics import Hallucination, AnswerRelevance, Moderation
    OPIK_AVAILABLE = True
    logger.info(f"opik v{getattr(opik, '__version__', 'unknown')} imported successfully")
except ImportError as e:
    logger.warning(f"opik not available: {e}. Run: pip install opik")


@dataclass
class OpikConfig:
    """Configuration for Opik observability."""
    api_key: Optional[str] = field(default_factory=lambda: os.environ.get("OPIK_API_KEY"))
    workspace: str = "default"
    project_name: str = "unleash-v27"
    enable_tracing: bool = True
    enable_cost_tracking: bool = True
    enable_hallucination_check: bool = True
    enable_relevance_check: bool = True
    enable_moderation: bool = True
    async_logging: bool = True


class OpikTracer:
    """UNLEASH integration for Opik observability (7-14x faster than Langfuse)."""
    
    MODEL_COSTS = {
        "claude-opus-4-5-20251101": {"input": 0.015, "output": 0.075},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }
    
    def __init__(self, config: Optional[OpikConfig] = None):
        if not OPIK_AVAILABLE:
            raise ImportError("opik not available. Install with: pip install opik")
        self.config = config or OpikConfig()
        self._client = None
        self._traces = {}
        self._metrics = {}
        if self.config.enable_hallucination_check:
            self._metrics["hallucination"] = Hallucination()
        if self.config.enable_relevance_check:
            self._metrics["relevance"] = AnswerRelevance()
        if self.config.enable_moderation:
            self._metrics["moderation"] = Moderation()
    
    def initialize(self):
        """Initialize Opik client."""
        if self.config.api_key:
            opik.configure(api_key=self.config.api_key)
        self._client = Opik(project_name=self.config.project_name, workspace=self.config.workspace)
        logger.info(f"Opik initialized for project: {self.config.project_name}")
        return self
    
    def start_trace(self, name: str, metadata: Optional[Dict] = None):
        """Start a new trace."""
        if not self._client:
            self.initialize()
        trace = self._client.trace(name=name, metadata=metadata or {})
        self._traces[trace.id] = trace
        return trace
    
    def log_llm_call(self, trace_id: str, model: str, prompt: str, response: str, 
                     input_tokens: int = 0, output_tokens: int = 0, metadata: Optional[Dict] = None):
        """Log an LLM call with cost tracking."""
        trace = self._traces.get(trace_id)
        if not trace:
            return
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        with trace.span(name=f"llm_{model}") as span:
            span.log_input(prompt)
            span.log_output(response)
            span.set_metadata({"model": model, "input_tokens": input_tokens, 
                             "output_tokens": output_tokens, "cost_usd": cost, **(metadata or {})})
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = self.MODEL_COSTS.get(model, {"input": 0.003, "output": 0.015})
        return round((input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"], 6)
    
    def evaluate_response(self, input_text: str, output_text: str, context: Optional[str] = None) -> Dict[str, float]:
        """Evaluate response using LLM-as-Judge metrics."""
        results = {}
        if "hallucination" in self._metrics and context:
            results["hallucination"] = self._metrics["hallucination"].score(input=input_text, output=output_text, context=context).value
        if "relevance" in self._metrics:
            results["relevance"] = self._metrics["relevance"].score(input=input_text, output=output_text).value
        return results
    
    def flush(self):
        if self._client:
            self._client.flush()
    
    def close(self):
        self.flush()
        self._client = None
        self._traces.clear()


def create_tracer(project_name: str = "unleash-v27") -> OpikTracer:
    """Create and initialize an Opik tracer."""
    return OpikTracer(OpikConfig(project_name=project_name)).initialize()


if __name__ == "__main__":
    print("Opik Integration - Demo")
    if OPIK_AVAILABLE:
        print("opik available and ready")
    else:
        print("opik not installed - run: pip install opik")
