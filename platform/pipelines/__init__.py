"""
Unleashed Platform Integrated Pipelines

This module provides high-level pipelines that combine multiple SDK adapters
to accomplish complex tasks:

- DeepResearchPipeline: Research + Reasoning + Memory
- CodeAnalysisPipeline: Serena + Aider integration
- SelfImprovementPipeline: EvoAgentX + TextGrad

Each pipeline orchestrates multiple adapters with proper error handling,
caching, and performance optimization.
"""

from typing import Dict, Any, Optional, List

# Pipeline availability tracking
PIPELINE_STATUS: Dict[str, Dict[str, Any]] = {}


def register_pipeline(name: str, available: bool, dependencies: Optional[List[str]] = None):
    """Register a pipeline's availability status."""
    PIPELINE_STATUS[name] = {
        "available": available,
        "initialized": False,
        "dependencies": dependencies or [],
    }


def get_pipeline_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all registered pipelines."""
    return PIPELINE_STATUS.copy()


# Lazy imports
def get_deep_research_pipeline():
    """Get DeepResearchPipeline if available."""
    try:
        from .deep_research_pipeline import DeepResearchPipeline, PIPELINE_AVAILABLE
        return DeepResearchPipeline if PIPELINE_AVAILABLE else None
    except ImportError:
        return None


def get_self_improvement_pipeline():
    """Get SelfImprovementPipeline if available."""
    try:
        from .self_improvement_pipeline import SelfImprovementPipeline, PIPELINE_AVAILABLE
        return SelfImprovementPipeline if PIPELINE_AVAILABLE else None
    except ImportError:
        return None


def get_code_analysis_pipeline():
    """Get CodeAnalysisPipeline if available."""
    try:
        from .code_analysis_pipeline import CodeAnalysisPipeline, PIPELINE_AVAILABLE
        return CodeAnalysisPipeline if PIPELINE_AVAILABLE else None
    except ImportError:
        return None


def get_agent_evolution_pipeline():
    """Get AgentEvolutionPipeline if available."""
    try:
        from .agent_evolution_pipeline import AgentEvolutionPipeline, PIPELINE_AVAILABLE
        return AgentEvolutionPipeline if PIPELINE_AVAILABLE else None
    except ImportError:
        return None


__all__ = [
    "PIPELINE_STATUS",
    "register_pipeline",
    "get_pipeline_status",
    "get_deep_research_pipeline",
    "get_self_improvement_pipeline",
    "get_code_analysis_pipeline",
    "get_agent_evolution_pipeline",
]
