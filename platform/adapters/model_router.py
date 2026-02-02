"""
UNLEASH V2 Intelligent Model Router
====================================

Performance-first model routing with 3-tier architecture:
- TIER 1 (LOCAL): GLM-4.7, DeepSeek Coder, Mistral - FREE, fast
- TIER 2 (HYBRID): DeepSeek API, Gemini Flash - Cheap, good quality
- TIER 3 (PREMIUM): Opus 4.5, GPT-5.2, Gemini 3 Pro - Best performance

Research-verified (2026-01-30):
- Context7: Model benchmarks from LMArena, LMCouncil
- Exa: Production patterns for model routing
- Architecture: Hybrid Mesh with intelligent classification

Philosophy: Best performance FIRST, then optimize cost through smart routing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Dict, List, TYPE_CHECKING
import logging

# V118: Import httpx for connection pooling
if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class ModelTier(Enum):
    """Model tiers for routing."""
    LOCAL = "local"       # Free, fast, 85-90% quality
    HYBRID = "hybrid"     # Cheap API, 90-95% quality
    PREMIUM = "premium"   # Best models, 100% quality


class TaskComplexity(Enum):
    """Task complexity levels."""
    TRIVIAL = 1      # Simple lookup, formatting
    SIMPLE = 2       # Basic coding, single-step
    MODERATE = 3     # Multi-step, some reasoning
    COMPLEX = 4      # Architecture, security, deep analysis
    EXPERT = 5       # Novel problems, research, critical decisions


class TaskDomain(Enum):
    """Task domains for specialized routing."""
    CODING = "coding"
    REASONING = "reasoning"
    CREATIVE = "creative"
    RESEARCH = "research"
    CHAT = "chat"
    MATH = "math"
    SECURITY = "security"
    ARCHITECTURE = "architecture"


class ModelProvider(Enum):
    """Model providers."""
    OLLAMA = "ollama"           # Local Ollama
    ANTHROPIC = "anthropic"     # Claude
    OPENAI = "openai"           # GPT
    GOOGLE = "google"           # Gemini
    DEEPSEEK = "deepseek"       # DeepSeek API
    MOONSHOT = "moonshot"       # Kimi


# =============================================================================
# Model Configurations (January 2026 - Verified)
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider: ModelProvider
    tier: ModelTier

    # Capabilities (0-100 scale)
    coding_score: int = 80
    reasoning_score: int = 80
    speed_score: int = 80

    # Cost
    cost_per_1k_tokens: float = 0.0  # 0 for local

    # Technical
    context_window: int = 32000
    max_output: int = 4096
    supports_vision: bool = False
    supports_tools: bool = True

    # Local-specific
    vram_required_gb: float = 0.0
    ollama_name: str = ""  # Name in Ollama

    # API-specific
    api_model_id: str = ""  # API model identifier

    @property
    def is_local(self) -> bool:
        return self.provider == ModelProvider.OLLAMA


# Tier 1: Local Models (FREE)
LOCAL_MODELS = {
    "glm-4.7": ModelConfig(
        name="GLM-4.7",
        provider=ModelProvider.OLLAMA,
        tier=ModelTier.LOCAL,
        coding_score=85,
        reasoning_score=82,
        speed_score=90,
        context_window=64000,
        vram_required_gb=8.0,
        ollama_name="glm-4.7",
    ),
    "deepseek-coder-v3": ModelConfig(
        name="DeepSeek Coder V3",
        provider=ModelProvider.OLLAMA,
        tier=ModelTier.LOCAL,
        coding_score=93,
        reasoning_score=78,
        speed_score=85,
        context_window=32000,
        vram_required_gb=16.0,
        ollama_name="deepseek-coder-v3:8b",
    ),
    "mistral-small": ModelConfig(
        name="Mistral Small 3.2",
        provider=ModelProvider.OLLAMA,
        tier=ModelTier.LOCAL,
        coding_score=78,
        reasoning_score=75,
        speed_score=95,
        context_window=32000,
        vram_required_gb=8.0,
        ollama_name="mistral-small:latest",
    ),
    "phi-4": ModelConfig(
        name="Phi-4 (Router)",
        provider=ModelProvider.OLLAMA,
        tier=ModelTier.LOCAL,
        coding_score=70,
        reasoning_score=72,
        speed_score=98,  # Fastest for routing
        context_window=16000,
        vram_required_gb=6.0,
        ollama_name="phi4",
    ),
    "qwen-coder-32b": ModelConfig(
        name="Qwen2.5-Coder-32B",
        provider=ModelProvider.OLLAMA,
        tier=ModelTier.LOCAL,
        coding_score=91,
        reasoning_score=80,
        speed_score=75,
        context_window=32000,
        vram_required_gb=24.0,
        ollama_name="qwen2.5-coder:32b",
    ),
}

# Tier 2: Hybrid Models (Cheap API)
# Following everything-claude-code pattern: Haiku for 70% of tasks
HYBRID_MODELS = {
    "claude-haiku": ModelConfig(
        name="Claude 3.5 Haiku",
        provider=ModelProvider.ANTHROPIC,
        tier=ModelTier.HYBRID,
        coding_score=88,
        reasoning_score=85,
        speed_score=98,  # Very fast
        cost_per_1k_tokens=0.00025,  # $0.25/M tokens - cheapest Claude
        context_window=200000,
        max_output=8192,
        supports_vision=True,
        supports_tools=True,
        api_model_id="claude-3-5-haiku-20241022",
    ),
    "deepseek-api": ModelConfig(
        name="DeepSeek V3 API",
        provider=ModelProvider.DEEPSEEK,
        tier=ModelTier.HYBRID,
        coding_score=92,
        reasoning_score=88,
        speed_score=85,
        cost_per_1k_tokens=0.0005,  # $0.50/M tokens
        context_window=64000,
        api_model_id="deepseek-chat",
    ),
    "gemini-flash": ModelConfig(
        name="Gemini 2.0 Flash",
        provider=ModelProvider.GOOGLE,
        tier=ModelTier.HYBRID,
        coding_score=85,
        reasoning_score=87,
        speed_score=92,
        cost_per_1k_tokens=0.0001,  # Very cheap
        context_window=1000000,  # 1M context
        supports_vision=True,
        api_model_id="gemini-2.0-flash",
    ),
    "kimi-k2.5": ModelConfig(
        name="Kimi K2.5",
        provider=ModelProvider.MOONSHOT,
        tier=ModelTier.HYBRID,
        coding_score=90,
        reasoning_score=92,
        speed_score=80,
        cost_per_1k_tokens=0.001,
        context_window=128000,
        api_model_id="moonshot-v1-128k",
    ),
}

# Tier 3: Premium Models (Best Performance)
PREMIUM_MODELS = {
    "claude-opus": ModelConfig(
        name="Claude Opus 4.5",
        provider=ModelProvider.ANTHROPIC,
        tier=ModelTier.PREMIUM,
        coding_score=98,
        reasoning_score=98,
        speed_score=70,
        cost_per_1k_tokens=0.015,
        context_window=200000,
        max_output=16384,
        supports_vision=True,
        api_model_id="claude-opus-4-5-20251101",
    ),
    "claude-sonnet": ModelConfig(
        name="Claude Sonnet 4",
        provider=ModelProvider.ANTHROPIC,
        tier=ModelTier.PREMIUM,
        coding_score=95,
        reasoning_score=94,
        speed_score=85,
        cost_per_1k_tokens=0.003,
        context_window=200000,
        supports_vision=True,
        api_model_id="claude-sonnet-4-20250514",
    ),
    "gpt-5.2": ModelConfig(
        name="GPT-5.2 Extended",
        provider=ModelProvider.OPENAI,
        tier=ModelTier.PREMIUM,
        coding_score=94,
        reasoning_score=99,  # Best reasoning
        speed_score=75,
        cost_per_1k_tokens=0.02,
        context_window=128000,
        supports_vision=True,
        api_model_id="gpt-5.2-extended",
    ),
    "gemini-3-pro": ModelConfig(
        name="Gemini 3 Pro",
        provider=ModelProvider.GOOGLE,
        tier=ModelTier.PREMIUM,
        coding_score=93,
        reasoning_score=95,
        speed_score=88,
        cost_per_1k_tokens=0.005,
        context_window=2000000,  # 2M context
        supports_vision=True,
        api_model_id="gemini-3-pro",
    ),
}

# All models combined
ALL_MODELS = {**LOCAL_MODELS, **HYBRID_MODELS, **PREMIUM_MODELS}


# =============================================================================
# Task Classification
# =============================================================================

@dataclass
class TaskClassification:
    """Result of task classification."""
    complexity: TaskComplexity
    domain: TaskDomain
    requires_vision: bool = False
    requires_tools: bool = False
    requires_long_context: bool = False
    estimated_tokens: int = 1000
    confidence: float = 0.8
    reasoning: str = ""


class TaskClassifier:
    """
    Classifies tasks to determine optimal model tier.

    Uses keyword analysis, pattern matching, and context inference
    to route tasks to the most efficient model.

    Following everything-claude-code pattern:
    - Haiku (70%): Exploration, simple edits, single-file changes
    - Sonnet (25%): Multi-file implementation, moderate complexity
    - Opus (5%): Security, architecture, complex reasoning
    """

    # Complexity indicators
    TRIVIAL_PATTERNS = [
        "what is", "define", "explain briefly", "list", "format",
        "convert", "translate simple", "summarize short",
    ]

    # EXPLORATION patterns (70% - route to Haiku)
    EXPLORATION_PATTERNS = [
        "find", "search", "where is", "list files", "show me",
        "glob", "grep", "explore", "look for", "check if",
        "read file", "show content", "what files", "directory",
        "quick", "simple", "just", "only", "single file",
    ]

    SIMPLE_PATTERNS = [
        "write a function", "fix this bug", "add a feature",
        "implement", "create a class", "write test",
    ]

    # MULTI-FILE patterns (25% - route to Sonnet)
    MULTI_FILE_PATTERNS = [
        "multiple files", "across files", "refactor", "rename across",
        "update all", "change everywhere", "full implementation",
        "create module", "new feature", "integration",
    ]

    COMPLEX_PATTERNS = [
        "architecture", "design system", "security audit",
        "optimize performance", "refactor large", "debug complex",
        "multi-step", "analyze codebase", "review security",
    ]

    EXPERT_PATTERNS = [
        "novel algorithm", "research", "vulnerability analysis",
        "critical decision", "production deployment", "scale to",
        "distributed system", "consensus protocol",
    ]

    # Domain indicators
    CODING_KEYWORDS = [
        "code", "function", "class", "implement", "bug", "error",
        "typescript", "python", "javascript", "rust", "api",
    ]

    REASONING_KEYWORDS = [
        "analyze", "evaluate", "compare", "decide", "strategy",
        "why", "how", "trade-off", "pros cons",
    ]

    SECURITY_KEYWORDS = [
        "security", "auth", "token", "password", "vulnerability", "vulnerabilities",
        "owasp", "injection", "xss", "csrf", "encryption", "exploit",
        "security review", "security audit", "penetration", "pentest",
    ]

    ARCHITECTURE_KEYWORDS = [
        "architecture", "design", "system", "scale", "microservice",
        "distributed", "database", "cache", "queue",
    ]

    def classify(self, task: str, context: Optional[str] = None) -> TaskClassification:
        """
        Classify a task to determine routing.

        Args:
            task: The task description
            context: Optional additional context

        Returns:
            TaskClassification with complexity, domain, and requirements
        """
        task_lower = task.lower()
        full_text = f"{task_lower} {(context or '').lower()}"

        # Determine complexity
        complexity = self._determine_complexity(full_text)

        # Determine domain
        domain = self._determine_domain(full_text)

        # Check requirements
        requires_vision = any(kw in full_text for kw in [
            "image", "screenshot", "diagram", "visual", "picture", "photo"
        ])

        requires_tools = any(kw in full_text for kw in [
            "search", "browse", "fetch", "api call", "execute", "run"
        ])

        requires_long_context = len(full_text) > 10000 or any(kw in full_text for kw in [
            "entire codebase", "all files", "full document", "complete analysis"
        ])

        # Estimate tokens
        estimated_tokens = self._estimate_tokens(task, context)

        # Build reasoning
        reasoning = (
            f"Complexity: {complexity.name} (domain: {domain.value}). "
            f"Vision: {requires_vision}, Tools: {requires_tools}, "
            f"Long context: {requires_long_context}"
        )

        return TaskClassification(
            complexity=complexity,
            domain=domain,
            requires_vision=requires_vision,
            requires_tools=requires_tools,
            requires_long_context=requires_long_context,
            estimated_tokens=estimated_tokens,
            confidence=0.85,
            reasoning=reasoning,
        )

    def _determine_complexity(self, text: str) -> TaskComplexity:
        """
        Determine task complexity from text.

        Following everything-claude-code 70/25/5 pattern:
        - TRIVIAL/exploration → Haiku (70%)
        - SIMPLE/MODERATE → Sonnet (25%)
        - COMPLEX/EXPERT → Opus (5%)
        """
        # Check patterns in order of complexity
        if any(p in text for p in self.EXPERT_PATTERNS):
            return TaskComplexity.EXPERT
        if any(p in text for p in self.COMPLEX_PATTERNS):
            return TaskComplexity.COMPLEX

        # Multi-file patterns indicate MODERATE (Sonnet territory)
        if any(p in text for p in self.MULTI_FILE_PATTERNS):
            return TaskComplexity.MODERATE

        if any(p in text for p in self.SIMPLE_PATTERNS):
            return TaskComplexity.SIMPLE

        # Exploration patterns are TRIVIAL (Haiku territory - 70%)
        if any(p in text for p in self.EXPLORATION_PATTERNS):
            return TaskComplexity.TRIVIAL

        if any(p in text for p in self.TRIVIAL_PATTERNS):
            return TaskComplexity.TRIVIAL

        # Default based on length - prefer TRIVIAL for shorter texts
        if len(text) > 5000:
            return TaskComplexity.COMPLEX
        if len(text) > 2000:
            return TaskComplexity.MODERATE
        if len(text) > 500:
            return TaskComplexity.SIMPLE
        return TaskComplexity.TRIVIAL  # Default to Haiku-appropriate

    def _determine_domain(self, text: str) -> TaskDomain:
        """Determine task domain from text."""
        # Count keyword matches for each domain
        scores = {
            TaskDomain.CODING: sum(1 for kw in self.CODING_KEYWORDS if kw in text),
            TaskDomain.REASONING: sum(1 for kw in self.REASONING_KEYWORDS if kw in text),
            TaskDomain.SECURITY: sum(1 for kw in self.SECURITY_KEYWORDS if kw in text),
            TaskDomain.ARCHITECTURE: sum(1 for kw in self.ARCHITECTURE_KEYWORDS if kw in text),
        }

        # Security and Architecture take priority (lower threshold for security - critical!)
        if scores[TaskDomain.SECURITY] >= 1:
            return TaskDomain.SECURITY
        if scores[TaskDomain.ARCHITECTURE] >= 2:
            return TaskDomain.ARCHITECTURE

        # Return highest scoring domain
        max_domain = max(scores.keys(), key=lambda d: scores[d])
        if scores[max_domain] > 0:
            return max_domain

        return TaskDomain.CHAT

    def _estimate_tokens(self, task: str, context: Optional[str]) -> int:
        """Estimate tokens for task completion."""
        # Rough estimation: 4 chars per token
        input_tokens = (len(task) + len(context or "")) // 4

        # Output estimation based on task type
        task_lower = task.lower()
        if "explain" in task_lower or "analyze" in task_lower:
            output_multiplier = 3
        elif "implement" in task_lower or "code" in task_lower:
            output_multiplier = 5
        elif "fix" in task_lower or "debug" in task_lower:
            output_multiplier = 2
        else:
            output_multiplier = 2

        return input_tokens + (input_tokens * output_multiplier)


# =============================================================================
# Model Router
# =============================================================================

@dataclass
class RoutingDecision:
    """Decision from the model router."""
    model: ModelConfig
    tier: ModelTier
    reason: str
    fallback_model: Optional[ModelConfig] = None
    estimated_cost: float = 0.0
    estimated_latency_ms: int = 0
    confidence: float = 0.9


class ModelRouter:
    """
    Intelligent model router for optimal performance/cost balance.

    Routing Strategy:
    1. Classify task (complexity, domain, requirements)
    2. Check tier requirements
    3. Select best model within tier
    4. Provide fallback for reliability

    Example:
        router = ModelRouter()
        decision = router.route("Implement JWT authentication", context="Express.js API")
        print(f"Using {decision.model.name} ({decision.tier.value})")
    """

    def __init__(
        self,
        prefer_local: bool = True,
        cost_sensitivity: float = 0.5,  # 0 = performance only, 1 = cost only
        available_vram_gb: float = 24.0,
    ):
        """
        Initialize the model router.

        Args:
            prefer_local: Prefer local models when possible
            cost_sensitivity: Balance between cost and performance
            available_vram_gb: Available GPU VRAM for local models
        """
        self.prefer_local = prefer_local
        self.cost_sensitivity = cost_sensitivity
        self.available_vram_gb = available_vram_gb
        self.classifier = TaskClassifier()

        # Filter available local models by VRAM
        self._available_local = {
            k: v for k, v in LOCAL_MODELS.items()
            if v.vram_required_gb <= available_vram_gb
        }

        # Statistics
        self._stats = {
            "total_routes": 0,
            "local_routes": 0,
            "hybrid_routes": 0,
            "premium_routes": 0,
            "total_estimated_cost": 0.0,
        }

    def route(
        self,
        task: str,
        context: Optional[str] = None,
        force_tier: Optional[ModelTier] = None,
        require_vision: bool = False,
        require_tools: bool = False,
    ) -> RoutingDecision:
        """
        Route a task to the optimal model.

        Args:
            task: Task description
            context: Optional additional context
            force_tier: Force a specific tier
            require_vision: Require vision capability
            require_tools: Require tool use capability

        Returns:
            RoutingDecision with selected model and metadata
        """
        self._stats["total_routes"] += 1

        # Classify the task
        classification = self.classifier.classify(task, context)

        # Override requirements
        if require_vision:
            classification.requires_vision = True
        if require_tools:
            classification.requires_tools = True

        # Determine required tier
        if force_tier:
            required_tier = force_tier
        else:
            required_tier = self._determine_tier(classification)

        # Select model within tier
        model = self._select_model(required_tier, classification)
        fallback = self._select_fallback(model, classification)

        # Calculate estimates
        estimated_cost = self._estimate_cost(model, classification.estimated_tokens)
        estimated_latency = self._estimate_latency(model, classification.estimated_tokens)

        # Update stats
        self._stats[f"{required_tier.value}_routes"] += 1
        self._stats["total_estimated_cost"] += estimated_cost

        reason = (
            f"Task complexity: {classification.complexity.name}, "
            f"Domain: {classification.domain.value}. "
            f"{classification.reasoning}"
        )

        return RoutingDecision(
            model=model,
            tier=required_tier,
            reason=reason,
            fallback_model=fallback,
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
            confidence=classification.confidence,
        )

    def _determine_tier(self, classification: TaskClassification) -> ModelTier:
        """
        Determine required tier based on classification.

        Following everything-claude-code 70/25/5 pattern:
        - HYBRID (Haiku): 70% - exploration, simple edits, single-file
        - PREMIUM (Sonnet): 25% - multi-file implementation
        - PREMIUM (Opus): 5% - security, architecture, complex reasoning

        The key insight: most tasks DON'T need Opus or even Sonnet.
        Prefer cheaper models and only escalate when truly needed.
        """
        # ===== 5% OPUS TERRITORY =====
        # Only for truly expert-level tasks
        if classification.complexity == TaskComplexity.EXPERT:
            return ModelTier.PREMIUM

        # Security and Architecture need careful review (Opus)
        if classification.domain in (TaskDomain.SECURITY, TaskDomain.ARCHITECTURE):
            return ModelTier.PREMIUM

        # ===== 25% SONNET TERRITORY =====
        # Complex tasks that aren't security/architecture critical
        if classification.complexity == TaskComplexity.COMPLEX:
            # Use Sonnet from PREMIUM tier
            return ModelTier.PREMIUM

        # ===== 70% HAIKU TERRITORY =====
        # Everything else goes to HYBRID which now includes Haiku

        # Vision: Haiku supports vision, use HYBRID
        if classification.requires_vision:
            return ModelTier.HYBRID

        # Moderate tasks: prefer HYBRID (Haiku) over LOCAL for reliability
        # unless we specifically want local (Ollama) inference
        if classification.complexity == TaskComplexity.MODERATE:
            # Only use LOCAL if explicitly preferring local AND local is available
            if self.prefer_local and self._available_local and self.cost_sensitivity > 0.8:
                return ModelTier.LOCAL
            return ModelTier.HYBRID

        # Simple/Trivial: HYBRID (Haiku) by default
        # This is the key change - Haiku handles 70% of tasks
        if classification.complexity in (TaskComplexity.SIMPLE, TaskComplexity.TRIVIAL):
            # Use LOCAL only if cost_sensitivity is very high
            if self.prefer_local and self._available_local and self.cost_sensitivity > 0.9:
                return ModelTier.LOCAL
            return ModelTier.HYBRID

        return ModelTier.HYBRID

    def _select_model(
        self,
        tier: ModelTier,
        classification: TaskClassification,
    ) -> ModelConfig:
        """Select best model within tier."""
        if tier == ModelTier.LOCAL:
            models = self._available_local
        elif tier == ModelTier.HYBRID:
            models = HYBRID_MODELS
        else:
            models = PREMIUM_MODELS

        if not models:
            # Fallback to hybrid if local not available
            if tier == ModelTier.LOCAL:
                models = HYBRID_MODELS
            else:
                models = PREMIUM_MODELS

        # Score models based on task domain
        def score_model(model: ModelConfig) -> float:
            if classification.domain == TaskDomain.CODING:
                return model.coding_score * 1.5 + model.reasoning_score
            elif classification.domain in (TaskDomain.REASONING, TaskDomain.ARCHITECTURE):
                return model.reasoning_score * 1.5 + model.coding_score
            else:
                return model.coding_score + model.reasoning_score

        # Filter by requirements
        valid_models = [
            m for m in models.values()
            if (not classification.requires_vision or m.supports_vision)
            and (not classification.requires_tools or m.supports_tools)
        ]

        if not valid_models:
            valid_models = list(models.values())

        # Return highest scoring model
        return max(valid_models, key=score_model)

    def _select_fallback(
        self,
        primary: ModelConfig,
        classification: TaskClassification,
    ) -> Optional[ModelConfig]:
        """Select fallback model for reliability."""
        # Get next tier up
        if primary.tier == ModelTier.LOCAL:
            fallback_tier = HYBRID_MODELS
        elif primary.tier == ModelTier.HYBRID:
            fallback_tier = PREMIUM_MODELS
        else:
            return None  # Premium has no fallback

        # Select best from fallback tier
        models = list(fallback_tier.values())
        if classification.requires_vision:
            models = [m for m in models if m.supports_vision]

        if models:
            return max(models, key=lambda m: m.coding_score + m.reasoning_score)

        return None

    def _estimate_cost(self, model: ModelConfig, tokens: int) -> float:
        """Estimate cost for model usage."""
        return (tokens / 1000) * model.cost_per_1k_tokens

    def _estimate_latency(self, model: ModelConfig, tokens: int) -> int:
        """Estimate latency in milliseconds."""
        # Base latency by tier
        base_latency = {
            ModelTier.LOCAL: 100,
            ModelTier.HYBRID: 500,
            ModelTier.PREMIUM: 1000,
        }

        # Tokens per second (rough)
        tps = {
            ModelTier.LOCAL: 100,
            ModelTier.HYBRID: 50,
            ModelTier.PREMIUM: 30,
        }

        base = base_latency.get(model.tier, 500)
        token_time = (tokens / tps.get(model.tier, 50)) * 1000

        return int(base + token_time)

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total = self._stats["total_routes"]
        return {
            **self._stats,
            "local_percentage": (self._stats["local_routes"] / total * 100) if total else 0,
            "hybrid_percentage": (self._stats["hybrid_routes"] / total * 100) if total else 0,
            "premium_percentage": (self._stats["premium_routes"] / total * 100) if total else 0,
            "avg_cost_per_route": self._stats["total_estimated_cost"] / total if total else 0,
        }

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models by tier."""
        return {
            "local": list(self._available_local.keys()),
            "hybrid": list(HYBRID_MODELS.keys()),
            "premium": list(PREMIUM_MODELS.keys()),
        }

    def get_claude_model(
        self,
        task: str,
        context: Optional[str] = None,
    ) -> ModelConfig:
        """
        Get the appropriate Claude model following 70/25/5 pattern.

        This method is specifically for when you want to use Claude
        (Anthropic) models and need to pick between Haiku/Sonnet/Opus.

        Distribution:
        - Haiku (70%): Exploration, simple edits, single-file changes
        - Sonnet (25%): Multi-file implementation, moderate complexity
        - Opus (5%): Security, architecture, complex reasoning

        Args:
            task: Task description
            context: Optional additional context

        Returns:
            ModelConfig for the appropriate Claude variant
        """
        classification = self.classifier.classify(task, context)

        # 5% Opus - Expert complexity or security/architecture
        if classification.complexity == TaskComplexity.EXPERT:
            return PREMIUM_MODELS["claude-opus"]

        if classification.domain in (TaskDomain.SECURITY, TaskDomain.ARCHITECTURE):
            return PREMIUM_MODELS["claude-opus"]

        # 25% Sonnet - Complex tasks, multi-file work
        if classification.complexity == TaskComplexity.COMPLEX:
            return PREMIUM_MODELS["claude-sonnet"]

        if classification.complexity == TaskComplexity.MODERATE:
            return PREMIUM_MODELS["claude-sonnet"]

        # 70% Haiku - Everything else (exploration, simple, trivial)
        return HYBRID_MODELS["claude-haiku"]

    def get_claude_model_name(
        self,
        task: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Get Claude model name string for direct API calls.

        Convenience wrapper around get_claude_model() that returns
        just the model ID string.

        Args:
            task: Task description
            context: Optional additional context

        Returns:
            Model ID string (e.g., "claude-3-5-haiku-20241022")
        """
        model = self.get_claude_model(task, context)
        return model.api_model_id


# =============================================================================
# Ollama Integration
# =============================================================================

class OllamaClient:
    """
    Client for Ollama local model inference (V118 optimized with connection pooling).

    Provides OpenAI-compatible API for local models.
    """

    # V118: Shared client for connection pooling
    _shared_client: Optional["httpx.AsyncClient"] = None

    def __init__(self, base_url: Optional[str] = None):
        import os
        # V45 FIX: Environment-configurable Ollama URL (was hardcoded)
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self._available_models: List[str] = []

    def _get_client(self) -> "httpx.AsyncClient":
        """V118: Get shared client with connection pooling."""
        import httpx
        if OllamaClient._shared_client is None:
            OllamaClient._shared_client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
                timeout=httpx.Timeout(120.0),
            )
        return OllamaClient._shared_client

    async def check_health(self) -> bool:
        """Check if Ollama is running."""
        try:
            client = self._get_client()
            response = await client.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self._available_models = [m["name"] for m in data.get("models", [])]
                return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
        return False

    async def list_models(self) -> List[str]:
        """List available models in Ollama."""
        await self.check_health()
        return self._available_models

    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """Generate completion from Ollama model."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }

        if system:
            payload["system"] = system

        client = self._get_client()
        response = await client.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("response", "")
        else:
            raise Exception(f"Ollama error: {response.status_code}")

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
    ) -> str:
        """Chat completion from Ollama model."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        client = self._get_client()
        response = await client.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("message", {}).get("content", "")
        else:
            raise Exception(f"Ollama error: {response.status_code}")


# =============================================================================
# Demo
# =============================================================================

async def demo():
    """Demonstrate the model router with Haiku-first pattern."""
    print("=" * 70)
    print("UNLEASH V2 Model Router - Haiku-First Demo")
    print("Following everything-claude-code 70/25/5 pattern")
    print("=" * 70)
    print()

    router = ModelRouter(available_vram_gb=24.0)

    # Test cases demonstrating 70/25/5 distribution
    test_cases = [
        # 70% Haiku territory - exploration, simple, trivial
        ("What is a variable?", None),  # Trivial → Haiku
        ("Find all Python files in src/", None),  # Exploration → Haiku
        ("Show me the contents of config.py", None),  # Exploration → Haiku
        ("Quick fix for this typo", None),  # Simple → Haiku

        # 25% Sonnet territory - moderate, multi-file
        ("Implement a new API endpoint with tests", None),  # Moderate → Sonnet
        ("Refactor across multiple files", None),  # Multi-file → Sonnet

        # 5% Opus territory - expert, security, architecture
        ("Design a distributed cache architecture for 1M QPS", None),  # Architecture → Opus
        ("Review this code for security vulnerabilities", None),  # Security → Opus
        ("Analyze the performance bottleneck in this distributed system", "Large context..."),  # Expert → Opus
    ]

    haiku_count = 0
    sonnet_count = 0
    opus_count = 0

    print("General Routing (any provider):")
    print("-" * 50)
    for task, context in test_cases:
        decision = router.route(task, context)
        print(f"Task: {task[:45]}...")
        print(f"  → Model: {decision.model.name} ({decision.tier.value})")
        print(f"  → Cost: ${decision.estimated_cost:.4f}")
        print()

    print("\n" + "=" * 70)
    print("Claude-Specific Routing (Haiku/Sonnet/Opus):")
    print("-" * 50)

    for task, context in test_cases:
        model = router.get_claude_model(task, context)
        print(f"Task: {task[:45]}...")
        print(f"  → Claude: {model.name} (${model.cost_per_1k_tokens}/1k)")

        if "haiku" in model.name.lower():
            haiku_count += 1
        elif "sonnet" in model.name.lower():
            sonnet_count += 1
        elif "opus" in model.name.lower():
            opus_count += 1

    total = len(test_cases)
    print("\n" + "=" * 70)
    print("Distribution Analysis:")
    print(f"  Haiku:  {haiku_count}/{total} ({haiku_count/total*100:.0f}%) - Target: 70%")
    print(f"  Sonnet: {sonnet_count}/{total} ({sonnet_count/total*100:.0f}%) - Target: 25%")
    print(f"  Opus:   {opus_count}/{total} ({opus_count/total*100:.0f}%) - Target: 5%")
    print("=" * 70)

    # Routing stats
    print("\nRouting Statistics:")
    stats = router.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Test Ollama
    print("\nTesting Ollama connection...")
    ollama = OllamaClient()
    if await ollama.check_health():
        print(f"Ollama available with models: {await ollama.list_models()}")
    else:
        print("Ollama not available (install with: curl -fsSL https://ollama.com/install.sh | sh)")


if __name__ == "__main__":
    asyncio.run(demo())
