#!/usr/bin/env python3
"""
Rails Compatibility Layer for Python 3.14+
Part of V34 Architecture - Phase 10 Fix.

This module provides a rule-based replacement for nemo-guardrails.
The nemo-guardrails package has complex dependencies (annoy, nemoguardrails)
that don't have Python 3.14+ wheels available.

Key features replaced:
- Colang-style flow definitions
- Input/output rails
- Topic filtering
- Response validation
- Action execution

Usage:
    from core.safety.rails_compat import (
        Guardrails,
        Rail,
        RailConfig,
        check_input,
        check_output,
    )

    # Create guardrails
    rails = Guardrails()

    # Add a rail
    rails.add_rail(Rail(
        name="no_politics",
        pattern=r"(?i)\\b(democrat|republican|election)\\b",
        action="block",
        message="I can't discuss political topics."
    ))

    # Check input
    result = rails.check_input("What about the election?")
    if result.blocked:
        print(result.message)
"""

from __future__ import annotations

import re
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class RailAction(str, Enum):
    """Actions a rail can take."""
    ALLOW = "allow"
    BLOCK = "block"
    REDIRECT = "redirect"
    MODIFY = "modify"
    WARN = "warn"
    LOG = "log"


class RailType(str, Enum):
    """Types of rails."""
    INPUT = "input"
    OUTPUT = "output"
    BOTH = "both"


class CheckPhase(str, Enum):
    """When to apply the rail."""
    PRE = "pre"      # Before LLM call
    POST = "post"    # After LLM call


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Rail:
    """A single guardrail rule."""
    name: str
    pattern: Optional[str] = None  # Regex pattern to match
    keywords: Optional[List[str]] = None  # Keywords to match
    action: RailAction = RailAction.BLOCK
    message: Optional[str] = None  # Message to return when triggered
    redirect_response: Optional[str] = None  # Response for redirect action
    rail_type: RailType = RailType.BOTH
    priority: int = 100  # Lower = higher priority
    enabled: bool = True
    custom_check: Optional[Callable[[str], bool]] = None  # Custom validation function

    def __post_init__(self):
        if self.pattern:
            self._compiled_pattern = re.compile(self.pattern, re.IGNORECASE)
        else:
            self._compiled_pattern = None

        if self.keywords:
            self._keyword_pattern = re.compile(
                r'\b(' + '|'.join(re.escape(k) for k in self.keywords) + r')\b',
                re.IGNORECASE
            )
        else:
            self._keyword_pattern = None

    def matches(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check if this rail matches the text."""
        if not self.enabled:
            return False, None

        # Custom check
        if self.custom_check:
            if self.custom_check(text):
                return True, self.message

        # Pattern matching
        if self._compiled_pattern:
            match = self._compiled_pattern.search(text)
            if match:
                return True, self.message

        # Keyword matching
        if self._keyword_pattern:
            match = self._keyword_pattern.search(text)
            if match:
                return True, self.message

        return False, None


@dataclass
class RailResult:
    """Result of applying rails to text."""
    original_text: str
    processed_text: str
    blocked: bool = False
    modified: bool = False
    warnings: List[str] = field(default_factory=list)
    triggered_rails: List[str] = field(default_factory=list)
    message: Optional[str] = None
    action_taken: Optional[RailAction] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocked": self.blocked,
            "modified": self.modified,
            "warnings": self.warnings,
            "triggered_rails": self.triggered_rails,
            "message": self.message,
            "action_taken": self.action_taken.value if self.action_taken else None
        }


@dataclass
class RailConfig:
    """Configuration for guardrails."""
    rails: List[Rail] = field(default_factory=list)
    default_input_action: RailAction = RailAction.BLOCK
    default_output_action: RailAction = RailAction.WARN
    log_triggers: bool = True
    strict_mode: bool = False  # Block on any match vs allow by default


# =============================================================================
# TOPIC DEFINITIONS
# =============================================================================

# Pre-defined topic patterns
TOPIC_PATTERNS = {
    "politics": [
        r'\b(democrat|republican|election|vote|voting|president|congress|senator|political|party|liberal|conservative)\b',
        r'\b(trump|biden|obama|clinton)\b',
    ],
    "religion": [
        r'\b(god|jesus|allah|buddha|christian|muslim|jewish|hindu|atheist|pray|worship|church|mosque|temple)\b',
    ],
    "violence": [
        r'\b(kill|murder|attack|assault|shoot|stab|bomb|weapon|gun|knife)\b',
        r'\b(hurt|harm|injure|beat|fight)\s+(someone|people|you|them)\b',
    ],
    "illegal_activities": [
        r'\b(hack|phish|scam|fraud|steal|rob|burgl|drug|cocaine|heroin|meth)\b',
        r'\bhow\s+to\s+(make|create|build)\s+(bomb|weapon|drug)\b',
    ],
    "self_harm": [
        r'\b(suicide|kill\s+myself|end\s+my\s+life|self[- ]?harm)\b',
    ],
    "sexual_content": [
        r'\b(porn|xxx|nude|naked|sex\s+with|erotic)\b',
    ],
    "personal_info": [
        r'\b(ssn|social\s+security|credit\s+card|bank\s+account|password)\b',
    ],
    "medical_advice": [
        r'\b(diagnose|prescribe|treatment\s+for|cure\s+for|medical\s+advice)\b',
    ],
    "legal_advice": [
        r'\b(legal\s+advice|sue|lawsuit|attorney|lawyer|court\s+case)\b',
    ],
    "financial_advice": [
        r'\b(invest|stock\s+tip|financial\s+advice|buy\s+stock|sell\s+stock)\b',
    ],
}


def create_topic_rail(
    topic: str,
    action: RailAction = RailAction.BLOCK,
    message: Optional[str] = None
) -> Rail:
    """Create a rail for a predefined topic."""
    if topic not in TOPIC_PATTERNS:
        raise ValueError(f"Unknown topic: {topic}. Available: {list(TOPIC_PATTERNS.keys())}")

    patterns = TOPIC_PATTERNS[topic]
    combined_pattern = '|'.join(f'({p})' for p in patterns)

    return Rail(
        name=f"topic_{topic}",
        pattern=combined_pattern,
        action=action,
        message=message or f"I'm not able to discuss {topic.replace('_', ' ')}.",
        rail_type=RailType.BOTH
    )


# =============================================================================
# FLOW DEFINITIONS (Simplified Colang replacement)
# =============================================================================

@dataclass
class Flow:
    """A conversation flow definition."""
    name: str
    triggers: List[str]  # Input patterns that trigger this flow
    responses: List[str]  # Possible responses
    conditions: Optional[List[Callable[[Dict], bool]]] = None
    next_flows: Optional[List[str]] = None

    def matches(self, text: str) -> bool:
        """Check if this flow matches the input."""
        for trigger in self.triggers:
            if re.search(trigger, text, re.IGNORECASE):
                return True
        return False

    def get_response(self) -> str:
        """Get a response from this flow."""
        import random
        return random.choice(self.responses)


class FlowEngine:
    """Simple flow execution engine."""

    def __init__(self):
        self.flows: Dict[str, Flow] = {}
        self.context: Dict[str, Any] = {}

    def add_flow(self, flow: Flow):
        """Add a flow to the engine."""
        self.flows[flow.name] = flow

    def process(self, text: str) -> Optional[str]:
        """Process input through flows."""
        for flow in self.flows.values():
            if flow.matches(text):
                # Check conditions
                if flow.conditions:
                    all_conditions_met = all(
                        cond(self.context) for cond in flow.conditions
                    )
                    if not all_conditions_met:
                        continue

                return flow.get_response()

        return None

    def set_context(self, key: str, value: Any):
        """Set a context variable."""
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.context.get(key, default)


# =============================================================================
# GUARDRAILS ENGINE
# =============================================================================

class Guardrails:
    """Main guardrails engine."""

    def __init__(self, config: Optional[RailConfig] = None):
        self.config = config or RailConfig()
        self._input_rails: List[Rail] = []
        self._output_rails: List[Rail] = []
        self.flow_engine = FlowEngine()

        # Load rails from config
        for rail in self.config.rails:
            self.add_rail(rail)

    def add_rail(self, rail: Rail):
        """Add a rail to the engine."""
        if rail.rail_type in (RailType.INPUT, RailType.BOTH):
            self._input_rails.append(rail)
            self._input_rails.sort(key=lambda r: r.priority)

        if rail.rail_type in (RailType.OUTPUT, RailType.BOTH):
            self._output_rails.append(rail)
            self._output_rails.sort(key=lambda r: r.priority)

    def remove_rail(self, name: str):
        """Remove a rail by name."""
        self._input_rails = [r for r in self._input_rails if r.name != name]
        self._output_rails = [r for r in self._output_rails if r.name != name]

    def add_topic_rail(
        self,
        topic: str,
        action: RailAction = RailAction.BLOCK,
        message: Optional[str] = None
    ):
        """Add a predefined topic rail."""
        rail = create_topic_rail(topic, action, message)
        self.add_rail(rail)

    def check_input(self, text: str) -> RailResult:
        """Check input against input rails."""
        return self._check(text, self._input_rails, CheckPhase.PRE)

    def check_output(self, text: str) -> RailResult:
        """Check output against output rails."""
        return self._check(text, self._output_rails, CheckPhase.POST)

    def _check(
        self,
        text: str,
        rails: List[Rail],
        phase: CheckPhase
    ) -> RailResult:
        """Apply rails to text."""
        result = RailResult(
            original_text=text,
            processed_text=text
        )

        for rail in rails:
            matches, message = rail.matches(text)

            if not matches:
                continue

            result.triggered_rails.append(rail.name)

            if self.config.log_triggers:
                logger.info(f"Rail triggered: {rail.name} in {phase.value} phase")

            if rail.action == RailAction.BLOCK:
                result.blocked = True
                result.message = message or f"Content blocked by {rail.name}"
                result.action_taken = RailAction.BLOCK
                # Stop processing on first block
                if self.config.strict_mode:
                    return result

            elif rail.action == RailAction.REDIRECT:
                result.blocked = True
                result.message = rail.redirect_response
                result.action_taken = RailAction.REDIRECT
                return result

            elif rail.action == RailAction.MODIFY:
                # For modify action, the custom_check should return modified text
                result.modified = True
                result.action_taken = RailAction.MODIFY

            elif rail.action == RailAction.WARN:
                result.warnings.append(message or f"Warning from {rail.name}")
                result.action_taken = RailAction.WARN

            elif rail.action == RailAction.LOG:
                logger.warning(f"Rail {rail.name} matched: {message}")
                result.action_taken = RailAction.LOG

        return result

    def add_flow(self, flow: Flow):
        """Add a conversation flow."""
        self.flow_engine.add_flow(flow)

    def process_with_flow(self, text: str) -> Optional[str]:
        """Process input through flow engine."""
        return self.flow_engine.process(text)


# =============================================================================
# PREDEFINED GUARDRAIL SETS
# =============================================================================

def create_safety_rails() -> Guardrails:
    """Create a guardrails instance with common safety rails."""
    rails = Guardrails()

    # Add topic rails
    rails.add_topic_rail("violence", RailAction.BLOCK)
    rails.add_topic_rail("illegal_activities", RailAction.BLOCK)
    rails.add_topic_rail("self_harm", RailAction.REDIRECT,
                          "I'm concerned about what you're sharing. Please reach out to a crisis helpline: 988 (Suicide & Crisis Lifeline)")

    # Add PII protection
    rails.add_rail(Rail(
        name="pii_protection",
        pattern=r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',  # SSN pattern
        action=RailAction.BLOCK,
        message="I cannot process personal identification numbers."
    ))

    # Add prompt injection protection
    rails.add_rail(Rail(
        name="prompt_injection",
        pattern=r'(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions?',
        action=RailAction.BLOCK,
        message="I detected a potential prompt injection attempt."
    ))

    return rails


def create_content_moderation_rails() -> Guardrails:
    """Create guardrails for content moderation."""
    rails = Guardrails()

    # Sexual content
    rails.add_topic_rail("sexual_content", RailAction.BLOCK)

    # Violence
    rails.add_topic_rail("violence", RailAction.WARN)

    # Hate speech patterns
    rails.add_rail(Rail(
        name="hate_speech",
        pattern=r'(?i)\b(hate|kill|eliminate)\s+all\s+\w+s?\b',
        action=RailAction.BLOCK,
        message="This content violates our community guidelines."
    ))

    return rails


def create_professional_rails() -> Guardrails:
    """Create guardrails for professional/business use."""
    rails = Guardrails()

    # No medical/legal/financial advice
    rails.add_topic_rail("medical_advice", RailAction.REDIRECT,
                          "I cannot provide medical advice. Please consult a healthcare professional.")
    rails.add_topic_rail("legal_advice", RailAction.REDIRECT,
                          "I cannot provide legal advice. Please consult an attorney.")
    rails.add_topic_rail("financial_advice", RailAction.REDIRECT,
                          "I cannot provide financial advice. Please consult a financial advisor.")

    # No politics
    rails.add_topic_rail("politics", RailAction.REDIRECT,
                          "I prefer not to discuss political topics in this context.")

    return rails


# =============================================================================
# YAML/JSON CONFIG LOADING
# =============================================================================

def load_rails_from_dict(config_dict: Dict[str, Any]) -> Guardrails:
    """Load guardrails from a dictionary configuration."""
    rails = Guardrails()

    for rail_config in config_dict.get("rails", []):
        rail = Rail(
            name=rail_config.get("name", "unnamed"),
            pattern=rail_config.get("pattern"),
            keywords=rail_config.get("keywords"),
            action=RailAction(rail_config.get("action", "block")),
            message=rail_config.get("message"),
            redirect_response=rail_config.get("redirect_response"),
            rail_type=RailType(rail_config.get("type", "both")),
            priority=rail_config.get("priority", 100),
            enabled=rail_config.get("enabled", True)
        )
        rails.add_rail(rail)

    # Load topic rails
    for topic_config in config_dict.get("topics", []):
        if isinstance(topic_config, str):
            rails.add_topic_rail(topic_config)
        else:
            rails.add_topic_rail(
                topic_config.get("name"),
                RailAction(topic_config.get("action", "block")),
                topic_config.get("message")
            )

    return rails


def load_rails_from_file(path: Union[str, Path]) -> Guardrails:
    """Load guardrails from a JSON file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    return load_rails_from_dict(config_dict)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Default guardrails instance
_default_guardrails: Optional[Guardrails] = None


def get_guardrails() -> Guardrails:
    """Get the default guardrails instance."""
    global _default_guardrails
    if _default_guardrails is None:
        _default_guardrails = create_safety_rails()
    return _default_guardrails


def set_guardrails(guardrails: Guardrails):
    """Set the default guardrails instance."""
    global _default_guardrails
    _default_guardrails = guardrails


def check_input(text: str) -> RailResult:
    """Check input using default guardrails."""
    return get_guardrails().check_input(text)


def check_output(text: str) -> RailResult:
    """Check output using default guardrails."""
    return get_guardrails().check_output(text)


def is_safe(text: str) -> bool:
    """Quick check if text passes all rails."""
    result = check_input(text)
    return not result.blocked


# =============================================================================
# EXPORTS
# =============================================================================

# Compatibility flag and alias for V35 validation
RAILS_COMPAT_AVAILABLE = True

# Alias for V35 validation compatibility
RailsCompat = Guardrails

__all__ = [
    # Enums
    "RailAction",
    "RailType",
    "CheckPhase",
    # Data classes
    "Rail",
    "RailResult",
    "RailConfig",
    # Flow
    "Flow",
    "FlowEngine",
    # Main engine
    "Guardrails",
    # V35 compat alias
    "RailsCompat",
    # Topic helpers
    "TOPIC_PATTERNS",
    "create_topic_rail",
    # Presets
    "create_safety_rails",
    "create_content_moderation_rails",
    "create_professional_rails",
    # Config loading
    "load_rails_from_dict",
    "load_rails_from_file",
    # Convenience
    "check_input",
    "check_output",
    "is_safe",
    "get_guardrails",
    "set_guardrails",
    # Compat flag
    "RAILS_COMPAT_AVAILABLE",
]
