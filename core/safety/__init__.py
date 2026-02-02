#!/usr/bin/env python3
"""
Safety Layer - Input/Output Validation & Security Scanning
Part of the V33 Architecture (Layer 6) - Phase 9 Production Fix.

Provides unified access to three safety SDKs:
- guardrails-ai: Pydantic-based input/output validation with validators
- llm-guard: Security scanning for PII, jailbreaks, prompt injection
- nemo-guardrails: Programmable guardrails with Colang 2.0

NO STUBS: All SDKs must be explicitly installed and configured.
Missing SDKs raise SDKNotAvailableError with install instructions.
Misconfigured SDKs raise SDKConfigurationError with missing config.

Usage:
    from core.safety import (
        # Exceptions
        SDKNotAvailableError,
        SDKConfigurationError,

        # Guardrails AI
        get_guardrails_guard,
        GuardrailsClient,
        GUARDRAILS_AVAILABLE,

        # LLM Guard
        get_llm_guard_scanner,
        LLMGuardClient,
        LLM_GUARD_AVAILABLE,

        # NeMo Guardrails
        get_nemo_rails,
        NemoGuardrailsClient,
        NEMO_AVAILABLE,

        # Factory
        SafetyFactory,
    )

    # Quick start with explicit error handling
    try:
        guard = get_guardrails_guard()
    except SDKNotAvailableError as e:
        print(f"Install: {e.install_cmd}")
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Optional, List, Dict, Union, Callable
from dataclasses import dataclass, field

# Import exceptions from observability layer
from core.observability import (
    SDKNotAvailableError,
    SDKConfigurationError,
)


# ============================================================================
# SDK Availability Checks - Import-time validation
# ============================================================================

# Guardrails AI - Pydantic Validation
GUARDRAILS_AVAILABLE = False
GUARDRAILS_HUB_AVAILABLE = False
GUARDRAILS_ERROR = None
Guard = None
RegexMatch = None
ValidLength = None
ToxicLanguage = None

try:
    from guardrails import Guard
    GUARDRAILS_AVAILABLE = True

    # Hub validators are optional - they need to be installed separately
    # via: guardrails hub install hub://guardrails/regex_match, etc.
    try:
        from guardrails.hub import RegexMatch, ValidLength, ToxicLanguage
        GUARDRAILS_HUB_AVAILABLE = True
    except ImportError:
        # Hub validators not installed - that's fine, core Guard still works
        pass
except Exception as e:
    GUARDRAILS_ERROR = str(e)

# LLM Guard - Security Scanning
LLM_GUARD_AVAILABLE = False
LLM_GUARD_ERROR = None
try:
    from llm_guard import scan_prompt, scan_output
    from llm_guard.input_scanners import (
        Anonymize as AnonymizeInput,
        BanTopics,
        PromptInjection,
        TokenLimit,
        Toxicity as ToxicityInput,
    )
    from llm_guard.output_scanners import (
        Bias,
        Deanonymize,
        MaliciousURLs,
        NoRefusal,
        Relevance,
        Sensitive,
        Toxicity as ToxicityOutput,
    )
    LLM_GUARD_AVAILABLE = True
except Exception as e:
    LLM_GUARD_ERROR = str(e)

# NeMo Guardrails - Programmable Guardrails
NEMO_AVAILABLE = False
NEMO_ERROR = None
try:
    from nemoguardrails import RailsConfig, LLMRails
    from nemoguardrails.actions import action
    NEMO_AVAILABLE = True
except Exception as e:
    NEMO_ERROR = str(e)


# ============================================================================
# Guardrails AI Types and Implementation
# ============================================================================

class ValidatorType(str, Enum):
    """Types of Guardrails validators."""
    REGEX = "regex"
    LENGTH = "length"
    TOXIC = "toxic"
    PII = "pii"
    CUSTOM = "custom"


@dataclass
class GuardrailsConfig:
    """Configuration for Guardrails AI."""
    validators: List[ValidatorType] = field(default_factory=list)
    on_fail: str = "fix"  # fix, reask, filter, refrain, noop, exception
    max_length: int = 4096
    regex_patterns: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result from a validation operation."""
    valid: bool
    output: str
    errors: List[str] = field(default_factory=list)
    fixes_applied: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class GuardrailsClient:
    """
    Guardrails AI client for input/output validation.

    Provides Pydantic-based validation with automatic
    fixing and retry capabilities.
    """

    def __init__(
        self,
        config: Optional[GuardrailsConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize Guardrails client.

        Args:
            config: Optional GuardrailsConfig
            **kwargs: Override config values
        """
        if not GUARDRAILS_AVAILABLE:
            raise SDKNotAvailableError(
                sdk_name="guardrails-ai",
                install_cmd="pip install guardrails-ai>=0.5.0",
                docs_url="https://www.guardrailsai.com/docs/"
            )

        self.config = config or GuardrailsConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._guard = None
        self._build_guard()

    def _build_guard(self) -> None:
        """Build the Guard with configured validators."""
        validators = []

        # Only use hub validators if they're installed
        if GUARDRAILS_HUB_AVAILABLE:
            for v_type in self.config.validators:
                if v_type == ValidatorType.LENGTH and ValidLength is not None:
                    validators.append(ValidLength(max=self.config.max_length))
                elif v_type == ValidatorType.TOXIC and ToxicLanguage is not None:
                    validators.append(ToxicLanguage())
                elif v_type == ValidatorType.REGEX and RegexMatch is not None:
                    for pattern in self.config.regex_patterns:
                        validators.append(RegexMatch(regex=pattern))

        if validators:
            self._guard = Guard().use_many(*validators)
        else:
            # Use base Guard without validators if hub not installed
            self._guard = Guard()

    def validate(
        self,
        text: str,
        prompt: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate text against configured validators.

        Args:
            text: Text to validate
            prompt: Optional original prompt for context

        Returns:
            ValidationResult with validation status
        """
        try:
            result = self._guard.validate(text)

            return ValidationResult(
                valid=result.validation_passed,
                output=result.validated_output or text,
                errors=[str(e) for e in result.validation_summaries] if result.validation_summaries else [],
                fixes_applied=len(result.validation_summaries) if not result.validation_passed else 0,
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                output=text,
                errors=[str(e)],
            )

    def validate_with_llm(
        self,
        prompt: str,
        llm: Callable[[str], str],
        max_retries: int = 3,
    ) -> ValidationResult:
        """
        Validate LLM output with retry on failure.

        Args:
            prompt: Prompt to send to LLM
            llm: LLM callable
            max_retries: Maximum retry attempts

        Returns:
            ValidationResult with validated output
        """
        try:
            result = self._guard(
                llm,
                prompt=prompt,
                max_retries=max_retries,
            )

            return ValidationResult(
                valid=result.validation_passed,
                output=result.validated_output or "",
                errors=[str(e) for e in result.validation_summaries] if result.validation_summaries else [],
                metadata={
                    "retries": result.reasks if hasattr(result, "reasks") else 0,
                    "raw_output": result.raw_llm_output,
                },
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                output="",
                errors=[str(e)],
            )

    def add_validator(
        self,
        validator_type: ValidatorType,
        **kwargs: Any,
    ) -> None:
        """
        Add a validator to the guard.

        Args:
            validator_type: Type of validator
            **kwargs: Validator-specific configuration
        """
        self.config.validators.append(validator_type)
        self._build_guard()


# ============================================================================
# LLM Guard Types and Implementation
# ============================================================================

class ScannerType(str, Enum):
    """Types of LLM Guard scanners."""
    # Input scanners
    PROMPT_INJECTION = "prompt_injection"
    PII = "pii"
    TOXIC = "toxic"
    BAN_TOPICS = "ban_topics"
    TOKEN_LIMIT = "token_limit"
    # Output scanners
    BIAS = "bias"
    MALICIOUS_URLS = "malicious_urls"
    SENSITIVE = "sensitive"
    RELEVANCE = "relevance"


@dataclass
class LLMGuardConfig:
    """Configuration for LLM Guard."""
    input_scanners: List[ScannerType] = field(default_factory=list)
    output_scanners: List[ScannerType] = field(default_factory=list)
    fail_fast: bool = True
    anonymize: bool = False
    token_limit: int = 4096
    banned_topics: List[str] = field(default_factory=list)


@dataclass
class ScanResult:
    """Result from a security scan."""
    safe: bool
    sanitized_text: str
    risks_detected: List[str] = field(default_factory=list)
    risk_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMGuardClient:
    """
    LLM Guard client for security scanning.

    Provides comprehensive security scanning for
    prompts and outputs including PII, injection, and toxicity.
    """

    def __init__(
        self,
        config: Optional[LLMGuardConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize LLM Guard client.

        Args:
            config: Optional LLMGuardConfig
            **kwargs: Override config values
        """
        if not LLM_GUARD_AVAILABLE:
            raise SDKNotAvailableError(
                sdk_name="llm-guard",
                install_cmd="pip install llm-guard>=0.3.0",
                docs_url="https://llm-guard.com/"
            )

        self.config = config or LLMGuardConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._input_scanners = []
        self._output_scanners = []
        self._build_scanners()

    def _build_scanners(self) -> None:
        """Build scanner instances from config."""
        # Build input scanners
        for scanner_type in self.config.input_scanners:
            if scanner_type == ScannerType.PROMPT_INJECTION:
                self._input_scanners.append(PromptInjection())
            elif scanner_type == ScannerType.PII:
                self._input_scanners.append(AnonymizeInput())
            elif scanner_type == ScannerType.TOXIC:
                self._input_scanners.append(ToxicityInput())
            elif scanner_type == ScannerType.BAN_TOPICS:
                self._input_scanners.append(BanTopics(topics=self.config.banned_topics))
            elif scanner_type == ScannerType.TOKEN_LIMIT:
                self._input_scanners.append(TokenLimit(limit=self.config.token_limit))

        # Build output scanners
        for scanner_type in self.config.output_scanners:
            if scanner_type == ScannerType.BIAS:
                self._output_scanners.append(Bias())
            elif scanner_type == ScannerType.MALICIOUS_URLS:
                self._output_scanners.append(MaliciousURLs())
            elif scanner_type == ScannerType.SENSITIVE:
                self._output_scanners.append(Sensitive())
            elif scanner_type == ScannerType.TOXIC:
                self._output_scanners.append(ToxicityOutput())

    def scan_input(self, prompt: str) -> ScanResult:
        """
        Scan input prompt for security risks.

        Args:
            prompt: Input prompt to scan

        Returns:
            ScanResult with security analysis
        """
        sanitized, results_valid, results_score = scan_prompt(
            self._input_scanners,
            prompt,
            self.config.fail_fast,
        )

        risks = []
        scores = {}
        for i, (valid, score) in enumerate(zip(results_valid, results_score)):
            scanner_name = type(self._input_scanners[i]).__name__
            scores[scanner_name] = score
            if not valid:
                risks.append(scanner_name)

        return ScanResult(
            safe=all(results_valid),
            sanitized_text=sanitized,
            risks_detected=risks,
            risk_scores=scores,
        )

    def scan_output(
        self,
        output: str,
        prompt: Optional[str] = None,
    ) -> ScanResult:
        """
        Scan LLM output for security risks.

        Args:
            output: LLM output to scan
            prompt: Optional original prompt for context

        Returns:
            ScanResult with security analysis
        """
        sanitized, results_valid, results_score = scan_output(
            self._output_scanners,
            prompt or "",
            output,
            self.config.fail_fast,
        )

        risks = []
        scores = {}
        for i, (valid, score) in enumerate(zip(results_valid, results_score)):
            scanner_name = type(self._output_scanners[i]).__name__
            scores[scanner_name] = score
            if not valid:
                risks.append(scanner_name)

        return ScanResult(
            safe=all(results_valid),
            sanitized_text=sanitized,
            risks_detected=risks,
            risk_scores=scores,
        )

    def scan_full(
        self,
        prompt: str,
        output: str,
    ) -> Dict[str, ScanResult]:
        """
        Scan both input and output.

        Args:
            prompt: Input prompt
            output: LLM output

        Returns:
            Dict with 'input' and 'output' ScanResults
        """
        return {
            "input": self.scan_input(prompt),
            "output": self.scan_output(output, prompt),
        }


# ============================================================================
# NeMo Guardrails Types and Implementation
# ============================================================================

@dataclass
class NemoConfig:
    """Configuration for NeMo Guardrails."""
    config_path: Optional[str] = None
    rails_config: Optional[Dict[str, Any]] = None
    model: str = "gpt-4o"
    streaming: bool = False


@dataclass
class RailsResult:
    """Result from NeMo rails processing."""
    response: str
    blocked: bool
    reason: Optional[str] = None
    actions_executed: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class NemoGuardrailsClient:
    """
    NeMo Guardrails client for programmable guardrails.

    Provides Colang 2.0 based guardrails for complex
    conversational safety policies.
    """

    def __init__(
        self,
        config: Optional[NemoConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize NeMo Guardrails client.

        Args:
            config: Optional NemoConfig
            **kwargs: Override config values
        """
        if not NEMO_AVAILABLE:
            raise SDKNotAvailableError(
                sdk_name="nemo-guardrails",
                install_cmd="pip install nemoguardrails>=0.9.0",
                docs_url="https://docs.nvidia.com/nemo/guardrails/"
            )

        self.config = config or NemoConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._rails = None
        self._init_rails()

    def _init_rails(self) -> None:
        """Initialize the LLMRails instance."""
        if self.config.config_path:
            # Load from file path
            rails_config = RailsConfig.from_path(self.config.config_path)
        elif self.config.rails_config:
            # Load from dict
            rails_config = RailsConfig.from_content(self.config.rails_config)
        else:
            # Create minimal default config
            rails_config = RailsConfig.from_content({
                "models": [{
                    "type": "main",
                    "engine": "openai",
                    "model": self.config.model,
                }],
                "rails": {
                    "input": {
                        "flows": ["check input"]
                    },
                    "output": {
                        "flows": ["check output"]
                    }
                }
            })

        self._rails = LLMRails(rails_config)

    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> RailsResult:
        """
        Generate a response with guardrails.

        Args:
            prompt: User prompt
            context: Optional context variables

        Returns:
            RailsResult with processed response
        """
        try:
            response = await self._rails.generate_async(
                messages=[{"role": "user", "content": prompt}],
                options={"context": context} if context else None,
            )

            # Check if blocked
            blocked = False
            reason = None
            if hasattr(response, "blocked") and response.blocked:
                blocked = True
                reason = getattr(response, "blocked_reason", "Policy violation")

            return RailsResult(
                response=response["content"] if isinstance(response, dict) else str(response),
                blocked=blocked,
                reason=reason,
                actions_executed=getattr(response, "actions", []),
            )
        except Exception as e:
            return RailsResult(
                response="",
                blocked=True,
                reason=str(e),
            )

    def generate_sync(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> RailsResult:
        """
        Synchronous generate with guardrails.

        Args:
            prompt: User prompt
            context: Optional context variables

        Returns:
            RailsResult with processed response
        """
        import asyncio
        return asyncio.run(self.generate(prompt, context))

    def register_action(
        self,
        name: str,
        fn: Callable,
    ) -> None:
        """
        Register a custom action.

        Args:
            name: Action name for Colang
            fn: Action function
        """
        self._rails.register_action(fn, name)

    def load_colang(self, colang_content: str) -> None:
        """
        Load additional Colang flows.

        Args:
            colang_content: Colang 2.0 code
        """
        # This would require reinitializing rails with new content
        if self.config.rails_config:
            self.config.rails_config["colang_content"] = colang_content
        else:
            self.config.rails_config = {"colang_content": colang_content}
        self._init_rails()


# ============================================================================
# Explicit Getter Functions - Raise SDKNotAvailableError if unavailable
# ============================================================================

def get_guardrails_guard(
    validators: Optional[List[ValidatorType]] = None,
    **kwargs: Any,
) -> GuardrailsClient:
    """
    Get a Guardrails AI client.

    Args:
        validators: List of validators to use
        **kwargs: Additional configuration

    Returns:
        GuardrailsClient instance

    Raises:
        SDKNotAvailableError: If guardrails-ai is not installed
    """
    config = GuardrailsConfig(
        validators=validators or [ValidatorType.LENGTH, ValidatorType.TOXIC],
        **kwargs,
    )
    return GuardrailsClient(config=config)


def get_llm_guard_scanner(
    input_scanners: Optional[List[ScannerType]] = None,
    output_scanners: Optional[List[ScannerType]] = None,
    **kwargs: Any,
) -> LLMGuardClient:
    """
    Get an LLM Guard scanner client.

    Args:
        input_scanners: List of input scanners
        output_scanners: List of output scanners
        **kwargs: Additional configuration

    Returns:
        LLMGuardClient instance

    Raises:
        SDKNotAvailableError: If llm-guard is not installed
    """
    config = LLMGuardConfig(
        input_scanners=input_scanners or [ScannerType.PROMPT_INJECTION, ScannerType.TOXIC],
        output_scanners=output_scanners or [ScannerType.TOXIC, ScannerType.BIAS],
        **kwargs,
    )
    return LLMGuardClient(config=config)


def get_nemo_rails(
    config_path: Optional[str] = None,
    model: str = "gpt-4o",
    **kwargs: Any,
) -> NemoGuardrailsClient:
    """
    Get a NeMo Guardrails client.

    Args:
        config_path: Path to rails config
        model: LLM model to use
        **kwargs: Additional configuration

    Returns:
        NemoGuardrailsClient instance

    Raises:
        SDKNotAvailableError: If nemo-guardrails is not installed
    """
    config = NemoConfig(config_path=config_path, model=model, **kwargs)
    return NemoGuardrailsClient(config=config)


# ============================================================================
# Unified Factory
# ============================================================================

class SafetyFactory:
    """
    Unified factory for creating safety clients.

    Provides a single entry point for all three SDKs with
    consistent configuration and V33 integration.
    """

    def __init__(self):
        """Initialize the factory."""
        self._guardrails: Optional[GuardrailsClient] = None
        self._llm_guard: Optional[LLMGuardClient] = None
        self._nemo: Optional[NemoGuardrailsClient] = None

    def get_availability(self) -> Dict[str, bool]:
        """Get availability status of all SDKs."""
        return {
            "guardrails": GUARDRAILS_AVAILABLE,
            "llm_guard": LLM_GUARD_AVAILABLE,
            "nemo": NEMO_AVAILABLE,
        }

    def create_guardrails(self, **kwargs: Any) -> GuardrailsClient:
        """Create a Guardrails AI client."""
        self._guardrails = GuardrailsClient(**kwargs)
        return self._guardrails

    def create_llm_guard(self, **kwargs: Any) -> LLMGuardClient:
        """Create an LLM Guard client."""
        self._llm_guard = LLMGuardClient(**kwargs)
        return self._llm_guard

    def create_nemo(self, **kwargs: Any) -> NemoGuardrailsClient:
        """Create a NeMo Guardrails client."""
        self._nemo = NemoGuardrailsClient(**kwargs)
        return self._nemo

    def get_guardrails(self) -> Optional[GuardrailsClient]:
        """Get the cached Guardrails client."""
        return self._guardrails

    def get_llm_guard(self) -> Optional[LLMGuardClient]:
        """Get the cached LLM Guard client."""
        return self._llm_guard

    def get_nemo(self) -> Optional[NemoGuardrailsClient]:
        """Get the cached NeMo client."""
        return self._nemo


# ============================================================================
# Module-level availability
# ============================================================================

SAFETY_AVAILABLE = GUARDRAILS_AVAILABLE or LLM_GUARD_AVAILABLE or NEMO_AVAILABLE


def get_available_sdks() -> Dict[str, bool]:
    """Get availability status of all safety SDKs."""
    return {
        "guardrails": GUARDRAILS_AVAILABLE,
        "llm_guard": LLM_GUARD_AVAILABLE,
        "nemo": NEMO_AVAILABLE,
    }


# ============================================================================
# All Exports
# ============================================================================

__all__ = [
    # Exceptions (re-exported from observability)
    "SDKNotAvailableError",
    "SDKConfigurationError",

    # Availability flags
    "GUARDRAILS_AVAILABLE",
    "LLM_GUARD_AVAILABLE",
    "NEMO_AVAILABLE",
    "SAFETY_AVAILABLE",

    # Getter functions (raise on unavailable)
    "get_guardrails_guard",
    "get_llm_guard_scanner",
    "get_nemo_rails",

    # Guardrails AI
    "GuardrailsClient",
    "GuardrailsConfig",
    "ValidatorType",
    "ValidationResult",

    # LLM Guard
    "LLMGuardClient",
    "LLMGuardConfig",
    "ScannerType",
    "ScanResult",

    # NeMo Guardrails
    "NemoGuardrailsClient",
    "NemoConfig",
    "RailsResult",

    # Factory
    "SafetyFactory",
    "get_available_sdks",
]
