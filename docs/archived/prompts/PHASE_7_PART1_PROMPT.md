# Phase 7 Part 1: Safety Layer (Guardrails AI)

## Overview
**Layer 6 - Safety Layer**
Part 1 covers: Guardrails AI integration for input/output validation
Part 2 will cover: Additional safety SDKs (llm-guard, nemo-guardrails, rebuff)

The Safety Layer provides comprehensive input validation, output verification, content safety checks, and guardrails enforcement for all LLM interactions in the Unleash platform.

---

## Prerequisites

Phase 6 (Observability Layer) must be complete before starting Phase 7.

### Pre-Flight Validation

```bash
# Verify Phase 6 is complete
cd /path/to/unleash

# Check Part 1 (tracing & monitoring)
python -c "from core.observability.langfuse_tracer import LangfuseTracer; print('Part 1 OK')"

# Check Part 2 (evaluation & testing)
python -c "from core.observability.deepeval_tests import DeepEvalRunner; print('Part 2 OK')"

# Full validation
python -c "from core.observability import ObservabilityFactory; print('Phase 6 OK')"
```

**Required Checks:**
- [ ] Observability Layer operational (`core/observability/`)
- [ ] LangfuseTracer functional (Phase 6 Part 1)
- [ ] DeepEvalRunner functional (Phase 6 Part 2)
- [ ] All Phase 6 tests passing (40/43 or better)

---

## Phase 7 Part 1 Objectives

Implement Guardrails AI integration with:
1. **SafetyFactory** - Unified interface for all safety providers
2. **GuardrailsValidator** - Core validation with Guardrails AI
3. **InputValidator** - Pre-flight input validation (length, format, language)
4. **OutputValidator** - Post-generation validation (compliance, schema, safety)
5. **Custom validators** - Extensible validation framework
6. **LLM Gateway integration** - Seamless validation in request flow
7. **Observability integration** - Safety events logged via Phase 6

---

## Step 1: Install Dependencies

```bash
# Activate virtual environment first
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate

# Install Guardrails AI
pip install guardrails-ai

# Install optional validators (recommended)
guardrails hub install hub://guardrails/regex_match
guardrails hub install hub://guardrails/valid_length
guardrails hub install hub://guardrails/toxic_language
guardrails hub install hub://guardrails/detect_pii

# Verify installation
python -c "import guardrails; print(f'Guardrails AI {guardrails.__version__} installed')"
```

---

## Step 2: Create Safety Directory Structure

```bash
mkdir -p core/safety
```

---

## Step 3: Create Safety Factory Interface

Create `core/safety/__init__.py`:

```python
"""
Safety Layer - Phase 7 Part 1
Provides comprehensive input/output validation and guardrails for LLM operations.

Part 1: Guardrails AI integration
Part 2: Additional safety SDKs (llm-guard, nemo-guardrails, rebuff)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ============================================
# SDK Availability Checks
# ============================================

def check_guardrails_available() -> bool:
    """Check if Guardrails AI is available."""
    try:
        import guardrails
        return True
    except ImportError:
        return False


def check_llm_guard_available() -> bool:
    """Check if LLM Guard is available (Part 2)."""
    try:
        import llm_guard
        return True
    except ImportError:
        return False


def check_nemo_guardrails_available() -> bool:
    """Check if NeMo Guardrails is available (Part 2)."""
    try:
        import nemoguardrails
        return True
    except ImportError:
        return False


def check_rebuff_available() -> bool:
    """Check if Rebuff is available (Part 2)."""
    try:
        import rebuff
        return True
    except ImportError:
        return False


# SDK availability flags
GUARDRAILS_AVAILABLE = check_guardrails_available()
LLM_GUARD_AVAILABLE = check_llm_guard_available()
NEMO_GUARDRAILS_AVAILABLE = check_nemo_guardrails_available()
REBUFF_AVAILABLE = check_rebuff_available()


# ============================================
# Enums and Base Models
# ============================================

class SafetyProvider(str, Enum):
    """Available safety providers."""
    GUARDRAILS = "guardrails"
    LLM_GUARD = "llm_guard"
    NEMO_GUARDRAILS = "nemo_guardrails"
    REBUFF = "rebuff"


class ValidationTarget(str, Enum):
    """Target of validation."""
    INPUT = "input"
    OUTPUT = "output"
    BOTH = "both"


class ValidationSeverity(str, Enum):
    """Severity of validation failures."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(str, Enum):
    """Status of validation result."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


class ValidationIssue(BaseModel):
    """A single validation issue."""
    issue_id: str = Field(default_factory=lambda: str(uuid4()))
    validator_name: str
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    fix_suggestion: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationResult(BaseModel):
    """Result of a validation operation."""
    result_id: str = Field(default_factory=lambda: str(uuid4()))
    status: ValidationStatus
    target: ValidationTarget
    provider: SafetyProvider
    original_content: str
    validated_content: Optional[str] = None
    issues: List[ValidationIssue] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def passed(self) -> bool:
        """Check if validation passed."""
        return self.status == ValidationStatus.PASSED
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)
    
    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return any(i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for i in self.issues)


class GuardExecutionResult(BaseModel):
    """Result from executing a guard."""
    guard_name: str
    passed: bool
    raw_output: Optional[str] = None
    validated_output: Optional[str] = None
    error: Optional[str] = None
    reasks: int = 0
    latency_ms: float = 0.0


# ============================================
# Safety Configuration
# ============================================

@dataclass
class SafetyConfig:
    """Configuration for safety layer."""
    # Provider selection
    default_provider: SafetyProvider = SafetyProvider.GUARDRAILS
    fallback_enabled: bool = True
    
    # Validation settings
    validate_inputs: bool = True
    validate_outputs: bool = True
    fail_on_error: bool = True
    max_retries: int = 2
    
    # Content limits
    max_input_length: int = 100000
    max_output_length: int = 50000
    
    # Logging
    log_validations: bool = True
    log_to_observability: bool = True
    
    # Performance
    timeout_seconds: int = 30
    async_mode: bool = True


# ============================================
# Safety Factory
# ============================================

class SafetyFactory:
    """
    Factory for creating safety validators.
    
    Provides unified interface for all safety providers.
    
    Usage:
        factory = SafetyFactory()
        validator = factory.create_validator(SafetyProvider.GUARDRAILS)
        result = validator.validate_input("user message")
    """
    
    def __init__(self, config: Optional[SafetyConfig] = None):
        """Initialize the safety factory."""
        self.config = config or SafetyConfig()
        self._validators: Dict[SafetyProvider, Any] = {}
        self._validation_count = 0
        self._pass_count = 0
        
        logger.info(
            "safety_factory_initialized",
            default_provider=self.config.default_provider.value,
        )
    
    def get_available_providers(self) -> Dict[str, bool]:
        """Get availability of all safety providers."""
        return {
            "guardrails": GUARDRAILS_AVAILABLE,
            "llm_guard": LLM_GUARD_AVAILABLE,
            "nemo_guardrails": NEMO_GUARDRAILS_AVAILABLE,
            "rebuff": REBUFF_AVAILABLE,
        }
    
    def create_validator(
        self,
        provider: Optional[SafetyProvider] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create a validator for the specified provider.
        
        Args:
            provider: Safety provider to use
            **kwargs: Provider-specific configuration
            
        Returns:
            Validator instance
        """
        provider = provider or self.config.default_provider
        
        if provider == SafetyProvider.GUARDRAILS:
            if not GUARDRAILS_AVAILABLE:
                raise ImportError("guardrails-ai not installed - pip install guardrails-ai")
            from core.safety.guardrails_validators import GuardrailsValidator
            return GuardrailsValidator(**kwargs)
        
        elif provider == SafetyProvider.LLM_GUARD:
            if not LLM_GUARD_AVAILABLE:
                raise ImportError("llm-guard not installed (Phase 7 Part 2)")
            raise NotImplementedError("LLM Guard integration in Phase 7 Part 2")
        
        elif provider == SafetyProvider.NEMO_GUARDRAILS:
            if not NEMO_GUARDRAILS_AVAILABLE:
                raise ImportError("nemoguardrails not installed (Phase 7 Part 2)")
            raise NotImplementedError("NeMo Guardrails integration in Phase 7 Part 2")
        
        elif provider == SafetyProvider.REBUFF:
            if not REBUFF_AVAILABLE:
                raise ImportError("rebuff not installed (Phase 7 Part 2)")
            raise NotImplementedError("Rebuff integration in Phase 7 Part 2")
        
        else:
            raise ValueError(f"Unknown safety provider: {provider}")
    
    def validate_input(
        self,
        content: str,
        provider: Optional[SafetyProvider] = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """
        Validate input content.
        
        Args:
            content: Content to validate
            provider: Provider to use
            **kwargs: Additional validation options
            
        Returns:
            ValidationResult
        """
        provider = provider or self.config.default_provider
        validator = self.create_validator(provider)
        
        self._validation_count += 1
        result = validator.validate_input(content, **kwargs)
        
        if result.passed:
            self._pass_count += 1
        
        # Log to observability if enabled
        if self.config.log_to_observability:
            self._log_to_observability(result)
        
        return result
    
    def validate_output(
        self,
        content: str,
        provider: Optional[SafetyProvider] = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """
        Validate output content.
        
        Args:
            content: Content to validate
            provider: Provider to use
            **kwargs: Additional validation options
            
        Returns:
            ValidationResult
        """
        provider = provider or self.config.default_provider
        validator = self.create_validator(provider)
        
        self._validation_count += 1
        result = validator.validate_output(content, **kwargs)
        
        if result.passed:
            self._pass_count += 1
        
        # Log to observability if enabled
        if self.config.log_to_observability:
            self._log_to_observability(result)
        
        return result
    
    async def validate_input_async(
        self,
        content: str,
        provider: Optional[SafetyProvider] = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Async version of validate_input."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.validate_input(content, provider, **kwargs),
        )
    
    async def validate_output_async(
        self,
        content: str,
        provider: Optional[SafetyProvider] = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Async version of validate_output."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.validate_output(content, provider, **kwargs),
        )
    
    def execute_guard(
        self,
        guard_name: str,
        content: str,
        provider: Optional[SafetyProvider] = None,
        **kwargs: Any,
    ) -> GuardExecutionResult:
        """
        Execute a specific guard.
        
        Args:
            guard_name: Name of guard to execute
            content: Content to process
            provider: Provider to use
            **kwargs: Guard-specific options
            
        Returns:
            GuardExecutionResult
        """
        provider = provider or self.config.default_provider
        validator = self.create_validator(provider)
        
        return validator.execute_guard(guard_name, content, **kwargs)
    
    async def execute_guard_async(
        self,
        guard_name: str,
        content: str,
        **kwargs: Any,
    ) -> GuardExecutionResult:
        """Async version of execute_guard."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.execute_guard(guard_name, content, **kwargs),
        )
    
    def _log_to_observability(self, result: ValidationResult) -> None:
        """Log validation result to observability layer."""
        try:
            from core.observability import LangfuseTracer, LangfuseConfig
            
            tracer = LangfuseTracer(LangfuseConfig(enabled=True))
            
            with tracer.trace_span(
                name=f"safety_validation_{result.target.value}",
                tags=["safety", result.provider.value],
            ) as span:
                span.metadata.update({
                    "status": result.status.value,
                    "issues_count": len(result.issues),
                    "latency_ms": result.latency_ms,
                })
            
        except ImportError:
            logger.debug("observability_layer_not_available")
        except Exception as e:
            logger.warning("observability_logging_failed", error=str(e))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get factory statistics."""
        return {
            "validations_run": self._validation_count,
            "validations_passed": self._pass_count,
            "pass_rate": self._pass_count / self._validation_count if self._validation_count > 0 else 0.0,
            "available_providers": self.get_available_providers(),
            "default_provider": self.config.default_provider.value,
        }


# ============================================
# Convenience Functions
# ============================================

def validate_input(content: str, **kwargs: Any) -> ValidationResult:
    """Quick input validation with default settings."""
    factory = SafetyFactory()
    return factory.validate_input(content, **kwargs)


def validate_output(content: str, **kwargs: Any) -> ValidationResult:
    """Quick output validation with default settings."""
    factory = SafetyFactory()
    return factory.validate_output(content, **kwargs)


async def validate_async(
    content: str,
    target: ValidationTarget = ValidationTarget.INPUT,
    **kwargs: Any,
) -> ValidationResult:
    """Quick async validation."""
    factory = SafetyFactory()
    if target == ValidationTarget.INPUT:
        return await factory.validate_input_async(content, **kwargs)
    return await factory.validate_output_async(content, **kwargs)


# ============================================
# Imports from submodules
# ============================================

try:
    from core.safety.guardrails_validators import (
        GuardrailsConfig,
        GuardrailsValidator,
        InputValidator,
        OutputValidator,
        create_guardrails_validator,
    )
except ImportError:
    GuardrailsConfig = None
    GuardrailsValidator = None
    InputValidator = None
    OutputValidator = None
    create_guardrails_validator = None


# ============================================
# Module Exports
# ============================================

__all__ = [
    # Enums
    "SafetyProvider",
    "ValidationTarget",
    "ValidationSeverity",
    "ValidationStatus",
    
    # Models
    "ValidationIssue",
    "ValidationResult",
    "GuardExecutionResult",
    "SafetyConfig",
    
    # Factory
    "SafetyFactory",
    
    # Availability
    "GUARDRAILS_AVAILABLE",
    "LLM_GUARD_AVAILABLE",
    "NEMO_GUARDRAILS_AVAILABLE",
    "REBUFF_AVAILABLE",
    
    # Convenience functions
    "validate_input",
    "validate_output",
    "validate_async",
    
    # Guardrails (Part 1)
    "GuardrailsConfig",
    "GuardrailsValidator",
    "InputValidator",
    "OutputValidator",
    "create_guardrails_validator",
]


if __name__ == "__main__":
    print("Safety Layer - Phase 7 Part 1")
    print("-" * 40)
    
    factory = SafetyFactory()
    status = factory.get_available_providers()
    
    print("\nProvider Availability:")
    for provider, available in status.items():
        status_str = "✓" if available else "✗"
        print(f"  {status_str} {provider}")
    
    print("\nUsage:")
    print("  from core.safety import SafetyFactory")
    print("  factory = SafetyFactory()")
    print("  result = factory.validate_input('Hello, world!')")
```

---

## Step 4: Create Guardrails Validators

Create `core/safety/guardrails_validators.py`:

```python
"""
Guardrails AI Validators - Phase 7 Part 1
Comprehensive input/output validation using Guardrails AI.

Features:
- Input validation (length, format, language detection)
- Output validation (compliance, JSON schema, content safety)
- Custom validator creation
- Async guard execution
- LLM Gateway integration
- Observability logging
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field, validator

logger = structlog.get_logger(__name__)


# ============================================
# Guardrails AI Imports
# ============================================

try:
    import guardrails as gd
    from guardrails import Guard, Validator
    from guardrails.validators import (
        FailResult,
        PassResult,
        ValidationResult as GRValidationResult,
    )
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    gd = None
    Guard = None
    Validator = None
    FailResult = None
    PassResult = None
    GRValidationResult = None
    logger.warning("guardrails-ai not available - install with: pip install guardrails-ai")


# ============================================
# Enums and Types
# ============================================

class ValidatorType(str, Enum):
    """Types of validators."""
    LENGTH = "length"
    REGEX = "regex"
    FORMAT = "format"
    LANGUAGE = "language"
    PII = "pii"
    TOXICITY = "toxicity"
    JSON_SCHEMA = "json_schema"
    CUSTOM = "custom"


class InputValidationType(str, Enum):
    """Input validation types."""
    LENGTH_CHECK = "length_check"
    FORMAT_CHECK = "format_check"
    LANGUAGE_DETECTION = "language_detection"
    INJECTION_DETECTION = "injection_detection"
    PII_DETECTION = "pii_detection"


class OutputValidationType(str, Enum):
    """Output validation types."""
    COMPLIANCE_CHECK = "compliance_check"
    JSON_SCHEMA_CHECK = "json_schema_check"
    CONTENT_SAFETY = "content_safety"
    FACTUALITY_CHECK = "factuality_check"
    HALLUCINATION_CHECK = "hallucination_check"


# ============================================
# Configuration
# ============================================

@dataclass
class GuardrailsConfig:
    """Configuration for Guardrails validator."""
    # General settings
    enabled: bool = True
    fail_fast: bool = False
    max_reasks: int = 2
    
    # Input validation
    validate_inputs: bool = True
    min_input_length: int = 1
    max_input_length: int = 100000
    allowed_languages: List[str] = field(default_factory=lambda: ["en"])
    block_injections: bool = True
    
    # Output validation
    validate_outputs: bool = True
    max_output_length: int = 50000
    enforce_json_schema: bool = False
    check_content_safety: bool = True
    
    # PII settings
    detect_pii: bool = True
    pii_types: List[str] = field(default_factory=lambda: [
        "email", "phone", "ssn", "credit_card", "address"
    ])
    redact_pii: bool = True
    
    # Toxicity settings
    check_toxicity: bool = True
    toxicity_threshold: float = 0.7
    
    # LLM settings for AI-powered validators
    llm_model: str = "gpt-4o"
    api_key: Optional[str] = None
    
    # Logging
    log_validations: bool = True
    output_dir: str = "./safety_logs"


# ============================================
# Result Models
# ============================================

class ValidationIssue(BaseModel):
    """A validation issue."""
    issue_id: str = Field(default_factory=lambda: str(uuid4()))
    validator_name: str
    validator_type: ValidatorType
    severity: str = "error"
    message: str
    details: Optional[Dict[str, Any]] = None
    original_value: Optional[str] = None
    fixed_value: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class InputValidationResult(BaseModel):
    """Result of input validation."""
    result_id: str = Field(default_factory=lambda: str(uuid4()))
    passed: bool = True
    original_input: str
    validated_input: Optional[str] = None
    issues: List[ValidationIssue] = Field(default_factory=list)
    checks_performed: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def has_pii(self) -> bool:
        return any(i.validator_type == ValidatorType.PII for i in self.issues)
    
    @property
    def has_injection(self) -> bool:
        return any("injection" in i.validator_name.lower() for i in self.issues)


class OutputValidationResult(BaseModel):
    """Result of output validation."""
    result_id: str = Field(default_factory=lambda: str(uuid4()))
    passed: bool = True
    original_output: str
    validated_output: Optional[str] = None
    issues: List[ValidationIssue] = Field(default_factory=list)
    checks_performed: List[str] = Field(default_factory=list)
    compliance_score: float = Field(1.0, ge=0.0, le=1.0)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def is_safe(self) -> bool:
        return self.passed and self.compliance_score >= 0.7


class GuardExecutionResult(BaseModel):
    """Result from guard execution."""
    guard_name: str
    passed: bool = True
    raw_output: Optional[str] = None
    validated_output: Optional[str] = None
    reask_count: int = 0
    error: Optional[str] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================
# Custom Validators
# ============================================

if GUARDRAILS_AVAILABLE:
    
    @gd.register_validator(name="custom/length_validator", data_type="string")
    class LengthValidator(Validator):
        """Validate string length."""
        
        def __init__(
            self,
            min_length: int = 1,
            max_length: int = 100000,
            on_fail: str = "fix",
            **kwargs,
        ):
            super().__init__(on_fail=on_fail, **kwargs)
            self.min_length = min_length
            self.max_length = max_length
        
        def validate(self, value: str, metadata: Dict) -> GRValidationResult:
            """Validate the value."""
            if len(value) < self.min_length:
                return FailResult(
                    error_message=f"Input too short: {len(value)} < {self.min_length}",
                    fix_value=value.ljust(self.min_length),
                )
            
            if len(value) > self.max_length:
                return FailResult(
                    error_message=f"Input too long: {len(value)} > {self.max_length}",
                    fix_value=value[:self.max_length],
                )
            
            return PassResult()
    
    
    @gd.register_validator(name="custom/injection_detector", data_type="string")
    class InjectionDetector(Validator):
        """Detect prompt injection attempts."""
        
        INJECTION_PATTERNS = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"disregard\s+(all\s+)?(previous|above|prior)",
            r"forget\s+(everything|all)",
            r"you\s+are\s+now\s+(a|an)",
            r"pretend\s+(to\s+be|you\s+are)",
            r"act\s+as\s+(if|though)",
            r"system\s*:\s*",
            r"<\s*system\s*>",
            r"\[\s*INST\s*\]",
            r"jailbreak",
            r"DAN\s+mode",
        ]
        
        def __init__(self, on_fail: str = "exception", **kwargs):
            super().__init__(on_fail=on_fail, **kwargs)
            self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
        
        def validate(self, value: str, metadata: Dict) -> GRValidationResult:
            """Check for injection patterns."""
            for pattern in self.patterns:
                if pattern.search(value):
                    return FailResult(
                        error_message=f"Potential prompt injection detected: {pattern.pattern}",
                    )
            
            return PassResult()
    
    
    @gd.register_validator(name="custom/pii_detector", data_type="string")
    class PIIDetector(Validator):
        """Detect and optionally redact PII."""
        
        PII_PATTERNS = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        }
        
        def __init__(
            self,
            pii_types: Optional[List[str]] = None,
            redact: bool = True,
            on_fail: str = "fix",
            **kwargs,
        ):
            super().__init__(on_fail=on_fail, **kwargs)
            self.pii_types = pii_types or list(self.PII_PATTERNS.keys())
            self.redact = redact
        
        def validate(self, value: str, metadata: Dict) -> GRValidationResult:
            """Detect PII in the value."""
            detected = []
            fixed_value = value
            
            for pii_type in self.pii_types:
                if pii_type in self.PII_PATTERNS:
                    pattern = re.compile(self.PII_PATTERNS[pii_type])
                    matches = pattern.findall(value)
                    
                    if matches:
                        detected.append({"type": pii_type, "count": len(matches)})
                        
                        if self.redact:
                            fixed_value = pattern.sub(f"[REDACTED_{pii_type.upper()}]", fixed_value)
            
            if detected:
                return FailResult(
                    error_message=f"PII detected: {detected}",
                    fix_value=fixed_value if self.redact else None,
                )
            
            return PassResult()
    
    
    @gd.register_validator(name="custom/toxicity_checker", data_type="string")
    class ToxicityChecker(Validator):
        """Check for toxic content."""
        
        TOXIC_PATTERNS = [
            r"\b(hate|kill|murder|attack)\b",
            r"\b(racist|sexist|homophobic)\b",
            r"\b(stupid|idiot|moron|dumb)\b",
            r"\b(threat|threaten|violence)\b",
        ]
        
        def __init__(
            self,
            threshold: float = 0.7,
            on_fail: str = "exception",
            **kwargs,
        ):
            super().__init__(on_fail=on_fail, **kwargs)
            self.threshold = threshold
            self.patterns = [re.compile(p, re.IGNORECASE) for p in self.TOXIC_PATTERNS]
        
        def validate(self, value: str, metadata: Dict) -> GRValidationResult:
            """Check for toxic content."""
            matches = []
            
            for pattern in self.patterns:
                found = pattern.findall(value)
                matches.extend(found)
            
            # Simple heuristic scoring
            toxicity_score = min(1.0, len(matches) * 0.2)
            
            if toxicity_score >= self.threshold:
                return FailResult(
                    error_message=f"Toxic content detected (score: {toxicity_score:.2f})",
                )
            
            return PassResult()
    
    
    @gd.register_validator(name="custom/json_schema_validator", data_type="string")
    class JSONSchemaValidator(Validator):
        """Validate JSON against a schema."""
        
        def __init__(
            self,
            schema: Dict[str, Any],
            on_fail: str = "exception",
            **kwargs,
        ):
            super().__init__(on_fail=on_fail, **kwargs)
            self.schema = schema
        
        def validate(self, value: str, metadata: Dict) -> GRValidationResult:
            """Validate JSON against schema."""
            try:
                import jsonschema
                
                # Parse JSON
                data = json.loads(value)
                
                # Validate against schema
                jsonschema.validate(data, self.schema)
                
                return PassResult()
                
            except json.JSONDecodeError as e:
                return FailResult(error_message=f"Invalid JSON: {e}")
            except jsonschema.ValidationError as e:
                return FailResult(error_message=f"Schema validation failed: {e.message}")
            except ImportError:
                # jsonschema not installed, skip validation
                return PassResult()


# ============================================
# Input Validator
# ============================================

class InputValidator:
    """
    Input validator using Guardrails AI.
    
    Validates:
    - Length constraints
    - Format compliance
    - Language detection
    - Injection detection
    - PII detection
    """
    
    def __init__(self, config: Optional[GuardrailsConfig] = None):
        """Initialize the input validator."""
        self.config = config or GuardrailsConfig()
        self._guards: Dict[str, Any] = {}
        self._validation_count = 0
        
        if GUARDRAILS_AVAILABLE and self.config.enabled:
            self._setup_guards()
        
        logger.info("input_validator_initialized", enabled=self.config.enabled)
    
    def _setup_guards(self) -> None:
        """Setup input validation guards."""
        # Length guard
        self._guards["length"] = Guard().use(
            LengthValidator(
                min_length=self.config.min_input_length,
                max_length=self.config.max_input_length,
            )
        )
        
        # Injection detection guard
        if self.config.block_injections:
            self._guards["injection"] = Guard().use(
                InjectionDetector()
            )
        
        # PII detection guard
        if self.config.detect_pii:
            self._guards["pii"] = Guard().use(
                PIIDetector(
                    pii_types=self.config.pii_types,
                    redact=self.config.redact_pii,
                )
            )
    
    def validate(self, input_text: str) -> InputValidationResult:
        """
        Validate input text.
        
        Args:
            input_text: Text to validate
            
        Returns:
            InputValidationResult
        """
        start_time = time.time()
        self._validation_count += 1
        
        result = InputValidationResult(
            original_input=input_text,
            validated_input=input_text,
        )
        
        if not self.config.enabled or not GUARDRAILS_AVAILABLE:
            result.checks_performed.append("none (disabled)")
            result.latency_ms = (time.time() - start_time) * 1000
            return result
        
        current_text = input_text
        
        # Run each guard
        for guard_name, guard in self._guards.items():
            try:
                guard_result = guard.validate(current_text)
                result.checks_performed.append(guard_name)
                
                if not guard_result.validation_passed:
                    issue = ValidationIssue(
                        validator_name=guard_name,
                        validator_type=self._get_validator_type(guard_name),
                        message=str(guard_result.error) if guard_result.error else "Validation failed",
                        original_value=current_text[:100],
                    )
                    result.issues.append(issue)
                    result.passed = False
                    
                    # Apply fix if available
                    if guard_result.validated_output:
                        current_text = guard_result.validated_output
                        issue.fixed_value = current_text[:100]
                    
                    if self.config.fail_fast:
                        break
                        
            except Exception as e:
                logger.error("guard_execution_error", guard=guard_name, error=str(e))
                result.issues.append(ValidationIssue(
                    validator_name=guard_name,
                    validator_type=ValidatorType.CUSTOM,
                    severity="error",
                    message=str(e),
                ))
                result.passed = False
        
        result.validated_input = current_text
        result.latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "input_validation_complete",
            passed=result.passed,
            issues=len(result.issues),
            latency_ms=result.latency_ms,
        )
        
        return result
    
    def _get_validator_type(self, guard_name: str) -> ValidatorType:
        """Map guard name to validator type."""
        mapping = {
            "length": ValidatorType.LENGTH,
            "injection": ValidatorType.REGEX,
            "pii": ValidatorType.PII,
            "toxicity": ValidatorType.TOXICITY,
        }
        return mapping.get(guard_name, ValidatorType.CUSTOM)
    
    async def validate_async(self, input_text: str) -> InputValidationResult:
        """Async version of validate."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.validate(input_text),
        )


# ============================================
# Output Validator
# ============================================

class OutputValidator:
    """
    Output validator using Guardrails AI.
    
    Validates:
    - Compliance with guidelines
    - JSON schema conformance
    - Content safety
    - Length constraints
    """
    
    def __init__(self, config: Optional[GuardrailsConfig] = None):
        """Initialize the output validator."""
        self.config = config or GuardrailsConfig()
        self._guards: Dict[str, Any] = {}
        self._validation_count = 0
        
        if GUARDRAILS_AVAILABLE and self.config.enabled:
            self._setup_guards()
        
        logger.info("output_validator_initialized", enabled=self.config.enabled)
    
    def _setup_guards(self) -> None:
        """Setup output validation guards."""
        # Length guard
        self._guards["length"] = Guard().use(
            LengthValidator(
                min_length=1,
                max_length=self.config.max_output_length,
            )
        )
        
        # Toxicity guard
        if self.config.check_toxicity:
            self._guards["toxicity"] = Guard().use(
                ToxicityChecker(threshold=self.config.toxicity_threshold)
            )
        
        # PII guard (for output)
        if self.config.detect_pii:
            self._guards["pii"] = Guard().use(
                PIIDetector(
                    pii_types=self.config.pii_types,
                    redact=self.config.redact_pii,
                )
            )
    
    def validate(
        self,
        output_text: str,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> OutputValidationResult:
        """
        Validate output text.
        
        Args:
            output_text: Text to validate
            json_schema: Optional JSON schema for validation
            
        Returns:
            OutputValidationResult
        """
        start_time = time.time()
        self._validation_count += 1
        
        result = OutputValidationResult(
            original_output=output_text,
            validated_output=output_text,
        )
        
        if not self.config.enabled or not GUARDRAILS_AVAILABLE:
            result.checks_performed.append("none (disabled)")
            result.latency_ms = (time.time() - start_time) * 1000
            return result
        
        current_text = output_text
        issues_count = 0
        
        # Run standard guards
        for guard_name, guard in self._guards.items():
            try:
                guard_result = guard.validate(current_text)
                result.checks_performed.append(guard_name)
                
                if not guard_result.validation_passed:
                    issues_count += 1
                    issue = ValidationIssue(
                        validator_name=guard_name,
                        validator_type=self._get_validator_type(guard_name),
                        message=str(guard_result.error) if guard_result.error else "Validation failed",
                        original_value=current_text[:100],
                    )
                    result.issues.append(issue)
                    
                    # Apply fix if available
                    if guard_result.validated_output:
                        current_text = guard_result.validated_output
                        issue.fixed_value = current_text[:100]
                    
                    if self.config.fail_fast:
                        break
                        
            except Exception as e:
                logger.error("guard_execution_error", guard=guard_name, error=str(e))
                issues_count += 1
        
        # JSON schema validation if provided
        if json_schema and self.config.enforce_json_schema:
            try:
                schema_guard = Guard().use(
                    JSONSchemaValidator(schema=json_schema)
                )
                schema_result = schema_guard.validate(current_text)
                result.checks_performed.append("json_schema")
                
                if not schema_result.validation_passed:
                    issues_count += 1
                    result.issues.append(ValidationIssue(
                        validator_name="json_schema",
                        validator_type=ValidatorType.JSON_SCHEMA,
                        message=str(schema_result.error),
                    ))
                    
            except Exception as e:
                logger.error("schema_validation_error", error=str(e))
                issues_count += 1
        
        result.validated_output = current_text
        result.passed = issues_count == 0
        result.compliance_score = max(0.0, 1.0 - (issues_count * 0.2))
        result.latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "output_validation_complete",
            passed=result.passed,
            compliance_score=result.compliance_score,
            issues=len(result.issues),
            latency_ms=result.latency_ms,
        )
        
        return result
    
    def _get_validator_type(self, guard_name: str) -> ValidatorType:
        """Map guard name to validator type."""
        mapping = {
            "length": ValidatorType.LENGTH,
            "toxicity": ValidatorType.TOXICITY,
            "pii": ValidatorType.PII,
            "json_schema": ValidatorType.JSON_SCHEMA,
        }
        return mapping.get(guard_name, ValidatorType.CUSTOM)
    
    async def validate_async(
        self,
        output_text: str,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> OutputValidationResult:
        """Async version of validate."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.validate(output_text, json_schema),
        )


# ============================================
# Main Guardrails Validator
# ============================================

class GuardrailsValidator:
    """
    Unified Guardrails AI validator.
    
    Provides comprehensive input/output validation with:
    - Configurable validators
    - Custom validator support
    - LLM Gateway integration
    - Observability logging
    
    Usage:
        validator = GuardrailsValidator()
        
        # Validate input
        input_result = validator.validate_input("Hello, world!")
        
        # Validate output
        output_result = validator.validate_output("AI response here")
        
        # Execute specific guard
        guard_result = validator.execute_guard("injection", "user content")
    """
    
    def __init__(self, config: Optional[GuardrailsConfig] = None):
        """Initialize the Guardrails validator."""
        self.config = config or GuardrailsConfig()
        self.input_validator = InputValidator(self.config)
        self.output_validator = OutputValidator(self.config)
        self._custom_guards: Dict[str, Any] = {}
        self._execution_count = 0
        
        logger.info(
            "guardrails_validator_initialized",
            enabled=self.config.enabled,
            guardrails_available=GUARDRAILS_AVAILABLE,
        )
    
    @property
    def is_available(self) -> bool:
        """Check if Guardrails AI is available."""
        return GUARDRAILS_AVAILABLE
    
    def validate_input(self, content: str, **kwargs: Any) -> InputValidationResult:
        """
        Validate input content.
        
        Args:
            content: Content to validate
            **kwargs: Additional options
            
        Returns:
            InputValidationResult
        """
        self._execution_count += 1
        return self.input_validator.validate(content)
    
    def validate_output(
        self,
        content: str,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> OutputValidationResult:
        """
        Validate output content.
        
        Args:
            content: Content to validate
            json_schema: Optional JSON schema
            **kwargs: Additional options
            
        Returns:
            OutputValidationResult
        """
        self._execution_count += 1
        return self.output_validator.validate(content, json_schema)
    
    async def validate_input_async(self, content: str, **kwargs: Any) -> InputValidationResult:
        """Async version of validate_input."""
        return await self.input_validator.validate_async(content)
    
    async def validate_output_async(
        self,
        content: str,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> OutputValidationResult:
        """Async version of validate_output."""
        return await self.output_validator.validate_async(content, json_schema)
    
    def create_custom_validator(
        self,
        name: str,
        validator_func: Callable[[str, Dict], bool],
        on_fail: str = "exception",
    ) -> None:
        """
        Create a custom validator.
        
        Args:
            name: Validator name
            validator_func: Function that takes (value, metadata) and returns bool
            on_fail: Action on failure ("exception", "fix", "filter")
        """
        if not GUARDRAILS_AVAILABLE:
            logger.warning("cannot_create_custom_validator", reason="guardrails not available")
            return
        
        @gd.register_validator(name=f"custom/{name}", data_type="string")
        class CustomValidator(Validator):
            def __init__(self, **kwargs):
                super().__init__(on_fail=on_fail, **kwargs)
                self.validate_func = validator_func
            
            def validate(self, value: str, metadata: Dict) -> GRValidationResult:
                try:
                    if self.validate_func(value, metadata):
                        return PassResult()
                    return FailResult(error_message=f"Custom validation '{name}' failed")
                except Exception as e:
                    return FailResult(error_message=str(e))
        
        self._custom_guards[name] = Guard().use(CustomValidator())
        logger.info("custom_validator_created", name=name)
    
    def execute_guard(
        self,
        guard_name: str,
        content: str,
        **kwargs: Any,
    ) -> GuardExecutionResult:
        """
        Execute a specific guard.
        
        Args:
            guard_name: Name of the guard
            content: Content to process
            **kwargs: Guard-specific options
            
        Returns:
            GuardExecutionResult
        """
        start_time = time.time()
        self._execution_count += 1
        
        result = GuardExecutionResult(
            guard_name=guard_name,
            raw_output=content,
        )
        
        if not GUARDRAILS_AVAILABLE:
            result.error = "Guardrails AI not available"
            result.passed = False
            return result
        
        # Find the guard
        guard = None
        
        # Check input validator guards
        if guard_name in self.input_validator._guards:
            guard = self.input_validator._guards[guard_name]
        # Check output validator guards
        elif guard_name in self.output_validator._guards:
            guard = self.output_validator._guards[guard_name]
        # Check custom guards
        elif guard_name in self._custom_guards:
            guard = self._custom_guards[guard_name]
        
        if guard is None:
            result.error = f"Guard '{guard_name}' not found"
            result.passed = False
            return result
        
        try:
            guard_output = guard.validate(content)
            result.passed = guard_output.validation_passed
            result.validated_output = guard_output.validated_output or content
            
            if not result.passed:
                result.error = str(guard_output.error) if guard_output.error else "Validation failed"
                
        except Exception as e:
            result.passed = False
            result.error = str(e)
        
        result.latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "guard_executed",
            guard=guard_name,
            passed=result.passed,
            latency_ms=result.latency_ms,
        )
        
        return result
    
    async def execute_guard_async(
        self,
        guard_name: str,
        content: str,
        **kwargs: Any,
    ) -> GuardExecutionResult:
        """Async version of execute_guard."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.execute_guard(guard_name, content, **kwargs),
        )
    
    def integrate_with_gateway(self) -> Dict[str, Callable]:
        """
        Get validation hooks for LLM Gateway integration.
        
        Returns:
            Dict with pre_request and post_response hooks
        """
        async def pre_request_hook(request: Dict[str, Any]) -> Dict[str, Any]:
            """Validate request before sending to LLM."""
            if "messages" in request:
                for msg in request["messages"]:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        result = await self.validate_input_async(content)
                        
                        if not result.passed:
                            raise ValueError(f"Input validation failed: {result.issues}")
                        
                        msg["content"] = result.validated_input
            
            return request
        
        async def post_response_hook(response: Dict[str, Any]) -> Dict[str, Any]:
            """Validate response from LLM."""
            content = response.get("content", "")
            result = await self.validate_output_async(content)
            
            if not result.passed:
                logger.warning("output_validation_issues", issues=len(result.issues))
            
            response["content"] = result.validated_output
            response["_validation"] = {
                "passed": result.passed,
                "compliance_score": result.compliance_score,
                "issues": len(result.issues),
            }
            
            return response
        
        return {
            "pre_request": pre_request_hook,
            "post_response": post_response_hook,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validator statistics."""
        return {
            "executions": self._execution_count,
            "guardrails_available": GUARDRAILS_AVAILABLE,
            "enabled": self.config.enabled,
            "input_guards": list(self.input_validator._guards.keys()),
            "output_guards": list(self.output_validator._guards.keys()),
            "custom_guards": list(self._custom_guards.keys()),
        }


# ============================================
# Factory Function
# ============================================

def create_guardrails_validator(
    validate_inputs: bool = True,
    validate_outputs: bool = True,
    detect_pii: bool = True,
    check_toxicity: bool = True,
    **kwargs: Any,
) -> GuardrailsValidator:
    """
    Create a configured GuardrailsValidator.
    
    Args:
        validate_inputs: Enable input validation
        validate_outputs: Enable output validation
        detect_pii: Enable PII detection
        check_toxicity: Enable toxicity checking
        **kwargs: Additional configuration options
        
    Returns:
        Configured GuardrailsValidator
    """
    config = GuardrailsConfig(
        validate_inputs=validate_inputs,
        validate_outputs=validate_outputs,
        detect_pii=detect_pii,
        check_toxicity=check_toxicity,
        **kwargs,
    )
    return GuardrailsValidator(config)


# ============================================
# Module Exports
# ============================================

__all__ = [
    # Configuration
    "GuardrailsConfig",
    
    # Validators
    "GuardrailsValidator",
    "InputValidator",
    "OutputValidator",
    
    # Results
    "ValidationIssue",
    "InputValidationResult",
    "OutputValidationResult",
    "GuardExecutionResult",
    
    # Enums
    "ValidatorType",
    "InputValidationType",
    "OutputValidationType",
    
    # Factory
    "create_guardrails_validator",
    
    # Availability
    "GUARDRAILS_AVAILABLE",
]


if __name__ == "__main__":
    print("Guardrails Validators Module")
    print("-" * 40)
    
    if not GUARDRAILS_AVAILABLE:
        print("Guardrails AI not installed. Install with: pip install guardrails-ai")
    else:
        print("Guardrails AI is available")
        
        # Quick test
        validator = GuardrailsValidator()
        
        # Test input validation
        input_result = validator.validate_input("Hello, my email is test@example.com")
        print(f"\nInput validation: {'PASS' if input_result.passed else 'FAIL'}")
        print(f"  Issues: {len(input_result.issues)}")
        if input_result.issues:
            for issue in input_result.issues:
                print(f"    - {issue.validator_name}: {issue.message}")
        
        # Test output validation
        output_result = validator.validate_output("This is a safe response.")
        print(f"\nOutput validation: {'PASS' if output_result.passed else 'FAIL'}")
        print(f"  Compliance score: {output_result.compliance_score:.2f}")
        
        print("\nStatistics:")
        stats = validator.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
```

---

## Step 5: Create Validation Script

Create `scripts/validate_phase7_part1.py`:

```python
#!/usr/bin/env python3
"""
Phase 7 Part 1 Validation Script
Validates Safety Layer - Guardrails AI integration.
"""

import sys
from pathlib import Path
from typing import Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_result(name: str, passed: bool, message: str = "") -> None:
    """Print a test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    msg = f" - {message}" if message else ""
    print(f"  {status}: {name}{msg}")


def check_file_exists(filepath: str) -> Tuple[bool, str]:
    """Check if a file exists."""
    path = project_root / filepath
    if path.exists():
        return True, f"Found"
    return False, f"Missing"


def validate_file_structure() -> Tuple[int, int]:
    """Validate all required files exist."""
    print_header("File Structure Validation")
    
    required_files = [
        "core/safety/__init__.py",
        "core/safety/guardrails_validators.py",
    ]
    
    passed = 0
    failed = 0
    
    for filepath in required_files:
        exists, message = check_file_exists(filepath)
        print_result(filepath, exists, message)
        if exists:
            passed += 1
        else:
            failed += 1
    
    return passed, failed


def validate_imports() -> Tuple[int, int]:
    """Validate imports work correctly."""
    print_header("Import Validation")
    
    passed = 0
    failed = 0
    
    # Test safety module imports
    try:
        from core.safety import (
            SafetyFactory,
            SafetyProvider,
            ValidationResult,
            GUARDRAILS_AVAILABLE,
        )
        print_result("core.safety imports", True)
        passed += 1
    except Exception as e:
        print_result("core.safety imports", False, str(e))
        failed += 1
    
    # Test guardrails_validators imports
    try:
        from core.safety.guardrails_validators import (
            GuardrailsValidator,
            GuardrailsConfig,
            InputValidator,
            OutputValidator,
        )
        print_result("guardrails_validators imports", True)
        passed += 1
    except Exception as e:
        print_result("guardrails_validators imports", False, str(e))
        failed += 1
    
    return passed, failed


def validate_instantiation() -> Tuple[int, int]:
    """Validate classes can be instantiated."""
    print_header("Instantiation Validation")
    
    passed = 0
    failed = 0
    
    # Test SafetyFactory
    try:
        from core.safety import SafetyFactory
        factory = SafetyFactory()
        stats = factory.get_statistics()
        print_result("SafetyFactory", True, f"providers: {len(stats['available_providers'])}")
        passed += 1
    except Exception as e:
        print_result("SafetyFactory", False, str(e))
        failed += 1
    
    # Test GuardrailsValidator
    try:
        from core.safety.guardrails_validators import GuardrailsValidator
        validator = GuardrailsValidator()
        stats = validator.get_statistics()
        print_result("GuardrailsValidator", True, f"available: {stats['guardrails_available']}")
        passed += 1
    except Exception as e:
        print_result("GuardrailsValidator", False, str(e))
        failed += 1
    
    # Test InputValidator
    try:
        from core.safety.guardrails_validators import InputValidator
        input_val = InputValidator()
        print_result("InputValidator", True)
        passed += 1
    except Exception as e:
        print_result("InputValidator", False, str(e))
        failed += 1
    
    # Test OutputValidator
    try:
        from core.safety.guardrails_validators import OutputValidator
        output_val = OutputValidator()
        print_result("OutputValidator", True)
        passed += 1
    except Exception as e:
        print_result("OutputValidator", False, str(e))
        failed += 1
    
    return passed, failed


def validate_basic_operations() -> Tuple[int, int]:
    """Validate basic operations work."""
    print_header("Basic Operations Validation")
    
    passed = 0
    failed = 0
    
    # Test input validation
    try:
        from core.safety.guardrails_validators import GuardrailsValidator
        
        validator = GuardrailsValidator()
        result = validator.validate_input("Hello, this is a test message.")
        
        print_result(
            "validate_input",
            True,
            f"passed={result.passed}, issues={len(result.issues)}"
        )
        passed += 1
    except Exception as e:
        print_result("validate_input", False, str(e))
        failed += 1
    
    # Test output validation
    try:
        from core.safety.guardrails_validators import GuardrailsValidator
        
        validator = GuardrailsValidator()
        result = validator.validate_output("This is a safe AI response.")
        
        print_result(
            "validate_output",
            True,
            f"passed={result.passed}, score={result.compliance_score:.2f}"
        )
        passed += 1
    except Exception as e:
        print_result("validate_output", False, str(e))
        failed += 1
    
    # Test PII detection
    try:
        from core.safety.guardrails_validators import GuardrailsValidator, GUARDRAILS_AVAILABLE
        
        if GUARDRAILS_AVAILABLE:
            validator = GuardrailsValidator()
            result = validator.validate_input("My email is test@example.com")
            
            has_pii = result.has_pii
            print_result("pii_detection", True, f"detected={has_pii}")
            passed += 1
        else:
            print_result("pii_detection", True, "skipped (SDK not installed)")
            passed += 1
    except Exception as e:
        print_result("pii_detection", False, str(e))
        failed += 1
    
    # Test guard execution
    try:
        from core.safety.guardrails_validators import GuardrailsValidator, GUARDRAILS_AVAILABLE
        
        if GUARDRAILS_AVAILABLE:
            validator = GuardrailsValidator()
            result = validator.execute_guard("length", "Test content")
            
            print_result(
                "execute_guard",
                True,
                f"passed={result.passed}, latency={result.latency_ms:.1f}ms"
            )
            passed += 1
        else:
            print_result("execute_guard", True, "skipped (SDK not installed)")
            passed += 1
    except Exception as e:
        print_result("execute_guard", False, str(e))
        failed += 1
    
    # Test factory create_validator
    try:
        from core.safety import SafetyFactory, SafetyProvider, GUARDRAILS_AVAILABLE
        
        if GUARDRAILS_AVAILABLE:
            factory = SafetyFactory()
            validator = factory.create_validator(SafetyProvider.GUARDRAILS)
            
            print_result("factory.create_validator", True, "GuardrailsValidator created")
            passed += 1
        else:
            print_result("factory.create_validator", True, "skipped (SDK not installed)")
            passed += 1
    except Exception as e:
        print_result("factory.create_validator", False, str(e))
        failed += 1
    
    return passed, failed


def validate_gateway_integration() -> Tuple[int, int]:
    """Validate LLM Gateway integration hooks."""
    print_header("Gateway Integration Validation")
    
    passed = 0
    failed = 0
    
    try:
        from core.safety.guardrails_validators import GuardrailsValidator
        
        validator = GuardrailsValidator()
        hooks = validator.integrate_with_gateway()
        
        has_pre = "pre_request" in hooks
        has_post = "post_response" in hooks
        
        print_result(
            "gateway_hooks",
            has_pre and has_post,
            f"pre_request={has_pre}, post_response={has_post}"
        )
        if has_pre and has_post:
            passed += 1
        else:
            failed += 1
    except Exception as e:
        print_result("gateway_hooks", False, str(e))
        failed += 1
    
    return passed, failed


def print_summary(results: list) -> int:
    """Print validation summary."""
    print_header("Validation Summary")
    
    total_passed = 0
    total_failed = 0
    
    print("  Category                    | Passed | Failed")
    print("  " + "-" * 50)
    
    for category, passed, failed in results:
        total_passed += passed
        total_failed += failed
        print(f"  {category:<28} |   {passed:>3}  |   {failed:>3}")
    
    print("  " + "-" * 50)
    print(f"  {'TOTAL':<28} |   {total_passed:>3}  |   {total_failed:>3}")
    
    print(f"\n  Overall: ", end="")
    if total_failed == 0:
        print("✓ ALL TESTS PASSED")
    else:
        print(f"✗ {total_failed} TESTS FAILED")
    
    # Check SDK availability
    try:
        from core.safety import GUARDRAILS_AVAILABLE
        sdk_status = "installed" if GUARDRAILS_AVAILABLE else "not installed (optional)"
        print(f"\n  Guardrails AI SDK: {sdk_status}")
    except:
        pass
    
    return 0 if total_failed == 0 else 1


def main() -> int:
    """Run all validations."""
    print("\n" + "=" * 60)
    print("  PHASE 7 PART 1 VALIDATION - Safety Layer")
    print("  Guardrails AI Integration")
    print("=" * 60)
    
    results = []
    
    # Run validations
    results.append(("File Structure", *validate_file_structure()))
    results.append(("Imports", *validate_imports()))
    results.append(("Instantiation", *validate_instantiation()))
    results.append(("Basic Operations", *validate_basic_operations()))
    results.append(("Gateway Integration", *validate_gateway_integration()))
    
    # Print summary and return exit code
    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
```

---

## Step 6: Part 1 Validation

Run the validation to confirm Part 1 is complete:

```python
from core.safety.guardrails_validators import GuardrailsValidator
print("Part 1 OK")
```

Full validation:

```bash
python scripts/validate_phase7_part1.py
```

Expected output:
```
  PHASE 7 PART 1 VALIDATION - Safety Layer
  Guardrails AI Integration
============================================================

  File Structure | PASS
  Imports        | PASS
  Instantiation  | PASS
  Basic Ops      | PASS
  Gateway        | PASS
  
  Overall: ✓ ALL TESTS PASSED
```

---

## Success Criteria

- [ ] `core/safety/` directory exists
- [ ] `core/safety/__init__.py` - Safety factory interface
- [ ] `core/safety/guardrails_validators.py` - Guardrails validators (~400 lines)
- [ ] `scripts/validate_phase7_part1.py` - Validation script
- [ ] All imports work correctly
- [ ] SafetyFactory can create validators
- [ ] GuardrailsValidator validates input/output
- [ ] Gateway integration hooks available

---

## Rollback

If issues occur:

```bash
# Remove safety directory
rm -rf core/safety

# Remove validation script
rm scripts/validate_phase7_part1.py
```

---

## Usage Examples

### Basic Validation

```python
from core.safety import SafetyFactory, SafetyProvider

# Create factory
factory = SafetyFactory()

# Validate input
input_result = factory.validate_input("User message here")
print(f"Input valid: {input_result.passed}")

# Validate output
output_result = factory.validate_output("AI response here")
print(f"Output valid: {output_result.passed}")
print(f"Compliance: {output_result.compliance_score:.2%}")
```

### With GuardrailsValidator Directly

```python
from core.safety.guardrails_validators import GuardrailsValidator, GuardrailsConfig

# Configure validator
config = GuardrailsConfig(
    detect_pii=True,
    redact_pii=True,
    check_toxicity=True,
    toxicity_threshold=0.5,
)

validator = GuardrailsValidator(config)

# Validate with PII detection
result = validator.validate_input("My email is john@example.com")
print(f"Has PII: {result.has_pii}")
print(f"Redacted: {result.validated_input}")
```

### Custom Validators

```python
from core.safety.guardrails_validators import GuardrailsValidator

validator = GuardrailsValidator()

# Create custom validator
def no_urls_validator(value: str, metadata: dict) -> bool:
    """Reject content with URLs."""
    import re
    return not bool(re.search(r'https?://', value))

validator.create_custom_validator(
    name="no_urls",
    validator_func=no_urls_validator,
)

# Use custom validator
result = validator.execute_guard("no_urls", "Check out https://example.com")
print(f"Passed: {result.passed}")  # False
```

### Async Validation

```python
import asyncio
from core.safety.guardrails_validators import GuardrailsValidator

async def validate_conversation():
    validator = GuardrailsValidator()
    
    # Validate input async
    input_result = await validator.validate_input_async("User question")
    
    # Validate output async
    output_result = await validator.validate_output_async("AI answer")
    
    return input_result.passed and output_result.passed

asyncio.run(validate_conversation())
```

### LLM Gateway Integration

```python
from core.safety.guardrails_validators import GuardrailsValidator

validator = GuardrailsValidator()
hooks = validator.integrate_with_gateway()

# Use with LLM Gateway
async def process_request(request):
    # Pre-request validation
    validated_request = await hooks["pre_request"](request)
    
    # ... send to LLM ...
    
    # Post-response validation
    validated_response = await hooks["post_response"](response)
    
    return validated_response
```

---

## Integration with Phase 6 (Observability)

```python
from core.safety import SafetyFactory, SafetyConfig
from core.observability import LangfuseTracer

# Create safety factory with observability
config = SafetyConfig(log_to_observability=True)
factory = SafetyFactory(config)

# Validations are automatically traced
tracer = LangfuseTracer()
with tracer.trace_span("conversation") as span:
    input_result = factory.validate_input("User message")
    # Validation event logged to Langfuse
```

---

## Next Steps

After completing Phase 7 Part 1:

1. Run validation: `python scripts/validate_phase7_part1.py`
2. Proceed to Phase 7 Part 2: Additional Safety SDKs
   - llm-guard
   - nemo-guardrails
   - rebuff

---

**End of Phase 7 Part 1 Prompt**
