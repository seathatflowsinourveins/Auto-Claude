"""
Unleashed Platform - Configuration Validation

Production-grade configuration validation and management with:
- Schema validation for all settings
- Environment variable integration
- Secrets management
- Configuration versioning
- Hot-reload support
"""

import os
import json
import re
from typing import (
    Dict, Any, Optional, List, Type, TypeVar, Generic,
    Callable, Union, get_type_hints, get_origin, get_args
)
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Validation Types
# ============================================================================

class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ConfigSource(Enum):
    """Sources for configuration values."""
    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    OVERRIDE = "override"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    field: str
    message: str
    severity: ValidationSeverity
    current_value: Any = None
    suggested_value: Any = None


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: int = 0
    errors: int = 0
    critical: int = 0

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the result."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.WARNING:
            self.warnings += 1
        elif issue.severity == ValidationSeverity.ERROR:
            self.errors += 1
            self.valid = False
        elif issue.severity == ValidationSeverity.CRITICAL:
            self.critical += 1
            self.valid = False


# ============================================================================
# Validators
# ============================================================================

class Validator(Generic[T]):
    """Base class for configuration validators."""

    def __init__(self, message: Optional[str] = None):
        self.message = message

    def validate(self, value: T, field_name: str) -> Optional[ValidationIssue]:
        """Validate a value. Returns None if valid, ValidationIssue otherwise."""
        raise NotImplementedError


class RequiredValidator(Validator[Any]):
    """Validates that a value is not None or empty."""

    def validate(self, value: Any, field_name: str) -> Optional[ValidationIssue]:
        if value is None or (isinstance(value, str) and not value.strip()):
            return ValidationIssue(
                field=field_name,
                message=self.message or f"{field_name} is required",
                severity=ValidationSeverity.ERROR,
                current_value=value
            )
        return None


class RangeValidator(Validator[Union[int, float]]):
    """Validates that a numeric value is within a range."""

    def __init__(
        self,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        message: Optional[str] = None
    ):
        super().__init__(message)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Union[int, float], field_name: str) -> Optional[ValidationIssue]:
        if value is None:
            return None

        if self.min_value is not None and value < self.min_value:
            return ValidationIssue(
                field=field_name,
                message=self.message or f"{field_name} must be >= {self.min_value}",
                severity=ValidationSeverity.ERROR,
                current_value=value,
                suggested_value=self.min_value
            )

        if self.max_value is not None and value > self.max_value:
            return ValidationIssue(
                field=field_name,
                message=self.message or f"{field_name} must be <= {self.max_value}",
                severity=ValidationSeverity.ERROR,
                current_value=value,
                suggested_value=self.max_value
            )

        return None


class PatternValidator(Validator[str]):
    """Validates that a string matches a regex pattern."""

    def __init__(self, pattern: str, message: Optional[str] = None):
        super().__init__(message)
        self.pattern = re.compile(pattern)

    def validate(self, value: str, field_name: str) -> Optional[ValidationIssue]:
        if value is None:
            return None

        if not self.pattern.match(value):
            return ValidationIssue(
                field=field_name,
                message=self.message or f"{field_name} does not match required pattern",
                severity=ValidationSeverity.ERROR,
                current_value=value
            )
        return None


class URLValidator(PatternValidator):
    """Validates that a string is a valid URL."""

    URL_PATTERN = r"^https?://[^\s/$.?#].[^\s]*$"

    def __init__(self, require_https: bool = False, message: Optional[str] = None):
        pattern = r"^https://[^\s/$.?#].[^\s]*$" if require_https else self.URL_PATTERN
        super().__init__(pattern, message or "Invalid URL format")


class ChoiceValidator(Validator[T]):
    """Validates that a value is one of allowed choices."""

    def __init__(self, choices: List[T], message: Optional[str] = None):
        super().__init__(message)
        self.choices = choices

    def validate(self, value: T, field_name: str) -> Optional[ValidationIssue]:
        if value is None:
            return None

        if value not in self.choices:
            return ValidationIssue(
                field=field_name,
                message=self.message or f"{field_name} must be one of: {self.choices}",
                severity=ValidationSeverity.ERROR,
                current_value=value
            )
        return None


class PathValidator(Validator[str]):
    """Validates that a path exists or is creatable."""

    def __init__(
        self,
        must_exist: bool = False,
        must_be_dir: bool = False,
        must_be_file: bool = False,
        message: Optional[str] = None
    ):
        super().__init__(message)
        self.must_exist = must_exist
        self.must_be_dir = must_be_dir
        self.must_be_file = must_be_file

    def validate(self, value: str, field_name: str) -> Optional[ValidationIssue]:
        if value is None:
            return None

        path = Path(value)

        if self.must_exist and not path.exists():
            return ValidationIssue(
                field=field_name,
                message=self.message or f"{field_name}: path does not exist",
                severity=ValidationSeverity.ERROR,
                current_value=value
            )

        if self.must_be_dir and path.exists() and not path.is_dir():
            return ValidationIssue(
                field=field_name,
                message=self.message or f"{field_name}: path is not a directory",
                severity=ValidationSeverity.ERROR,
                current_value=value
            )

        if self.must_be_file and path.exists() and not path.is_file():
            return ValidationIssue(
                field=field_name,
                message=self.message or f"{field_name}: path is not a file",
                severity=ValidationSeverity.ERROR,
                current_value=value
            )

        return None


# ============================================================================
# Production Validators
# ============================================================================

class ProductionRequiredValidator(Validator[Any]):
    """Validates that a value is set in production mode."""

    def __init__(self, is_production: bool = False, message: Optional[str] = None):
        super().__init__(message)
        self.is_production = is_production

    def validate(self, value: Any, field_name: str) -> Optional[ValidationIssue]:
        if not self.is_production:
            return None

        if value is None or (isinstance(value, str) and not value.strip()):
            return ValidationIssue(
                field=field_name,
                message=self.message or f"{field_name} is required in production",
                severity=ValidationSeverity.CRITICAL,
                current_value=value
            )
        return None


class SecureValueValidator(Validator[str]):
    """Validates that sensitive values meet security requirements."""

    def __init__(
        self,
        min_length: int = 16,
        require_special: bool = True,
        message: Optional[str] = None
    ):
        super().__init__(message)
        self.min_length = min_length
        self.require_special = require_special

    def validate(self, value: str, field_name: str) -> Optional[ValidationIssue]:
        if value is None:
            return None

        if len(value) < self.min_length:
            return ValidationIssue(
                field=field_name,
                message=self.message or f"{field_name} must be at least {self.min_length} characters",
                severity=ValidationSeverity.ERROR,
                current_value="[REDACTED]"
            )

        if self.require_special and not re.search(r"[!@#$%^&*(),.?\":{}|<>]", value):
            return ValidationIssue(
                field=field_name,
                message=self.message or f"{field_name} should contain special characters",
                severity=ValidationSeverity.WARNING,
                current_value="[REDACTED]"
            )

        return None


# ============================================================================
# Configuration Schema
# ============================================================================

@dataclass
class ConfigField:
    """Definition of a configuration field."""
    name: str
    type: Type
    default: Any = None
    required: bool = False
    secret: bool = False
    env_var: Optional[str] = None
    validators: List[Validator] = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class ConfigSchema:
    """Schema for configuration validation."""
    name: str
    version: str
    fields: List[ConfigField]
    description: Optional[str] = None

    def validate(self, config: Dict[str, Any], is_production: bool = False) -> ValidationResult:
        """Validate a configuration dictionary against this schema."""
        result = ValidationResult(valid=True)

        for field_def in self.fields:
            value = config.get(field_def.name, field_def.default)

            # Check required
            if field_def.required and (value is None or value == ""):
                result.add_issue(ValidationIssue(
                    field=field_def.name,
                    message=f"{field_def.name} is required",
                    severity=ValidationSeverity.ERROR
                ))
                continue

            # Run validators
            for validator in field_def.validators:
                # Handle production-specific validators
                if isinstance(validator, ProductionRequiredValidator):
                    validator.is_production = is_production

                issue = validator.validate(value, field_def.name)
                if issue:
                    result.add_issue(issue)

        return result


# ============================================================================
# Environment Variable Integration
# ============================================================================

class EnvironmentLoader:
    """Loads configuration from environment variables."""

    def __init__(self, prefix: str = "UNLEASHED"):
        self.prefix = prefix

    def load_value(self, env_var: str, field_type: Type) -> Any:
        """Load a value from an environment variable."""
        full_var = f"{self.prefix}_{env_var}" if self.prefix else env_var
        raw_value = os.environ.get(full_var)

        if raw_value is None:
            return None

        # Type conversion
        origin = get_origin(field_type)
        if origin is not None:
            # Handle Optional, List, etc.
            args = get_args(field_type)
            if origin is Union and type(None) in args:
                # Optional type
                actual_type = [a for a in args if a is not type(None)][0]
                return self._convert(raw_value, actual_type)
            elif origin is list:
                return json.loads(raw_value)
            elif origin is dict:
                return json.loads(raw_value)
        else:
            return self._convert(raw_value, field_type)

        return raw_value

    def _convert(self, value: str, target_type: Type) -> Any:
        """Convert a string value to the target type."""
        if target_type is bool:
            return value.lower() in ("true", "1", "yes", "on")
        elif target_type is int:
            return int(value)
        elif target_type is float:
            return float(value)
        elif target_type is str:
            return value
        elif issubclass(target_type, Enum):
            return target_type(value)
        return value

    def load_config(self, schema: ConfigSchema) -> Dict[str, Any]:
        """Load configuration from environment variables based on schema."""
        config = {}

        for field_def in schema.fields:
            env_var = field_def.env_var or field_def.name.upper()
            value = self.load_value(env_var, field_def.type)

            if value is not None:
                config[field_def.name] = value
            elif field_def.default is not None:
                config[field_def.name] = field_def.default

        return config


# ============================================================================
# Configuration Manager
# ============================================================================

class ConfigurationManager:
    """Manages configuration with validation and hot-reload."""

    def __init__(
        self,
        schema: ConfigSchema,
        env_prefix: str = "UNLEASHED",
        config_file: Optional[str] = None
    ):
        self.schema = schema
        self.env_loader = EnvironmentLoader(env_prefix)
        self.config_file = config_file

        self._config: Dict[str, Any] = {}
        self._sources: Dict[str, ConfigSource] = {}
        self._last_loaded: Optional[datetime] = None
        self._is_production = os.environ.get("UNLEASHED_ENV", "development") == "production"

    def load(self) -> ValidationResult:
        """Load configuration from all sources."""
        # Start with defaults
        for field_def in self.schema.fields:
            if field_def.default is not None:
                self._config[field_def.name] = field_def.default
                self._sources[field_def.name] = ConfigSource.DEFAULT

        # Load from file if specified
        if self.config_file and Path(self.config_file).exists():
            self._load_from_file()

        # Load from environment (highest priority)
        env_config = self.env_loader.load_config(self.schema)
        for key, value in env_config.items():
            self._config[key] = value
            self._sources[key] = ConfigSource.ENVIRONMENT

        self._last_loaded = datetime.now(timezone.utc)

        # Validate
        return self.validate()

    def _load_from_file(self) -> None:
        """Load configuration from file."""
        if not self.config_file:
            return

        path = Path(self.config_file)
        if not path.exists():
            return

        with open(path, "r") as f:
            if path.suffix == ".json":
                file_config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")

        for key, value in file_config.items():
            self._config[key] = value
            self._sources[key] = ConfigSource.FILE

    def validate(self) -> ValidationResult:
        """Validate current configuration."""
        return self.schema.validate(self._config, self._is_production)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any, source: ConfigSource = ConfigSource.OVERRIDE) -> None:
        """Set a configuration value."""
        self._config[key] = value
        self._sources[key] = source

    def get_all(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Get all configuration values."""
        result = {}
        for field_def in self.schema.fields:
            value = self._config.get(field_def.name)
            if field_def.secret and not include_secrets:
                result[field_def.name] = "[REDACTED]" if value else None
            else:
                result[field_def.name] = value
        return result

    def get_source(self, key: str) -> Optional[ConfigSource]:
        """Get the source of a configuration value."""
        return self._sources.get(key)

    def export_template(self) -> str:
        """Export a template configuration file."""
        template = {
            "_schema": self.schema.name,
            "_version": self.schema.version,
            "_description": self.schema.description
        }

        for field_def in self.schema.fields:
            if field_def.secret:
                template[field_def.name] = f"<{field_def.env_var or field_def.name.upper()}>"
            else:
                template[field_def.name] = field_def.default

        return json.dumps(template, indent=2)


# ============================================================================
# Pre-defined Schemas
# ============================================================================

def create_adapter_config_schema() -> ConfigSchema:
    """Create configuration schema for SDK adapters."""
    return ConfigSchema(
        name="adapter_config",
        version="1.0",
        description="Configuration for SDK adapters",
        fields=[
            ConfigField(
                name="openai_api_key",
                type=str,
                secret=True,
                env_var="OPENAI_API_KEY",
                validators=[ProductionRequiredValidator()],
                description="OpenAI API key"
            ),
            ConfigField(
                name="anthropic_api_key",
                type=str,
                secret=True,
                env_var="ANTHROPIC_API_KEY",
                validators=[ProductionRequiredValidator()],
                description="Anthropic API key"
            ),
            ConfigField(
                name="exa_api_key",
                type=str,
                secret=True,
                env_var="EXA_API_KEY",
                description="Exa search API key"
            ),
            ConfigField(
                name="firecrawl_api_key",
                type=str,
                secret=True,
                env_var="FIRECRAWL_API_KEY",
                description="Firecrawl API key"
            ),
            ConfigField(
                name="default_model",
                type=str,
                default="gpt-4o",
                validators=[ChoiceValidator([
                    "gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20241022",
                    "claude-3-opus-20240229", "claude-3-haiku-20240307"
                ])],
                description="Default LLM model"
            ),
            ConfigField(
                name="request_timeout",
                type=int,
                default=60,
                validators=[RangeValidator(1, 300)],
                description="Request timeout in seconds"
            ),
            ConfigField(
                name="max_retries",
                type=int,
                default=3,
                validators=[RangeValidator(0, 10)],
                description="Maximum retry attempts"
            ),
            ConfigField(
                name="cache_enabled",
                type=bool,
                default=True,
                description="Enable response caching"
            ),
            ConfigField(
                name="cache_ttl",
                type=int,
                default=3600,
                validators=[RangeValidator(60, 86400)],
                description="Cache TTL in seconds"
            ),
        ]
    )


def create_platform_config_schema() -> ConfigSchema:
    """Create configuration schema for the platform."""
    return ConfigSchema(
        name="platform_config",
        version="1.0",
        description="Configuration for the Unleashed platform",
        fields=[
            ConfigField(
                name="environment",
                type=str,
                default="development",
                env_var="UNLEASHED_ENV",
                validators=[ChoiceValidator(["development", "staging", "production"])],
                description="Deployment environment"
            ),
            ConfigField(
                name="log_level",
                type=str,
                default="INFO",
                validators=[ChoiceValidator(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])],
                description="Logging level"
            ),
            ConfigField(
                name="log_format",
                type=str,
                default="json",
                validators=[ChoiceValidator(["json", "text"])],
                description="Log format"
            ),
            ConfigField(
                name="metrics_enabled",
                type=bool,
                default=True,
                description="Enable metrics collection"
            ),
            ConfigField(
                name="tracing_enabled",
                type=bool,
                default=True,
                description="Enable distributed tracing"
            ),
            ConfigField(
                name="security_level",
                type=str,
                default="standard",
                validators=[ChoiceValidator(["permissive", "standard", "strict", "paranoid"])],
                description="Security enforcement level"
            ),
            ConfigField(
                name="rate_limit_requests",
                type=int,
                default=1000,
                validators=[RangeValidator(100, 100000)],
                description="Rate limit requests per hour"
            ),
            ConfigField(
                name="max_concurrent_tasks",
                type=int,
                default=10,
                validators=[RangeValidator(1, 100)],
                description="Maximum concurrent tasks"
            ),
        ]
    )


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Enums
    "ValidationSeverity",
    "ConfigSource",

    # Data structures
    "ValidationIssue",
    "ValidationResult",
    "ConfigField",
    "ConfigSchema",

    # Validators
    "Validator",
    "RequiredValidator",
    "RangeValidator",
    "PatternValidator",
    "URLValidator",
    "ChoiceValidator",
    "PathValidator",
    "ProductionRequiredValidator",
    "SecureValueValidator",

    # Core
    "EnvironmentLoader",
    "ConfigurationManager",

    # Pre-defined schemas
    "create_adapter_config_schema",
    "create_platform_config_schema",
]
