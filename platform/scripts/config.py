#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.5.0",
#     "pydantic-settings>=2.1.0",
# ]
# ///
"""
Platform Configuration - Environment-aware Configuration Management

Provides centralized configuration for the Ultimate Autonomous Platform.
Supports multiple environments (development, staging, production) with
environment variable overrides.

Features:
- Type-safe configuration with Pydantic
- Environment variable support (UAP_ prefix)
- Hierarchical configuration (defaults → env-specific → env vars)
- Secrets management integration
- Configuration validation

Usage:
    from config import get_config, Environment

    config = get_config()  # Auto-detects environment
    print(config.qdrant.url)
    print(config.neo4j.uri)
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""
    url: str = "http://localhost:6333"
    api_key: Optional[SecretStr] = None
    collection_prefix: str = "uap_"
    timeout_seconds: float = 30.0
    grpc_port: int = 6334


class Neo4jConfig(BaseModel):
    """Neo4j graph database configuration."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: SecretStr = SecretStr("alphaforge2024")
    database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50


class LettaConfig(BaseModel):
    """Letta memory server configuration."""
    url: str = os.environ.get("LETTA_URL", "http://localhost:8500")
    api_key: Optional[SecretStr] = None
    default_agent_name: str = "uap_agent"
    memory_human: str = "User"
    memory_persona: str = "Platform Assistant"


class RedisConfig(BaseModel):
    """Redis cache configuration."""
    url: str = "redis://localhost:6379"
    password: Optional[SecretStr] = None
    db: int = 0
    socket_timeout: float = 5.0
    max_connections: int = 10


class AutoClaudeConfig(BaseModel):
    """Auto-Claude IDE backend configuration."""
    url: str = "http://localhost:3000"
    api_key: Optional[SecretStr] = None
    timeout_seconds: float = 60.0
    enabled: bool = True


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4
    shutdown_timeout: float = 30.0
    drain_timeout: float = 5.0


class ResilienceConfig(BaseModel):
    """Resilience pattern configuration."""
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 30.0
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0
    bulkhead_health_check_concurrent: int = 10
    bulkhead_swarm_tasks_concurrent: int = 50
    bulkhead_knowledge_graph_concurrent: int = 20


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    api_rate_per_second: float = 100.0
    api_burst_capacity: float = 200.0
    database_rate_per_second: float = 50.0
    external_rate_per_second: float = 10.0


class TracingConfig(BaseModel):
    """Distributed tracing configuration."""
    enabled: bool = True
    service_name: str = "ultimate-autonomous-platform"
    service_version: str = "1.0.0"
    otlp_endpoint: Optional[str] = None
    console_export: bool = True
    sampling_rate: float = 1.0


class MetricsConfig(BaseModel):
    """Prometheus metrics configuration."""
    enabled: bool = True
    port: int = 9090
    path: str = "/metrics"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"  # "json" or "console"
    include_timestamp: bool = True


class PlatformConfig(BaseSettings):
    """
    Main platform configuration.

    Environment variables are read with UAP_ prefix:
    - UAP_ENVIRONMENT=production
    - UAP_QDRANT__URL=http://qdrant:6333
    - UAP_NEO4J__PASSWORD=secret
    """

    model_config = SettingsConfigDict(
        env_prefix="UAP_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )

    # Environment
    environment: Environment = Environment.DEVELOPMENT

    # Component configs
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    letta: LettaConfig = Field(default_factory=LettaConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    auto_claude: AutoClaudeConfig = Field(default_factory=AutoClaudeConfig)

    # Server config
    server: ServerConfig = Field(default_factory=ServerConfig)

    # Feature configs
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v):
        """Allow string environment values."""
        if isinstance(v, str):
            return Environment(v.lower())
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT

    def get_component_urls(self) -> Dict[str, str]:
        """Get all component URLs for health checks."""
        return {
            "qdrant": self.qdrant.url,
            "neo4j": self.neo4j.uri,
            "letta": self.letta.url,
            "redis": self.redis.url,
            "auto_claude": self.auto_claude.url,
        }


# Environment-specific configuration overrides
_ENV_OVERRIDES: Dict[Environment, Dict[str, Any]] = {
    Environment.PRODUCTION: {
        "server": {"workers": 8},
        "resilience": {
            "circuit_breaker_failure_threshold": 3,
            "circuit_breaker_recovery_timeout": 60.0,
        },
        "tracing": {"console_export": False, "sampling_rate": 0.1},
        "logging": {"level": "WARNING"},
    },
    Environment.STAGING: {
        "server": {"workers": 4},
        "tracing": {"sampling_rate": 0.5},
        "logging": {"level": "INFO"},
    },
    Environment.TESTING: {
        "server": {"workers": 1},
        "tracing": {"enabled": False},
        "metrics": {"enabled": False},
    },
}


@lru_cache()
def get_config(environment: Optional[Environment] = None) -> PlatformConfig:
    """
    Get platform configuration.

    Configuration is loaded in this order (later overrides earlier):
    1. Default values
    2. Environment-specific overrides
    3. Environment variables (UAP_ prefix)

    Args:
        environment: Override environment detection

    Returns:
        Configured PlatformConfig instance
    """
    # Detect environment from env var if not specified
    if environment is None:
        env_str = os.getenv("UAP_ENVIRONMENT", "development")
        try:
            environment = Environment(env_str.lower())
        except ValueError:
            environment = Environment.DEVELOPMENT

    # Create base config (picks up env vars automatically)
    config = PlatformConfig(environment=environment)

    # Apply environment-specific overrides
    if environment in _ENV_OVERRIDES:
        overrides = _ENV_OVERRIDES[environment]
        for section, values in overrides.items():
            if hasattr(config, section):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)

    return config


def validate_config(config: PlatformConfig) -> Dict[str, Any]:
    """
    Validate configuration and return issues.

    Returns:
        Dict with validation results
    """
    issues = []
    warnings = []

    # Check for production security
    if config.is_production:
        if config.neo4j.password.get_secret_value() == "alphaforge2024":
            issues.append("Neo4j using default password in production")

        if config.tracing.console_export:
            warnings.append("Tracing console export enabled in production")

        if config.logging.level == "DEBUG":
            warnings.append("Debug logging enabled in production")

    # Check service connectivity
    urls = config.get_component_urls()
    for name, url in urls.items():
        if not url:
            issues.append(f"{name} URL not configured")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "environment": config.environment.value,
    }


def print_config_summary(config: PlatformConfig) -> None:
    """Print configuration summary."""
    print("=" * 60)
    print("PLATFORM CONFIGURATION")
    print("=" * 60)
    print(f"Environment: {config.environment.value.upper()}")
    print()

    print("Component URLs:")
    for name, url in config.get_component_urls().items():
        print(f"  {name:<12}: {url}")
    print()

    print("Server:")
    print(f"  Host:     {config.server.host}:{config.server.port}")
    print(f"  Workers:  {config.server.workers}")
    print()

    print("Features:")
    print(f"  Tracing:  {'Enabled' if config.tracing.enabled else 'Disabled'}")
    print(f"  Metrics:  {'Enabled' if config.metrics.enabled else 'Disabled'}")
    print(f"  Sampling: {config.tracing.sampling_rate:.0%}")
    print()

    # Validation
    validation = validate_config(config)
    if not validation["valid"]:
        print("Issues:")
        for issue in validation["issues"]:
            print(f"  [!] {issue}")
    if validation["warnings"]:
        print("Warnings:")
        for warning in validation["warnings"]:
            print(f"  [?] {warning}")


def main():
    """Demo configuration management."""
    print("[>>] Configuration Management Demo")
    print()

    # Test default (development) config
    config = get_config()
    print_config_summary(config)
    print()

    # Test environment variable override
    print("[>>] Testing environment variable override...")
    os.environ["UAP_ENVIRONMENT"] = "production"
    os.environ["UAP_QDRANT__URL"] = "http://qdrant-prod:6333"

    # Clear cache to get fresh config
    get_config.cache_clear()
    prod_config = get_config()

    print(f"Environment: {prod_config.environment.value}")
    print(f"Qdrant URL:  {prod_config.qdrant.url}")
    print()

    # Cleanup
    del os.environ["UAP_ENVIRONMENT"]
    del os.environ["UAP_QDRANT__URL"]
    get_config.cache_clear()

    print("[OK] Configuration demo complete")


if __name__ == "__main__":
    main()
