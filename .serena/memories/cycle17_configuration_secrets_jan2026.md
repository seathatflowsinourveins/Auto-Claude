# Cycle 17: Configuration Management & Secrets (January 2026)

## CONFIGURATION MANAGEMENT

### Pydantic Settings v2
**The standard for type-safe configuration in Python**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_nested_delimiter='__',  # DATABASE__HOST → database.host
        extra='ignore'
    )
    
    # Type-safe with validation
    database_url: str = Field(..., description="PostgreSQL connection string")
    api_key: SecretStr  # Never logged, repr shows '**********'
    debug: bool = False
    max_connections: int = Field(default=10, ge=1, le=100)
    
    # Nested settings
    redis: RedisSettings = Field(default_factory=RedisSettings)

# Usage - loads from environment automatically
settings = AppSettings()
```

**Key patterns**:
- `SecretStr` for credentials (never exposed in logs/repr)
- `env_nested_delimiter` for structured config
- Validation via Pydantic constraints (ge, le, regex, etc.)
- Multiple sources: .env files, environment variables, secrets managers

### dynaconf v3 (2025)
**Multi-environment configuration management**

```python
from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="MYAPP",           # MYAPP_DEBUG=true
    settings_files=['settings.toml', '.secrets.toml'],
    environments=True,                # [development], [production]
    load_dotenv=True,
    merge_enabled=True,               # Deep merge configs
)

# Environment switching
# ENV_FOR_DYNACONF=production python app.py

# Access with defaults
debug = settings.get('DEBUG', False)
database = settings.DATABASE.HOST  # Nested access
```

**Configuration layering** (priority order):
1. Environment variables (highest)
2. .secrets.toml (git-ignored)
3. settings.local.toml
4. settings.toml
5. defaults.toml (lowest)

## SECRETS MANAGEMENT

### Doppler - Universal Secrets Platform
**Single source of truth across environments**

```python
# Install: pip install doppler-sdk
from doppler_sdk import DopplerSDK

doppler = DopplerSDK()
doppler.access_token = os.environ["DOPPLER_TOKEN"]

# Fetch secrets for specific config
secrets = doppler.secrets.list(
    project="my-project",
    config="production"  # dev, staging, production
)

# Integration with Pydantic Settings
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        secrets_dir='/run/secrets'  # Docker secrets mount
    )
```

**Doppler CLI for local development**:
```bash
# Run with injected secrets
doppler run -- python app.py

# Sync to .env for offline
doppler secrets download --no-file --format env > .env
```

### AWS Secrets Manager Best Practices

```python
import boto3
from botocore.exceptions import ClientError
import json

def get_secret(secret_name: str, region: str = "us-east-1") -> dict:
    """Retrieve and cache secrets from AWS Secrets Manager."""
    client = boto3.client("secretsmanager", region_name=region)
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response["SecretString"])
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            raise ValueError(f"Secret {secret_name} not found")
        raise

# With caching (recommended for Lambda)
from aws_lambda_powertools.utilities.parameters import get_secret

@get_secret(name="my-api-key", max_age=300)  # 5-minute cache
def handler(event, context):
    api_key = get_secret("my-api-key")
```

**Best practices**:
- **Rotation**: Enable automatic rotation for database credentials
- **Versioning**: Use AWSCURRENT/AWSPREVIOUS for zero-downtime rotation
- **Caching**: Always cache secrets (5-15 minute TTL typical)
- **IAM**: Least-privilege access with resource-based policies
- **Cross-account**: Use resource policies for multi-account access

### HashiCorp Vault Patterns

```python
import hvac

client = hvac.Client(url='https://vault.example.com:8200')
client.token = os.environ['VAULT_TOKEN']

# KV secrets engine (v2)
secret = client.secrets.kv.v2.read_secret_version(
    path='myapp/database',
    mount_point='secret'
)
db_password = secret['data']['data']['password']

# Dynamic database credentials (auto-rotating)
creds = client.secrets.database.generate_credentials(
    name='my-role',
    mount_point='database'
)
# creds['data']['username'], creds['data']['password']
# Automatically revoked after TTL
```

## FEATURE FLAGS

### OpenFeature - CNCF Standard
**Vendor-neutral feature flagging API**

```python
from openfeature import api
from openfeature.contrib.provider.flagd import FlagdProvider

# Set provider (LaunchDarkly, Flagsmith, flagd, etc.)
api.set_provider(FlagdProvider())

# Get client for namespace
client = api.get_client("my-service")

# Evaluate flags with context
context = EvaluationContext(
    targeting_key="user-123",
    attributes={
        "email": "user@example.com",
        "tier": "premium"
    }
)

# Type-safe evaluations
enabled = client.get_boolean_value("new-feature", False, context)
variant = client.get_string_value("checkout-flow", "control", context)
limit = client.get_integer_value("rate-limit", 100, context)
```

### LaunchDarkly 2026 Patterns

```python
import ldclient
from ldclient.config import Config

ldclient.set_config(Config("sdk-key-xxx"))
client = ldclient.get()

# Feature evaluation with user context
user = ldclient.Context.builder("user-123").kind("user").set("email", "user@example.com").set("plan", "enterprise").build()

# Boolean flag
if client.variation("new-dashboard", user, False):
    show_new_dashboard()

# Multivariate flag
variant = client.variation("pricing-experiment", user, "control")
# Returns: "control", "variant-a", or "variant-b"

# Track custom events for experimentation
client.track("purchase-completed", user, data={"amount": 99.99})
```

**Feature flag best practices**:
- **Short-lived flags**: Remove after rollout complete (flag debt)
- **Kill switches**: Always have instant-off capability for new features
- **Targeting rules**: Start narrow (internal → beta → 10% → 100%)
- **Fallback defaults**: Always provide sensible defaults
- **Audit trail**: Log flag evaluations for debugging

## CONFIGURATION PATTERNS

### 12-Factor App Configuration
```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Required - fail fast if missing
    database_url: str
    redis_url: str
    
    # Optional with defaults
    log_level: str = "INFO"
    workers: int = 4
    
    # Feature flags as config
    enable_new_feature: bool = False

@lru_cache()  # Singleton pattern
def get_settings() -> Settings:
    return Settings()

# Dependency injection (FastAPI)
from fastapi import Depends

def get_db(settings: Settings = Depends(get_settings)):
    return create_engine(settings.database_url)
```

### Environment-Specific Configuration
```toml
# settings.toml
[default]
debug = false
log_level = "INFO"

[development]
debug = true
log_level = "DEBUG"
database_url = "postgresql://localhost/dev"

[production]
log_level = "WARNING"
database_url = "@vault secrets/data/prod/database:url"
```

### Secrets in Ephemeral Environments
```yaml
# Kubernetes with External Secrets Operator
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: app-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  target:
    name: app-secrets
    creationPolicy: Owner
  data:
    - secretKey: DATABASE_URL
      remoteRef:
        key: prod/database
        property: url
```

## KEY PATTERNS SUMMARY

| Pattern | Tool | Use Case |
|---------|------|----------|
| Type-safe config | Pydantic Settings | Application configuration |
| Multi-env config | dynaconf | Environment switching |
| Secrets injection | Doppler | Universal secrets sync |
| Cloud secrets | AWS Secrets Manager | AWS-native apps |
| Dynamic secrets | HashiCorp Vault | Auto-rotating credentials |
| Feature flags | OpenFeature + LaunchDarkly | Progressive rollouts |
| K8s secrets | External Secrets Operator | Cloud-native orchestration |

## ANTI-PATTERNS TO AVOID

1. **Hardcoded secrets**: Never commit credentials to git
2. **Secrets in environment files**: .env files get committed accidentally
3. **Long-lived credentials**: Prefer short TTL with auto-rotation
4. **Logging secrets**: Use SecretStr, redact in structured logs
5. **Feature flag debt**: Remove flags after full rollout
6. **Missing defaults**: Always provide fallback values
7. **Synchronous secret fetching**: Cache secrets, refresh async

---
*Cycle 17 - Configuration Management & Secrets - January 2026*
