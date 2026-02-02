# Python Configuration Management Patterns (January 2026)

Production-grade configuration management patterns from official documentation research.

## 1. Pydantic Settings (pydantic-settings 2.7+)

### Installation
```bash
pip install pydantic-settings
```

### Basic Settings Class
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """Application settings with automatic env var loading."""
    
    # Simple fields - loaded from APP_NAME, APP_DEBUG env vars
    app_name: str = "MyApp"
    debug: bool = False
    
    # With validation alias (multiple env var names)
    api_key: str = Field(validation_alias="MY_API_KEY")
    
    # Nested configuration
    database_url: str = Field(default="sqlite:///app.db")
    redis_url: str = "redis://localhost:6379/0"
    
    # Configuration
    model_config = SettingsConfigDict(
        env_prefix="APP_",           # All vars prefixed with APP_
        env_file=".env",             # Load from .env file
        env_file_encoding="utf-8",
        case_sensitive=False,        # APP_DEBUG = app_debug
        extra="ignore",              # Ignore unknown env vars
    )

# Usage
settings = Settings()
print(settings.app_name)  # From APP_APP_NAME or default
```

### Nested Settings with Delimiter
```python
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    name: str = "mydb"
    user: str = "postgres"
    password: str = ""

class RedisSettings(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0

class Settings(BaseSettings):
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_nested_delimiter="__",  # APP_DATABASE__HOST
        env_file=".env",
    )

# Environment variables:
# APP_DATABASE__HOST=db.example.com
# APP_DATABASE__PORT=5432
# APP_REDIS__HOST=cache.example.com
```

### Multiple Env File Sources
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    api_key: str
    database_url: str
    
    model_config = SettingsConfigDict(
        # Load in order (later files override earlier)
        env_file=(".env.defaults", ".env", ".env.local"),
        env_file_encoding="utf-8",
    )
```

### Validation Aliases (Multiple Names)
```python
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Accept any of these env var names
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias=AliasChoices(
            "REDIS_URL",
            "REDIS_DSN", 
            "CACHE_URL"
        )
    )
    
    # With prefix still applied
    api_key: str = Field(validation_alias="MY_SECRET_API_KEY")
```

### Secrets from Files (Docker/K8s)
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_password: str
    api_key: str
    
    model_config = SettingsConfigDict(
        # Read secrets from /run/secrets/ directory
        secrets_dir="/run/secrets",
        env_file=".env",
    )

# File: /run/secrets/database_password contains the secret value
# This is the pattern for Docker secrets and K8s secrets
```

### Custom Settings Sources
```python
from typing import Any, Dict, Tuple, Type
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

class JsonConfigSettingsSource(PydanticBaseSettingsSource):
    """Load settings from a JSON file."""
    
    def get_field_value(
        self, field, field_name
    ) -> Tuple[Any, str, bool]:
        # Return (value, field_key, is_complex)
        import json
        with open("config.json") as f:
            data = json.load(f)
        return data.get(field_name), field_name, False
    
    def __call__(self) -> Dict[str, Any]:
        import json
        with open("config.json") as f:
            return json.load(f)

class Settings(BaseSettings):
    app_name: str
    debug: bool = False
    
    model_config = SettingsConfigDict(env_file=".env")
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,           # Highest priority
            env_settings,            # Environment variables
            dotenv_settings,         # .env file
            JsonConfigSettingsSource(settings_cls),  # Custom JSON
            file_secret_settings,    # Secrets directory
        )
```

### Settings with Computed Fields
```python
from pydantic import computed_field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "mydb"
    db_user: str = "postgres"
    db_password: str = ""
    
    @computed_field
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
```

### FastAPI Integration Pattern
```python
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "My API"
    debug: bool = False
    database_url: str
    redis_url: str = "redis://localhost:6379/0"

@lru_cache
def get_settings() -> Settings:
    """Cached settings instance (singleton pattern)."""
    return Settings()

# In FastAPI
from fastapi import Depends, FastAPI

app = FastAPI()

@app.get("/info")
def get_info(settings: Settings = Depends(get_settings)):
    return {"app_name": settings.app_name, "debug": settings.debug}
```

## 2. Dynaconf (3.2.11+)

### Installation & Initialization
```bash
pip install dynaconf
cd your_project/
dynaconf init -f toml  # Creates config.py, settings.toml, .secrets.toml
```

### Basic Usage
```python
# config.py (generated)
from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=["settings.toml", ".secrets.toml"],
    envvar_prefix="MYAPP",  # Export MYAPP_DEBUG=true
    environments=True,       # Enable [development], [production] sections
    load_dotenv=True,        # Load .env file
)

# your_app.py
from config import settings

print(settings.DEBUG)  # Access settings
print(settings.DATABASE.HOST)  # Nested access
print(settings.get("OPTIONAL", default="fallback"))
```

### Settings Files (TOML Recommended)
```toml
# settings.toml
[default]
debug = false
app_name = "MyApp"

[default.database]
host = "localhost"
port = 5432
name = "mydb"

[development]
debug = true

[development.database]
host = "localhost"
name = "mydb_dev"

[production]
debug = false

[production.database]
host = "db.prod.example.com"
name = "mydb_prod"
```

```toml
# .secrets.toml (add to .gitignore!)
[development]
database_password = "dev_secret"
api_key = "dev_key_123"

[production]
database_password = "@vault secrets/prod/db:password"
api_key = "@vault secrets/prod/api:key"
```

### Environment Switching
```bash
# Switch environments via env var
export ENV_FOR_DYNACONF=production
python app.py

# Or via DYNACONF_ prefix
export DYNACONF_DEBUG=false
export DYNACONF_DATABASE__HOST=custom.db.com  # Nested with __
```

### Validation
```python
from dynaconf import Dynaconf, Validator

settings = Dynaconf(
    settings_files=["settings.toml"],
    validators=[
        # Must exist
        Validator("DATABASE.HOST", must_exist=True),
        Validator("DATABASE.PORT", must_exist=True),
        
        # Type and range validation
        Validator("DATABASE.PORT", gte=1, lte=65535),
        Validator("DEBUG", is_type_of=bool),
        
        # Conditional validation
        Validator("API_KEY", must_exist=True, when=Validator("ENV", eq="production")),
        
        # Default values
        Validator("TIMEOUT", default=30),
        
        # Combined conditions
        Validator("HOST", must_exist=True) | Validator("BIND", must_exist=True),
    ]
)

# Explicit validation (useful in CI)
settings.validators.validate()
```

### CLI Commands
```bash
# List all settings
dynaconf -i config.settings list

# Write a setting
dynaconf -i config.settings write --env development DEBUG=true

# Validate settings
dynaconf -i config.settings validate

# Export to different formats
dynaconf -i config.settings list --output json > config.json
```

### Flask Integration
```python
from flask import Flask
from dynaconf import FlaskDynaconf

app = Flask(__name__)
FlaskDynaconf(app, settings_files=["settings.toml"])

# Access via app.config
@app.route("/")
def index():
    return {"debug": app.config.DEBUG}
```

### Django Integration
```python
# settings.py (bottom of file)
import dynaconf
settings = dynaconf.DjangoDynaconf(__name__)
```

```bash
# Switch Django environment
export DJANGO_ENV=production
python manage.py runserver
```

### Vault Integration (Secrets)
```python
from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=["settings.toml"],
    vault_enabled=True,
    vault_url="https://vault.example.com",
    vault_token="s.xxxxx",  # Or use VAULT_TOKEN env var
    vault_path="secret/data/myapp",
)

# In settings.toml, reference vault secrets
# api_key = "@vault secret/data/myapp:api_key"
```

### Redis Integration (Dynamic Settings)
```python
from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=["settings.toml"],
    redis_enabled=True,
    redis_host="localhost",
    redis_port=6379,
    redis_db=0,
)

# Settings can be updated at runtime in Redis
# and Dynaconf will pick them up
```

## 3. python-dotenv (1.0.1+)

### Basic Usage
```python
from dotenv import load_dotenv
import os

# Load .env file into os.environ
load_dotenv()

# Access variables
database_url = os.getenv("DATABASE_URL")
api_key = os.getenv("API_KEY", "default_key")
```

### Advanced Loading Options
```python
from dotenv import load_dotenv
from pathlib import Path

# Explicit path
load_dotenv(Path("/path/to/.env"))

# Override existing env vars
load_dotenv(override=True)

# Multiple .env files (manual layering)
load_dotenv(".env.defaults")
load_dotenv(".env")  # Overrides defaults
load_dotenv(".env.local", override=True)  # Overrides everything
```

### Without Modifying Environment
```python
from dotenv import dotenv_values

# Returns dict without touching os.environ
config = dotenv_values(".env")

# Advanced layering pattern
import os
from dotenv import dotenv_values

config = {
    **dotenv_values(".env.shared"),   # Shared defaults
    **dotenv_values(".env.secret"),   # Secrets (gitignored)
    **os.environ,                      # OS env vars override all
}
```

### .env File Format
```bash
# .env file

# Simple key-value
DATABASE_URL=postgresql://localhost/mydb
DEBUG=true

# Quoted values (preserve spaces)
APP_NAME="My Application"
MESSAGE='Hello World'

# Multiline values
PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
-----END RSA PRIVATE KEY-----"

# Variable expansion
DOMAIN=example.com
API_URL=https://${DOMAIN}/api
ADMIN_EMAIL=admin@${DOMAIN}

# Export directive (optional, ignored)
export SECRET_KEY=mysecret

# Comments
# This is a comment
API_KEY=abc123  # Inline comment NOT supported
```

### CLI Usage
```bash
pip install "python-dotenv[cli]"

# Set values
dotenv set DATABASE_URL postgresql://localhost/mydb
dotenv set DEBUG true

# List values
dotenv list
dotenv list --format=json

# Run command with .env loaded
dotenv run -- python app.py
dotenv run -- flask run
```

## 4. Combined Patterns

### Production Configuration Stack
```python
# config.py - Complete production pattern
from functools import lru_cache
from typing import Optional
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

class DatabaseSettings(BaseSettings):
    host: str = "localhost"
    port: int = 5432
    name: str = "mydb"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 5
    pool_timeout: int = 30
    
    model_config = SettingsConfigDict(env_prefix="DB_")
    
    @computed_field
    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

class RedisSettings(BaseSettings):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    
    @computed_field
    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"

class Settings(BaseSettings):
    # Application
    app_name: str = "MyApp"
    environment: str = Field(default="development", alias="ENV")
    debug: bool = False
    log_level: str = "INFO"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Security
    secret_key: str
    allowed_hosts: list[str] = ["*"]
    cors_origins: list[str] = ["http://localhost:3000"]
    
    # Nested configs
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    
    model_config = SettingsConfigDict(
        env_file=(".env.defaults", ".env", ".env.local"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

@lru_cache
def get_settings() -> Settings:
    return Settings()

# Usage
settings = get_settings()
```

### Environment-Specific Configuration
```python
# config/__init__.py
import os
from pathlib import Path

ENV = os.getenv("ENV", "development")
CONFIG_DIR = Path(__file__).parent

def get_env_file() -> Path:
    """Get environment-specific config file."""
    env_file = CONFIG_DIR / f".env.{ENV}"
    if env_file.exists():
        return env_file
    return CONFIG_DIR / ".env"
```

### Docker Secrets Pattern
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # These will be read from /run/secrets/ in Docker
    db_password: str
    api_key: str
    jwt_secret: str
    
    # Regular env vars
    db_host: str = "localhost"
    db_port: int = 5432
    
    model_config = SettingsConfigDict(
        secrets_dir="/run/secrets",
        env_file=".env",
    )

# docker-compose.yml
# secrets:
#   db_password:
#     file: ./secrets/db_password.txt
# services:
#   app:
#     secrets:
#       - db_password
```

## Quick Reference

### When to Use Each Tool

| Tool | Best For |
|------|----------|
| **Pydantic Settings** | Type-safe configs, FastAPI apps, validation |
| **Dynaconf** | Multi-environment, Flask/Django, Vault/Redis |
| **python-dotenv** | Simple .env loading, scripts, legacy apps |

### Best Practices

```python
# 1. Always validate required settings on startup
settings = Settings()  # Pydantic validates immediately

# 2. Use computed fields for derived values
@computed_field
@property
def database_url(self) -> str: ...

# 3. Cache settings instance
@lru_cache
def get_settings() -> Settings:
    return Settings()

# 4. Layer configs appropriately
env_file=(".env.defaults", ".env", ".env.local")

# 5. Keep secrets in separate files
# .secrets.toml, /run/secrets/, .env.local
```

### Anti-Patterns to Avoid

```python
# BAD: Hardcoded secrets
API_KEY = "sk-abc123"

# BAD: No validation
config = dotenv_values(".env")
port = int(config.get("PORT", "8000"))  # Could fail!

# BAD: Global mutable settings
settings = Settings()
settings.debug = True  # Mutation!

# BAD: No default for optional settings
required_key = os.getenv("MAYBE_SET")  # Returns None!

# GOOD: Explicit defaults and validation
class Settings(BaseSettings):
    maybe_set: str = "default_value"
```

## Version Reference

- **pydantic-settings**: 2.7.0+ (latest stable)
- **dynaconf**: 3.2.11+ (latest stable)
- **python-dotenv**: 1.0.1+ (latest stable)
- **Python**: 3.10+ recommended
