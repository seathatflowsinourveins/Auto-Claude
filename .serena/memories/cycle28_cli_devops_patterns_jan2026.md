# Cycle 28: CLI & DevOps Patterns (January 25, 2026)

**Focus**: Production CLI development, CI/CD automation, containerization

---

## 1. Python CLI Development (Typer/Click)

### Typer Fundamentals (Type-Hint Powered)
```python
import typer
from typing import Annotated, Optional
from pathlib import Path

app = typer.Typer(help="Production CLI application")

@app.command()
def process(
    input_file: Annotated[Path, typer.Argument(help="Input file path")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
):
    """Process input file with optional verbosity."""
    if verbose:
        typer.echo(f"Processing {input_file}")
    # Implementation
```

### Command Groups (Subcommands)
```python
app = typer.Typer()
users_app = typer.Typer()
app.add_typer(users_app, name="users", help="User management")

@users_app.command("list")
def list_users(): ...

@users_app.command("create")
def create_user(name: str): ...

# Usage: cli users list, cli users create "John"
```

### Context Object Pattern (Shared State)
```python
class Context:
    def __init__(self):
        self.verbose = False
        self.config: dict = {}

@app.callback()
def main(ctx: typer.Context, verbose: bool = False, config: Path = None):
    """Initialize context for all commands."""
    ctx.ensure_object(Context)
    ctx.obj.verbose = verbose
    if config:
        ctx.obj.config = load_config(config)

@app.command()
def run(ctx: typer.Context):
    if ctx.obj.verbose:
        typer.echo("Verbose mode enabled")
```

### Environment-Based Configuration
```python
from pydantic_settings import BaseSettings

class CLISettings(BaseSettings):
    api_key: str
    debug: bool = False
    
    class Config:
        env_prefix = "MYAPP_"

# MYAPP_API_KEY=xxx MYAPP_DEBUG=true cli run
```

### CLI Testing (CliRunner)
```python
from typer.testing import CliRunner

runner = CliRunner()

def test_process_command():
    result = runner.invoke(app, ["process", "input.txt", "--verbose"])
    assert result.exit_code == 0
    assert "Processing" in result.stdout

def test_missing_argument():
    result = runner.invoke(app, ["process"])
    assert result.exit_code != 0
    assert "Missing argument" in result.stdout
```

### Packaging with Entry Points
```toml
# pyproject.toml
[project.scripts]
myapp = "myapp.cli:app"

# Or for Click
[project.scripts]
myapp = "myapp.cli:main"
```

---

## 2. GitHub Actions CI/CD Patterns

### Reusable Workflow Definition
```yaml
# .github/workflows/reusable-build.yml
name: Reusable Build

on:
  workflow_call:
    inputs:
      python-version:
        required: false
        type: string
        default: "3.11"
      environment:
        required: true
        type: string
    secrets:
      deploy-key:
        required: true
    outputs:
      artifact-name:
        value: ${{ jobs.build.outputs.artifact }}

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      artifact: ${{ steps.build.outputs.name }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ inputs.python-version }}
      - id: build
        run: echo "name=build-${{ github.sha }}" >> $GITHUB_OUTPUT
```

### Calling Reusable Workflows
```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  build:
    uses: ./.github/workflows/reusable-build.yml
    with:
      python-version: "3.12"
      environment: production
    secrets:
      deploy-key: ${{ secrets.DEPLOY_KEY }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - run: echo "Deploying ${{ needs.build.outputs.artifact-name }}"
```

### Composite Actions (Reusable Steps)
```yaml
# .github/actions/python-setup/action.yml
name: Python Setup
description: Setup Python with caching

inputs:
  python-version:
    default: "3.11"

runs:
  using: composite
  steps:
    - uses: actions/setup-python@v5
      with:
        python-version: ${{ inputs.python-version }}
    
    - uses: actions/cache@v4
      with:
        path: ~/.cache/pip
        key: pip-${{ runner.os }}-${{ hashFiles('**/requirements*.txt') }}
    
    - shell: bash
      run: pip install -r requirements.txt
```

### Matrix Strategy with Fail-Fast
```yaml
jobs:
  test:
    strategy:
      fail-fast: false  # Continue other jobs if one fails
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ["3.10", "3.11", "3.12"]
        exclude:
          - os: windows-latest
            python: "3.10"
    runs-on: ${{ matrix.os }}
```

### Workflow Versioning
```yaml
# Pin to specific commit (most secure)
uses: actions/checkout@8ade135a41bc03ea155e62e844d188df1ea18608

# Pin to major version (balance of security and updates)
uses: actions/checkout@v4

# AVOID: Floating tag (security risk)
# uses: actions/checkout@main
```

### Concurrency Control
```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true  # Cancel previous runs on same branch
```

---

## 3. Docker Best Practices (2026)

### Multi-Stage Build (90% Size Reduction)
```dockerfile
# Stage 1: Build
FROM python:3.12-slim AS builder

WORKDIR /app
RUN pip install --user pipx && pipx install poetry

COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --output requirements.txt

COPY . .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Stage 2: Runtime (minimal)
FROM python:3.12-slim AS runtime

# Security: Non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy only wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --user --no-cache-dir /wheels/*

COPY --chown=app:app ./src ./src

ENTRYPOINT ["python", "-m", "src.main"]
```

### Layer Optimization Order
```dockerfile
# CORRECT ORDER (most stable → most changing)
FROM python:3.12-slim

# 1. System dependencies (rarely change)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# 2. Python dependencies (change occasionally)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Application code (changes frequently)
COPY . .
```

### BuildKit Optimizations
```dockerfile
# syntax=docker/dockerfile:1.6

# Cache mounts (persist across builds)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Secret mounts (never stored in layer)
RUN --mount=type=secret,id=api_key \
    cat /run/secrets/api_key > /app/.env

# Bind mounts (for build-time only)
RUN --mount=type=bind,source=scripts,target=/scripts \
    /scripts/build.sh
```

### Security Hardening
```dockerfile
# 1. Use specific version tags (not :latest)
FROM python:3.12.1-slim-bookworm

# 2. Non-root user
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# 3. Read-only filesystem (at runtime)
# docker run --read-only --tmpfs /tmp myapp

# 4. No new privileges
# docker run --security-opt=no-new-privileges myapp

# 5. Drop all capabilities
# docker run --cap-drop=ALL myapp
```

### Health Checks
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# For Python without curl
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
```

### .dockerignore (Critical)
```
# .dockerignore
.git
.gitignore
__pycache__
*.pyc
.pytest_cache
.mypy_cache
.venv
venv
.env
*.md
!README.md
Dockerfile
docker-compose*.yml
.dockerignore
tests/
docs/
```

### Production docker-compose Pattern
```yaml
version: "3.9"

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
      target: runtime
      cache_from:
        - myapp:cache
    image: myapp:${VERSION:-latest}
    environment:
      - DATABASE_URL
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 2G
        reservations:
          cpus: "0.5"
          memory: 512M
    restart: unless-stopped
```

---

## 4. Integration Patterns

### CLI + Docker Development Workflow
```python
# cli.py - Docker commands integrated
@app.command()
def docker_build(tag: str = "latest", no_cache: bool = False):
    """Build Docker image."""
    cmd = ["docker", "build", "-t", f"myapp:{tag}", "."]
    if no_cache:
        cmd.insert(2, "--no-cache")
    subprocess.run(cmd, check=True)
```

### GitHub Actions + Docker
```yaml
jobs:
  build:
    steps:
      - uses: docker/setup-buildx-action@v3
      
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## Key Takeaways

| Area | Critical Pattern |
|------|------------------|
| CLI | Typer + Pydantic Settings + CliRunner testing |
| CI/CD | Reusable workflows + Composite actions + Version pinning |
| Docker | Multi-stage + BuildKit cache + Non-root + Health checks |
| Integration | CLI wraps Docker, GHA builds/pushes images |

## Anti-Patterns to Avoid

1. **CLI**: Hardcoded paths, no --help documentation
2. **GHA**: Floating action versions, no concurrency control
3. **Docker**: Running as root, :latest tags, no .dockerignore

---

*Next: Cycle 29 - Packaging & Distribution Patterns*
