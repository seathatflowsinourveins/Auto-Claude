# Phase 1: Environment Setup & Validation
## Self-Contained Claude Code CLI Executable Prompt

> **Version**: 1.0
> **Created**: 2026-01-24
> **Estimated Duration**: 2-4 hours
> **Dependencies**: None (Starting Phase)

---

## Instructions for Claude Code CLI

Copy everything below the line and paste into Claude Code CLI as a single prompt:

---

```
You are tasked with setting up the development environment for the Unleash SDK platform.
This is Phase 1 of an 8-phase implementation plan.

## Context

The Unleash platform manages 34 best-of-breed SDKs across 8 layers:
- Layer 0: Protocol (mcp-python-sdk, fastmcp, litellm, anthropic, openai-sdk)
- Layer 1: Orchestration (temporal-python, langgraph, claude-flow, crewai, autogen)
- Layer 2: Memory (letta, zep, mem0)
- Layer 3: Structured (instructor, baml, outlines, pydantic-ai)
- Layer 4: Reasoning (dspy, serena)
- Layer 5: Observability (langfuse, opik, arize-phoenix, deepeval, ragas, promptfoo)
- Layer 6: Safety (guardrails-ai, llm-guard, nemo-guardrails)
- Layer 7: Processing (aider, ast-grep, crawl4ai, firecrawl)
- Layer 8: Knowledge (graphrag, pyribs)

## Phase 1 Objectives

1. Validate Python/Node environment meets requirements
2. Create unified environment configuration
3. Set up Python virtual environment with uv
4. Create base configuration files
5. Validate all directories exist

## Step 1: Pre-flight Checks

Run each command and verify output:

### 1.1 Python Version Check
```bash
python --version
```
**Expected**: Python 3.11.x or higher
**If fails**: Install Python 3.11+ from python.org or use pyenv

### 1.2 Node Version Check
```bash
node --version
```
**Expected**: v18.x.x or higher
**If fails**: Install Node 18+ from nodejs.org or use nvm

### 1.3 uv Package Manager Check
```bash
uv --version
```
**Expected**: uv 0.x.x
**If fails, install uv**:
```bash
# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.4 Directory Structure Check
```bash
# Verify SDK directories exist
ls -la sdks/ | head -5
ls -la stack/ | head -5
ls -la core/
ls -la platform/ | head -5
ls -la docs/ | head -5
```
**Expected**: All directories exist and contain files

## Step 2: Create Environment Template

Create file `.env.template`:
```bash
cat > .env.template << 'EOF'
# ============================================
# Unleash Platform Environment Configuration
# ============================================
# Copy this file to .env and fill in values

# ===========================================
# PROVIDER API KEYS (Required)
# ===========================================
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENAI_API_KEY=your-openai-api-key

# ===========================================
# SERVICE URLS
# ===========================================
# Letta Memory Server
LETTA_URL=http://localhost:8500

# Temporal Orchestration
TEMPORAL_HOST=localhost:7233

# Neo4j for GraphRAG
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# ===========================================
# OBSERVABILITY
# ===========================================
# Langfuse (https://cloud.langfuse.com)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com

# Arize Phoenix
PHOENIX_HOST=http://localhost:6006

# ===========================================
# CRAWLING & SCRAPING
# ===========================================
# Firecrawl API (https://firecrawl.dev)
FIRECRAWL_API_KEY=

# ===========================================
# FEATURE FLAGS
# ===========================================
ENABLE_MEMORY=true
ENABLE_GUARDRAILS=true
ENABLE_OBSERVABILITY=true
DEBUG_MODE=false

# ===========================================
# PATHS
# ===========================================
SDK_BASE_PATH=./sdks
STACK_PATH=./stack
CORE_PATH=./core
PLATFORM_PATH=./platform
EOF
```

## Step 3: Create Local Environment File

Create file `.env` from template (user fills in actual values):
```bash
cp .env.template .env
echo "Created .env - Please edit with your API keys"
```

## Step 4: Update .gitignore

Ensure .env is gitignored:
```bash
if ! grep -q "^\.env$" .gitignore 2>/dev/null; then
    echo -e "\n# Environment files\n.env\n.env.local\n.env.*.local" >> .gitignore
    echo "Added .env to .gitignore"
else
    echo ".env already in .gitignore"
fi
```

## Step 5: Create Base pyproject.toml

Create file `pyproject.toml`:
```bash
cat > pyproject.toml << 'EOF'
[project]
name = "unleash-platform"
version = "1.0.0"
description = "Unified SDK Integration Platform"
requires-python = ">=3.11"
readme = "README.md"

dependencies = [
    # Core dependencies only - SDKs installed per phase
    "structlog>=24.1.0",
    "httpx>=0.26.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
# Phase 2: Protocol Layer
protocol = [
    "mcp>=0.1.0",
    "anthropic>=0.18.0",
    "openai>=1.12.0",
    "litellm>=1.30.0",
    "fastmcp>=0.1.0",
]

# Phase 3: Orchestration Layer
orchestration = [
    "temporalio>=1.4.0",
    "langgraph>=0.0.30",
    "crewai>=0.28.0",
    "pyautogen>=0.2.0",
]

# Phase 4: Memory Layer
memory = [
    "letta>=0.3.0",
    "zep-python>=2.0.0",
    "mem0ai>=0.0.20",
]

# Phase 5: Intelligence Layer
intelligence = [
    "instructor>=1.0.0",
    "outlines>=0.0.40",
    "pydantic-ai>=0.0.10",
    "dspy-ai>=2.4.0",
]

# Phase 6: Observability Layer
observability = [
    "langfuse>=2.0.0",
    "opik>=1.0.0",
    "arize-phoenix>=3.0.0",
    "deepeval>=0.21.0",
    "ragas>=0.1.0",
]

# Phase 7: Safety Layer
safety = [
    "guardrails-ai>=0.4.0",
    "llm-guard>=0.3.0",
    "nemoguardrails>=0.8.0",
]

# Phase 8: Processing & Knowledge Layer
processing = [
    "aider-chat>=0.30.0",
    "crawl4ai>=0.3.0",
    "graphrag>=0.3.0",
    "ribs>=0.7.0",
]

# All SDKs
all = [
    "unleash-platform[protocol,orchestration,memory,intelligence,observability,safety,processing]",
]

dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "mypy>=1.8.0",
    "ruff>=0.2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.mypy]
python_version = "3.11"
strict = true
EOF
```

## Step 6: Create Python Virtual Environment

```bash
# Create virtual environment with uv
uv venv .venv --python 3.11

# Activate (Windows)
.venv\Scripts\activate

# OR Activate (macOS/Linux)
source .venv/bin/activate

# Verify activation
which python
python --version
```

## Step 7: Install Base Dependencies

```bash
# Install base dependencies only
uv pip install structlog httpx pydantic python-dotenv rich

# Verify installation
python -c "import structlog; import httpx; import pydantic; print('Base deps OK')"
```

## Step 8: Create Validation Script

Create file `scripts/validate_environment.py`:
```bash
mkdir -p scripts

cat > scripts/validate_environment.py << 'EOF'
#!/usr/bin/env python3
"""
Phase 1 Environment Validation Script
Validates all prerequisites for Unleash platform.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple

def check_python_version() -> Tuple[bool, str]:
    """Check Python version >= 3.11"""
    version = sys.version_info
    if version >= (3, 11):
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor} (need 3.11+)"

def check_node_version() -> Tuple[bool, str]:
    """Check Node.js version >= 18"""
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True
        )
        version = result.stdout.strip()
        major = int(version.lstrip("v").split(".")[0])
        if major >= 18:
            return True, version
        return False, f"{version} (need v18+)"
    except FileNotFoundError:
        return False, "Node.js not found"

def check_uv_available() -> Tuple[bool, str]:
    """Check if uv is available"""
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True
        )
        return True, result.stdout.strip()
    except FileNotFoundError:
        return False, "uv not found"

def check_directory(path: str) -> Tuple[bool, str]:
    """Check if directory exists and is not empty"""
    p = Path(path)
    if not p.exists():
        return False, f"Directory {path} does not exist"
    if not p.is_dir():
        return False, f"{path} is not a directory"
    contents = list(p.iterdir())
    if len(contents) == 0:
        return False, f"Directory {path} is empty"
    return True, f"{len(contents)} items"

def check_env_file() -> Tuple[bool, str]:
    """Check if .env file exists"""
    if Path(".env").exists():
        return True, ".env exists"
    if Path(".env.template").exists():
        return False, ".env.template exists, copy to .env"
    return False, "No .env or .env.template found"

def check_imports() -> Tuple[bool, str]:
    """Check base Python imports"""
    try:
        import structlog
        import httpx
        import pydantic
        from dotenv import load_dotenv
        from rich.console import Console
        return True, "All base imports OK"
    except ImportError as e:
        return False, f"Import error: {e}"

def main():
    """Run all validation checks."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("\n[bold blue]Phase 1: Environment Validation[/bold blue]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    checks = [
        ("Python Version", check_python_version),
        ("Node.js Version", check_node_version),
        ("uv Package Manager", check_uv_available),
        ("sdks/ Directory", lambda: check_directory("sdks")),
        ("stack/ Directory", lambda: check_directory("stack")),
        ("core/ Directory", lambda: check_directory("core")),
        ("platform/ Directory", lambda: check_directory("platform")),
        ("docs/ Directory", lambda: check_directory("docs")),
        ("Environment File", check_env_file),
        ("Python Imports", check_imports),
    ]

    all_passed = True
    for name, check_fn in checks:
        passed, details = check_fn()
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        table.add_row(name, status, details)
        if not passed:
            all_passed = False

    console.print(table)
    console.print()

    if all_passed:
        console.print("[bold green]✅ All checks passed! Ready for Phase 2.[/bold green]")
        return 0
    else:
        console.print("[bold red]❌ Some checks failed. Please fix before proceeding.[/bold red]")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x scripts/validate_environment.py
```

## Step 9: Run Validation

```bash
python scripts/validate_environment.py
```

**Expected Output**:
```
Phase 1: Environment Validation

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Check              ┃ Status   ┃ Details                ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Python Version     │ ✓ PASS   │ Python 3.11.x          │
│ Node.js Version    │ ✓ PASS   │ v18.x.x                │
│ uv Package Manager │ ✓ PASS   │ uv 0.x.x               │
│ sdks/ Directory    │ ✓ PASS   │ 34 items               │
│ stack/ Directory   │ ✓ PASS   │ 10 items               │
│ core/ Directory    │ ✓ PASS   │ 1 items                │
│ platform/ Directory│ ✓ PASS   │ xx items               │
│ docs/ Directory    │ ✓ PASS   │ xx items               │
│ Environment File   │ ✓ PASS   │ .env exists            │
│ Python Imports     │ ✓ PASS   │ All base imports OK    │
└────────────────────┴──────────┴────────────────────────┘

✅ All checks passed! Ready for Phase 2.
```

## Step 10: Configure API Keys

Edit `.env` file with actual API keys:
```bash
# Open in editor
code .env
# Or
nano .env
```

Required keys for Phase 2:
- `ANTHROPIC_API_KEY` - Get from https://console.anthropic.com
- `OPENAI_API_KEY` - Get from https://platform.openai.com

## Verification Checklist

Before proceeding to Phase 2, verify:

- [ ] Python 3.11+ installed and working
- [ ] Node 18+ installed and working
- [ ] uv package manager available
- [ ] Virtual environment created at `.venv/`
- [ ] Base dependencies installed
- [ ] `.env.template` exists
- [ ] `.env` created with API keys
- [ ] `.gitignore` updated
- [ ] `pyproject.toml` created
- [ ] `scripts/validate_environment.py` passes all checks
- [ ] All SDK directories verified (34 SDKs in `sdks/`)

## Rollback Procedure

If something goes wrong, reset to initial state:
```bash
# Remove created files
rm -f .env .env.template pyproject.toml
rm -rf .venv/
rm -f scripts/validate_environment.py

# Restore .gitignore
git checkout .gitignore

echo "Environment reset to initial state"
```

## Success Criteria

Phase 1 is complete when:
1. All validation checks pass
2. Virtual environment is active
3. Base dependencies import successfully
4. API keys are configured in `.env`
5. Ready to proceed to Phase 2

## Next Phase

Once all checks pass, proceed to **Phase 2: Protocol Layer Setup**:
```bash
# Install Phase 2 dependencies
uv pip install mcp anthropic openai litellm fastmcp
```

---

## Files Created in This Phase

| File | Purpose |
|------|---------|
| `.env.template` | Environment variable template |
| `.env` | Local configuration (gitignored) |
| `pyproject.toml` | Project configuration |
| `scripts/validate_environment.py` | Validation script |

## Total Commands Summary

```bash
# Complete Phase 1 in one script (copy entire block)
python --version && \
node --version && \
uv --version && \
ls sdks/ | wc -l && \
cat > .env.template << 'ENVEOF'
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
LETTA_URL=http://localhost:8500
TEMPORAL_HOST=localhost:7233
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
FIRECRAWL_API_KEY=
ENABLE_MEMORY=true
ENABLE_GUARDRAILS=true
DEBUG_MODE=false
ENVEOF
cp .env.template .env && \
echo ".env" >> .gitignore && \
uv venv .venv --python 3.11 && \
source .venv/bin/activate && \
uv pip install structlog httpx pydantic python-dotenv rich && \
echo "Phase 1 Complete - Edit .env with your API keys"
```

---

*End of Phase 1 Executable Prompt*
```
