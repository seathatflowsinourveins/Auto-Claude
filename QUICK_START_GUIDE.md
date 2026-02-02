# UNLEASH Quick Start Guide
## Get Claude's Full Potential in 30 Minutes

**Version**: 30.0 | **Date**: 2026-01-23

---

## Prerequisites

- Python 3.11+
- Node.js 20+ (for MCP servers)
- Claude Code CLI installed
- ~2GB disk space for core SDKs

---

## Phase 1: Foundation (10 minutes)

### Step 1.1: Create Python Environment

```powershell
# Create dedicated environment
python -m venv $HOME\.unleash-env

# Activate (PowerShell)
& $HOME\.unleash-env\Scripts\Activate.ps1

# Or for bash/zsh:
# source ~/.unleash-env/bin/activate
```

### Step 1.2: Install Critical Stack (P0)

```bash
# The Indestructible Agent Stack
pip install temporalio langgraph "fastmcp[auth]" instructor litellm anthropic

# Verify installation
python -c "
import temporalio
import langgraph
import fastmcp
import instructor
import litellm
print('Critical stack installed successfully!')
"
```

### Step 1.3: Configure Environment Variables

```powershell
# PowerShell - Add to profile
$env:ANTHROPIC_API_KEY = "your-api-key"
$env:OPENAI_API_KEY = "your-openai-key"  # Optional for LiteLLM fallback
$env:LANGFUSE_SECRET_KEY = "your-langfuse-key"  # Optional

# Or create .env file
@"
ANTHROPIC_API_KEY=your-api-key
OPENAI_API_KEY=your-openai-key
LANGFUSE_SECRET_KEY=your-langfuse-key
"@ | Out-File -FilePath ".env" -Encoding UTF8
```

---

## Phase 2: Orchestration & Observability (10 minutes)

### Step 2.1: Install High-Value SDKs (P1)

```bash
# Orchestration + Observability + Memory
pip install langfuse pydantic-ai crewai zep-python opik

# Verify
python -c "
import langfuse
import pydantic_ai
import crewai
print('Orchestration stack installed!')
"
```

### Step 2.2: Install Claude Code Plugins

```bash
# In Claude Code CLI
/plugin marketplace add obra/superpowers-marketplace
/plugin install superpowers@superpowers-marketplace

# Memory persistence
/plugin install claude-mem

# Verify plugins
/plugins list
```

### Step 2.3: Configure CLAUDE.md Hierarchy

```bash
# Create user-level memory
mkdir -p ~/.claude
cat > ~/.claude/CLAUDE.md << 'EOF'
# User Preferences

## Claude's Role
- Use extended thinking for complex tasks
- Always validate with tests before committing
- Prefer type-safe patterns (Pydantic, TypeScript)

## Project Defaults
- Python 3.11+ with type hints
- pytest for testing
- structlog for logging

## Active Stacks
- Temporal for durable execution
- LangGraph for workflows
- FastMCP for MCP servers
EOF
```

---

## Phase 3: Enhancement (10 minutes)

### Step 3.1: Install Enhancement SDKs (P2)

```bash
# Evaluation + Safety + QD
pip install deepeval ribs nemoguardrails dspy-ai guardrails-ai

# Code intelligence
pip install ast-grep-py

# Memory extensions
pip install mem0ai letta graphiti-core

# Verify
python -c "
import deepeval
import ribs
print('Enhancement stack installed!')
"
```

### Step 3.2: Run Reorganization Script

```powershell
# Navigate to unleash
cd "Z:\insider\AUTO CLAUDE\unleash"

# Run reorganization
.\scripts\reorganize-unleash.ps1
```

### Step 3.3: Verify Installation

```python
# test_unleash.py
"""Verify UNLEASH stack installation"""

def test_critical_stack():
    """Test P0 critical SDKs"""
    import temporalio
    import langgraph
    import fastmcp
    import instructor
    import litellm
    print("P0 Critical: OK")

def test_orchestration_stack():
    """Test P1 orchestration SDKs"""
    import langfuse
    import pydantic_ai
    import crewai
    print("P1 Orchestration: OK")

def test_enhancement_stack():
    """Test P2 enhancement SDKs"""
    import deepeval
    import ribs
    print("P2 Enhancement: OK")

if __name__ == "__main__":
    test_critical_stack()
    test_orchestration_stack()
    test_enhancement_stack()
    print("\n=== All UNLEASH stacks verified! ===")
```

Run with:
```bash
python test_unleash.py
```

---

## Quick Reference: SDK Locations

### Tier 0: Critical (14 SDKs)
```
stack/tier-0-critical/
├── temporal/       # Durable execution
├── langgraph/      # Workflow graphs
├── fastmcp/        # MCP servers
├── instructor/     # Schema enforcement
├── litellm/        # 400+ LLM providers
├── langfuse/       # Observability
├── pydantic-ai/    # Type-safe agents
├── crewai/         # Multi-agent teams
├── zep/            # Semantic memory
├── deepeval/       # LLM evaluation
├── pyribs/         # Quality-Diversity
├── ast-grep/       # Code validation
├── nemo-guardrails/# Safety rails
└── dspy/           # Declarative prompts
```

### Skills & Plugins
```
skills/
├── superpowers/          # 33.8k star framework
├── everything-claude-code/# Hackathon winner
├── workflow-skills/      # 80+ specialized
└── anthropic-official/   # Official Anthropic
```

---

## First Project: Indestructible Agent

Create your first crash-proof agent:

```python
# indestructible_agent.py
from temporalio import workflow, activity
from temporalio.client import Client
from temporalio.worker import Worker
from datetime import timedelta
import instructor
import anthropic

# Patch Anthropic for structured output
client = instructor.from_anthropic(anthropic.Anthropic())

@activity.defn
async def analyze_task(task: str) -> dict:
    """Analyze task with structured output"""
    from pydantic import BaseModel

    class TaskAnalysis(BaseModel):
        summary: str
        steps: list[str]
        estimated_time: str

    return client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"Analyze this task: {task}"}],
        response_model=TaskAnalysis
    ).model_dump()

@workflow.defn
class IndestructibleWorkflow:
    @workflow.run
    async def run(self, task: str) -> dict:
        # This survives crashes, restarts, deploys
        return await workflow.execute_activity(
            analyze_task,
            task,
            start_to_close_timeout=timedelta(minutes=5),
        )

async def main():
    # Connect to Temporal
    client = await Client.connect("localhost:7233")

    # Start worker
    async with Worker(
        client,
        task_queue="unleash-queue",
        workflows=[IndestructibleWorkflow],
        activities=[analyze_task]
    ):
        # Execute workflow
        result = await client.execute_workflow(
            IndestructibleWorkflow.run,
            "Build a REST API with authentication",
            id="my-first-unleash-workflow",
            task_queue="unleash-queue"
        )
        print(f"Result: {result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## Common Commands

### Claude Code Skills
```bash
/brainstorm        # Socratic design session
/plan              # Create implementation plan
/tdd               # Test-driven development
/code-review       # Review code quality
/verify            # Validation loop
/learn             # Extract patterns
```

### Memory Operations
```bash
# claude-mem plugin
/memory search "previous decisions"
/memory save "key insight"
/memory status
```

### MCP Server Management
```bash
# List MCP servers
claude mcp list

# Add FastMCP server
claude mcp add my-server -- python -m my_mcp_server

# Remove server
claude mcp remove my-server
```

---

## Next Steps

1. **Explore Superpowers**: Read `skills/superpowers/README.md`
2. **Configure Projects**: Create project-specific `CLAUDE.md`
3. **Build Workflows**: Use LangGraph for complex orchestration
4. **Add Memory**: Integrate Zep for semantic persistence
5. **Run Evaluations**: Use DeepEval for quality assurance

---

## Troubleshooting

### Import Errors
```bash
# Ensure correct environment
which python  # Should be ~/.unleash-env/bin/python
pip list | grep -E "temporal|langgraph|fastmcp"
```

### MCP Connection Issues
```bash
# Check MCP server status
claude mcp status

# Restart servers
claude mcp restart
```

### Memory Not Persisting
```bash
# Verify claude-mem installation
/plugins list | grep claude-mem

# Check CLAUDE.md locations
cat ~/.claude/CLAUDE.md
cat ./CLAUDE.md
```

---

## Support Resources

- **Architecture Reference**: `ULTIMATE_UNLEASH_ARCHITECTURE_V30.md`
- **SDK Index**: `sdks/SDK_INDEX.md`
- **Patterns**: `sdks/SDK_INTEGRATION_PATTERNS_V30.md`
- **Superpowers Guide**: `skills/superpowers/SKILL.md`

---

*Quick Start Guide v30.0 | 2026-01-23*
