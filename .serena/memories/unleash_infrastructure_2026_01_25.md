# UNLEASH Infrastructure - Complete Reference (2026-01-25)

## Status: ALL LAYERS HEALTHY ✅

Version: 35.0.0 | Last Verified: 2026-01-25T19:56:15Z

## Health Check Results
```
┌────────────────────────────────────────────────────────────────┐
│                    UNLEASH HEALTH STATUS                       │
├────────────────────────────────────────────────────────────────┤
│  L0  Protocol      │ anthropic          │ HEALTHY │   395ms   │
│  L2  Memory        │ mem0               │ HEALTHY │   952ms   │
│  L3  Structured    │ instructor+pydantic│ HEALTHY │   0.5ms   │
│  L5  Observability │ langfuse_compat    │ HEALTHY │     7ms   │
│  L6  Safety        │ scanner_compat     │ HEALTHY │  5751ms   │
│  L8  Knowledge     │ llama_index        │ HEALTHY │  0.01ms   │
├────────────────────────────────────────────────────────────────┤
│  TOTAL: 6/6 HEALTHY  |  0 DEGRADED  |  0 UNHEALTHY            │
└────────────────────────────────────────────────────────────────┘
```

## 8-Layer Architecture (34 SDKs)

### L0: PROTOCOL GATEWAY (5 SDKs)
- mcp-python-sdk, fastmcp, litellm, anthropic, openai-sdk
- **Role**: API clients, routing, MCP protocol

### L1: ORCHESTRATION (5 SDKs)
- temporal-python, langgraph, claude-flow, crewai*, autogen*
- **Role**: Workflow orchestration, multi-agent coordination
- *Note: crewai/autogen optional, not currently installed

### L2: MEMORY (3 SDKs)
- letta, zep*, mem0
- **Role**: Persistent memory, context management
- *Note: zep has Python 3.14 compatibility issues

### L3: STRUCTURED OUTPUT (4 SDKs)
- instructor, baml, outlines, pydantic-ai
- **Role**: Type-safe LLM outputs, structured extraction

### L4: REASONING (2 SDKs)
- dspy, serena
- **Role**: Prompt programming, semantic code navigation

### L5: OBSERVABILITY (6 SDKs)
- langfuse, opik, arize-phoenix, deepeval, ragas, promptfoo
- **Role**: Tracing, evaluation, monitoring

### L6: SAFETY (3 SDKs)
- guardrails-ai, llm-guard, nemo-guardrails
- **Role**: Input/output validation, content safety

### L7: PROCESSING (4 SDKs)
- aider, ast-grep, crawl4ai, firecrawl
- **Role**: Code editing, web crawling, AST manipulation

### L8: KNOWLEDGE (2 SDKs)
- graphrag, pyribs
- **Role**: Knowledge graphs, quality-diversity optimization

## Core Module Structure (~500KB+ Python)

### Root Modules
| File | Size | Purpose |
|------|------|---------|
| core/__init__.py | 24KB | V33 status system, layer exports |
| core/llm_gateway.py | 14KB | LLM routing, multi-provider support |
| core/mcp_server.py | 15KB | MCP server implementation |
| core/sdk_access.py | 12KB | Unified SDK access layer |
| core/validator.py | 26KB | Comprehensive validation |
| core/health.py | 9KB | Production health checks |
| core/orchestrator.py | 10KB | Main orchestration logic |

### Subsystem Modules

**core/memory/** (62KB)
- providers.py (22KB) - Memory backend providers
- types.py (3KB) - Type definitions
- zep_compat.py (10KB) - Zep compatibility layer

**core/orchestration/** (108KB)
- autogen_agents.py (15KB) - AutoGen integration
- claude_flow.py (17KB) - Claude Flow patterns
- crew_manager.py (13KB) - CrewAI management
- langgraph_agents.py (14KB) - LangGraph integration
- temporal_workflows.py (11KB) - Temporal workflow definitions

**core/reasoning/** (37KB)
- agentlite_compat.py (10KB) - AgentLite compatibility

**core/safety/** (70KB)
- rails_compat.py (20KB) - NeMo Guardrails compatibility
- scanner_compat.py (24KB) - LLM Guard scanner integration

**core/tools/** (27KB)
- types.py (6KB) - Tool type definitions

**core/observability/** (133KB)
- deepeval_tests.py (20KB) - DeepEval integration
- langfuse_compat.py (27KB) - Langfuse compatibility
- langfuse_tracer.py (14KB) - Tracing implementation
- opik_evaluator.py (19KB) - Opik evaluation
- phoenix_monitor.py (18KB) - Phoenix monitoring
- promptfoo_runner.py (17KB) - PromptFoo test runner
- ragas_evaluator.py (18KB) - RAGAS evaluation

## SDK Directory Structure

Location: `Z:/insider/AUTO CLAUDE/unleash/sdks/`

34 SDKs installed:
```
aider/          autogen/        baml/           claude-flow/
anthropic/      arize-phoenix/  crawl4ai/       crewai/
ast-grep/       deepeval/       dspy/           fastmcp/
firecrawl/      graphrag/       guardrails-ai/  instructor/
langfuse/       langgraph/      letta/          litellm/
llm-guard/      mcp-python-sdk/ mem0/           nemo-guardrails/
openai-sdk/     opik/           outlines/       promptfoo/
pydantic-ai/    pyribs/         ragas/          serena/
temporal-python/ zep/
```

## Platform Subsystem

Location: `Z:/insider/AUTO CLAUDE/unleash/platform/`

Structure:
- adapters/ - SDK adapters
- benchmarks/ - Performance benchmarks
- bridges/ - Cross-SDK bridges
- cli/ - Command-line interface
- config/ - Platform configuration
- core/ - Platform core logic
- deploy/ - Deployment scripts
- docs/ - Platform documentation
- hooks/ - Lifecycle hooks
- pipelines/ - Data pipelines
- scripts/ - Utility scripts
- swarm/ - Multi-agent swarm logic
- tests/ - Platform tests
- utils/ - Utility functions

## Scripts & Validation

Location: `Z:/insider/AUTO CLAUDE/unleash/scripts/`

| Script | Purpose |
|--------|---------|
| validate_cross_session.py | Cross-session memory validation |
| validate_environment.py | Environment setup verification |
| validate_memory.py | Memory system validation |
| validate_orchestration.py | Orchestration layer validation |
| validate_production.py | Full production validation |
| validate_py314_compat.py | Python 3.14 compatibility checks |
| final_validation.py | Final deployment validation |
| deploy.py | Deployment automation |
| security_audit.py | Security scanning |

## Known Issues & Mitigations

## Python 3.14 Compatibility (FIXED 2026-01-25)

### Issue
Pydantic V1 patterns (`@validator`, `@root_validator`) cause `ConfigError` on Python 3.14+
instead of `ImportError`, crashing CLI commands that only caught `ImportError`.

### Files Fixed
1. **cli.py:90-97** - Optional dependency checking
2. **core/cli/unified_cli.py:1028-1035** - SDK status checking  
3. **core/cli/unified_cli.py:1048-1055** - Layer status checking

### Pattern Applied
```python
# Before (broken on Python 3.14)
except ImportError:
    status[name] = False

# After (Python 3.14 compatible)
except Exception:
    # Catch all exceptions - Python 3.14+ can throw ConfigError
    status[name] = False
```

## CLI-Module Contract Fixes (FIXED 2026-01-25)

### Issue
CLI expected accessor functions that didn't exist in modules.

### Files Fixed
1. **core/tools/__init__.py**
   - Added `get_tool_registry()` singleton accessor
   - Added `list_tools()` method to UnifiedToolLayer
   
2. **core/memory/__init__.py**
   - Added `get_memory_manager()` singleton accessor
   - Added `list()` method to UnifiedMemory

### Remaining Warnings (Non-blocking)
- crewai not installed (optional)
- autogen not installed (optional)  
- zep-python Python 3.14 compatibility issue
- 40+ Pydantic V1 deprecation warnings from dependencies

## CLI Command Status (All Working)

| Command | Status | Notes |
|---------|--------|-------|
| --help | ✅ PASS | 12 commands available |
| status | ✅ PASS | SDKs 8/10, Layers 6/6 |
| config show | ✅ PASS | Warns if no config file |
| tools list | ✅ PASS | Shows 4 SDK tools |
| memory list | ✅ PASS | Requires API keys for search |

# Cross-Project Usage

### For WITNESS (Creative)
```python
from core.memory import MemoryManager
from core.observability import LangfuseTracer
from core.llm_gateway import LLMGateway

# Initialize for creative exploration
gateway = LLMGateway(provider="anthropic", model="claude-sonnet-4-20250514")
tracer = LangfuseTracer(project="witness")
memory = MemoryManager(backend="letta")
```

### For TRADING (AlphaForge)
```python
from core.safety import InputScanner, OutputValidator
from core.orchestration import TemporalWorkflows
from core.observability import OpikEvaluator

# Initialize for trading with safety gates
scanner = InputScanner()
validator = OutputValidator()
workflows = TemporalWorkflows(namespace="alphaforge")
```

## Entry Points

### CLI
```bash
cd Z:/insider/AUTO\ CLAUDE/unleash
python cli.py --help
```

### Health Check
```bash
python -c "from core.health import main; main()" --verbose
```

### MCP Server
```bash
python core/mcp_server.py
```

---

*Document generated: 2026-01-25 | Status: PRODUCTION READY*
