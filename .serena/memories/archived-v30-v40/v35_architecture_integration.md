# V35 Architecture Integration Memory

**Created**: 2026-01-23
**Version**: 35.0.0
**Status**: Production-Ready

## Quick Reference

### Key V35 Innovations
| Innovation | Impact |
|------------|--------|
| FastMCP 3.0 | Component-based MCP with mounting/proxying |
| GEPA Optimizer | 10%+ improvement over MIPROv2 |
| Code Agents | 30% more efficient than JSON tool calls |
| LangGraph Checkpointing | Crash-proof state persistence |
| TextGrad | Gradient-based text optimization |
| EvoAgentX | Self-evolving workflow generation |

### Critical Imports
```python
# Multi-Agent
from pydantic_ai import Agent, RunContext
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.tools import AgentTool

# Structured Outputs
import instructor
from instructor import from_provider

# MCP 3.0
from fastmcp import FastMCP, Context

# Optimization
import dspy
from dspy.teleprompt import GEPA
import textgrad as tg

# State Management
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres import PostgresSaver

# Self-Evolution
from evoagentx.workflow import WorkFlowGenerator

# RAG
from crawl4ai import AsyncWebCrawler
from lightrag import LightRAG

# Memory
from mem0 import Memory
from letta import create_client

# Observability
import opik
from opik.integrations.anthropic import track_anthropic
```

### Document Locations
- Full Architecture: `Z:\insider\AUTO CLAUDE\unleash\ULTIMATE_UNLEASH_ARCHITECTURE_V35.md`
- Bootstrap: `Z:\insider\AUTO CLAUDE\unleash\CROSS_SESSION_BOOTSTRAP_V35.md`
- Tests: `Z:\insider\AUTO CLAUDE\unleash\tests\v35_verification_tests.py`

### Three Project Stacks
1. **UNLEASH**: Meta-project at `Z:\insider\AUTO CLAUDE\unleash`
2. **WITNESS**: Creative AI at `Z:\insider\AUTO CLAUDE\Touchdesigner-createANDBE`
3. **TRADING**: AlphaForge at `Z:\insider\AUTO CLAUDE\autonomous AI trading system`

### Verification
```bash
pytest tests/v35_verification_tests.py -v
# Should pass all tests
```

---
*Serena Memory for V35 Cross-Session Access*
