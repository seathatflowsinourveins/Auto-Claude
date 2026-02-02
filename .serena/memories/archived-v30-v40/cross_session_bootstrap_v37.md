# Cross-Session Bootstrap V37

## Instant SDK Access

### V37 NEW - Enterprise Orchestration
| SDK | Import | Pattern |
|-----|--------|---------|
| Claude SDK | `from claude_agent_sdk import query` | In-process MCP |
| Google ADK | `from google.adk.agents import Agent` | A2A protocol |
| Agent Squad | `from agent_squad.orchestrator import AgentSquad` | AWS teams |
| kagent | `kagent create agent` | K8s declarative |
| Ralph Orch | `ralph init --preset tdd` | Hat-based loops |
| KServe | `InferenceService` | Scale-to-zero |
| Graphiti | `from graphiti_core import Graphiti` | Temporal graphs |

### V36 - Voice & Multimodal
| SDK | Import | Pattern |
|-----|--------|---------|
| Pipecat | `from pipecat.pipeline import Pipeline` | 70+ services |
| LiveKit | `from livekit.agents import AgentServer` | MCP + telephony |
| LMQL | `import lmql` | Constrained gen |
| Outlines | `import outlines` | Logit control |
| Docling | `from docling.document_converter import DocumentConverter` | PDF + MCP |
| LLM Guard | `from llm_guard.input_scanners import *` | Security |

### V35 - Foundation
| SDK | Import | Pattern |
|-----|--------|---------|
| PydanticAI | `from pydantic_ai import Agent` | Type-safe |
| AutoGen | `from autogen_agentchat.agents import AssistantAgent` | AgentTool |
| FastMCP | `from fastmcp import FastMCP` | Components |
| LangGraph | `from langgraph.graph import StateGraph` | Checkpoints |
| DSPy | `from dspy.teleprompt import GEPA` | 10%+ MIPROv2 |
| pyribs | `from ribs.archives import GridArchive` | MAP-Elites |

## Installation
```bash
pip install claude-agent-sdk google-adk agent-squad kserve graphiti-core pydantic-ai instructor fastmcp autogen-agentchat crewai langgraph dspy textgrad pyribs pipecat-ai lmql outlines docling llm-guard
```

## Verification
```bash
pytest tests/v37_verification_tests.py -v
```
