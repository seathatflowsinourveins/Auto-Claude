# COMPREHENSIVE SDK ARCHITECTURE V40+ SUMMARY

**Version**: 40.1.0 | **SDKs**: 154+ | **Layers**: 40

## Quick Layer Reference

### V40 NEW (Layers 36-40)
- L40: 3D Synthesis - NVSynth, AudioGen, Vision-Agents, MAGMA
- L39: Adversarial Robustness - adv-robust, Rebuff, PurpleLlama
- L38: Neuromorphic Memory - NeuroAIKit, Engram
- L37: Agent Discovery - OASF, AGORA Marketplace
- L36: Commerce Protocols - UCP (Google+Shopify)

### V39 (Layers 29-35)
- L35: Sub-ms Infrastructure - TensorZero (<1ms), ZCG (700µs)
- L34: Agent Protocols - ANP, OACP, AG-UI, MCP 3.0
- L33: Bi-Temporal Memory - Graphiti, BioT-AMG
- L32: Advanced RAG - Chonkie (100+ GB/s), NCF
- L31: Self-Evolution - MAAF, QD-PromptEvo
- L30: Constrained Gen - GCG, FSTGen, JunoQL
- L29: Voice/Multimodal - Pipecat, MMAgentPipe

### V38 (Layers 26-28)
- L28: Enterprise - Agent Squad, Fast-Agent, Claude SDK
- L27: A2A/ACP Protocols - Google A2A, Internet of Agents
- L26: Google ADK - kagent, AutoAgent

### V30 Foundation (Layers 1-21)
Full 21-layer foundation with 80+ core SDKs

## Key Imports (Most Used)

```python
# Orchestration
from pydantic_ai import Agent
from autogen_agentchat.agents import AssistantAgent
from langgraph.graph import StateGraph
from crewai.flow import Flow

# MCP
from fastmcp import FastMCP
from mcp import Server

# Memory
from letta import create_client
from mem0 import Memory
from graphiti_core import Graphiti
from zep_cloud.client import Zep

# RAG
from crawl4ai import AsyncWebCrawler
from lightrag import LightRAG
from chonkie import Pipeline

# Schema
from instructor import from_provider
import outlines

# Safety
from nemoguardrails import LLMRails
from llm_guard.input_scanners import *

# Observability
import opik
from langfuse import Langfuse

# Code
import ast-grep
from serena import analyze
```

Full docs: COMPREHENSIVE_SDK_ARCHITECTURE_V40.md
