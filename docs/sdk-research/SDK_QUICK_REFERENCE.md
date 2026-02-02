# ULTRAMAX SDK Quick Reference
## Project-Specific Recommendations

---

## 🆕 V29.4 New Integrations (2026-01-22)

### Opik - AI Observability (Must-Have for All Projects)
| Capability | Usage |
|------------|-------|
| **Install** | `pip install opik` |
| **Configure** | `opik configure` |
| **Trace LLM** | `@opik.track` decorator |
| **Evaluate** | Hallucination, relevance, precision metrics |
| **Self-host** | `./opik.sh` (Docker) |
| **Location** | `opik-full/` |

### Everything Claude Code - Production Configs (Must-Have)
| Component | Access |
|-----------|--------|
| **As Plugin** | `/plugin install everything-claude-code@everything-claude-code` |
| **Agents** | 9 subagents (planner, architect, code-reviewer, etc.) |
| **Skills** | 11 domains (continuous-learning, eval-harness, etc.) |
| **Commands** | `/plan`, `/tdd`, `/verify`, `/eval`, `/orchestrate` |
| **Location** | `../everything-claude-code-full/` |

---

## 🏦 AlphaForge (Trading/Risk)

### Must-Have SDKs:
| SDK | Location | Purpose |
|-----|----------|---------|
| **Serena** | Clone from GitHub | Navigate 12-layer architecture, semantic code understanding |
| **LightZero** | `lightzero/` | MCTS-based decision making for trading strategies |
| **Temporal** | `temporal-python/` | Durable execution for long-running trading workflows |
| **DSPy** | `dspy/` | Optimize risk analysis prompts |
| **LiteLLM** | `litellm/` | Route between Claude/GPT for cost optimization |
| **Guardrails AI** | `guardrails-ai/` | Validate trading decisions |
| **NeMo Guardrails** | `nemo-guardrails/` | Safety rails for automated trading |

### Recommended Stack:
```
Serena → Code Navigation
    ↓
LightZero → Strategic Planning (MuZero for market modeling)
    ↓
Temporal → Durable Trade Execution
    ↓
DSPy + LiteLLM → Optimized LLM Calls
    ↓
Guardrails → Risk Validation
```

---

## 🎭 State of Witness (MediaPipe/Real-time)

### Must-Have SDKs:
| SDK | Location | Purpose |
|-----|----------|---------|
| **Serena** | Clone from GitHub | Python LSP for MediaPipe code |
| **LiveKit Agents** | `livekit-agents/` | Real-time AI voice/video |
| **Pipecat** | `pipecat/` | Voice AI pipelines |
| **EvoTorch** | `evotorch/` | Neuroevolution for gesture models |
| **Vision Agents** | `vision-agents/` | Computer vision tools |

### Recommended Stack:
```
Serena → Navigate MediaPipe/PyTorch code
    ↓
EvoTorch → Optimize gesture recognition models
    ↓
LiveKit/Pipecat → Real-time processing
    ↓
Vision Agents → Visual understanding
```

---

## 🤖 General Agent Development

### Core Stack (Always Use):
1. **Temporal** (`temporal-python/`) - Durable execution
2. **DSPy** (`dspy/`) - Prompt optimization
3. **LiteLLM** (`litellm/`) - LLM routing
4. **Pydantic AI** (`pydantic-ai/`) - Type-safe agents
5. **Mem0** (`mem0/`) - Memory layer
6. **Langfuse** (`langfuse/`) - Observability

### Agent Framework Comparison:
| Framework | Best For | Location |
|-----------|----------|----------|
| **Pydantic AI** | Production, type safety | `pydantic-ai/` |
| **OpenAI Agents** | OpenAI integration | `openai-agents/` |
| **CrewAI** | Multi-agent teams | `crewai/` |
| **AutoGen** | Microsoft ecosystem | `autogen/` |
| **SmolaAgents** | HuggingFace models | `smolagents/` |
| **LangGraph** | Complex state machines | `langgraph/` |

---

## 📊 RAG Pipeline

### Data Ingestion:
| SDK | Best For |
|-----|----------|
| **Crawl4AI** | Web scraping → Markdown |
| **Firecrawl** | API-based scraping |
| **Docling** | PDF/DOCX processing |
| **Unstructured** | Multi-format ETL |

### Knowledge Management:
| SDK | Best For |
|-----|----------|
| **GraphRAG** | Graph-based retrieval |
| **LightRAG** | Lightweight RAG |
| **LlamaIndex** | Full RAG framework |
| **Graphiti** | Temporal knowledge graphs |

### Chunking:
| SDK | Best For |
|-----|----------|
| **Chonkie** | Smart chunking |
| **LLMLingua** | Prompt compression |

---

## 🔬 Research/Experimentation

### Reasoning:
| SDK | Algorithm |
|-----|-----------|
| **LightZero** | MCTS, MuZero, AlphaZero |
| **LLM Reasoners** | ToT, CoT, RAP |
| **Reflexion** | Self-reflection |
| **Tree of Thoughts** | Branching reasoning |

### Evolutionary:
| SDK | Purpose |
|-----|---------|
| **EvoTorch** | GPU evolutionary algorithms |
| **QDax** | Quality-diversity (JAX) |
| **Pyribs** | Quality-diversity (Python) |

### Prompt Optimization:
| SDK | Method |
|-----|--------|
| **DSPy** | MIPROv2, GEPA, BootstrapFinetune |
| **TextGrad** | Gradient-based text optimization |
| **AdalFlow** | Task pipelines |

---

## 🛡️ Safety & Production

### Guardrails:
| SDK | Focus |
|-----|-------|
| **NeMo Guardrails** | Comprehensive safety (NVIDIA) |
| **Guardrails AI** | Output validation |
| **LLM Guard** | Input/output scanning |
| **Purple Llama** | Meta safety suite |
| **Rebuff** | Prompt injection defense |

### Observability:
| SDK | Features |
|-----|----------|
| **Arize Phoenix** | Full LLM observability |
| **Langfuse** | Open-source analytics |
| **OpenLLMetry** | OpenTelemetry for LLMs |
| **AgentOps** | Agent-specific monitoring |

### Evaluation:
| SDK | Focus |
|-----|-------|
| **DeepEval** | LLM testing |
| **RAGAS** | RAG evaluation |
| **PromptFoo** | Prompt testing |
| **Braintrust** | AI evaluation platform |

---

## 💻 Code Agents

### IDE Assistants:
| SDK | Platform |
|-----|----------|
| **Aider** | CLI pair programming |
| **Cline** | VS Code |
| **Continue** | Multi-IDE |
| **SWE-Agent** | Autonomous SWE |

### Code Analysis:
| SDK | Purpose |
|-----|---------|
| **Serena** | LSP-based intelligence |
| **AST-Grep** | Structural code search |

---

## 🔌 MCP Development

### Core:
- `mcp-python-sdk/` - Python SDK
- `mcp-typescript-sdk/` - TypeScript SDK
- `mcp-servers/` - Reference servers

### Frameworks:
- `fastmcp/` - Fast MCP server framework
- `mcp-agent/` - Agent over MCP

---

## Quick Install Commands

```powershell
# Core Stack
pip install dspy-ai litellm temporalio pydantic-ai instructor

# RAG Stack
pip install "crawl4ai[all]" docling llama-index

# Evaluation Stack
pip install deepeval ragas langfuse

# Safety Stack
pip install nemoguardrails guardrails-ai llm-guard
```

---

*Quick Reference v1.0 | January 2026*
