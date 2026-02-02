# V36 Architecture Integration Summary

**Version**: 36.0.0
**Date**: 2026-01-23
**Author**: Claude Opus 4.5 (Ralph Loop V11)

## V36 Key Innovations

### NEW LAYERS (22-24)

#### Layer 22: Voice & Multimodal Agents
- **Pipecat** (Daily.co): 70+ service integrations for real-time voice
  - STT: Deepgram, AssemblyAI, Whisper, Azure, Google, Gladia
  - LLM: Anthropic, OpenAI, Gemini, Groq, Together, Fireworks
  - TTS: Deepgram, ElevenLabs, Cartesia, PlayHT, LMNT
  - S2S: AWS Nova Sonic, Gemini Live, OpenAI Realtime
  - Client SDKs: JavaScript, React, Swift, Kotlin, C++, ESP32

- **LiveKit Agents**: Real-time voice with MCP support
  - Semantic turn detection
  - Multi-agent handoff
  - Telephony integration (Twilio, Telnyx)
  - Builtin test framework

#### Layer 23: Constrained Generation
- **LMQL** (ETH Zurich): Python superset for LLMs
  - `where` constraints: stops_at, len, in, regex
  - Decoders: argmax, sample, beam, best_k
  - Async API for parallel execution

- **Outlines** (.txt): Structured generation via logit manipulation
  - Pydantic models → guaranteed valid JSON
  - Regex patterns for custom formats
  - Context-free grammars for complex syntax
  - Trusted by NVIDIA, Cohere, HuggingFace, vLLM

- **Sketch-of-Thought** (KAIST): Efficient reasoning
  - 3 paradigms: Conceptual Chaining, Chunked Symbolism, Expert Lexicons
  - 76% fewer output tokens than Chain-of-Thought
  - DistilBERT classifier for paradigm selection

#### Layer 24: Document Intelligence
- **Docling** (IBM/LF AI): Universal document processing
  - Formats: PDF, DOCX, PPTX, XLSX, images, HTML, Markdown
  - Layout understanding + table extraction
  - MCP server: `docling-mcp serve`
  - VLM support with GraniteDocling

- **LLMLingua** (Microsoft): Prompt compression
  - Up to 20x compression ratio
  - LLMLingua-2: 3-6x faster via data distillation
  - LongLLMLingua: Addresses "lost in middle" problem
  - SecurityLingua: Jailbreak defense

### ENHANCED SECURITY (Layer 19)
- **LLM Guard** (Protect AI): Comprehensive security toolkit
  - Input scanners: Anonymize, BanCode, PromptInjection, Secrets, Toxicity
  - Output scanners: Bias, FactualConsistency, MaliciousURLs, NoRefusal
  - API server: `llm-guard-api`

### NEW CONTEXT ENGINEERING
- **Zep**: Graph RAG with temporal knowledge graphs
  - Powered by Graphiti
  - <200ms latency
  - Automatic entity/relationship extraction

### NEW AUTO-OPTIMIZATION
- **AdalFlow** (SylphAI): PyTorch-like LLM workflow optimization
  - Auto-differentiation for text
  - TextGradientDescent optimizer
  - DEMO few-shot optimization

## Critical Imports (V36)

```python
# Voice & Multimodal
from pipecat.pipeline import Pipeline
from livekit.agents import Agent, AgentServer, function_tool

# Constrained Generation
import lmql
import outlines
from guidance import models, gen, select
from sketch_of_thought import SoT

# Document Intelligence
from docling.document_converter import DocumentConverter
from llmlingua import PromptCompressor

# Security
from llm_guard.input_scanners import Anonymize, PromptInjection, Secrets
from llm_guard.output_scanners import Bias, FactualConsistency, MaliciousURLs

# Context Engineering
from zep_cloud.client import Zep

# Auto-Optimization
from adalflow import Agent, Runner
from adalflow.optim import TextGradientDescent
```

## Installation

```bash
# V36 New SDKs
pip install "pipecat-ai[openai,silero,deepgram,cartesia]" livekit-agents
pip install lmql outlines guidance sketch-of-thought
pip install docling llmlingua
pip install llm-guard adalflow zep-cloud
```

## Document Locations

- Architecture: `Z:/insider/AUTO CLAUDE/unleash/ULTIMATE_UNLEASH_ARCHITECTURE_V36.md`
- Bootstrap: `Z:/insider/AUTO CLAUDE/unleash/CROSS_SESSION_BOOTSTRAP_V36.md`
- Tests: `Z:/insider/AUTO CLAUDE/unleash/tests/v36_verification_tests.py`

## SDK Count

- V35: 170+
- V36: 175+ (10 new SDKs)

## Key Patterns

1. **Voice Agent Pattern**: Pipecat Pipeline → STT → LLM → TTS → Output
2. **Constrained Generation**: LMQL `where` clauses + Outlines Pydantic
3. **Document Pipeline**: Docling convert → LLMLingua compress → LLM process
4. **Security Pattern**: Input scanners → LLM → Output scanners
5. **Context Engineering**: Zep memory.add → memory.search with Graph RAG
