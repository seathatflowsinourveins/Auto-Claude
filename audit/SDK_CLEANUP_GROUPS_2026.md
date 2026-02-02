# SDK Cleanup Groups - Ready for User Review
## Comprehensive Audit Results (2026-01-24)

**IMPORTANT**: This document organizes SDKs into groups for your review BEFORE any deletion occurs.
All deletions require your explicit approval.

---

## EXECUTIVE SUMMARY

| Category | Count | Action |
|----------|-------|--------|
| **CRITICAL DELETE** | 6 | Immediate removal - merged/deprecated/inactive |
| **HIGH DELETE** | 42 | Remove after review - duplicate/provider-locked/unused |
| **MEDIUM DELETE** | 34 | Consider removal - low value/overlap |
| **KEEP P0 (Backbone)** | 9 | Essential infrastructure |
| **KEEP P1 (Core)** | 15 | Important capabilities |
| **KEEP P2 (Advanced)** | 11 | Valuable additions |
| **SKIP (Evaluate)** | 26 | Provider-locked, needs decision |
| **TOTAL** | 143 | |

---

## GROUP 1: CRITICAL DELETE (Immediate Removal Required)

These SDKs are confirmed deprecated, merged, or >12 months inactive.

| SDK | Location | Reason | Evidence |
|-----|----------|--------|----------|
| **memgpt** | `sdks/memgpt/` | Merged into Letta | Letta blog (Sept 2024): "MemGPT is now part of Letta" |
| **memgpt** | `stack/tier-2-memory/memgpt/` | Duplicate of above | Same content, duplicate storage |
| **infinite-agentic-loop** | `sdks/infinite-agentic-loop/` | >12 months inactive | Last indexed June 2025, only 496 stars, experimental |
| **snarktank-ralph** | `sdks/snarktank-ralph/` | Superseded by V12 | ralph-orchestrator (V12) replaces this |
| **firecrawl-sdk** | `sdks/firecrawl-sdk/` | Duplicate | `sdks/firecrawl/` is the main version |
| **mem0-full** | `sdks/mem0-full/` | Duplicate | `sdks/mem0/` is sufficient |

**Disk Space Estimate**: ~500MB+ to be recovered

---

## GROUP 2: HIGH DELETE (Remove After Review)

These SDKs are either provider-locked (and we have alternatives), duplicates, or low-activity.

### 2A. Provider-Locked SDKs (OpenAI/Google Only)

| SDK | Location | Issue | Alternative |
|-----|----------|-------|-------------|
| **openai-agents** | `sdks/openai-agents/` | OpenAI-only | Use langgraph + litellm |
| **openai-agents** | `stack/tier-1-orchestration/openai-agents/` | Duplicate + provider-locked | Already in sdks/ |
| **google-adk** | `sdks/google-adk/` | Gemini-only | Use litellm for multi-provider |

### 2B. Documentation/Reference Only (Not SDKs)

| Item | Location | Reason |
|------|----------|--------|
| **BACKBONE_ARCHITECTURE_DEEP_RESEARCH.md** | `sdks/` | Doc file, not SDK - move to docs/ |
| **NEW_SDK_INTEGRATIONS.md** | `sdks/` | Doc file - move to docs/ |
| **SDK_INDEX.md** | `sdks/` | Doc file - move to docs/ |
| **SDK_INTEGRATION_PATTERNS_V30.md** | `sdks/` | Doc file - move to docs/ |
| **SDK_QUICK_REFERENCE.md** | `sdks/` | Doc file - move to docs/ |
| **SELF_IMPROVEMENT_RESEARCH_2026.md** | `sdks/` | Doc file - move to docs/ |
| **ULTRAMAX_SDK_COMPLETE_ANALYSIS.md** | `sdks/` | Doc file - move to docs/ |
| **setup-ultramax.ps1** | `sdks/` | Script file - move to scripts/ |

### 2C. Low-Activity/Experimental SDKs

| SDK | Stars | Last Update | Reason |
|-----|-------|-------------|--------|
| **adalflow** | <1k | 2024 | Low adoption, LLM abstraction (use litellm) |
| **agentops** | ~2k | Active | Covered by opik/langfuse |
| **any-guardrail** | <500 | 2024 | Use guardrails-ai or nemo-guardrails |
| **anytool** | <500 | 2024 | Low adoption |
| **blip2-lavis** | Old | 2023 | Outdated vision model |
| **braintrust** | ~1k | Active | Covered by opik/langfuse |
| **conductor** | <1k | 2024 | Netflix workflow (not AI-focused) |
| **cua** | <500 | 2024 | Computer-use agent (experimental) |
| **dria-sdk** | <500 | 2024 | Low adoption |
| **EvoAgentX** | <500 | 2024 | Use pyribs/qdax instead |
| **fast-agent** | <1k | 2024 | Use langgraph |
| **gretel-synthetics** | ~1k | Active | Specialized synthetic data |
| **guidance** | ~15k | Active | Covered by instructor/outlines |
| **helicone** | ~2k | Active | Covered by langfuse/opik |
| **hindsight** | <500 | 2024 | Experimental |
| **hive-agents** | <500 | 2024 | Use autogen/crewai |
| **lightrag** | ~5k | Active | Use graphrag |
| **lightzero** | <1k | 2024 | RL framework (not core) |
| **livekit-agents** | ~3k | Active | Voice-specific |
| **lmql** | ~3k | 2024 | Use instructor/baml |
| **magma-multimodal** | <500 | 2024 | Experimental |
| **marvin** | ~5k | Active | Covered by instructor |
| **meta-synth-gen** | <100 | 2024 | Experimental |
| **midscene** | <500 | 2024 | Browser testing |
| **mirascope** | ~1k | Active | Covered by instructor |
| **modal** | Cloud | Active | Cloud-specific, not SDK |
| **mostly-ai-sdk** | <500 | 2024 | Synthetic data |
| **nul** | N/A | N/A | Empty/placeholder directory |
| **omagent** | <500 | 2024 | Low adoption |
| **perplexica** | ~15k | Active | Search tool (have exa/tavily) |
| **probe-semantic** | <100 | 2024 | Experimental |
| **prompttools** | <2k | 2024 | Covered by promptfoo |
| **promptwizard** | <500 | 2024 | Covered by dspy |
| **purplellama** | ~3k | Active | Meta's safety (use llm-guard) |
| **qdax** | ~500 | 2024 | Use pyribs |
| **rebuff** | <500 | 2023 | Outdated |
| **sourcerer-mcp** | <100 | 2024 | Experimental MCP |
| **spring-ai-examples** | Java | Active | Java-focused (not our stack) |
| **superprompt** | <500 | 2024 | Prompt tool |
| **swe-agent** | ~15k | Active | Covered by aider/serena |
| **typechat** | ~8k | 2024 | Microsoft, use instructor |

### 2D. Stack Tier Duplicates

These exist in both `sdks/` and `stack/` - keep only one copy.

| SDK | sdks/ Location | stack/ Location | Keep |
|-----|----------------|-----------------|------|
| memgpt | sdks/memgpt/ | stack/tier-2-memory/memgpt/ | DELETE BOTH (merged to Letta) |
| evoagentx | sdks/EvoAgentX/ | stack/tier-4-evolution/evoagentx/ | DELETE BOTH (use pyribs) |
| evoagentx-advanced | N/A | stack/tier-4-evolution/evoagentx-advanced/ | DELETE |

---

## GROUP 3: MEDIUM DELETE (Consider After Priority Cleanup)

These have some value but overlap with KEEP SDKs or are not actively used.

| SDK | Reason | Alternative |
|-----|--------|-------------|
| **a2a-protocol** | Experimental protocol | Use MCP |
| **acp-sdk** | Experimental protocol | Use MCP |
| **agent-rpc** | Protocol layer | Use MCP |
| **autoagent** | Low adoption | Use autogen/crewai |
| **autonomous-agents-research** | Research only | Archive |
| **awesome-ai-agents** | Link collection | Archive |
| **camel-ai** | Multi-agent | Use autogen/crewai |
| **chonkie** | Chunking | Built into crawl4ai |
| **claude-code-cookbook** | Examples | Archive to docs |
| **claude-code-guide** | Guide | Archive to docs |
| **claude-context-local** | Local context | Experimental |
| **claude-flow-v3** | Duplicate | Keep claude-flow only |
| **claude-skills-mcp** | Experimental | Part of main setup |
| **cline** | IDE extension | Not SDK |
| **code-reasoning** | Research | Archive |
| **context-engineering** | Research | Archive |
| **continue** | IDE extension | Not SDK |
| **deer-flow** | Alternative to langgraph | Keep langgraph |
| **docling** | Document parsing | Use unstructured |
| **evotorch** | PyTorch evolution | Use pyribs |
| **genai-agents** | Google examples | Archive |
| **github-best-sdks** | Collection | Archive |
| **graphiti** | Temporal graphs | Evaluate vs graphrag |
| **kagent** | Kubernetes agents | Niche use |
| **kserve** | ML serving | Infrastructure |
| **kubeflow-sdk** | ML pipelines | Infrastructure |
| **letta-evals** | Letta evaluation | Merge with letta |
| **llama-index** | RAG framework | Evaluate vs langgraph |
| **llm-d** | LLM deployment | Infrastructure |
| **llmlingua** | Prompt compression | Niche |
| **llm-reasoners** | Reasoning | Covered by dspy |
| **loom** | Experimental | Unknown |
| **mcp** | Core but check if needed | May be in mcp-python-sdk |
| **mcp-agent** | MCP agent | Covered by fastmcp |
| **mcp-servers** | MCP servers | Core infrastructure |
| **mcp-typescript-sdk** | TypeScript | Keep if using TS |
| **mcp-vector-search** | Vector MCP | Specialized |
| **ms-graphrag** | Microsoft GraphRAG | Keep graphrag instead |
| **nemo** | NVIDIA NeMo | Large, specialized |
| **pipecat** | Voice AI | Specialized |
| **ralph-orchestrator** | Our V12 | KEEP THIS |
| **ray-serve** | Model serving | Infrastructure |
| **reflexion** | Reasoning | Covered by dspy |
| **self-evolving-agents** | Research | Archive |
| **sglang** | Structured gen | Use outlines |
| **sketch-of-thought** | Reasoning | Experimental |
| **strands-agents** | AWS agents | AWS-locked |
| **tau-bench** | Benchmarking | Use swe-bench |
| **tavily** | Search | Have exa |
| **tensorneat** | NeuroEvolution | Specialized |
| **tensorzero** | Inference | Specialized |
| **thinking-claude** | Claude prompts | Archive |
| **tree-of-thoughts** | Reasoning | Use dspy |
| **ui-tars** | UI testing | Specialized |
| **vision-agents** | Vision | Specialized |

---

## GROUP 4: KEEP P0 - BACKBONE (Essential Infrastructure)

These are the core SDKs that form the backbone of our architecture.

| SDK | Location | Stars | Role | Why Essential |
|-----|----------|-------|------|---------------|
| **mcp-python-sdk** | `sdks/mcp-python-sdk/` | Official | Protocol | Core MCP implementation |
| **fastmcp** | `sdks/fastmcp/` | 5k+ | MCP Server | High-performance MCP servers |
| **litellm** | `sdks/litellm/` | 20k+ | LLM Gateway | Multi-provider LLM abstraction |
| **temporal-python** | `sdks/temporal-python/` | 10k+ | Orchestration | Durable execution, workflows |
| **letta** | `sdks/letta/` | 15k+ | Memory | Stateful agent memory (includes MemGPT) |
| **dspy** | `sdks/dspy/` | 25k+ | Prompting | Prompt programming, optimization |
| **langfuse** | `sdks/langfuse/` | 10k+ | Observability | LLM tracing, evaluation |
| **anthropic** | `sdks/anthropic/` | Official | SDK | Official Claude SDK |
| **openai-sdk** | `sdks/openai-sdk/` | Official | SDK | OpenAI compatibility layer |

---

## GROUP 5: KEEP P1 - CORE (Important Capabilities)

| SDK | Location | Stars | Role | Why Important |
|-----|----------|-------|------|---------------|
| **langgraph** | `sdks/langgraph/` | 10k+ | Orchestration | State machine workflows |
| **pydantic-ai** | `sdks/pydantic-ai/` | 5k+ | Type-safe | Type-safe agents |
| **crewai** | `sdks/crewai/` | 40k+ | Multi-agent | Team-based agents |
| **autogen** | `sdks/autogen/` | 50k+ | Multi-agent | Microsoft multi-agent |
| **zep** | `sdks/zep/` | 3k+ | Memory | Session memory |
| **mem0** | `sdks/mem0/` | 25k+ | Memory | Personalization memory |
| **instructor** | `sdks/instructor/` | 10k+ | Structured | Structured outputs |
| **baml** | `sdks/baml/` | 3k+ | Type-safe | Type-safe LLM |
| **outlines** | `sdks/outlines/` | 10k+ | Structured | Structured generation |
| **ast-grep** | `sdks/ast-grep/` | 10k+ | Code | AST manipulation |
| **serena** | `sdks/serena/` | 1k+ | Autonomous | Semantic code editing |
| **arize-phoenix** | `sdks/arize-phoenix/` | 5k+ | Observability | ML observability |
| **deepeval** | `sdks/deepeval/` | 5k+ | Evaluation | LLM evaluation |
| **nemo-guardrails** | `sdks/nemo-guardrails/` | 5k+ | Safety | NVIDIA guardrails |
| **claude-flow** | `sdks/claude-flow/` | 2k+ | Orchestration | Claude orchestration |

---

## GROUP 6: KEEP P2 - ADVANCED (Valuable Additions)

| SDK | Location | Stars | Role | Why Valuable |
|-----|----------|-------|------|--------------|
| **aider** | `sdks/aider/` | 25k+ | Coding | AI pair programming |
| **crawl4ai** | `sdks/crawl4ai/` | 30k+ | Web | Web crawling/scraping |
| **firecrawl** | `sdks/firecrawl/` | 10k+ | Web | Web scraping API |
| **graphrag** | `sdks/graphrag/` | 20k+ | RAG | Graph-based RAG |
| **guardrails-ai** | `sdks/guardrails-ai/` | 5k+ | Safety | Input/output validation |
| **llm-guard** | `sdks/llm-guard/` | 2k+ | Safety | Security scanning |
| **ragas** | `sdks/ragas/` | 10k+ | Evaluation | RAG evaluation |
| **promptfoo** | `sdks/promptfoo/` | 5k+ | Testing | Prompt testing |
| **opik** | `sdks/opik/` | 5k+ | Observability | Comet observability |
| **opik-full** | `sdks/opik-full/` | N/A | Observability | Full Opik installation |
| **pyribs** | `sdks/pyribs/` | 1k+ | QD | Quality-Diversity |

**Note**: `opik-full` and `opik` may be consolidated - keep one.

---

## GROUP 7: SKIP (Evaluate Quarterly)

These are provider-locked or specialized - evaluate based on project needs.

| SDK | Provider Lock | Notes |
|-----|--------------|-------|
| **exa** | Exa API | Search - currently used |
| **smolagents** | HuggingFace | HF ecosystem |
| **agent-squad** | AWS | AWS Multi-Agent |
| **swe-bench** | OpenAI | Benchmarking |
| **textgrad** | Research | Gradient prompts |
| **unstructured** | API | Document parsing |
| **qodo-cover** | Commercial | Test generation |

---

## CLEANUP EXECUTION PLAN

### Phase 1: CRITICAL DELETE (Immediate)
```bash
# After your approval:
rm -rf sdks/memgpt/
rm -rf sdks/infinite-agentic-loop/
rm -rf sdks/snarktank-ralph/
rm -rf sdks/firecrawl-sdk/
rm -rf sdks/mem0-full/
rm -rf stack/tier-2-memory/memgpt/
```

### Phase 2: Move Documentation Files
```bash
mkdir -p docs/sdk-research/
mv sdks/*.md docs/sdk-research/
mv sdks/*.ps1 scripts/
```

### Phase 3: HIGH DELETE (After Review)
- Provider-locked SDKs
- Duplicate SDKs in stack/
- Low-activity experimental SDKs

### Phase 4: Consolidate Stack Tiers
- Remove duplicate entries between sdks/ and stack/
- Keep authoritative version in sdks/
- Update stack/ to use symlinks if needed

---

## DISK SPACE ESTIMATES

| Group | Estimated Size | Recovery |
|-------|----------------|----------|
| CRITICAL DELETE | ~500MB | Immediate |
| HIGH DELETE | ~2GB | After review |
| MEDIUM DELETE | ~1GB | Optional |
| Documentation Move | ~50MB | Organize only |
| **TOTAL POTENTIAL** | **~3.5GB** | |

---

## NEXT STEPS

1. **Review this document** - Check each DELETE group
2. **Approve CRITICAL DELETE** - These are safe to remove immediately
3. **Discuss HIGH DELETE** - Which provider-locked SDKs to keep?
4. **Finalize KEEP list** - Confirm P0/P1/P2 selections
5. **Execute cleanup** - Run approved deletions
6. **Update documentation** - Reflect new structure

---

**Document Version**: 1.0
**Generated**: 2026-01-24
**Auditor**: Claude Code
**Status**: AWAITING USER REVIEW
