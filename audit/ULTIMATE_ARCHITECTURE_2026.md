# ULTIMATE ARCHITECTURE - Post-Cleanup Vision
## Unleash Platform Architecture (2026-01-24)

---

## EXECUTIVE VISION

After cleanup, Unleash becomes a **lean, focused platform** with:
- **35 carefully curated SDKs** (down from 154)
- **8 architectural layers** (clear separation of concerns)
- **3 project configurations** (Witness, AlphaForge, Unleash)
- **~3.5GB recovered disk space**

---

## DIRECTORY STRUCTURE (POST-CLEANUP)

```
Z:\insider\AUTO CLAUDE\unleash\
│
├── .claude/                    # Claude Code configuration
├── .serena/                    # Serena memories (cleaned)
├── .github/                    # GitHub workflows
│
├── audit/                      # Audit documentation (NEW)
│   ├── SDK_CLEANUP_GROUPS_2026.md
│   ├── SDK_KEEP_ARCHITECTURE_2026.md
│   ├── ULTIMATE_ARCHITECTURE_2026.md
│   ├── CLEANUP_EXECUTION_LOG.md
│   └── [previous audit files]
│
├── config/                     # Configuration files
│   ├── projects/
│   │   ├── witness.yaml
│   │   ├── alphaforge.yaml
│   │   └── unleash.yaml
│   └── sdk-versions.lock      # Pinned SDK versions
│
├── core/                       # Core platform code
│   ├── orchestrator.py        # Main orchestrator
│   ├── memory/                # Memory integration
│   └── hooks/                 # Platform hooks
│
├── docs/                       # Documentation (reorganized)
│   ├── architecture/
│   │   ├── 8-LAYER-ARCHITECTURE.md
│   │   └── SDK-INTEGRATION-GUIDE.md
│   ├── sdk-research/          # Moved from sdks/
│   │   ├── BACKBONE_ARCHITECTURE_DEEP_RESEARCH.md
│   │   ├── SDK_INTEGRATION_PATTERNS_V30.md
│   │   └── ULTRAMAX_SDK_COMPLETE_ANALYSIS.md
│   └── guides/
│
├── sdks/                       # CURATED SDK COLLECTION (35 total)
│   │
│   │── # LAYER 0: PROTOCOL & GATEWAY (5)
│   ├── mcp-python-sdk/        # Official MCP
│   ├── fastmcp/               # MCP server framework
│   ├── litellm/               # Multi-provider gateway
│   ├── anthropic/             # Claude SDK
│   ├── openai-sdk/            # OpenAI compatibility
│   │
│   │── # LAYER 1: ORCHESTRATION (5)
│   ├── temporal-python/       # Durable execution
│   ├── langgraph/             # State machine agents
│   ├── claude-flow/           # Claude orchestration
│   ├── crewai/                # Team-based agents
│   ├── autogen/               # Microsoft multi-agent
│   │
│   │── # LAYER 2: MEMORY (3)
│   ├── letta/                 # Stateful agents (includes MemGPT)
│   ├── zep/                   # Session memory
│   ├── mem0/                  # Personalization
│   │
│   │── # LAYER 3: STRUCTURED OUTPUT (4)
│   ├── instructor/            # Pydantic extraction
│   ├── baml/                  # Type-safe DSL
│   ├── outlines/              # Grammar constrained
│   ├── pydantic-ai/           # Type-safe agents
│   │
│   │── # LAYER 4: REASONING (2)
│   ├── dspy/                  # Prompt programming
│   ├── serena/                # Semantic editing
│   │
│   │── # LAYER 5: OBSERVABILITY (6)
│   ├── langfuse/              # LLM tracing
│   ├── opik/                  # Comet observability
│   ├── arize-phoenix/         # ML observability
│   ├── deepeval/              # LLM evaluation
│   ├── ragas/                 # RAG evaluation
│   ├── promptfoo/             # Prompt testing
│   │
│   │── # LAYER 6: SAFETY (3)
│   ├── guardrails-ai/         # Validators
│   ├── llm-guard/             # Security scanning
│   ├── nemo-guardrails/       # Dialogue rails
│   │
│   │── # LAYER 7: PROCESSING (4)
│   ├── aider/                 # AI pair programming
│   ├── ast-grep/              # AST manipulation
│   ├── crawl4ai/              # Web crawling
│   ├── firecrawl/             # Web scraping API
│   │
│   │── # LAYER 8: KNOWLEDGE (2)
│   ├── graphrag/              # Graph RAG
│   └── pyribs/                # Quality-Diversity
│
├── stack/                      # TIER SYMLINKS (organized view)
│   ├── tier-0-protocol/       # -> sdks L0
│   ├── tier-1-orchestration/  # -> sdks L1
│   ├── tier-2-memory/         # -> sdks L2
│   ├── tier-3-structured/     # -> sdks L3
│   ├── tier-4-reasoning/      # -> sdks L4
│   ├── tier-5-observability/  # -> sdks L5
│   ├── tier-6-safety/         # -> sdks L6
│   ├── tier-7-processing/     # -> sdks L7
│   └── tier-8-knowledge/      # -> sdks L8
│
├── scripts/                    # Utility scripts
│   ├── setup-ultramax.ps1     # Moved from sdks/
│   ├── cleanup/
│   │   └── execute-cleanup.ps1
│   └── sdk-management/
│       ├── update-sdks.py
│       └── verify-versions.py
│
├── skills/                     # Claude skills
├── tools/                      # Platform tools
├── tests/                      # Test suites
│
├── mcp/                        # MCP configurations
├── mcp_servers/                # MCP server implementations
│
├── projects/                   # Project-specific configs
│   ├── witness/
│   ├── alphaforge/
│   └── unleash/
│
├── archive/                    # Archived content
│   ├── deprecated-sdks/       # Removed SDKs (for reference)
│   └── research/              # Research materials
│
├── CLAUDE.md                   # Project intelligence
├── README.md                   # Project overview
├── pyproject.toml              # Python config
└── requirements.txt            # Dependencies
```

---

## 8-LAYER ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                     UNLEASH PLATFORM ARCHITECTURE                    │
│                        8 Layers, 35 SDKs                            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ L8: KNOWLEDGE                                                        │
│ ┌─────────────┐ ┌─────────────┐                                     │
│ │  graphrag   │ │   pyribs    │  Graph RAG + Quality-Diversity     │
│ └─────────────┘ └─────────────┘                                     │
├─────────────────────────────────────────────────────────────────────┤
│ L7: PROCESSING                                                       │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │
│ │   aider     │ │  ast-grep   │ │  crawl4ai   │ │  firecrawl  │    │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│ L6: SAFETY                                                           │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                     │
│ │guardrails-ai│ │  llm-guard  │ │nemo-guardrails│                   │
│ └─────────────┘ └─────────────┘ └─────────────┘                     │
├─────────────────────────────────────────────────────────────────────┤
│ L5: OBSERVABILITY                                                    │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │
│ │  langfuse   │ │    opik     │ │arize-phoenix│ │  deepeval   │    │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │
│ ┌─────────────┐ ┌─────────────┐                                     │
│ │    ragas    │ │  promptfoo  │                                     │
│ └─────────────┘ └─────────────┘                                     │
├─────────────────────────────────────────────────────────────────────┤
│ L4: REASONING                                                        │
│ ┌─────────────┐ ┌─────────────┐                                     │
│ │    dspy     │ │   serena    │  Prompt optimization + Semantic    │
│ └─────────────┘ └─────────────┘                                     │
├─────────────────────────────────────────────────────────────────────┤
│ L3: STRUCTURED OUTPUT                                                │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │
│ │ instructor  │ │    baml     │ │  outlines   │ │ pydantic-ai │    │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│ L2: MEMORY                                                           │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                     │
│ │   letta     │ │    zep      │ │    mem0     │                     │
│ └─────────────┘ └─────────────┘ └─────────────┘                     │
├─────────────────────────────────────────────────────────────────────┤
│ L1: ORCHESTRATION                                                    │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │
│ │  temporal   │ │  langgraph  │ │ claude-flow │ │crewai/autogen│   │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│ L0: PROTOCOL & GATEWAY                                               │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │
│ │ mcp-python  │ │  fastmcp    │ │  litellm    │ │anthropic/oai│    │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## PROJECT CONFIGURATIONS

### State of Witness Configuration
```yaml
# config/projects/witness.yaml
project: witness
description: Creative AI + TouchDesigner visualization

sdk_profile:
  required:
    - pyribs          # MAP-Elites exploration
    - langgraph       # Creative workflows
    - letta           # Aesthetic memory
    - opik            # Generation monitoring
    - fastmcp         # TouchDesigner MCP
    - litellm         # Multi-model creativity

  optional:
    - outlines        # Shader generation
    - crawl4ai        # Reference gathering

  excluded:
    - temporal-python # Not needed for creative
    - guardrails-ai   # Less critical for art

features:
  - particle_systems
  - shader_generation
  - archetype_mapping
  - qd_exploration

memory:
  type: letta
  archival: aesthetic_discoveries
  core: creative_context
```

### AlphaForge Configuration
```yaml
# config/projects/alphaforge.yaml
project: alphaforge
description: Autonomous trading system

sdk_profile:
  required:
    - temporal-python   # Durable trading workflows
    - guardrails-ai     # Trade validation CRITICAL
    - langfuse          # Decision tracing
    - deepeval          # Strategy evaluation
    - litellm           # Multi-provider routing
    - instructor        # Structured trade signals

  optional:
    - dspy              # Signal optimization
    - graphrag          # Market knowledge

  excluded:
    - pyribs            # Not for production trading

features:
  - risk_management
  - position_sizing
  - backtesting
  - live_execution

safety:
  validation: guardrails-ai + llm-guard
  monitoring: langfuse + opik
  kill_switch: rust (100ns latency)
```

### Unleash Configuration
```yaml
# config/projects/unleash.yaml
project: unleash
description: Meta-project for Claude enhancement

sdk_profile:
  required:
    - ALL_P0_BACKBONE
    - ALL_P1_CORE
    - serena           # Self-modification
    - aider            # Code generation

  optional:
    - ALL_P2_ADVANCED

features:
  - self_improvement
  - sdk_integration
  - cross_project_coordination
  - ralph_loop

memory:
  type: letta + zep + mem0
  cross_session: enabled
  project_context: full
```

---

## DATA FLOW ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER REQUEST                                │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    L6: SAFETY (Input Validation)                     │
│                  llm-guard → guardrails-ai                          │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    L2: MEMORY (Context Loading)                      │
│              letta.recall() + zep.get_history() + mem0.get()        │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    L8: KNOWLEDGE (RAG if needed)                     │
│                         graphrag.query()                            │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    L1: ORCHESTRATION (Workflow)                      │
│        temporal.workflow() → langgraph.graph() → claude-flow        │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    L0: PROTOCOL (LLM Call)                           │
│              litellm.completion() via mcp-python-sdk                │
│                    L5: OBSERVABILITY (Trace)                        │
│                         langfuse.observe()                          │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    L3: STRUCTURED OUTPUT                             │
│               instructor.create() → pydantic.validate()             │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    L6: SAFETY (Output Validation)                    │
│              nemo-guardrails.check() → guardrails-ai.validate()     │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    L2: MEMORY (State Update)                         │
│                 letta.memory.insert() + mem0.add()                  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    L5: OBSERVABILITY (Metrics)                       │
│              opik.track() + deepeval.measure() + ragas.score()      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          RESPONSE                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## VERSION MANAGEMENT

### SDK Version Lock File
```yaml
# config/sdk-versions.lock
# Pinned versions for production stability

layer_0_protocol:
  mcp-python-sdk: ">=1.0.0,<2.0.0"
  fastmcp: ">=2.0.0,<3.0.0"
  litellm: ">=1.50.0,<2.0.0"
  anthropic: ">=0.40.0,<1.0.0"
  openai: ">=1.50.0,<2.0.0"

layer_1_orchestration:
  temporalio: ">=1.8.0,<2.0.0"
  langgraph: ">=0.2.0,<1.0.0"
  crewai: ">=0.80.0,<1.0.0"
  autogen: ">=0.4.0,<1.0.0"

layer_2_memory:
  letta: ">=0.6.0,<1.0.0"
  zep-python: ">=2.0.0,<3.0.0"
  mem0ai: ">=0.1.0,<1.0.0"

layer_3_structured:
  instructor: ">=1.5.0,<2.0.0"
  baml: ">=0.70.0,<1.0.0"
  outlines: ">=0.1.0,<1.0.0"
  pydantic-ai: ">=0.0.30,<1.0.0"

layer_4_reasoning:
  dspy: ">=2.5.0,<3.0.0"
  # serena: local installation

layer_5_observability:
  langfuse: ">=2.50.0,<3.0.0"
  opik: ">=1.3.0,<2.0.0"
  arize-phoenix: ">=5.0.0,<6.0.0"
  deepeval: ">=1.5.0,<2.0.0"
  ragas: ">=0.2.0,<1.0.0"
  promptfoo: ">=0.90.0,<1.0.0"

layer_6_safety:
  guardrails-ai: ">=0.5.0,<1.0.0"
  llm-guard: ">=0.3.15,<1.0.0"
  nemoguardrails: ">=0.10.0,<1.0.0"

layer_7_processing:
  aider-chat: ">=0.70.0,<1.0.0"
  ast-grep-py: ">=0.30.0,<1.0.0"
  crawl4ai: ">=0.4.0,<1.0.0"
  firecrawl-py: ">=1.5.0,<2.0.0"

layer_8_knowledge:
  graphrag: ">=1.0.0,<2.0.0"
  ribs: ">=0.7.0,<1.0.0"
```

---

## MIGRATION PATH

### Phase 1: Critical Cleanup (Day 1)
- Delete 6 CRITICAL SDKs
- Move documentation files
- Update .serena/memories

### Phase 2: High Priority Cleanup (Week 1)
- Remove duplicate SDKs
- Delete low-activity SDKs
- Consolidate stack/ structure

### Phase 3: Architecture Alignment (Week 2)
- Reorganize sdks/ by layer
- Create stack/ symlinks
- Update project configs

### Phase 4: Version Standardization (Week 3)
- Pin all SDK versions
- Resolve conflicts
- Update requirements.txt

### Phase 5: Documentation (Week 4)
- Update CLAUDE.md
- Create integration guides
- Archive research materials

---

## SUCCESS METRICS

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| SDK Count | 154 | 35 | 77% reduction |
| Disk Usage | ~8GB | ~4.5GB | 44% reduction |
| Layer Clarity | None | 8 layers | Structured |
| Version Conflicts | 5+ | 0 | Resolved |
| Documentation | Scattered | Centralized | Organized |

---

## SECURITY IMPROVEMENTS

With cleanup, address the 5 CRITICAL blockers:

| Blocker | Solution | SDK Support |
|---------|----------|-------------|
| SEC-001: API key exposure | Vault integration | litellm secrets |
| SEC-002: No credential rotation | Temporal workflows | temporal-python |
| SEC-003: Missing RBAC | Guardrails policies | guardrails-ai |
| SEC-004: Input validation | Multi-layer safety | llm-guard + nemo |
| SEC-005: Rate limiting | LiteLLM config | litellm |

---

**Document Version**: 1.0
**Generated**: 2026-01-24
**Architect**: Claude Code
**Status**: READY FOR IMPLEMENTATION
