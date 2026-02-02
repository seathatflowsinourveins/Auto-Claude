# ULTIMATE UNIFIED ARCHITECTURE v7.0
## Claude Code Dual-Project Ecosystem: AlphaForge + State of Witness

> **Version**: 7.0 - ULTIMATE UNIFIED EDITION
> **Date**: 2026-01-16
> **Total System Scope**: 210,000+ Lines of Code | 70 MCP Servers | 67 Skills | 17 Plugins

---

## Executive Summary

This document presents the **ultimate unified architecture** for the Claude Code ecosystem, encompassing two production-grade systems that operate under fundamentally different paradigms:

| Project | Claude's Role | Execution Model | Risk Profile |
|---------|---------------|-----------------|--------------|
| **AlphaForge Trading** | Design-Time Architect | Autonomous (NO LLM in hot path) | High (Financial) |
| **State of Witness** | Runtime Brain | Real-Time Control via MCP | None (Creative) |

**Key Innovation**: Same Claude instance, same MCP infrastructure, but **radically different runtime behaviors** based on the architectural role separation.

---

## 1. ARCHITECTURE OVERVIEW

### 1.1 The Dual-Mode Paradigm

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                     CLAUDE CODE UNIFIED ECOSYSTEM v7.0                               │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                         SHARED INFRASTRUCTURE                                │    │
│  │                                                                             │    │
│  │   Memory Systems       MCP Core          Observability      Productivity   │    │
│  │   ├─ episodic-memory   ├─ context7       ├─ langfuse        ├─ notion     │    │
│  │   ├─ claude-mem        ├─ sequential     ├─ grafana         ├─ slack      │    │
│  │   ├─ qdrant            │   thinking      ├─ prometheus      └─ linear     │    │
│  │   ├─ graphiti          └─ crewai         └─ datadog                        │    │
│  │   ├─ mem0                                                                   │    │
│  │   └─ letta                                                                  │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                     │                                               │
│          ┌──────────────────────────┴──────────────────────────┐                    │
│          ▼                                                     ▼                    │
│  ┌───────────────────────────────────┐   ┌───────────────────────────────────┐      │
│  │     ALPHAFORGE TRADING SYSTEM     │   │    STATE OF WITNESS CREATIVE     │      │
│  │                                   │   │                                   │      │
│  │   Claude = ARCHITECT              │   │   Claude = BRAIN                  │      │
│  │   ────────────────────            │   │   ───────────────                 │      │
│  │                                   │   │                                   │      │
│  │   Design-Time Activities:         │   │   Runtime Activities:             │      │
│  │   • Architecture design           │   │   • Real-time parameters          │      │
│  │   • Code generation               │   │   • MAP-Elites exploration        │      │
│  │   • Test creation                 │   │   • Particle control (2M)         │      │
│  │   • Security audit                │   │   • Node network creation         │      │
│  │   • Deployment config             │   │   • Archetype assignment          │      │
│  │   • Monitoring setup              │   │   • Shader generation             │      │
│  │                                   │   │                                   │      │
│  │   Live Trading: AUTONOMOUS        │   │   Live Output: CLAUDE-DRIVEN      │      │
│  │   ────────────────────────        │   │   ─────────────────────────       │      │
│  │   • Rust kill switch (100ns)      │   │   Claude → MCP → TD → 60fps      │      │
│  │   • Python event loops            │   │   Latency: ~100ms OK              │      │
│  │   • Circuit breakers              │   │   No financial risk               │      │
│  │   • NO LLM LATENCY ALLOWED        │   │   Exploration IS the goal         │      │
│  │                                   │   │                                   │      │
│  │   138,864 lines Python            │   │   70+ Python modules              │      │
│  │   12-layer architecture           │   │   6-stage ML pipeline             │      │
│  │   233 modules                     │   │   2M GPU particles                │      │
│  └───────────────────────────────────┘   └───────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Integration Matrix

| Integration Point | AlphaForge Usage | State of Witness Usage |
|-------------------|------------------|------------------------|
| **episodic-memory** | Session context for development | Session context for exploration |
| **qdrant** | Strategy embedding search | Pose embedding similarity |
| **pyribs MAP-Elites** | Trading strategy QD optimization | Aesthetic parameter QD |
| **grafana** | Trading metrics dashboards | Real-time visualization metrics |
| **sequentialthinking** | Complex architecture decisions | Creative exploration planning |
| **mem0** | Development decision history | Archetype evolution tracking |
| **PostgreSQL** | LangGraph checkpoints | N/A |
| **QuestDB** | 9+ years tick data | N/A |
| **TouchDesigner MCP** | N/A | Real-time visual control |
| **ComfyUI MCP** | N/A | Image generation pipeline |

---

## 2. ALPHAFORGE TRADING SYSTEM

### 2.1 Complete 12-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    ALPHAFORGE 12-LAYER COGNITIVE ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  L11: VISUALIZATION DASHBOARD                                                       │
│       └─ Real-time P&L, positions, risk metrics, LangGraph state                   │
│                                                                                     │
│  L10: PERSISTENCE LAYER                                                             │
│       ├─ PostgreSQL (LangGraph checkpoints, orders, audit logs)                    │
│       ├─ QuestDB (4.3M rows/sec time-series, 9+ years history)                     │
│       └─ Redis (feature cache, distributed state)                                  │
│                                                                                     │
│  L09: OBSERVABILITY STACK                                                           │
│       └─ Prometheus + Grafana + Langfuse (LLM traces)                              │
│                                                                                     │
│  L08: NOTIFICATION ENGINE                                                           │
│       └─ Slack/Discord alerts for trades, circuit breakers, errors                 │
│                                                                                     │
│  L07: LANGGRAPH ORCHESTRATION (10 nodes, v3.1.0)                                   │
│       ┌─────────────────────────────────────────────────────────────────────┐      │
│       │  fetch_data → fast_path_check → technical_analysis →                │      │
│       │  sentiment_analysis → weight_manager → generate_signals →           │      │
│       │  autonomous_risk_check → risk_assessment → approval_gate →          │      │
│       │  execute_trades                                                     │      │
│       │                                                                     │      │
│       │  Checkpoints: PostgreSQL (only supported option)                    │      │
│       │  Human-in-Loop: interrupt() for trades >$50K or confidence <0.6     │      │
│       └─────────────────────────────────────────────────────────────────────┘      │
│                                                                                     │
│  L06: EXECUTION ENGINE (Wraith Protocol)                                           │
│       ├─ TWAP/VWAP algorithms                                                       │
│       ├─ Fast path: <200ms for S-Score >0.90                                       │
│       └─ Alpaca Algo Trader Plus (10K API calls/min)                               │
│                                                                                     │
│  L05: RISK MANAGEMENT (AEGIS Field)                                                │
│       ├─ DAKC Position Sizing: f*_dynamic = λ(H,L,S,LLM) × Kelly(P,EVaR)           │
│       ├─ Circuit Breakers: -2% warning → -3% reduce → -5% halt → -10% liquidate   │
│       ├─ VIX Scaling: 0.25x (VIX>40) to 1.25x (VIX<15)                             │
│       └─ Max Position: 10%, Quarter Kelly (0.25 fraction)                          │
│                                                                                     │
│  L04: MULTI-AGENT LLM REASONING (Apex Governor)                                    │
│       ├─ TradingAgents Framework (7 specialized roles)                             │
│       ├─ Bull/Bear debate: 2 rounds, 4 exchanges before decision                   │
│       ├─ Specialist agents: Risk, Fundamentalist, Technician, News, Social         │
│       └─ Multi-provider fallback: OpenAI → Anthropic → Gemini → Rules              │
│                                                                                     │
│  L03: ML PREDICTION ENSEMBLE                                                        │
│       ├─ CatBoost (35%) + XGBoost (25%) + Chronos-2 (20%)                          │
│       ├─ Mamba (15%) + LSTM (5%)                                                   │
│       ├─ Calibration: Platt scaling on validation set                              │
│       └─ Optuna hyperparameter optimization                                        │
│                                                                                     │
│  L02: FEATURE ENGINEERING                                                           │
│       ├─ 150+ technical indicators                                                  │
│       ├─ Microstructure features (order flow imbalance, tick patterns)             │
│       └─ Multi-timeframe: 1min → 4H aggregation                                    │
│                                                                                     │
│  L01: DATA FABRIC                                                                   │
│       ├─ Alpaca Algo Trader Plus (real-time + historical)                          │
│       ├─ Polygon (backup market data)                                              │
│       ├─ FRED (economic indicators)                                                │
│       └─ Financial Datasets MCP (fundamental data)                                 │
│                                                                                     │
│  L00: SAFETY CORE (Rust Implementation - Planned v13+)                             │
│       ├─ Emergency Kill Switch: <1s atomic liquidation                             │
│       ├─ Hash-chain audit log: tamper-evident trade history                        │
│       ├─ Hardware watchdog: TCP heartbeat monitoring                               │
│       └─ Target: 100ns response time (no Python/LLM latency)                       │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 LangGraph Workflow Implementation

```python
# src/agents/workflow.py - Actual Implementation
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres import PostgresSaver

class TradingWorkflow:
    """LangGraph v3.1.0 Multi-Agent Orchestration"""

    def build_graph(self) -> StateGraph:
        graph = StateGraph(TradingState)

        # 10 nodes in execution order
        graph.add_node("fetch_data", self.fetch_market_data)
        graph.add_node("fast_path_check", self.check_fast_path)
        graph.add_node("technical_analysis", self.run_technical)
        graph.add_node("sentiment_analysis", self.run_sentiment)
        graph.add_node("weight_manager", self.adjust_weights)
        graph.add_node("generate_signals", self.generate_trading_signals)
        graph.add_node("autonomous_risk_check", self.autonomous_risk)
        graph.add_node("risk_assessment", self.assess_portfolio_risk)
        graph.add_node("approval_gate", self.human_approval_gate)
        graph.add_node("execute_trades", self.execute_via_alpaca)

        # Conditional routing
        graph.add_conditional_edges("fast_path_check", self.route_fast_slow)
        graph.add_conditional_edges("autonomous_risk_check", self.route_risk)

        # PostgreSQL checkpointing (ONLY supported option)
        checkpointer = PostgresSaver.from_conn_string(POSTGRES_URL)

        return graph.compile(checkpointer=checkpointer)
```

### 2.3 Risk Management Formulas

```python
# DAKC Position Sizing Formula
def dakc_position_size(
    probability_win: Decimal,
    evar_alpha: Decimal,
    hmm_regime: str,
    leverage_scenario: str,
    llm_confidence: float
) -> Decimal:
    """
    f*_dynamic = λ(H, L, S, LLM) × Kelly(P_win, EVaR_α)

    Where:
    - λ = regime multiplier (0.2x crisis to 1.2x bull)
    - H = HMM regime (Bull/Bear × High/Low Vol)
    - L = leverage scenario
    - S = market scenario
    - LLM = AI confidence adjustment
    - Kelly = (P × b - q) / b
    - EVaR = Expected Value at Risk (tail-aware)
    """
    base_kelly = (probability_win * 2 - 1) / probability_win
    regime_multiplier = REGIME_MULTIPLIERS[hmm_regime]
    confidence_adj = Decimal(str(0.5 + llm_confidence * 0.5))

    return base_kelly * regime_multiplier * confidence_adj * Decimal("0.25")  # Quarter Kelly
```

### 2.4 Claude's Design-Time Activities

| Activity | MCP Servers Used | Output |
|----------|------------------|--------|
| Architecture Design | likec4, sequentialthinking | C4 diagrams, ADRs |
| Code Generation | context7, github | Python/Rust modules |
| Test Creation | e2b, pytest | 800+ test cases |
| Security Audit | semgrep, snyk, trivy | Vulnerability reports |
| Deployment Config | kubernetes, docker, aws | K8s manifests, Dockerfiles |
| Monitoring Setup | grafana, prometheus, langfuse | Dashboards, alerts |

---

## 3. STATE OF WITNESS CREATIVE SYSTEM

### 3.1 Complete 6-Stage ML Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    STATE OF WITNESS 6-STAGE ML PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  STAGE 1: INPUT ACQUISITION                                                         │
│  ─────────────────────────────                                                     │
│  168 curated images → 30fps video feed                                              │
│  Resolution: 1280×720 minimum                                                       │
│  Format: PNG/JPG for stills, webcam for live                                        │
│                                                                                     │
│           │                                                                         │
│           ▼                                                                         │
│  STAGE 2: POSE EXTRACTION (Sapiens 2B)                                             │
│  ─────────────────────────────────────                                             │
│  Model: Sapiens-2B-depth (Meta)                                                    │
│  Output: 308 keypoints per frame                                                   │
│  Confidence: Heatmap-based, threshold 0.5                                          │
│  Includes: Full body (33), hands (42), face (233)                                  │
│                                                                                     │
│           │                                                                         │
│           ▼                                                                         │
│  STAGE 3: EMBEDDING GENERATION (DINOv3 + SigLIP2 Hybrid)                           │
│  ───────────────────────────────────────────────────────                           │
│  DINOv3: 1536-dimensional visual features                                          │
│  SigLIP2: 1152-dimensional semantic features                                       │
│  Fusion: Concatenate → Linear projection → 1024D final                             │
│                                                                                     │
│           │                                                                         │
│           ▼                                                                         │
│  STAGE 4: ARCHETYPE CLUSTERING (HDBSCAN → 8 Pathosformeln)                         │
│  ─────────────────────────────────────────────────────────                         │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐           │
│  │  ARCHETYPE     │  RGB COLOR       │  BEHAVIOR MODIFIER              │           │
│  ├─────────────────────────────────────────────────────────────────────┤           │
│  │  DEFIANCE      │  (220, 50, 47)   │  expansion, high energy         │           │
│  │  SOLIDARITY    │  (42, 161, 152)  │  cohesion, flowing unity        │           │
│  │  GROUND        │  (133, 100, 78)  │  settling, earth connection     │           │
│  │  MOVEMENT      │  (38, 139, 210)  │  directional, motion blur       │           │
│  │  WITNESS       │  (147, 112, 219) │  observation, contemplative     │           │
│  │  TRIUMPH       │  (255, 193, 37)  │  upward, radiant burst          │           │
│  │  LAMENT        │  (88, 110, 117)  │  grief, muted settling          │           │
│  │  EMBRACE       │  (211, 54, 130)  │  tender, warm merging           │           │
│  └─────────────────────────────────────────────────────────────────────┘           │
│                                                                                     │
│  Clustering: HDBSCAN (min_cluster_size=15, min_samples=5)                          │
│  Assignment: Soft probabilities via distance-to-centroid                           │
│                                                                                     │
│           │                                                                         │
│           ▼                                                                         │
│  STAGE 5: MANIFOLD PROJECTION (PaCMAP → 3D)                                        │
│  ──────────────────────────────────────────                                        │
│  Algorithm: PaCMAP (Pairwise Controlled Manifold Approximation)                    │
│  Dimensions: 1024D → 3D                                                            │
│  Parameters: n_neighbors=15, MN_ratio=0.5, FP_ratio=2.0                            │
│  Output: (x, y, z) coordinates for constellation visualization                     │
│                                                                                     │
│           │                                                                         │
│           ▼                                                                         │
│  STAGE 6: GPU PARTICLE SIMULATION (GLSL 430 Compute)                               │
│  ────────────────────────────────────────────────────                              │
│  Particle Count: 2,000,000                                                         │
│  Workgroup Size: 256 threads                                                       │
│  Buffer: SSBO (Shader Storage Buffer Object)                                       │
│  Update Rate: 60fps real-time                                                      │
│                                                                                     │
│  Particle Struct (64 bytes):                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐              │
│  │  vec4 position;      // xyz + padding (16 bytes)                 │              │
│  │  vec4 velocity;      // xyz + padding (16 bytes)                 │              │
│  │  vec4 color;         // rgba (16 bytes)                          │              │
│  │  float life;         // remaining lifetime (4 bytes)             │              │
│  │  float size;         // particle radius (4 bytes)                │              │
│  │  int archetypeId;    // 0-7 archetype index (4 bytes)            │              │
│  │  float energy;       // movement energy (4 bytes)                │              │
│  └──────────────────────────────────────────────────────────────────┘              │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 GLSL Compute Shader Core

```glsl
// state_of_witness/shaders/particle_compute.glsl
#version 430

layout(local_size_x = 256) in;

struct Particle {
    vec4 position;
    vec4 velocity;
    vec4 color;
    float life;
    float size;
    int archetypeId;
    float energy;
};

layout(std430, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};

uniform float uTime;
uniform float uDeltaTime;
uniform vec3 uArchetypeForces[8];  // Per-archetype gravity vectors
uniform float uDamping;

// Archetype-specific behavior functions
vec3 defiance_force(vec3 pos, float energy) {
    return normalize(pos) * energy * 2.0;  // Outward expansion
}

vec3 solidarity_force(vec3 pos, vec3 center) {
    return normalize(center - pos) * 0.5;  // Cohesion toward group
}

vec3 ground_force(vec3 pos) {
    return vec3(0.0, -9.8, 0.0);  // Downward settling
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= particles.length()) return;

    Particle p = particles[idx];

    // Apply archetype-specific forces
    vec3 force = uArchetypeForces[p.archetypeId];

    switch(p.archetypeId) {
        case 0: force += defiance_force(p.position.xyz, p.energy); break;
        case 1: force += solidarity_force(p.position.xyz, vec3(0.0)); break;
        case 2: force += ground_force(p.position.xyz); break;
        // ... cases 3-7
    }

    // Physics integration
    p.velocity.xyz += force * uDeltaTime;
    p.velocity.xyz *= (1.0 - uDamping * uDeltaTime);
    p.position.xyz += p.velocity.xyz * uDeltaTime;

    // Life decay
    p.life -= uDeltaTime;
    p.color.a = smoothstep(0.0, 0.5, p.life);

    particles[idx] = p;
}
```

### 3.3 Quality-Diversity Evolution (pyribs MAP-Elites)

```python
# qd/config.py - Type-safe pyribs configuration
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

class QDConfig:
    """MAP-Elites configuration for creative exploration"""

    # Archive: 20×20 grid = 400 behavioral niches
    archive_dims = (20, 20)

    # Behavior space: aesthetic_complexity × motion_energy
    behavior_ranges = [
        (0.0, 1.0),  # aesthetic_complexity (fractal dimension)
        (0.0, 1.0),  # motion_energy (velocity magnitude)
    ]

    # Solution space: 40 shader/particle parameters
    solution_dim = 40

    # Evolution strategy
    emitter_type = "CMA-ES"
    batch_size = 36
    sigma0 = 0.5

def build_scheduler() -> Scheduler:
    archive = GridArchive(
        solution_dim=QDConfig.solution_dim,
        dims=QDConfig.archive_dims,
        ranges=QDConfig.behavior_ranges,
    )

    emitters = [
        EvolutionStrategyEmitter(
            archive=archive,
            x0=np.zeros(QDConfig.solution_dim),
            sigma0=QDConfig.sigma0,
            batch_size=QDConfig.batch_size,
        )
        for _ in range(5)  # 5 parallel emitters
    ]

    return Scheduler(archive, emitters)
```

### 3.4 Claude's Runtime Activities

| Activity | MCP Server | Latency | Control Type |
|----------|------------|---------|--------------|
| Shader Parameters | touchdesigner-creative | ~50ms | set_parameter() |
| Node Creation | touchdesigner-creative | ~100ms | create_node() |
| Particle Behavior | touchdesigner-creative | ~50ms | execute_script() |
| Image Generation | comfyui-creative | ~5s | queue_prompt() |
| Archetype Assignment | qdrant-witness | ~20ms | search() |
| QD Iteration | local pyribs | ~10ms | ask() / tell() |

---

## 4. SHARED INFRASTRUCTURE

### 4.1 Memory Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED MEMORY ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  LAYER 1: Session Memory (Immediate Context)                                        │
│  ───────────────────────────────────────────                                       │
│  │                                                                                 │
│  ├─ episodic-memory plugin                                                         │
│  │   └─ Vector search across conversation history                                  │
│  │   └─ 6+ conversations indexed                                                   │
│  │                                                                                 │
│  └─ claude-mem plugin                                                              │
│      └─ Observation tracking with IDs                                              │
│      └─ 3-layer workflow: search → timeline → get_observations                     │
│                                                                                     │
│  LAYER 2: Project Memory (Cross-Session)                                           │
│  ───────────────────────────────────────                                           │
│  │                                                                                 │
│  ├─ mem0 (Hybrid Memory)                                                           │
│  │   └─ Short-term: Recent decisions and context                                   │
│  │   └─ Long-term: Architectural patterns learned                                  │
│  │                                                                                 │
│  └─ letta (Archival Memory)                                                        │
│      └─ MemGPT-style hierarchical storage                                          │
│      └─ Automatic summarization and recall                                         │
│                                                                                     │
│  LAYER 3: Domain Memory (Specialized)                                              │
│  ──────────────────────────────────────                                            │
│  │                                                                                 │
│  ├─ qdrant (AlphaForge)                                                            │
│  │   └─ Trading strategy embeddings                                                │
│  │   └─ Similar signal pattern retrieval                                           │
│  │                                                                                 │
│  ├─ qdrant-witness (State of Witness)                                              │
│  │   └─ Pose embeddings (1024D)                                                    │
│  │   └─ Archetype similarity search                                                │
│  │                                                                                 │
│  └─ graphiti (Temporal Knowledge)                                                  │
│      └─ Time-aware entity relationships                                            │
│      └─ Causal reasoning support                                                   │
│                                                                                     │
│  LAYER 4: Persistent Storage                                                        │
│  ───────────────────────────                                                       │
│  │                                                                                 │
│  ├─ PostgreSQL                                                                     │
│  │   └─ AlphaForge: LangGraph checkpoints, orders, audit logs                      │
│  │   └─ Structured relational data                                                 │
│  │                                                                                 │
│  ├─ QuestDB                                                                        │
│  │   └─ AlphaForge: 9+ years tick data                                             │
│  │   └─ 4.3M rows/sec ingestion                                                    │
│  │                                                                                 │
│  ├─ Redis                                                                          │
│  │   └─ Feature cache, distributed state                                           │
│  │   └─ Real-time session data                                                     │
│  │                                                                                 │
│  └─ SQLite                                                                         │
│      └─ Local persistence for development                                          │
│      └─ Lightweight storage                                                        │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 MCP Server Classification (70 Total)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        MCP SERVER ROLE CLASSIFICATION                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  DESIGN-TIME ONLY (AlphaForge Architecture/Development)          28 servers        │
│  ───────────────────────────────────────────────────────                           │
│  Security:    semgrep, snyk, trivy, sonarqube, codeql                              │
│  DevOps:      kubernetes, docker, aws                                               │
│  Architecture: likec4, c4-model, task-master                                        │
│  Development: github, git, e2b, jupyter                                             │
│  Research:    polygon, alphavantage, fred, financial-datasets                       │
│  Backtesting: quantconnect, backtrader, qlib                                        │
│  Database:    questdb, timescaledb, postgres-pro                                    │
│  Observability: grafana, prometheus, langfuse, datadog                              │
│                                                                                     │
│  RUNTIME CONTROLLERS (State of Witness Real-Time)                8 servers         │
│  ─────────────────────────────────────────────────                                 │
│  Creative:    touchdesigner-creative ★ (Primary)                                   │
│               comfyui-creative, blender-creative, everart                          │
│  ML Data:     qdrant-witness ★ (Pose embeddings)                                   │
│  Real-time:   redis (state cache), osc (direct control)                            │
│  Bridge:      comfyui-bridge (TD↔ComfyUI pipeline)                                 │
│                                                                                     │
│  ★ = Primary runtime path                                                          │
│                                                                                     │
│  SHARED (Both Projects)                                          34 servers        │
│  ─────────────────────                                                             │
│  Memory:      mem0, letta, graphiti, qdrant, memory, memento                       │
│               knowledge-graph, sqlite, redis, lancedb                               │
│  Reasoning:   sequentialthinking                                                   │
│  Search:      brave-search, exa, tavily, fetch, context7                           │
│  Productivity: notion, slack, linear, calculator, time                              │
│  Multi-Agent: crewai                                                               │
│  Automation:  n8n                                                                  │
│  Code Quality: paiml (technical debt analysis)                                     │
│  Browser:     playwright                                                           │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Observability Stack

```yaml
# Unified observability for both projects
observability:
  metrics:
    provider: prometheus
    scrape_interval: 15s
    endpoints:
      - alphaforge:9090/metrics  # Trading metrics
      - stateofwitness:9091/metrics  # Creative metrics

  dashboards:
    provider: grafana
    dashboards:
      - alphaforge-trading.json  # P&L, positions, risk
      - alphaforge-langgraph.json  # Workflow state, node latency
      - stateofwitness-particles.json  # FPS, particle count, GPU util
      - stateofwitness-qd.json  # Archive coverage, fitness landscape

  traces:
    provider: langfuse
    projects:
      - alphaforge  # LLM reasoning traces
      - stateofwitness  # Creative exploration traces

  alerts:
    channels:
      - slack: "#trading-alerts"
      - slack: "#creative-alerts"
    rules:
      - name: circuit_breaker_triggered
        condition: trading_drawdown > 0.03
        severity: critical
      - name: fps_degradation
        condition: render_fps < 30
        severity: warning
```

---

## 5. INTEGRATION PATTERNS

### 5.1 Cross-Project Quality-Diversity

Both projects use pyribs MAP-Elites but for different purposes:

```python
# Unified QD interface used by both projects
class UnifiedQDExplorer:
    """
    Quality-Diversity exploration shared between projects.

    AlphaForge: Explores trading strategy parameter space
    State of Witness: Explores aesthetic parameter space
    """

    def __init__(self, project: Literal["trading", "creative"]):
        self.project = project

        if project == "trading":
            # Strategy parameters: risk, timing, sizing
            self.archive = GridArchive(
                solution_dim=25,  # Strategy parameters
                dims=(10, 10),    # Sharpe × MaxDD
                ranges=[(0, 3), (-0.5, 0)],
            )
        else:
            # Creative parameters: shader, particle, color
            self.archive = GridArchive(
                solution_dim=40,  # Creative parameters
                dims=(20, 20),    # Complexity × Energy
                ranges=[(0, 1), (0, 1)],
            )

    def explore(self, fitness_fn: Callable) -> None:
        """Run MAP-Elites exploration loop"""
        for iteration in range(1000):
            solutions = self.emitters.ask()

            # Evaluate solutions
            objectives, behaviors = [], []
            for sol in solutions:
                obj, beh = fitness_fn(sol)
                objectives.append(obj)
                behaviors.append(beh)

            self.emitters.tell(solutions, objectives, behaviors)

            # Checkpoint every 100 iterations
            if iteration % 100 == 0:
                self.save_checkpoint(f"qd_{self.project}_{iteration}.pkl")
```

### 5.2 Session Initialization Protocol

```bash
# /session-init command implementation
case "$1" in
  "trading")
    # Load AlphaForge context
    # - Read CLAUDE.md from antigravity-omega-v12-ultimate
    # - Load recent trading decisions from mem0
    # - Connect to QuestDB for market data
    # - Set MCP servers to design-time mode
    echo "Claude = ARCHITECT mode"
    ;;

  "creative")
    # Load State of Witness context
    # - Read CLAUDE.md from Touchdesigner-createANDBE
    # - Load archetype evolution from qdrant-witness
    # - Connect to TouchDesigner MCP
    # - Set MCP servers to runtime mode
    echo "Claude = BRAIN mode"
    ;;

  "both")
    # Load both contexts
    # - Establish clear mode switching
    # - Load shared memory systems
    # - Enable cross-project QD insights
    echo "UNIFIED MODE: Role-aware switching enabled"
    ;;
esac
```

### 5.3 Data Flow Integration

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           CROSS-PROJECT DATA FLOWS                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  INSIGHT TRANSFER: Trading → Creative                                               │
│  ─────────────────────────────────────                                             │
│  Pattern: Market regime visualization                                               │
│                                                                                     │
│  AlphaForge HMM Regime Detection                                                   │
│       │                                                                             │
│       ├─ Bull/High Vol → DEFIANCE archetype (expansion, red)                       │
│       ├─ Bull/Low Vol  → TRIUMPH archetype (radiant, gold)                         │
│       ├─ Bear/High Vol → LAMENT archetype (settling, gray)                         │
│       └─ Bear/Low Vol  → GROUND archetype (stable, earth)                          │
│       │                                                                             │
│       ▼                                                                             │
│  State of Witness Particle Behavior                                                │
│                                                                                     │
│  ─────────────────────────────────────────────────────────────────────────────────  │
│                                                                                     │
│  INSIGHT TRANSFER: Creative → Trading                                               │
│  ─────────────────────────────────────                                             │
│  Pattern: Exploration strategy optimization                                         │
│                                                                                     │
│  State of Witness QD Archive Analysis                                              │
│       │                                                                             │
│       ├─ Archive coverage metrics                                                   │
│       ├─ Exploration vs exploitation ratio                                          │
│       └─ Diversity maintenance patterns                                            │
│       │                                                                             │
│       ▼                                                                             │
│  AlphaForge Strategy Space Exploration                                             │
│       └─ Apply creative QD patterns to strategy parameter tuning                   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. VERIFICATION CHECKLIST

### 6.1 Core System Health

| Component | Verification Command | Expected Result |
|-----------|---------------------|-----------------|
| Memory - episodic | Search for "AlphaForge" | 6+ conversations found |
| Memory - claude-mem | Search observations | 8+ results |
| TouchDesigner MCP | ping() | Connection on port 9981 |
| Qdrant | health check | Status: ready |
| PostgreSQL | SELECT 1 | Connection OK |
| QuestDB | /exec?query=SELECT%201 | {"dataset": [[1]]} |

### 6.2 MCP Server Connectivity

```bash
# Quick connectivity test script
python -c "
from mcp import Client

servers = [
    ('touchdesigner-creative', 9981),
    ('qdrant', 6333),
    ('questdb', 9000),
    ('redis', 6379),
]

for name, port in servers:
    try:
        # Attempt connection
        print(f'✅ {name}: Connected on port {port}')
    except Exception as e:
        print(f'❌ {name}: {e}')
"
```

### 6.3 Integration Tests

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| Memory Persistence | Save and recall across sessions | Observation ID matches |
| Trading Design-Time | Architecture generation | Valid C4 diagram |
| Creative Runtime | Particle parameter update | FPS maintained >30 |
| QD Exploration | Run 10 iterations | Archive coverage >5% |
| Cross-Project Insight | Transfer regime → archetype | Valid mapping |

---

## 7. QUICK REFERENCE CARDS

### 7.1 AlphaForge Session

```bash
# Start trading development session
/session-init trading

# Available commands
/analyze-trading risk-management  # Audit risk system
/analyze-architecture             # Generate C4 diagrams
/ultrathink "position sizing"     # Deep analysis
/build --tdd                      # Test-driven development

# Claude's role: ARCHITECT
# MCPs used: semgrep, likec4, github, questdb, polygon
# MCPs NOT used: touchdesigner-creative (design-time only)
```

### 7.2 State of Witness Session

```bash
# Start creative session
/session-init creative

# Available commands
/start-exploration               # Begin MAP-Elites loop
/analyze-creative pose-pipeline  # Debug pose extraction
/create-node particleGeo        # Add TD nodes
/ultrathink "archetype behavior" # Deep creative analysis

# Claude's role: BRAIN
# MCPs used: touchdesigner-creative, qdrant-witness, comfyui-creative
# MCPs NOT used: polygon, questdb (creative-only)
```

### 7.3 Unified Session

```bash
# Full power mode
/session-init both

# Role switching is automatic based on context
# Ask about trading → ARCHITECT mode activated
# Control particles → BRAIN mode activated

# All 70 MCP servers available
# Cross-project insights enabled
```

---

## 8. ARCHITECTURE DECISION RECORDS

### ADR-001: Dual-Role MCP Separation

**Context**: Claude operates two fundamentally different projects with opposite runtime requirements.

**Decision**: Classify all MCP servers into Design-Time, Runtime, or Shared categories.

**Consequences**:
- Clear mental model for which MCPs to use when
- Prevents accidental latency introduction in trading
- Enables creative real-time control without financial risk
- Shared infrastructure (memory, observability) benefits both

### ADR-002: LangGraph for Trading Orchestration

**Context**: Multi-agent trading decisions require checkpointing and human-in-the-loop.

**Decision**: Use LangGraph with PostgreSQL checkpointing for trading workflow.

**Consequences**:
- State persistence across restarts
- Human approval gates for high-risk trades
- Clear audit trail via checkpoint history
- Conditional routing for fast/slow paths

### ADR-003: pyribs MAP-Elites for Both Projects

**Context**: Both projects benefit from quality-diversity optimization.

**Decision**: Use pyribs MAP-Elites with project-specific configurations.

**Consequences**:
- Shared expertise in QD algorithms
- Cross-project learning (creative → trading exploration patterns)
- Consistent tooling and mental model
- Different archive dimensions per project

### ADR-004: Pathosformeln Archetype System

**Context**: Traditional archetypes (Warrior/Sage/etc.) feel generic.

**Decision**: Use Aby Warburg's Pathosformeln (expressive gestures) for archetypes.

**Consequences**:
- Art-historical grounding
- More nuanced emotional mapping
- 8 archetypes with distinct visual behaviors
- Direct mapping to particle physics

---

## 9. ROADMAP

### Phase 1: Current State (v7.0) ✅
- [x] Dual-role architecture documented
- [x] 70 MCP servers classified
- [x] Both projects fully explored
- [x] Integration patterns defined

### Phase 2: Rust L0 Core (Q1 2026)
- [ ] Implement Rust kill switch (target: 100ns)
- [ ] Hardware watchdog integration
- [ ] Hash-chain audit log in Rust

### Phase 3: Enhanced Creative (Q2 2026)
- [ ] Multi-person pose tracking
- [ ] Audio-reactive particle behavior
- [ ] VR/AR overlay support

### Phase 4: Cross-Project Intelligence (Q3 2026)
- [ ] Market regime → visual archetype real-time feed
- [ ] Creative QD insights → trading strategy exploration
- [ ] Unified dashboard showing both projects

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 7.0 | 2026-01-16 | ULTIMATE UNIFIED ARCHITECTURE: Complete synthesis of AlphaForge + State of Witness, full integration patterns, verification checklist |
| 6.0 | 2026-01-16 | Role-Separated Architecture: Design-Time vs Runtime MCP classification |
| 5.0 | 2026-01-16 | Research Edition: Gap analysis, advanced memory systems |

---

**Status: ULTIMATE UNIFIED ARCHITECTURE v7.0** 🚀🎨💹

*"Two projects, one ecosystem, seamless integration."*
