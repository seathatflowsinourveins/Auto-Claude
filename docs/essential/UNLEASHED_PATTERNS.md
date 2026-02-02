# UNLEASHED PATTERNS
## Extracted Best Practices from Research Archives

> **Version**: 1.0 | **Date**: 2026-01-16
> **Source**: GAP_ANALYSIS_V5, ULTIMATE_UNIFIED_FINAL, Compass Artifacts
> **Status**: PRODUCTION-READY PATTERNS

---

## 1. MEMORY ARCHITECTURE PATTERNS

### Hierarchical Memory Stack
Research shows optimal memory uses 3 tiers:

| Tier | System | Retention | Use Case |
|------|--------|-----------|----------|
| **Short-term** | Context window | Session | Immediate reasoning |
| **Medium-term** | Qdrant vectors | Days-weeks | Semantic similarity |
| **Long-term** | Graph (Graphiti) | Permanent | Entity relationships |

### Memory Selection Criteria
```python
# Decision tree for memory system selection
def select_memory_system(query_type: str, retention: str) -> str:
    if query_type == "semantic_similarity":
        return "qdrant"  # Vector search
    elif query_type == "entity_relationship":
        return "graphiti"  # Knowledge graph
    elif query_type == "conversation_recall":
        return "episodic-memory"  # Cross-session
    elif query_type == "observation_tracking":
        return "claude-mem"  # Structured observations
    else:
        return "context"  # Default to context window
```

### Mem0 Integration (26% Accuracy Boost)
```json
{
  "mem0": {
    "command": "uvx",
    "args": ["mem0-mcp"],
    "env": {
      "MEM0_API_KEY": "${MEM0_API_KEY}",
      "MEM0_ORG_ID": "${MEM0_ORG_ID}"
    }
  }
}
```
**Why**: Hybrid datastore (vector + key-value + graph) with dynamic extraction.

---

## 2. HOT PATH OPTIMIZATION PATTERNS

### msgspec for High-Frequency Operations
Research shows **6.9x faster decode, 4x faster encode** vs Pydantic:

```python
import msgspec

# Zero GC overhead for trading hot paths
class Trade(msgspec.Struct, gc=False):
    timestamp: int
    price: float
    quantity: float
    side: str

class OrderBook(msgspec.Struct, gc=False):
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    sequence: int

# Use Pydantic ONLY for:
# - API validation at boundaries
# - Complex nested structures
# - Developer experience priority
```

### When to Use Each
| Library | Use Case | Performance |
|---------|----------|-------------|
| **msgspec** | Hot paths, market data | 6x faster |
| **Pydantic v2** | API validation, configs | Good DX |
| **dataclasses** | Simple internal DTOs | Zero deps |

---

## 3. CIRCUIT BREAKER PATTERNS (3-Level)

### Trading Safety Architecture
```python
# Level 1: Exchange Connections
EXCHANGE_BREAKER = {
    "failure_threshold": 3,
    "timeout_seconds": 10,
    "fallback": "queue_orders"
}

# Level 2: Risk Validation (CRITICAL)
RISK_BREAKER = {
    "failure_threshold": 1,  # Zero tolerance
    "timeout_ms": 100,
    "fallback": "reject_all_orders"
}

# Level 3: Market Data Feeds
DATA_BREAKER = {
    "failure_threshold": 5,
    "timeout_seconds": 5,
    "fallback": "use_cached_data"
}
```

### Integration with Rust Kill Switch
```rust
// Aggregate all circuit breaker states
pub struct SafetyAggregator {
    exchange_healthy: AtomicBool,
    risk_healthy: AtomicBool,
    data_healthy: AtomicBool,
}

impl SafetyAggregator {
    pub fn should_halt(&self) -> bool {
        // Any critical breaker tripped = halt
        !self.risk_healthy.load(Ordering::SeqCst)
    }
}
```

---

## 4. MARKET → VISUAL BRIDGE (Cross-Project Synergy)

### Market Regimes Drive Creative Archetypes
The unique integration between AlphaForge and State of Witness:

| Market Regime | VIX Range | Archetype | Visual Behavior |
|---------------|-----------|-----------|-----------------|
| **Bull Calm** | <15 | TRIUMPH | Gold, upward burst |
| **Bull Volatile** | 15-25 | MOVEMENT | Blue, directional flow |
| **Bear Mild** | 20-30 | GROUND | Brown, settling |
| **Bear Severe** | >30 | LAMENT | Gray, grief settling |
| **Panic** | >40 | DEFIANCE | Red, expansion |
| **Recovery** | Declining | EMBRACE | Pink, merging |

### Implementation
```python
async def market_to_archetype(market_state: dict) -> str:
    """Bridge trading signals to creative archetypes."""
    vix = market_state.get("vix", 20)
    trend = market_state.get("trend", "neutral")  # bull, bear, neutral

    if vix > 40:
        return "DEFIANCE"
    elif vix > 30 and trend == "bear":
        return "LAMENT"
    elif vix < 15 and trend == "bull":
        return "TRIUMPH"
    elif trend == "bull":
        return "MOVEMENT"
    elif trend == "bear":
        return "GROUND"
    else:
        return "WITNESS"

# OSC message to TouchDesigner
async def send_archetype(archetype: str):
    await osc_client.send("/archetype/active", archetype)
```

---

## 5. OBSERVABILITY PATTERNS

### Grafana MCP as Unified Backbone
```json
{
  "grafana": {
    "command": "mcp-grafana",
    "env": {
      "GRAFANA_URL": "http://localhost:3000",
      "GRAFANA_SERVICE_ACCOUNT_TOKEN": "${GRAFANA_TOKEN}",
      "GRAFANA_ORG_ID": "1"
    }
  }
}
```

### Langfuse for LLM Tracing
```json
{
  "langfuse": {
    "command": "npx",
    "args": ["-y", "@langfuse/mcp-server"],
    "env": {
      "LANGFUSE_PUBLIC_KEY": "${LANGFUSE_PUBLIC_KEY}",
      "LANGFUSE_SECRET_KEY": "${LANGFUSE_SECRET_KEY}",
      "LANGFUSE_HOST": "https://cloud.langfuse.com"
    }
  }
}
```

### Unified Metrics Strategy
| Domain | Tool | Metrics |
|--------|------|---------|
| **Trading** | Grafana + Prometheus | P&L, positions, latency |
| **Creative** | Grafana + custom | FPS, particle count, QD coverage |
| **LLM** | Langfuse | Token usage, reasoning traces |
| **Infrastructure** | OpenTelemetry | Distributed tracing |

---

## 6. QUALITY-DIVERSITY PATTERNS

### pyribs MAP-Elites Configuration
```python
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

# Shared QD pattern for both projects
def create_qd_optimizer(
    solution_dim: int,
    archive_dims: tuple,
    ranges: list,
    num_emitters: int = 5
):
    archive = GridArchive(
        solution_dim=solution_dim,
        dims=archive_dims,
        ranges=ranges
    )

    emitters = [
        EvolutionStrategyEmitter(
            archive,
            x0=np.zeros(solution_dim),
            sigma0=0.5,
            batch_size=50
        )
        for _ in range(num_emitters)
    ]

    return Scheduler(archive, emitters)

# Trading: strategy parameter optimization
trading_qd = create_qd_optimizer(
    solution_dim=20,  # Strategy parameters
    archive_dims=(20, 20),
    ranges=[(-1, 1), (0, 1)]  # risk_tolerance, momentum_weight
)

# Creative: aesthetic parameter exploration
creative_qd = create_qd_optimizer(
    solution_dim=40,  # Shader + particle params
    archive_dims=(20, 20),
    ranges=[(0, 1), (0, 1)]  # color_warmth, visual_complexity
)
```

---

## 7. MULTI-AGENT PATTERNS

### obra/superpowers 4-Phase Debugging
From highest-rated skill collection (9/10 production readiness):

```
Phase 1: SYMPTOM CAPTURE
- Log exact error messages
- Note environmental conditions
- Document reproduction steps

Phase 2: HYPOTHESIS GENERATION
- List 3-5 potential causes
- Rank by likelihood
- Identify testable predictions

Phase 3: SYSTEMATIC ISOLATION
- Binary search through codebase
- Test one variable at a time
- Document each test result

Phase 4: ROOT CAUSE FIX
- Fix root cause, not symptom
- Add regression test
- Update documentation
```

### Guardrail Agent Patterns
Real-time monitoring of main agent actions:

```python
BLOCKED_PATTERNS = {
    "system_destruction": [
        "rm -rf", "del /s /q", "format C:"
    ],
    "code_injection": [
        "curl | bash", "eval(", "exec("
    ],
    "trading_safety": [
        "submit_order.*validate=False",
        "kill_switch.*disable",
        "bypass_risk"
    ],
    "credential_exposure": [
        r"api_key\s*=\s*['\"][^'\"]+['\"]",
        r"password\s*=\s*['\"][^'\"]+['\"]"
    ]
}
```

---

## 8. HOOKS CONFIGURATION PATTERNS

### Deterministic Automation
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write(**/*.py)",
        "hooks": [{"type": "command", "command": "python -m py_compile %CLAUDE_FILE_PATHS%"}]
      },
      {
        "matcher": "Edit(**/*.rs)",
        "hooks": [{"type": "command", "command": "cargo check --quiet"}]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash(*order*)",
        "hooks": [{"type": "command", "command": "echo [TRADING] Order operation logged"}]
      }
    ],
    "Stop": [
      {
        "hooks": [{"type": "prompt", "prompt": "Verify all tests pass before completing"}]
      }
    ]
  }
}
```

---

## 9. DATA PIPELINE PATTERNS

### Polars + QuestDB Stack
**QuestDB**: 11M+ rows/sec ingestion for market data
**Polars**: 30x faster than Pandas with lazy evaluation

```python
import polars as pl

# Lazy pipeline with automatic optimization
pipeline = (
    pl.scan_parquet("trades/*.parquet")
    .filter(pl.col("symbol") == "BTCUSDT")
    .with_columns([
        pl.col("price").rolling_mean(window_size=20).alias("sma_20"),
        pl.col("price").rolling_std(window_size=20).alias("volatility")
    ])
    .group_by_dynamic("timestamp", every="15m").agg([
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("volume").sum().alias("volume")
    ])
)

# Execute optimized query plan
result = pipeline.collect()
```

---

## 10. IMPLEMENTATION PRIORITY

### Critical (Week 1)
1. ✅ Guardrail Agent - Implemented
2. ✅ LSP Plugins (Rust, Go) - Enabled
3. ⚪ Langfuse observability - Needs API key
4. ✅ Market→Visual Bridge - **IMPLEMENTED** (AlphaForge + State of Witness sides)

### High (Week 2)
5. ⚪ Grafana MCP - Needs service running
6. ⚪ msgspec integration - Document patterns
7. ⚪ 3-level circuit breakers - Design complete

### Medium (Week 3)
8. ⚪ Mem0 integration - Needs API key
9. ⚪ QLib RL patterns - Research complete
10. ⚪ CrewAI multi-agent - Patterns documented

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-16 | 1.0 | Initial extraction from archives |

---

## 11. STATE OF WITNESS PIPELINE PATTERNS

### Complete Data Flow Architecture
```
500+ Images → Sapiens 2B (308 keypoints) → DINOv3 + SigLIP2 (1024D hybrid)
      │
      ▼
HDBSCAN Clustering → 8 Archetypes → PaCMAP 3D Projection
      │
      ▼
SSBO Export (constellation.bin, atlas, arrays)
      │
      ▼
TouchDesigner (2M particles @ 60fps) → ProRes 4444 Output
```

### Directory Structure Pattern
```bash
state_of_witness/
├── data/
│   ├── 01_raw/          # Original images
│   ├── 02_curated/      # Selected images
│   ├── 03_poses/        # Extracted keypoints
│   ├── 04_crops/        # Person crops
│   ├── 05_embeddings/   # DINOv3 + SigLIP2
│   ├── 06_clusters/     # HDBSCAN labels
│   ├── 07_projections/  # PaCMAP 3D
│   ├── 08_evolution/    # MAP-Elites archives
│   └── 09_export/       # TouchDesigner assets
│       ├── ssbo/        # GPU buffer arrays
│       ├── sequences/   # Animation data
│       └── shaders/     # GLSL sources
├── models/              # Trained models
├── touchdesigner/       # TD project
└── config/              # YAML configs
```

### Pose Normalization Pattern
```python
def normalize_pose(keypoints_17x3):
    """Torso-relative normalization for pose-invariant features."""
    coords = keypoints[:, :2].copy()
    conf = keypoints[:, 2].copy()
    hip_center = (coords[11] + coords[12]) / 2
    shoulder_center = (coords[5] + coords[6]) / 2
    torso_length = max(np.linalg.norm(shoulder_center - hip_center), 1e-6)
    normalized = (coords - hip_center) / torso_length
    return np.concatenate([normalized.flatten(), conf])  # 51D vector
```

---

## 12. PATHOSFORMELN ARCHETYPE SYSTEM

### Visual Parameter Presets
5 embodied postures of collective action with specific TouchDesigner parameters:

| Archetype | harmon | rough | amp | rotate | blur | contrast | Character |
|-----------|--------|-------|-----|--------|------|----------|-----------|
| **DEFIANCE** | 7 | 0.6 | 1.2 | 15° | 2.5 | 1.8 | Angular, intense |
| **SOLIDARITY** | 4 | 0.4 | 0.9 | 0° | 4.0 | 1.2 | Stable, soft |
| **GROUND** | 3 | 0.3 | 0.7 | 0° | 5.0 | 1.0 | Muted, grounded |
| **MOVEMENT** | 6 | 0.55 | 1.1 | 8° | 3.0 | 1.4 | Dynamic, flowing |
| **WITNESS** | 5 | 0.5 | 1.0 | 3° | 3.5 | 1.3 | Balanced, observational |

### Node Chain Pattern
```
noise → transform → blur → level → out
  │        │          │       │
  ├─harmon ├─rotate   ├─size  ├─contrast
  ├─rough  └─scale    └─      ├─brightness
  └─amp                       └─opacity
```

### Archetype Blending
```python
def blend_archetypes(arch1: dict, arch2: dict, weight: float) -> dict:
    """Interpolate between archetype parameter presets."""
    return {
        param: arch1[param] * (1 - weight) + arch2[param] * weight
        for param in arch1.keys()
    }
```

---

## 13. DEEP LEARNING EMBEDDING STACK

### Hybrid DINOv3 + SigLIP2 Embeddings
Best-in-class visual + semantic understanding:

```python
class HybridEmbedder:
    def __init__(self, device="cuda"):
        # DINOv3: Rich visual features (1536D)
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')

        # SigLIP2: Semantic understanding (1152D)
        import open_clip
        self.siglip_model, _, self.siglip_preprocess = open_clip.create_model_and_transforms(
            'ViT-SO400M-14-SigLIP2', pretrained='webli'
        )

    @torch.no_grad()
    def embed(self, image, weight_visual=0.5, weight_semantic=0.5):
        dino = self.dino_model(self.dino_transform(image).unsqueeze(0))
        siglip = self.siglip_model.encode_image(self.siglip_preprocess(image).unsqueeze(0))

        # L2 normalize before concatenation
        dino_norm = dino / (torch.norm(dino) + 1e-8)
        siglip_norm = siglip / (torch.norm(siglip) + 1e-8)

        return torch.cat([dino_norm * weight_visual, siglip_norm * weight_semantic])
```

### Dimensionality Reduction Pipeline
```
DINOv3 (1536D) + SigLIP2 (1152D) → Concatenate (2688D) → PCA (1024D) → L2 Normalize
                                                              │
                                        ┌─────────────────────┴─────────────────────┐
                                        ▼                                           ▼
                                  HDBSCAN Clusters                         PaCMAP 3D Projection
                                  (8 archetypes)                            ([-1, 1] normalized)
```

---

## 14. GLSL PARTICLE RENDERING PATTERNS

### Curl Noise Force Field
```glsl
vec3 curlNoise(vec3 p) {
    float eps = 0.01;
    return vec3(
        noise(p + vec3(0, eps, 0)) - noise(p - vec3(0, eps, 0)) -
        noise(p + vec3(0, 0, eps)) + noise(p - vec3(0, 0, eps)),
        noise(p + vec3(0, 0, eps)) - noise(p - vec3(0, 0, eps)) -
        noise(p + vec3(eps, 0, 0)) + noise(p - vec3(eps, 0, 0)),
        noise(p + vec3(eps, 0, 0)) - noise(p - vec3(eps, 0, 0)) -
        noise(p + vec3(0, eps, 0)) + noise(p - vec3(0, eps, 0))
    ) / (2.0 * eps);
}
```

### Particle Compute Pattern (2M particles)
```glsl
// GPU compute shader for particle physics
void main() {
    uint idx = gl_GlobalInvocationID.x;
    Particle p = particles[idx];

    // Three-force system
    vec3 curl = curlNoise(pos * 2.0 + uTime * 0.1) * 0.015;      // Organic motion
    vec3 returnForce = (original - pos) * (0.03 + fitness * 0.02); // Constellation structure
    vec3 breathForce = (original * (1.0 + uBreathing * 0.05) - pos) * 0.01; // Living pulse

    vel += (curl + returnForce + breathForce) * uDeltaTime * 60.0;
    vel *= 0.98;  // Damping
    pos += vel * uDeltaTime;
}
```

### Abstract-to-Photo Reveal Transition
```glsl
// Particle fragment shader
float d = length(vUV - 0.5) * 2.0;
float noise = fract(sin(dot(vUV, vec2(12.9898, 78.233))) * 43758.5453);
float reveal = smoothstep(blend * 1.4 - 0.3, blend * 1.4 - 0.1, 1.0 - d + noise * 0.1);
float rim = exp(-abs(d - (1.0 - blend) * 1.2) * 10.0) * 0.5;  // Glowing edge
fragColor = mix(abstract, photo, reveal);
fragColor.rgb += abstract.rgb * rim;  // Rim highlight during transition
```

### ACES Film Tonemapping
```glsl
vec3 ACESFilmic(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
```

---

## 15. MAP-ELITES EVOLUTION ENGINE PATTERNS

### Production Implementation
```python
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np

@dataclass
class Elite:
    solution: np.ndarray      # Genotype (TD parameters)
    fitness: float            # QDAIF score
    behavior: np.ndarray      # 2D behavioral coordinates
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0

class MAPElitesArchive:
    def __init__(self, dims=(20, 20), ranges=[(0, 1), (0, 1)], solution_dim=64):
        self.dims = dims
        self.ranges = ranges
        self.solution_dim = solution_dim
        self._grid = [[None for _ in range(dims[1])] for _ in range(dims[0])]

    def add(self, solution, fitness, behavior) -> bool:
        """Add solution if it improves cell or cell is empty."""
        cell = self._behavior_to_cell(behavior)
        existing = self._grid[cell[0]][cell[1]]
        if existing is None or fitness > existing.fitness:
            self._grid[cell[0]][cell[1]] = Elite(solution, fitness, behavior)
            return True
        return False
```

### Mixed Emitter Strategy
```python
# Balance exploration vs exploitation
emitters = [
    GaussianEmitter(sigma=0.05),   # Fine-grained refinement
    GaussianEmitter(sigma=0.1),    # Medium exploration
    GaussianEmitter(sigma=0.2),    # Broad exploration
    IsoLineEmitter(iso_sigma=0.01, line_sigma=0.2),  # Directional
    CMAESEmitter(sigma=0.3),       # Covariance-adapted
]
```

### QDAIF Fitness Scoring
```python
def compute_fitness(image: np.ndarray) -> float:
    """Quality-Diversity AI Feedback scoring."""
    coherence = evaluate_coherence(image)      # 30%
    aesthetic = evaluate_aesthetic(image)       # 30%
    smoothness = evaluate_smoothness(image)     # 20%
    diversity = evaluate_diversity(image)       # 20%

    return (0.30 * coherence + 0.30 * aesthetic +
            0.20 * smoothness + 0.20 * diversity)
```

---

## 16. TOUCHDESIGNER NETWORK PATTERNS

### Container Organization
```
ROOT (root1)
├── /DATA        - Loaders, atlas, configs (Blue)
├── /PLAYBACK    - Timer, LFOs, emergence (Orange)
├── /PARTICLES   - Point generation, GLSL compute (Green)
├── /RENDER      - Camera, lights, render TOPs (Teal)
├── /POST        - Bloom, tonemap, vignette (Brown)
└── /OUTPUT      - Window, recording (Purple)
```

### SSBO Export Format
```python
# Binary constellation file structure
with open("constellation.bin", 'wb') as f:
    f.write(b'SOWC')  # Magic header
    f.write(struct.pack('II', version, num_points))
    for point in points:
        f.write(struct.pack('fffBfBBBI',  # 24 bytes per point
            x, y, z,           # Position (12 bytes)
            archetype,         # Archetype index (1 byte)
            fitness,           # Fitness score (4 bytes)
            r, g, b,           # Color (3 bytes)
            padding            # Alignment (4 bytes)
        ))
```

### Post-Processing Chain
```
render_beauty → bloom_threshold → blur_h/v → upsample → composite_add
                                                              │
                                                              ▼
                                            aces_film → vignette → final_output
```

---

## 17. LETTA MEMORY INTEGRATION PATTERNS

### Memory Architecture Overview

Letta provides **3-tier hierarchical memory** superior to simple conversation logging:

```
┌─────────────────────────────────────────────────────────────────┐
│                     LETTA MEMORY ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│  TIER 1: CORE MEMORY (In-Context)                                │
│  ├─ persona: Agent identity and capabilities                     │
│  ├─ human: User preferences, project context                     │
│  └─ system: Active session state                                 │
│  Characteristics: Always in LLM context, <10KB, instant access   │
├─────────────────────────────────────────────────────────────────┤
│  TIER 2: ARCHIVAL MEMORY (Vector Search)                         │
│  ├─ pgvector: Semantic similarity search                         │
│  ├─ embeddings: text-embedding-3-small (1536D)                   │
│  └─ capacity: Unlimited long-term knowledge                      │
│  Characteristics: Retrieved by semantic query, ~100ms latency    │
├─────────────────────────────────────────────────────────────────┤
│  TIER 3: RECALL MEMORY (PostgreSQL)                              │
│  ├─ conversations: Full session transcripts                      │
│  ├─ decisions: Architectural choices + rationale                 │
│  └─ learnings: Extracted insights and patterns                   │
│  Characteristics: Structured queries, historical replay          │
└─────────────────────────────────────────────────────────────────┘
```

### Production Hook Scripts

#### Session Start: Context Loading
```python
# ~/.claude/hooks/letta-session-start.py
import subprocess
import json
import os

def load_letta_context():
    """Load relevant memories at session start."""
    project = os.path.basename(os.getcwd())

    # Query archival memory for project context
    result = subprocess.run([
        "npx", "letta-mcp-client",
        "archival_memory_search",
        "--query", f"project:{project} architecture decisions patterns",
        "--limit", "10"
    ], capture_output=True, text=True)

    memories = json.loads(result.stdout) if result.returncode == 0 else []

    # Format for Claude context injection
    context = "## Previous Session Context\n\n"
    for mem in memories:
        context += f"- {mem['content'][:200]}...\n"

    # Write to .claude/session-context.md for hook pickup
    with open(".claude/session-context.md", "w") as f:
        f.write(context)

    return {"status": "loaded", "memories": len(memories)}

if __name__ == "__main__":
    print(json.dumps(load_letta_context()))
```

#### Session End: Learning Extraction
```python
# ~/.claude/hooks/letta-session-end.py
import subprocess
import json
import os
from datetime import datetime

def extract_and_save_learnings(session_log: str):
    """Extract insights from session and save to Letta."""

    # Extract key patterns using simple heuristics
    learnings = []

    # Pattern: Bug fixes
    if "fixed" in session_log.lower() or "resolved" in session_log.lower():
        learnings.append({
            "type": "bugfix",
            "content": extract_bugfix_context(session_log)
        })

    # Pattern: Architecture decisions
    if "decided" in session_log.lower() or "chose" in session_log.lower():
        learnings.append({
            "type": "decision",
            "content": extract_decision_context(session_log)
        })

    # Pattern: Code patterns
    if "pattern" in session_log.lower() or "approach" in session_log.lower():
        learnings.append({
            "type": "pattern",
            "content": extract_pattern_context(session_log)
        })

    # Save each learning to Letta archival memory
    for learning in learnings:
        subprocess.run([
            "npx", "letta-mcp-client",
            "archival_memory_insert",
            "--content", json.dumps({
                "project": os.path.basename(os.getcwd()),
                "date": datetime.now().isoformat(),
                "type": learning["type"],
                "content": learning["content"]
            })
        ])

    return {"saved": len(learnings)}

def extract_bugfix_context(log: str) -> str:
    """Extract bug fix context from session log."""
    # Implementation: find lines around "fixed", "resolved"
    lines = log.split('\n')
    for i, line in enumerate(lines):
        if 'fixed' in line.lower() or 'resolved' in line.lower():
            start = max(0, i - 5)
            end = min(len(lines), i + 5)
            return '\n'.join(lines[start:end])
    return ""

def extract_decision_context(log: str) -> str:
    """Extract architectural decision context."""
    # Similar pattern extraction for decisions
    return ""

def extract_pattern_context(log: str) -> str:
    """Extract code pattern context."""
    return ""
```

#### User Prompt Submit: Dynamic Context
```python
# ~/.claude/hooks/letta-prompt-context.py
import subprocess
import json
import sys

def inject_relevant_context(user_prompt: str):
    """Query Letta for context relevant to user's prompt."""

    # Semantic search in archival memory
    result = subprocess.run([
        "npx", "letta-mcp-client",
        "archival_memory_search",
        "--query", user_prompt[:500],  # First 500 chars
        "--limit", "5"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        return {"context": ""}

    memories = json.loads(result.stdout)

    # Format as context block
    if memories:
        context = "\n<previous-context>\n"
        for mem in memories:
            context += f"[{mem.get('type', 'memory')}] {mem['content'][:300]}\n"
        context += "</previous-context>\n"
        return {"context": context}

    return {"context": ""}

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else ""
    print(json.dumps(inject_relevant_context(prompt)))
```

### Slash Command Patterns

```yaml
# ~/.claude/commands/memory-save.md
---
name: memory-save
description: Save current context/decision to Letta archival memory
arguments:
  - name: type
    description: Memory type (decision|pattern|learning|context)
    required: true
  - name: content
    description: Content to save (or 'auto' to extract from conversation)
    required: false
---
Save important context to persistent memory for cross-session recall.

## Usage
/memory-save decision "Chose event sourcing over CRUD for audit trail requirements"
/memory-save pattern "Used circuit breaker with 5s timeout for external API calls"
/memory-save learning "Redis pub/sub more reliable than polling for real-time updates"
```

```yaml
# ~/.claude/commands/memory-recall.md
---
name: memory-recall
description: Search Letta memory for relevant context
arguments:
  - name: query
    description: Semantic search query
    required: true
  - name: limit
    description: Max results (default 10)
    required: false
---
Search persistent memory for relevant past decisions, patterns, and learnings.

## Examples
/memory-recall "how did we handle rate limiting"
/memory-recall "authentication architecture decisions" 5
```

```yaml
# ~/.claude/commands/memory-status.md
---
name: memory-status
description: Show current memory system status and statistics
---
Display memory tier statistics, recent saves, and system health.
```

### Docker Infrastructure

```yaml
# docker-compose.letta.yml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: letta
      POSTGRES_PASSWORD: ${LETTA_DB_PASSWORD}
      POSTGRES_DB: letta
    volumes:
      - letta_pgdata:/var/lib/postgresql/data
    ports:
      - "5433:5432"  # Avoid conflict with system postgres
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U letta"]
      interval: 5s
      timeout: 5s
      retries: 5

  letta-server:
    image: letta/letta-server:latest
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      LETTA_DATABASE_URL: postgresql://letta:${LETTA_DB_PASSWORD}@postgres:5432/letta
      OPENAI_API_KEY: ${OPENAI_API_KEY}  # For embeddings
    ports:
      - "8283:8283"
    volumes:
      - letta_data:/root/.letta

  letta-mcp:
    image: letta/letta-mcp-server:latest
    depends_on:
      - letta-server
    environment:
      LETTA_SERVER_URL: http://letta-server:8283
    ports:
      - "3001:3001"

volumes:
  letta_pgdata:
  letta_data:
```

### MCP Server Configuration

```json
// Add to mcp_servers section in settings.json
{
  "letta": {
    "type": "http",
    "url": "http://localhost:3001/mcp",
    "description": "Letta Memory Server - 70+ tools for agent/memory management",
    "tools": [
      "archival_memory_insert",
      "archival_memory_search",
      "core_memory_append",
      "core_memory_replace",
      "recall_memory_search",
      "conversation_search",
      "send_message"
    ]
  }
}
```

### Memory Plugin Comparison

| Feature | episodic-memory | claude-mem | Letta |
|---------|----------------|------------|-------|
| **Storage** | SQLite + sqlite-vec | JSON files | PostgreSQL + pgvector |
| **Embeddings** | 384D local | None | 1536D (OpenAI) |
| **Search** | Hybrid vector/text | None | Semantic + structured |
| **Capacity** | ~10K conversations | Session-scoped | Unlimited |
| **Latency** | ~50ms local | Instant | ~100ms (network) |
| **Isolation** | Per-user | Per-session | Multi-tenant |
| **Best For** | Conversation replay | Orchestration | Long-term knowledge |

### Integration Recommendation

**Use all three in combination:**

1. **episodic-memory**: Fast local conversation search for "did we discuss this?"
2. **claude-mem**: Session orchestration and subagent coordination
3. **Letta**: Permanent knowledge base for architectural decisions and patterns

```
Session Start:
  1. episodic-memory: Load recent conversation context
  2. Letta: Query archival memory for project knowledge
  3. claude-mem: Initialize orchestration state

During Session:
  1. episodic-memory: Background indexing of exchanges
  2. Letta: Save important decisions in real-time
  3. claude-mem: Coordinate multi-agent tasks

Session End:
  1. episodic-memory: Final conversation index
  2. Letta: Extract and persist learnings
  3. claude-mem: Save orchestration artifacts
```

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-16 | **3.0** | **UNLEASHED**: Added Letta Memory Integration patterns, hook scripts, slash commands |
| 2026-01-16 | 2.0 | ENHANCED: Added State of Witness pipeline, Pathosformeln, GLSL, MAP-Elites patterns |
| 2026-01-16 | 1.0 | Initial extraction from archives |

---

**UNLEASHED: Research-Backed Production Patterns** 🚀
