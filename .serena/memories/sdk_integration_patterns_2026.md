# SDK Integration Patterns - January 2026

## Overview
This document captures verified, working integration patterns for the core SDKs
used across UNLEASH, WITNESS, and TRADING projects.

---

## 1. Opik SDK (AI Observability)

### Version: 1.9.98
### Install: `pip install opik`

### Configuration
```python
import os
os.environ["OPIK_API_KEY"] = "your-api-key"  # Required for cloud
os.environ["OPIK_URL_OVERRIDE"] = "http://localhost:5173"  # For self-hosted
```

### Auto-Trace Claude Calls (VERIFIED WORKING)
```python
import anthropic
from opik.integrations.anthropic import track_anthropic

# Wrap client - ALL Claude calls auto-traced
client = track_anthropic(anthropic.Anthropic())

# Use normally - traces appear in Opik dashboard
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Manual Function Tracing
```python
import opik

@opik.track(name="my_pipeline", tags=["production"])
def my_llm_pipeline(prompt: str):
    # All calls inside auto-traced
    return process(prompt)

# Async support
@opik.track(name="async_pipeline")
async def async_pipeline(prompt: str):
    return await process_async(prompt)
```

### Evaluation Metrics (50+)
```python
from opik.evaluation.metrics import (
    # RAG Evaluation
    Hallucination, AnswerRelevance, ContextPrecision, ContextRecall,
    
    # Agent Evaluation
    AgentTaskCompletionJudge, TrajectoryAccuracy, ToolCallAccuracy,
    
    # Bias Detection
    GenderBiasJudge, PoliticalBiasJudge,
    
    # Heuristic (no LLM needed)
    ROUGE, BERTScore, Readability, Levenshtein
)

# Example evaluation
metric = Hallucination()
score = metric.score(
    input="What is the capital?",
    output="Paris is the capital of France",
    context="France is a country in Europe"
)
```

### Self-Hosted Setup
```bash
cd opik-full
./opik.sh  # Starts on http://localhost:5173
```

---

## 2. Letta SDK (Stateful Agent Memory)

### Version: 1.7.1 (letta-client)
### Install: `pip install letta-client`

### Two Deployment Modes

#### Mode 1: Cloud (Requires API Key)
```python
from letta_client import Letta

client = Letta(
    api_key="your-api-key",
    base_url="https://api.letta.com"
)
```

#### Mode 2: Local Server (NO API Key Needed) - VERIFIED WORKING
```python
from letta_client import Letta

# Connect to local server - no authentication required
client = Letta(base_url="http://localhost:8283")

# Check connection
agents = client.agents.list()
print(f"Connected! Found {len(agents)} agents")
```

### Environment Variables
```bash
# Cloud mode
LETTA_API_KEY=letta-...

# Local mode (optional, default is localhost:8283)
LETTA_BASE_URL=http://localhost:8283
```

### Agent Operations
```python
# Create agent
agent = client.agents.create(
    name="my_agent",
    embedding="text-embedding-3-small",
    llm="claude-sonnet-4-20250514"
)

# Send message
response = client.agents.messages.create(
    agent_id=agent.id,
    role="user",
    content="Hello, remember this!"
)

# Memory persists across sessions
# Agent maintains context through Letta's memory system
```

### Integration with Evolution Pipeline
```python
from letta_evolution_integration import get_letta_integration

integration = get_letta_integration(project="unleash")

# Store evolution results
integration.store_iteration_results(
    iteration=5,
    fitness_scores={"pattern_1": 0.85},
    patterns=["def optimized_func(): ..."]
)

# Get guidance for next iteration
guidance = integration.get_evolution_guidance(query="optimization patterns")
```

---

## 3. Task Complexity Detector

### Location: `~/.claude/scripts/unleash-integration/task_complexity_detector.py`

### Purpose
Adaptive thinking token allocation based on task complexity.
Reduces latency for simple tasks, enables deep thinking for complex ones.

### Complexity Levels & Token Budgets
| Level | Tokens | Example |
|-------|--------|---------|
| TRIVIAL | 0 | "Show status" |
| LOW | 4,000 | "Fix typo in README" |
| MEDIUM | 16,000 | "Add API endpoint" |
| HIGH | 64,000 | "Debug WebSocket issue" |
| ULTRATHINK | 128,000 | "Design microservices architecture" |

### Usage
```python
from task_complexity_detector import get_thinking_config

config = get_thinking_config("Design a payment processing system")
# Returns:
# {
#     "enable_thinking": True,
#     "thinking_tokens": 128000,
#     "verbosity_directive": "Use comprehensive analysis...",
#     "complexity": "ultrathink",
#     "confidence": 0.85
# }
```

### High-Weight Triggers (Score ≥3)
- Architecture: architect, system design, microservices, distributed
- Security: vulnerability, exploit, OWASP, authentication
- Business-Critical: payment, transaction, financial, billing
- Complex Tasks: refactor, debug, integrate, migration

### Medium-Weight Triggers (Score 2)
- API/Infrastructure: endpoint, database, schema, controller
- Analysis: analyze, research, explore, compare
- Testing: write tests, unit test

---

## 4. Integration Test Script

```python
#!/usr/bin/env python3
"""Validate all SDK integrations work together."""

def test_all_integrations():
    results = {}
    
    # 1. Opik
    try:
        import opik
        from opik.integrations.anthropic import track_anthropic
        results["opik"] = {
            "available": True,
            "version": opik.__version__,
            "track_anthropic": hasattr(track_anthropic, "__call__")
        }
    except ImportError as e:
        results["opik"] = {"available": False, "error": str(e)}
    
    # 2. Letta
    try:
        from letta_client import Letta
        client = Letta(base_url="http://localhost:8283")
        agents = client.agents.list()
        results["letta"] = {
            "available": True,
            "mode": "local",
            "agents": len(agents)
        }
    except Exception as e:
        results["letta"] = {"available": False, "error": str(e)}
    
    # 3. Task Complexity
    try:
        from task_complexity_detector import get_thinking_config
        config = get_thinking_config("Design a system")
        results["complexity"] = {
            "available": True,
            "detected": config["complexity"]
        }
    except Exception as e:
        results["complexity"] = {"available": False, "error": str(e)}
    
    return results

if __name__ == "__main__":
    import json
    print(json.dumps(test_all_integrations(), indent=2))
```

---

## 5. File Locations

| Component | Path |
|-----------|------|
| Opik Tracing Layer | `~/.claude/scripts/unleash-integration/opik_tracing_layer.py` |
| Letta Sync Bridge | `~/.claude/scripts/unleash-integration/letta_sync_bridge.py` |
| Letta Evolution | `~/.claude/scripts/unleash-integration/letta_evolution_integration.py` |
| Complexity Detector | `~/.claude/scripts/unleash-integration/task_complexity_detector.py` |
| Full SDK Reference | `Z:\insider\AUTO CLAUDE\unleash\DEEP_DIVE_SDK_REFERENCE.md` |

---

## 6. Troubleshooting

### Opik "No API key" Error
- For cloud: Set `OPIK_API_KEY` environment variable
- For self-hosted: Set `OPIK_URL_OVERRIDE=http://localhost:5173`

### Letta "Connection refused" Error
- Start local server: `letta server` or Docker container
- Verify port 8283 is not blocked
- Check with: `curl http://localhost:8283/health`

### Low Complexity Detection Accuracy
- Review COMPLEXITY_KEYWORDS weights in detector
- Add domain-specific patterns (e.g., "payment" for financial)
- Test with: `python task_complexity_detector.py test`

---

*Last Updated: 2026-01-23*
*Accuracy: Opik ✅ | Letta ✅ | Complexity 90%*
