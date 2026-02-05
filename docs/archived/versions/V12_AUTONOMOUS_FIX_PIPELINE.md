# V12 Autonomous Fix Pipeline: 100-Iteration Claude CLI Guide

**Document Version:** 1.0  
**Created:** 2026-01-22  
**Target:** platform/core/ralph_loop.py (4,147 lines)  
**Starting Iteration:** 24  
**Target Iteration:** 124+  
**Status:** Ready for Claude CLI Execution

---

## Table of Contents

1. [Overview](#1-overview)
2. [CLAUDE.md Integration](#2-claudemd-integration)
3. [Iteration Strategy (100 Iterations)](#3-iteration-strategy-100-iterations)
4. [Exa Search Queries per Phase](#4-exa-search-queries-per-phase)
5. [iteration-state.json Update Protocol](#5-iteration-statejson-update-protocol)
6. [Claude CLI Commands](#6-claude-cli-commands)
7. [Self-Improvement Loop Pattern](#7-self-improvement-loop-pattern)
8. [Recovery Protocols](#8-recovery-protocols)
9. [Success Criteria](#9-success-criteria)
10. [Quick Reference](#10-quick-reference)

---

## 1. Overview

### Mission Statement

Implement all missing V12 methods in `ralph_loop.py` using Claude CLI in autonomous mode, leveraging Exa search for real-time research on cutting-edge patterns.

### V12 Gap Summary

| Gap ID | Method | Pattern | Severity | Lines Est. |
|--------|--------|---------|----------|------------|
| GAP-001 | `_run_communication_round()` | RIAL/DIAL Emergent Language | 🔴 Critical | 150-200 |
| GAP-002 | `_evaluate_architecture_candidate()` | DARTS NAS Evaluation | 🔴 Critical | 100-150 |
| GAP-003 | `_run_memory_consolidation()` | VAE Generative Replay | 🔴 Critical | 200-250 |
| GAP-004 | `get_v12_insights()` | Full V12 Reporting | 🟡 High | 150-200 |
| GAP-005 | V12 in `run_iteration()` | Main Loop Integration | 🔴 Critical | 100-150 |
| GAP-006 | V12 Artifact Metrics | Observable Progress | 🟡 High | 50-100 |

### Current State Reference

```
iteration-state.json status:
- version: "12.0"
- iteration: 24
- v12_data_structures: 18 (ALL implemented ✅)
- v12_methods_implemented: 7/13 (54%)
- v12_serialization_lines: 370
```

---

## 2. CLAUDE.md Integration

### Required Environment Setup

Create or update `CLAUDE.md` in the workspace root with these sections:

```markdown
# CLAUDE.md - Unleash Platform V12 Autonomous Fix

## Project Context
- **Target File:** platform/core/ralph_loop.py
- **Current Lines:** 4,147
- **Version:** V12 (Partial Implementation)
- **State Tracking:** iteration-state.json

## V12 Gap Resolution Mission
The agent must implement 6 missing V12 methods following patterns from V4-V11.

### Critical Patterns to Follow
1. All V4-V11 methods use `self.state.{subsystem}_state` for state access
2. Type hints are required for all parameters and return values
3. Dataclasses are defined at top of file before RalphLoop class
4. Integration points use iteration modulo triggers (e.g., `if iteration % 10 == 0`)

## File Structure Reference
```
platform/
├── core/
│   ├── ralph_loop.py          # TARGET - 4,147 lines
│   ├── test_ralph_loop_v12.py # V12 tests - 231 lines
│   └── ultimate_orchestrator.py
├── scripts/
└── research_artifacts/
```

## Commands
- `pytest platform/core/test_ralph_loop_v12.py -v` - Run V12 tests
- `python -c "from platform.core.ralph_loop import RalphLoop; print('Import OK')"` - Verify import

## Style Guide
- Use Google-style docstrings
- Maximum line length: 100 characters
- Imports: stdlib, third-party, local (separated by blank lines)
```

### Required MCP Servers

Configure these MCP servers in your Claude Desktop/CLI environment:

```json
{
  "mcpServers": {
    "exa": {
      "command": "npx",
      "args": ["-y", "exa-mcp-server"],
      "env": {
        "EXA_API_KEY": "${EXA_API_KEY}"
      }
    },
    "firecrawl": {
      "command": "npx",
      "args": ["-y", "firecrawl-mcp"],
      "env": {
        "FIRECRAWL_API_KEY": "${FIRECRAWL_API_KEY}"
      }
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git", "--repository", "."]
    },
    "sequentialthinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    }
  }
}
```

### API Keys Required

| Service | Environment Variable | Purpose |
|---------|---------------------|---------|
| Exa | `EXA_API_KEY` | Research paper and code search |
| Firecrawl | `FIRECRAWL_API_KEY` | Deep documentation scraping |
| Anthropic | `ANTHROPIC_API_KEY` | Claude API access |

---

## 3. Iteration Strategy (100 Iterations)

### Phase Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Research & Planning           │ Iterations 25-34  │ 10 iterations │
│ PHASE 2: _run_communication_round()    │ Iterations 35-54  │ 20 iterations │
│ PHASE 3: _evaluate_architecture()      │ Iterations 55-74  │ 20 iterations │
│ PHASE 4: _run_memory_consolidation()   │ Iterations 75-94  │ 20 iterations │
│ PHASE 5: get_v12_insights() + wiring   │ Iterations 95-109 │ 15 iterations │
│ PHASE 6: Testing & Validation          │ Iterations 110-124│ 15 iterations │
└────────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Research & Planning (Iterations 25-34)

**Goal:** Deep research on V12 patterns using Exa search

| Iteration | Focus | Deliverable |
|-----------|-------|-------------|
| 25 | Read entire ralph_loop.py, understand V4-V11 patterns | Pattern analysis doc |
| 26 | Research RIAL/DIAL emergent communication | Research notes in memory |
| 27 | Research DARTS differentiable architecture search | Research notes in memory |
| 28 | Research VAE generative replay, memory consolidation | Research notes in memory |
| 29 | Research world models (Dreamer V4, IRIS) | Research notes in memory |
| 30 | Research predictive coding (PCX, Free Energy) | Research notes in memory |
| 31 | Research active inference patterns | Research notes in memory |
| 32 | Analyze existing V12 dataclasses and their fields | Structure map |
| 33 | Map integration points in run_iteration() | Integration plan |
| 34 | Create implementation blueprint | Blueprint document |

**State Update Template (Phase 1):**
```json
{
  "current_iteration": [
    "Phase 1: Research RIAL/DIAL emergent communication patterns",
    "Document findings in MCP memory",
    "Identify integration requirements"
  ]
}
```

### Phase 2: Implement `_run_communication_round()` (Iterations 35-54)

**Goal:** Implement emergent language exchange between agents

| Iteration | Focus | Code Location |
|-----------|-------|---------------|
| 35-36 | Design message encoding/decoding functions | Helper functions |
| 37-39 | Implement RIAL (Reinforced Inter-Agent Learning) path | Main method |
| 40-42 | Implement DIAL (Differentiable Inter-Agent Learning) path | Main method |
| 43-45 | Add protocol evolution tracking | Protocol state |
| 46-48 | Implement compositionality scoring | Metrics |
| 49-51 | Add communication success rate tracking | Metrics |
| 52-54 | Unit tests and refinement | Tests + fixes |

**Expected Method Signature:**
```python
def _run_communication_round(self) -> Dict[str, Any]:
    """
    Execute emergent language exchange between agents.
    
    Implements RIAL/DIAL protocols for multi-agent communication
    where agents develop shared vocabulary through interaction.
    
    Returns:
        Dict containing:
        - messages_exchanged: int
        - protocol_vocabulary_size: int
        - compositionality_score: float
        - communication_success_rate: float
        - evolved_protocols: List[str]
    """
```

**State Update Template (Phase 2):**
```json
{
  "current_iteration": [
    "Phase 2: Implement _run_communication_round()",
    "Current sub-phase: DIAL implementation",
    "Lines added: ~150"
  ]
}
```

### Phase 3: Implement `_evaluate_architecture_candidate()` (Iterations 55-74)

**Goal:** Implement DARTS-style NAS evaluation

| Iteration | Focus | Code Location |
|-----------|-------|---------------|
| 55-57 | Design architecture encoding (operation scores) | Helper functions |
| 58-60 | Implement continuous relaxation mechanism | Main method |
| 61-63 | Add Pareto-optimal tracking | State management |
| 64-66 | Implement architecture mutation operators | Mutation functions |
| 67-69 | Add performance vs latency tradeoff scoring | Metrics |
| 70-72 | Implement architecture discretization | Post-processing |
| 73-74 | Unit tests and refinement | Tests + fixes |

**Expected Method Signature:**
```python
def _evaluate_architecture_candidate(
    self, 
    candidate: ArchitectureCandidate
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate a neural architecture candidate using DARTS-style scoring.
    
    Args:
        candidate: ArchitectureCandidate with operation_scores
        
    Returns:
        Tuple of (fitness_score, metrics_dict) where metrics_dict contains:
        - accuracy_estimate: float
        - latency_estimate: float
        - parameter_count: int
        - pareto_rank: int
    """
```

### Phase 4: Implement `_run_memory_consolidation()` (Iterations 75-94)

**Goal:** Implement sleep-like VAE compression and generative replay

| Iteration | Focus | Code Location |
|-----------|-------|---------------|
| 75-77 | Design VAE compression interface | Helper functions |
| 78-80 | Implement memory importance scoring | Scoring logic |
| 81-83 | Add generative replay mechanism | Replay functions |
| 84-86 | Implement catastrophic forgetting prevention (EWC) | Protection logic |
| 87-89 | Add knowledge distillation steps | Distillation |
| 90-92 | Implement consolidation scheduling | Scheduler |
| 93-94 | Unit tests and refinement | Tests + fixes |

**Expected Method Signature:**
```python
def _run_memory_consolidation(self) -> Dict[str, Any]:
    """
    Execute sleep-like memory consolidation cycle.
    
    Implements VAE compression and generative replay to consolidate
    important experiences while preventing catastrophic forgetting.
    
    Returns:
        Dict containing:
        - memories_consolidated: int
        - memories_pruned: int
        - compression_ratio: float
        - replay_loss: float
        - ewc_regularization: float
        - consolidated_knowledge: List[str]
    """
```

### Phase 5: Implement `get_v12_insights()` + Integration (Iterations 95-109)

**Goal:** Complete V12 reporting and wire into main loop

| Iteration | Focus | Code Location |
|-----------|-------|---------------|
| 95-97 | Implement get_v12_insights() method | New method |
| 98-100 | Wire V12 initialization in run() | run() method |
| 101-103 | Wire V12 updates in run_iteration() success | run_iteration() |
| 104-106 | Wire V12 updates in run_iteration() failure | run_iteration() |
| 107-109 | Add V12 metrics to artifact data | Artifact creation |

**Expected Method Signature:**
```python
def get_v12_insights(self) -> Dict[str, Any]:
    """
    Get comprehensive V12 subsystem insights.
    
    Returns:
        Dict containing all V12 metrics:
        - world_model: WorldModelState summary
        - predictive_coding: PredictiveCodingState summary
        - active_inference: ActiveInferenceState summary
        - emergent_communication: EmergentCommunicationState summary
        - neural_architecture_search: NeuralArchitectureSearchState summary
        - memory_consolidation: MemoryConsolidationState summary
        - v12_health_score: float (0.0-1.0)
        - recommendations: List[str]
    """
```

### Phase 6: Testing & Validation (Iterations 110-124)

**Goal:** Comprehensive testing and validation

| Iteration | Focus | Deliverable |
|-----------|-------|-------------|
| 110-112 | Run full test suite, fix failures | All tests green |
| 113-115 | E2E test: Run 10-iteration Ralph Loop | Successful run |
| 116-118 | Performance profiling | Timing data |
| 119-121 | Documentation updates | Updated docstrings |
| 122-124 | Final validation, state update | iteration-state.json at 124+ |

---

## 4. Exa Search Queries per Phase

### Phase 1: Research Queries

```bash
# Emergent Communication
"RIAL DIAL emergent communication multi-agent reinforcement learning"
"emergent language neural network multi-agent compositionality"
"differentiable inter-agent learning ICLR 2017"

# DARTS Neural Architecture Search
"DARTS differentiable architecture search continuous relaxation"
"neural architecture search gradient-based ICLR 2019"
"DARTS implementation PyTorch bilevel optimization"

# Memory Consolidation
"memory consolidation VAE generative replay continual learning"
"catastrophic forgetting prevention EWC generative replay"
"sleep-like memory consolidation neural networks"

# World Models
"Dreamer V4 world models latent dynamics"
"RSSM recurrent state space model imagination"
"IRIS transformer world models ICML 2023"

# Predictive Coding
"predictive coding free energy principle implementation"
"PCX predictive coding library Python"
"hierarchical predictive coding precision weighting"

# Active Inference
"active inference expected free energy implementation"
"active inference action selection pymdp"
"epistemic pragmatic value active inference"
```

### Example Exa Search Call

```javascript
// Use with mcp--exa--web_search_exa tool
{
  "query": "RIAL DIAL emergent communication multi-agent implementation code",
  "numResults": 10,
  "type": "auto"
}
```

### Example Firecrawl Scrape for Code

```javascript
// Use with mcp--firecrawl--firecrawl_scrape tool
{
  "url": "https://github.com/facebookresearch/EGG",
  "formats": ["markdown"],
  "onlyMainContent": true
}
```

### Phase-Specific Search Strategy

| Phase | Primary Search | Fallback Search |
|-------|---------------|-----------------|
| Phase 2 (Comm) | Exa: "RIAL DIAL implementation" | Firecrawl: EGG library |
| Phase 3 (NAS) | Exa: "DARTS PyTorch implementation" | Firecrawl: pt.darts repo |
| Phase 4 (Memory) | Exa: "VAE generative replay" | Firecrawl: Avalanche library |

---

## 5. iteration-state.json Update Protocol

### Update Frequency

- **After each phase completion**: Full state update
- **After significant milestones**: Metrics update
- **Every 5 iterations**: Checkpoint update

### Update Template

```json
{
  "version": "12.0",
  "iteration": 35,
  "last_updated": "2026-01-22T15:30:00Z",
  "target": "platform/core/ralph_loop.py",
  "goals": {
    "current_iteration": [
      "Phase 2: Implement _run_communication_round()",
      "Current: RIAL path implementation",
      "Lines added: 75/200"
    ],
    "completed": [
      "... (previous items)",
      "Phase 1: Research & Planning complete",
      "RIAL/DIAL research documented"
    ],
    "next_iteration": [
      "Complete DIAL path implementation",
      "Add compositionality scoring"
    ]
  },
  "metrics": {
    "... (previous metrics)",
    "v12_methods_implemented": 8,
    "v12_communication_round_complete": false,
    "v12_architecture_eval_complete": false,
    "v12_memory_consolidation_complete": false,
    "v12_insights_complete": false,
    "v12_integration_complete": false,
    "v12_artifact_metrics_complete": false,
    "current_phase": 2,
    "phase_progress_percent": 35
  },
  "v12_implementation_status": {
    "_run_communication_round": {
      "status": "in_progress",
      "lines_written": 75,
      "tests_passing": false,
      "blockers": []
    },
    "_evaluate_architecture_candidate": {
      "status": "not_started",
      "lines_written": 0,
      "tests_passing": false,
      "blockers": []
    },
    "_run_memory_consolidation": {
      "status": "not_started",
      "lines_written": 0,
      "tests_passing": false,
      "blockers": []
    },
    "get_v12_insights": {
      "status": "not_started",
      "lines_written": 0,
      "tests_passing": false,
      "blockers": []
    },
    "run_iteration_v12_integration": {
      "status": "not_started",
      "lines_written": 0,
      "tests_passing": false,
      "blockers": []
    },
    "artifact_v12_metrics": {
      "status": "not_started",
      "lines_written": 0,
      "tests_passing": false,
      "blockers": []
    }
  }
}
```

### Update Commands

```bash
# Read current state
cat iteration-state.json | jq '.'

# Update iteration number
jq '.iteration = 35' iteration-state.json > tmp.json && mv tmp.json iteration-state.json

# Add to completed goals
jq '.goals.completed += ["Phase 1: Research complete"]' iteration-state.json > tmp.json && mv tmp.json iteration-state.json

# Update metrics
jq '.metrics.v12_methods_implemented = 8' iteration-state.json > tmp.json && mv tmp.json iteration-state.json
```

### State File Validation

Before each iteration, validate the state file:

```python
import json
from datetime import datetime

with open('iteration-state.json', 'r') as f:
    state = json.load(f)

# Validate structure
assert 'version' in state, "Missing version"
assert 'iteration' in state, "Missing iteration"
assert state['iteration'] >= 24, f"Iteration {state['iteration']} too low"
assert 'goals' in state, "Missing goals"
assert 'metrics' in state, "Missing metrics"

print(f"✅ State valid - Iteration {state['iteration']}")
```

---

## 6. Claude CLI Commands

### Basic Invocation Pattern

```bash
# Start a new session for V12 fix
claude --print "Read iteration-state.json and begin iteration $(cat iteration-state.json | jq '.iteration + 1'). Focus: $(cat iteration-state.json | jq -r '.goals.current_iteration[0]')"
```

### Continue Session

```bash
# Continue existing session
claude --continue --print "Continue V12 gap fix. Current iteration: $(cat iteration-state.json | jq '.iteration'). Status: $(cat iteration-state.json | jq -r '.v12_implementation_status | to_entries | map(select(.value.status == \"in_progress\")) | .[0].key // \"none\"')"
```

### Phase-Specific Commands

```bash
# Phase 1: Research
claude --print "Execute Phase 1 research iteration. Use Exa to search: 'RIAL DIAL emergent communication implementation'. Document findings in MCP memory. Update iteration-state.json when complete."

# Phase 2: Communication Round
claude --print "Execute Phase 2 iteration: Implement _run_communication_round() in ralph_loop.py. Follow V11 method patterns. Use Exa search for RIAL/DIAL reference implementations."

# Phase 3: Architecture Evaluation
claude --print "Execute Phase 3 iteration: Implement _evaluate_architecture_candidate() in ralph_loop.py. Research DARTS bilevel optimization. Follow existing NAS patterns in V12 dataclasses."

# Phase 4: Memory Consolidation
claude --print "Execute Phase 4 iteration: Implement _run_memory_consolidation() in ralph_loop.py. Research VAE generative replay. Follow ConsolidatedMemory and MemoryConsolidationState dataclass patterns."

# Phase 5: Integration
claude --print "Execute Phase 5 iteration: Implement get_v12_insights() and wire V12 into run_iteration(). Follow V11 get_v11_insights() pattern. Add V12 metrics to artifact data."

# Phase 6: Testing
claude --print "Execute Phase 6 iteration: Run pytest platform/core/test_ralph_loop_v12.py -v. Fix any failures. Ensure all V12 methods pass import validation."
```

### Batch Execution Script

```bash
#!/bin/bash
# v12_fix_batch.sh - Run 5 iterations at a time

CURRENT_ITERATION=$(cat iteration-state.json | jq '.iteration')
TARGET_ITERATION=$((CURRENT_ITERATION + 5))

for i in $(seq $((CURRENT_ITERATION + 1)) $TARGET_ITERATION); do
    echo "========== ITERATION $i =========="
    
    # Determine phase
    if [ $i -le 34 ]; then
        PHASE="Phase 1: Research"
    elif [ $i -le 54 ]; then
        PHASE="Phase 2: Communication Round"
    elif [ $i -le 74 ]; then
        PHASE="Phase 3: Architecture Evaluation"
    elif [ $i -le 94 ]; then
        PHASE="Phase 4: Memory Consolidation"
    elif [ $i -le 109 ]; then
        PHASE="Phase 5: Integration"
    else
        PHASE="Phase 6: Testing"
    fi
    
    # Run Claude
    claude --print "$PHASE - Iteration $i. Read iteration-state.json for context. Execute the iteration and update state when complete."
    
    # Wait for completion signal
    read -p "Press Enter when iteration $i is complete..."
    
    # Update iteration number
    jq ".iteration = $i" iteration-state.json > tmp.json && mv tmp.json iteration-state.json
    
    echo "Updated to iteration $i"
done
```

### Automated Loop with Checkpoints

```bash
#!/bin/bash
# v12_autonomous_loop.sh - Fully autonomous execution

MAX_ITERATIONS=124
CHECKPOINT_INTERVAL=5

while [ $(cat iteration-state.json | jq '.iteration') -lt $MAX_ITERATIONS ]; do
    CURRENT=$(cat iteration-state.json | jq '.iteration')
    NEXT=$((CURRENT + 1))
    
    # Create checkpoint if needed
    if [ $((NEXT % CHECKPOINT_INTERVAL)) -eq 0 ]; then
        cp iteration-state.json "checkpoints/iteration-state-$CURRENT.json"
        echo "Checkpoint saved at iteration $CURRENT"
    fi
    
    # Run iteration
    claude --print "Autonomous iteration $NEXT for V12 gap fix. Read iteration-state.json, execute appropriate phase action, update state file when complete. If tests fail, debug and fix before proceeding."
    
    # Verify state was updated
    NEW_ITERATION=$(cat iteration-state.json | jq '.iteration')
    if [ $NEW_ITERATION -eq $CURRENT ]; then
        echo "WARNING: State not updated after iteration $NEXT"
        read -p "Press Enter to continue or Ctrl+C to abort..."
    fi
    
    # Check for fatal errors
    if [ -f "v12_error.flag" ]; then
        echo "Fatal error detected. Check logs and run recovery."
        exit 1
    fi
done

echo "V12 gap fix complete! Reached iteration $MAX_ITERATIONS"
```

---

## 7. Self-Improvement Loop Pattern

### Pre-Iteration Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRE-ITERATION CHECKLIST                      │
├─────────────────────────────────────────────────────────────────┤
│ 1. Read iteration-state.json                                     │
│ 2. Verify current phase and sub-task                            │
│ 3. Check for blockers in v12_implementation_status              │
│ 4. Load relevant research from MCP memory                       │
│ 5. Plan specific action for this iteration                      │
└─────────────────────────────────────────────────────────────────┘
```

### Execute Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│                     EXECUTION PROTOCOL                           │
├─────────────────────────────────────────────────────────────────┤
│ 1. RESEARCH (if needed):                                         │
│    - Use Exa search for pattern references                      │
│    - Use Firecrawl to scrape implementation examples            │
│    - Store findings in MCP memory                               │
│                                                                  │
│ 2. IMPLEMENT:                                                    │
│    - Read target method location in ralph_loop.py               │
│    - Write code following V4-V11 patterns                       │
│    - Use proper type hints and docstrings                       │
│                                                                  │
│ 3. TEST:                                                         │
│    - Run pytest platform/core/test_ralph_loop_v12.py            │
│    - Verify no import errors                                    │
│    - Check for regressions in existing tests                    │
└─────────────────────────────────────────────────────────────────┘
```

### Post-Iteration Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│                    POST-ITERATION CHECKLIST                      │
├─────────────────────────────────────────────────────────────────┤
│ 1. Update iteration number in iteration-state.json             │
│ 2. Move completed goals to completed array                      │
│ 3. Update v12_implementation_status for affected methods        │
│ 4. Update metrics (lines added, methods complete, etc.)         │
│ 5. Log any blockers or issues discovered                        │
│ 6. Set next_iteration goals if phase complete                   │
│ 7. Git commit if milestone reached                              │
└─────────────────────────────────────────────────────────────────┘
```

### Convergence Check

```python
def check_convergence() -> Tuple[bool, str]:
    """Check if V12 implementation is complete."""
    with open('iteration-state.json', 'r') as f:
        state = json.load(f)
    
    v12_status = state.get('v12_implementation_status', {})
    
    # Check all methods
    incomplete = []
    for method, status in v12_status.items():
        if status['status'] != 'complete':
            incomplete.append(method)
    
    if not incomplete:
        return True, "All V12 methods complete!"
    
    # Check tests
    test_result = subprocess.run(
        ['pytest', 'platform/core/test_ralph_loop_v12.py', '-v', '--tb=short'],
        capture_output=True
    )
    
    if test_result.returncode != 0:
        return False, f"Tests failing. Incomplete: {incomplete}"
    
    return False, f"Incomplete methods: {incomplete}"
```

---

## 8. Recovery Protocols

### Checkpoint Recovery

```bash
#!/bin/bash
# recover_from_checkpoint.sh

# Find latest checkpoint
LATEST=$(ls -t checkpoints/iteration-state-*.json 2>/dev/null | head -1)

if [ -z "$LATEST" ]; then
    echo "No checkpoints found!"
    exit 1
fi

echo "Recovering from checkpoint: $LATEST"
cp "$LATEST" iteration-state.json

# Get iteration number
ITERATION=$(cat iteration-state.json | jq '.iteration')
echo "Recovered to iteration $ITERATION"

# Run validation
python -c "
import json
with open('iteration-state.json') as f:
    s = json.load(f)
print(f'State valid: iteration={s[\"iteration\"]}, version={s[\"version\"]}')
"
```

### Rollback Strategy for Test Failures

```bash
#!/bin/bash
# rollback_on_failure.sh

# Save current state
cp iteration-state.json iteration-state.json.failed
cp platform/core/ralph_loop.py platform/core/ralph_loop.py.failed

# Find last known good state
LAST_GOOD=$(jq -r '.last_known_good_iteration // 24' iteration-state.json)
echo "Rolling back to iteration $LAST_GOOD"

# Restore from git
git checkout HEAD~1 -- platform/core/ralph_loop.py

# Restore state
jq ".iteration = $LAST_GOOD" iteration-state.json > tmp.json && mv tmp.json iteration-state.json

# Mark methods as needing review
jq '.v12_implementation_status |= with_entries(if .value.status == "complete" then .value.status = "needs_review" else . end)' \
    iteration-state.json > tmp.json && mv tmp.json iteration-state.json

echo "Rolled back. Review iteration-state.json.failed for what went wrong."
```

### Manual Intervention Triggers

| Condition | Action |
|-----------|--------|
| 3 consecutive test failures | Pause, switch to debug mode |
| Import error in ralph_loop.py | Immediate rollback |
| Iteration stuck > 30 min | Human review required |
| Memory API unavailable | Fall back to local notes |
| Exa API rate limited | Use cached research, slow down |

### Debug Mode Entry

```bash
# Enter debug mode
claude --print "ENTERING DEBUG MODE. Last iteration: $(cat iteration-state.json | jq '.iteration'). 

Read the following files:
1. iteration-state.json - current state
2. platform/core/ralph_loop.py - target file  
3. platform/core/test_ralph_loop_v12.py - test file

Run tests with: pytest platform/core/test_ralph_loop_v12.py -v --tb=long

Identify the root cause of failure and propose a fix before proceeding."
```

### Emergency Stop

Create `v12_error.flag` file to halt automated execution:

```bash
# Emergency stop
echo "STOP: $(date) - Manual review required" > v12_error.flag

# Resume after fix
rm v12_error.flag
```

---

## 9. Success Criteria

### Method Completion Criteria

| Method | Lines | Tests | Integration | Status |
|--------|-------|-------|-------------|--------|
| `_run_communication_round()` | 150+ | 3+ | Called in run_iteration | ⬜ |
| `_evaluate_architecture_candidate()` | 100+ | 2+ | Called by NAS state | ⬜ |
| `_run_memory_consolidation()` | 200+ | 3+ | Triggered every N iterations | ⬜ |
| `get_v12_insights()` | 150+ | 2+ | Returns full V12 state | ⬜ |
| V12 integration in run_iteration() | 100+ | E2E | All V12 subsystems active | ⬜ |
| V12 metrics in artifacts | 50+ | 1+ | Observable in logs | ⬜ |

### Test Validation Commands

```bash
# Run all V12 tests
pytest platform/core/test_ralph_loop_v12.py -v

# Check import
python -c "from platform.core.ralph_loop import RalphLoop; r = RalphLoop(); print('V12 methods:', [m for m in dir(r) if 'v12' in m.lower() or m.startswith('_run_') or m.startswith('_evaluate_')])"

# Verify get_v12_insights exists and returns dict
python -c "
from platform.core.ralph_loop import RalphLoop
r = RalphLoop()
insights = r.get_v12_insights()
assert isinstance(insights, dict), 'Should return dict'
assert 'world_model' in insights, 'Should have world_model'
print('✅ get_v12_insights() valid')
"

# E2E test: Run 5 iterations
python -c "
from platform.core.ralph_loop import RalphLoop
r = RalphLoop()
for i in range(5):
    result = r.run_iteration()
    print(f'Iteration {i}: {\"success\" if result else \"failure\"}')
print('✅ E2E test complete')
"
```

### Final Validation Checklist

```
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL VALIDATION CHECKLIST                    │
├─────────────────────────────────────────────────────────────────┤
│ [ ] All 6 V12 methods implemented                               │
│ [ ] test_ralph_loop_v12.py: ALL TESTS PASSING                   │
│ [ ] No import errors in ralph_loop.py                           │
│ [ ] iteration-state.json at iteration 124+                      │
│ [ ] V12 metrics appearing in artifact data                      │
│ [ ] get_v12_insights() returns complete state                   │
│ [ ] E2E: 10 iterations run successfully                         │
│ [ ] Git commit: "V12 complete - all gaps resolved"              │
│ [ ] Documentation updated                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Success Metrics

| Metric | Target | Verification |
|--------|--------|--------------|
| V12 Methods | 13/13 | Code inspection |
| Test Pass Rate | 100% | pytest output |
| Lines Added | ~750-1000 | git diff --stat |
| Iteration Count | 124+ | iteration-state.json |
| E2E Success | 10/10 iterations | E2E test script |

---

## 10. Quick Reference

### Key Files

| File | Purpose |
|------|---------|
| `platform/core/ralph_loop.py` | Target file (4,147 lines) |
| `platform/core/test_ralph_loop_v12.py` | V12 test suite |
| `iteration-state.json` | Iteration tracking |
| `docs/V12_AUTONOMOUS_FIX_PIPELINE.md` | This document |

### V12 Dataclasses (Already Implemented)

```python
# World Models
LatentState, ImaginedTrajectory, WorldModelState

# Predictive Coding  
PredictiveCodingLayer, PredictionError, PredictiveCodingState

# Active Inference
ExpectedFreeEnergy, ActiveInferenceState

# Emergent Communication
EmergentMessage, CommunicationProtocol, EmergentCommunicationState

# Neural Architecture Search
ArchitectureCandidate, NeuralArchitectureSearchState

# Memory Consolidation
ConsolidatedMemory, MemoryConsolidationState
```

### Research Paper Quick Reference

| Pattern | Paper | Key Insight |
|---------|-------|-------------|
| RIAL/DIAL | ICLR 2017 | Emergent compositional language via RL |
| DARTS | ICLR 2019 | Continuous relaxation for NAS |
| VAE Replay | ICLR 2017 | Prevent forgetting via generation |
| Dreamer V4 | 2025 | RSSM world model imagination |
| PCX | 2024 | Free energy minimization |
| Active Inference | Friston | EFE-based action selection |

### Claude CLI Quick Commands

```bash
# Start V12 fix
claude --print "Begin V12 gap fix from iteration 24"

# Continue session
claude --continue

# Phase-specific
claude --print "Phase 2: Implement _run_communication_round()"

# Debug mode
claude --print "Debug V12 test failures"

# Validation
claude --print "Run complete V12 validation suite"
```

### State Update Quick Commands

```bash
# Increment iteration
jq '.iteration += 1' iteration-state.json > tmp.json && mv tmp.json iteration-state.json

# Mark method complete
jq '.v12_implementation_status._run_communication_round.status = "complete"' iteration-state.json > tmp.json && mv tmp.json iteration-state.json

# Add to completed goals
jq '.goals.completed += ["GAP-001 resolved"]' iteration-state.json > tmp.json && mv tmp.json iteration-state.json
```

---

## Appendix: Research Sources

From iteration-state.json `research_sources`:

| Source | Relevance |
|--------|-----------|
| Dreamer V4 | World Models implementation |
| PCX Library | Predictive Coding reference |
| Free Energy Principle | Active Inference foundation |
| Active Inference MIT Press | EFE calculation |
| RIAL/DIAL ICLR 2017 | Emergent communication |
| DARTS ICLR 2019 | Architecture search |
| EWC PNAS 2017 | Forgetting prevention |
| Generative Replay | Memory consolidation |

---

*Document generated for Claude CLI autonomous execution of V12 gap resolution.*
