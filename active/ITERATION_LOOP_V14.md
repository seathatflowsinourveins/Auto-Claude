# UNLEASH Iteration Loop V14 — Production-Ready Autonomous Improvement

> **Version**: V14 | **Based On**: Ralph Loop V13 + Production Research Synthesis
> **Sources**: frankbria/ralph-claude-code, affaan-m/everything-claude-code, claudefa.st L-thread architecture, letta-ai/letta, comet-ml/opik, memory-mcp two-tier pattern
> **Date**: 2026-02-02

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                 UNLEASH ITERATION LOOP V14                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ OBSERVE  │──▶│ ORIENT   │──▶│ DECIDE   │──▶│  ACT     │    │
│  │ (Scan)   │   │ (Analyze)│   │ (Plan)   │   │(Execute) │    │
│  └────┬─────┘   └──────────┘   └──────────┘   └────┬─────┘    │
│       │                                              │          │
│       │         ┌──────────┐   ┌──────────┐         │          │
│       └─────────│ VERIFY   │◀──│ LEARN    │◀────────┘          │
│                 │ (6-Phase)│   │(Persist) │                     │
│                 └──────────┘   └──────────┘                     │
│                                                                   │
│  CROSS-SESSION LAYER                                             │
│  ┌─────────────┬──────────────┬──────────────┐                  │
│  │ Letta Cloud │ File-Based   │ Opik Traces  │                  │
│  │ (Archival)  │ (Checkpoint) │ (Observable) │                  │
│  └─────────────┴──────────────┴──────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: OBSERVE — System Scan

**Goal**: Understand current system state, detect gaps, identify next targets.

```
FOR EACH iteration:
1. Read iteration-state.json for current goals and completed work
2. Run gap detection:
   - Import failures (missing dependencies, broken imports)
   - Test failures (run existing test suites)
   - Stale references (SDK versions vs current)
   - Cross-session memory gaps (Letta agent block freshness)
3. Read ~/.claude/learnings/ for accumulated patterns
4. Check platform/core/ for modules with TODO/FIXME markers
5. Score each gap by: impact (H/M/L) × effort (H/M/L) × dependency-count
```

**Output**: Prioritized gap list with scores.

---

## Phase 2: ORIENT — Research & Analysis

**Goal**: Research solutions using ALL tools in parallel. No assumptions.

```
FOR EACH high-priority gap:
1. PARALLEL RESEARCH (mandatory):
   - Context7: Official SDK docs for affected libraries
   - Exa: Deep search for production patterns
   - Tavily: Advanced search for best practices
   - Jina: Academic papers if architectural
2. CROSS-REFERENCE findings (min 2 sources)
3. VERIFY against real API signatures (no mocks)
4. DOCUMENT corrections in CLAUDE.md if SDK patterns wrong
```

**Anti-patterns**:
- ❌ Skip research → implement from memory
- ❌ Query one source → fallback to another
- ❌ Use heuristic/mock verification
- ✅ ALL sources in parallel → synthesize → verify with real API

---

## Phase 3: DECIDE — Plan Changes

**Goal**: Create concrete, measurable improvement plan.

```
FOR EACH researched solution:
1. Define WHAT changes (files, functions, lines)
2. Define WHY (gap it resolves, expected gain)
3. Define HOW TO VERIFY (specific test command or API call)
4. Quantify expected gains:
   - Latency: "Reduces from Xms to Yms" (or "N/A")
   - Throughput: "Handles X→Y concurrent ops"
   - Reliability: "Failure rate from X% to Y%"
   - Token efficiency: "Reduces by X tokens per call"
   - Memory persistence: "Cross-session recall from X% to Y%"
5. Define measurement methodology:
   - Before/after benchmarks with specific commands
   - Confidence level: HIGH (measured) / MEDIUM (estimated) / LOW (theoretical)
```

---

## Phase 4: ACT — Execute Changes

**Goal**: Implement changes with real code, real APIs.

```
FOR EACH planned change:
1. Create feature branch: git checkout -b v14/[change-name]
2. Implement the change
3. Run 6-Phase Verification Gate:
   BUILD → TYPES → LINT → TEST → SECRETS → DIFF
     ↓       ↓       ↓       ↓        ↓       ↓
    0 err   0 err  <10 warn  pass   0 match  review
4. If verification fails: research why, fix, re-verify (max 3 attempts)
5. If 3 attempts fail: document as blocked, move to next gap
6. Commit with descriptive message
```

---

## Phase 5: LEARN — Persist Knowledge

**Goal**: Ensure every insight survives session boundaries.

### 5A: File-Based Checkpoint (Immediate)
```python
# Update iteration-state.json with:
{
  "version": "14.0",
  "iteration": N+1,
  "last_updated": "ISO-timestamp",
  "completed_this_iteration": ["description of each change"],
  "gaps_resolved": ["gap-id: resolution summary"],
  "gaps_remaining": ["gap-id: why blocked"],
  "learnings": ["new learning from this iteration"],
  "metrics": { "files_changed": N, "tests_added": N, "tests_passing": N }
}
```

### 5B: Letta Cloud Persistence (Cross-Session)
```python
# Using letta-client 1.7.7 verified patterns:
from letta_client import Letta

client = Letta(api_key=LETTA_API_KEY)

# Update learnings block on ECOSYSTEM agent
agent_id = "agent-daee71d2-193b-485e-bda4-ee44752635fe"
client.agents.blocks.update(
    "learnings",            # label (NOT block_id)
    agent_id=agent_id,
    value=json.dumps({
        "iteration": N,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gaps_resolved": [...],
        "patterns_verified": [...],
        "next_priorities": [...]
    })
)

# Store detailed findings in archival memory
client.agents.passages.create(
    agent_id=agent_id,
    content=f"V14 Iteration {N}: {detailed_findings}",
    tags=["v14", "iteration", f"iter-{N}"]
)
```

### 5C: Opik Trace Logging (Observable)
```python
import opik

@opik.track(name="unleash-iteration", project_name="unleash-v14")
def run_iteration(iteration_num: int, gaps: list):
    """Each iteration is a traced span in Opik."""
    results = []
    for gap in gaps:
        with opik.track(name=f"gap-{gap['id']}"):
            result = resolve_gap(gap)
            results.append(result)
    return results
```

### 5D: Instinct Extraction (Learning Engine)
```python
# From platform/core/learning.py
engine = LearningEngine()
engine.record_success(
    task_type="gap_resolution",
    pattern="Verified pattern description",
    context={"sdk": "...", "version": "...", "method": "..."}
)
```

---

## Phase 6: VERIFY — End-of-Iteration Gate

**Goal**: Confirm iteration produced real, measurable improvement.

```
VERIFICATION CHECKLIST:
□ All changes committed to git (no uncommitted work)
□ iteration-state.json updated with new state
□ At least 1 gap resolved with real verification (not mock)
□ Letta Cloud blocks updated with iteration learnings
□ No regressions (existing tests still pass)
□ New learnings documented (either CLAUDE.md or learnings/)
□ Next iteration priorities identified

FORBIDDEN until ALL pass:
- "COMPLETE", "SUCCESS", "Done", "Verified"
```

---

## Dynamic Direction Monitoring

The loop monitors direction by checking at each iteration boundary:

```python
DIRECTION_SIGNALS = {
    "convergence": {
        # If last 3 iterations resolved < 1 gap each
        "action": "shift_focus",
        "trigger": "gaps_resolved_avg < 1.0 over 3 iterations"
    },
    "regression": {
        # If tests that passed now fail
        "action": "stop_and_fix",
        "trigger": "test_pass_count < previous_test_pass_count"
    },
    "diminishing_returns": {
        # If improvement magnitude declining
        "action": "explore_new_area",
        "trigger": "improvement_score declining for 3 iterations"
    },
    "blocked_gaps": {
        # If same gap blocked for 2+ iterations
        "action": "escalate_or_archive",
        "trigger": "gap.blocked_count >= 2"
    },
    "user_redirect": {
        # If user changes priorities mid-loop
        "action": "reload_priorities",
        "trigger": "priority_file_changed"
    }
}
```

---

## Iteration Prompt Template

Use this as the Claude Code prompt for each iteration:

```
You are executing UNLEASH Iteration Loop V14, iteration {N}.

SYSTEM STATE:
- Project: Z:\insider\AUTO CLAUDE\unleash
- Last iteration: {timestamp}
- Gaps resolved last iteration: {count}
- Current priorities: {priority_list}

PHASE 1 - OBSERVE:
Read iteration-state.json. Scan platform/core/ for import errors,
test failures, and stale patterns. Check Letta Cloud for cross-session
learnings from other sessions.

PHASE 2 - ORIENT:
For top 3 gaps, research solutions using ALL tools in parallel
(Context7 + Exa + Tavily + Jina). Cross-reference minimum 2 sources.
Verify against real APIs.

PHASE 3 - DECIDE:
For each solution, define: files to change, expected gains (with
numbers), verification method, and measurement confidence.

PHASE 4 - ACT:
Implement changes. Run 6-phase verification. Commit to git.
No mocks. No heuristic fallbacks. Real API calls only.

PHASE 5 - LEARN:
Update iteration-state.json, Letta Cloud blocks, Opik traces,
and instinct patterns. Document every correction.

PHASE 6 - VERIFY:
Confirm all changes verified, no regressions, state persisted.

DIRECTION CHECK:
After completing, assess: Are we converging? Regressing? Blocked?
Adjust next iteration priorities based on signals.

RULES:
- Research FIRST, implement SECOND
- ALL research tools in PARALLEL
- NO mocks, NO stubs, NO heuristic fallbacks
- Real API verification REQUIRED
- Cross-reference minimum 2 sources
- Update CLAUDE.md for any SDK corrections found
- Commit each meaningful change
```

---

## Expected Gains Per Iteration

| Metric | Target | Measurement |
|--------|--------|-------------|
| Gaps resolved | 2-5 per iteration | Count from iteration-state.json |
| Test coverage | +5% per iteration | pytest --cov delta |
| Import errors | -50% per iteration | python -c "import platform.core.X" |
| Cross-session recall | 90%+ | Letta Cloud query hit rate |
| Token efficiency | -10% per iteration | Opik trace token tracking |
| Regression rate | 0% | Previous tests still pass |

---

## SDK Integration Priority for V14

| Priority | SDK | Gap | Action |
|----------|-----|-----|--------|
| P0 | Letta 1.7.7 | Cross-session blocks not updating | Fix blocks.update() calls |
| P0 | Opik | No tracing integrated | Add @track decorators |
| P1 | Anthropic | Claude API patterns | Verify against Context7 |
| P1 | LangGraph | Checkpoint patterns | Fix CVE-2025-64439 |
| P2 | DSPy | Prompt optimization | Research current patterns |
| P2 | Instructor | Structured output | Verify Pydantic v2 compat |
