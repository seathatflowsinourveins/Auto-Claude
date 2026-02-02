# /iterate - True Iterative Protocol Skill

## Description
Apply the True Iterative Protocol (TIP) for genuine cross-session iteration. Unlike simple loops, TIP uses explicit state persistence to maintain continuity.

## When to Use
- Long-running tasks spanning multiple sessions
- Complex projects requiring iterative refinement
- Ralph Loop style continuous improvement
- Building on previous work systematically

## TIP Protocol (What I Actually Do)

### Step 1: Load State
```
TOOL: Read iteration-state.json

CHECK:
- What iteration am I on?
- What was completed previously?
- What are the current goals?
- What learnings should I apply?
```

### Step 2: Execute Iteration
```
PROCESS:
1. Reference previous work (don't regenerate)
2. Build on existing progress
3. Apply learnings from past iterations
4. Track new discoveries

TOOLS:
- TodoWrite: Track this iteration's progress
- Read/Edit: Modify existing work
- Memory: Recall relevant context
```

### Step 3: Persist State
```
TOOL: Write iteration-state.json

UPDATE:
- Increment iteration number
- Move completed goals to completed
- Add new learnings
- Set next_iteration goals
- Update metrics
```

## State File Structure

```json
{
  "version": "1.0",
  "iteration": 8,
  "last_updated": "2026-01-20T17:30:00Z",
  "target": "CLAUDE_SELF_ENHANCEMENT_V30.md",
  "goals": {
    "current_iteration": ["Goal 1", "Goal 2"],
    "completed": ["Past goal 1", "Past goal 2"],
    "next_iteration": ["Future goal 1"]
  },
  "metrics": {
    "sections_added": 18,
    "patterns_documented": 72
  },
  "learnings": [
    "Key insight 1",
    "Key insight 2"
  ]
}
```

## Why State Files Matter

```
WITHOUT STATE:
Session 1: Generate solution A
Session 2: Generate solution B (no memory of A)
Session 3: Generate solution C (no memory of A or B)
→ No actual iteration, just repeated generation

WITH STATE:
Session 1: Generate A, save to state
Session 2: Read state, improve A → A'
Session 3: Read state, improve A' → A''
→ Genuine iterative improvement
```

## Memory + State Integration

```
SESSION START:
1. Read iteration-state.json (explicit state)
2. Search episodic memory (implicit context)
3. Load CLAUDE.md (project context)
→ Full context restored

SESSION END:
1. Update iteration-state.json (persist explicit state)
2. Save learnings to memory (optional for significant insights)
→ State ready for next session
```

## Example: V30 Document Iteration

```
Iteration 7: Added sections 37-50 (OODA, RISE, etc.)
  State: {iteration: 7, sections: 50}

Iteration 8: Added sections 51-60 (Voice, Structured, Actionable)
  Read state → Know I'm on iteration 8
  Build on 50 sections → Add 10 more
  Update state → {iteration: 8, sections: 60}

Iteration 9: Create skill files, benchmark patterns
  Read state → Know sections 60 complete
  Focus on skills → New work
  Update state → {iteration: 9, skills_created: 5}
```

## Anti-Patterns

```
DON'T:
- Regenerate from scratch each session
- Ignore previous work
- Forget to update state file
- Skip reading state at session start

DO:
- Always read state first
- Build on previous iterations
- Update state immediately after progress
- Track learnings explicitly
```

## Integration with V30.8

This skill implements Section 45 (True Iterative Protocol) and Section 59 (Actionable Workflow) of CLAUDE_SELF_ENHANCEMENT_V30.md

Key insight: Iteration requires explicit state persistence. Memory alone isn't enough - I need a structured state file to know EXACTLY where I left off.
