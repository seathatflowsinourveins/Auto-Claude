# Advanced Claude Code Patterns - Implementation Guide

**Created**: 2026-01-23
**Purpose**: Code-level patterns for maximum capability enhancement

---

## 1. SKILL STRUCTURE (Progressive Disclosure)

### Minimal Skill (~100 tokens)
```
skills/my-skill/
├── SKILL.md      # Instructions + triggering conditions
├── scripts/      # Optional: executable scripts
└── references/   # Optional: reference docs
```

### SKILL.md Template
```markdown
# Skill Name

## Description
One-line purpose statement.

## Triggering Conditions
- When user asks about X
- Before performing Y operation
- After Z event occurs

## Instructions
Step-by-step guidance with:
1. Concrete actions
2. Specific file paths
3. Exact commands

## Anti-patterns
What NOT to do.
```

---

## 2. AGENT DEFINITION PATTERNS

### Simple Agent
```markdown
# Agent: code-reviewer

## Purpose
Review code for security, quality, and performance issues.

## Triggering
Activate when user requests code review or after significant changes.

## Process
1. Scan for CRITICAL: hardcoded credentials, SQL injection, XSS
2. Check for HIGH: large functions, deep nesting, missing error handling
3. Identify MEDIUM: O(n²) algorithms, N+1 queries, missing memoization

## Output Format
CRITICAL/WARNING/SUGGESTION with file:line references
```

### Subagent Delegation Pattern
```markdown
## When to Delegate
- Task requires specialized knowledge
- Parallel execution beneficial
- Fresh context needed (prevent context rot)

## Delegation Template
Use the Task tool with:
- Clear objective
- Required context only
- Expected output format
- Success criteria
```

---

## 3. HOOK PATTERNS

### PreToolUse Hook
```json
{
  "event": "PreToolUse",
  "tools": ["Bash"],
  "condition": "command contains 'npm run dev'",
  "action": "warn",
  "message": "Long-running process detected. Use tmux?"
}
```

### PostToolUse Hook
```json
{
  "event": "PostToolUse",
  "tools": ["Edit", "Write"],
  "condition": "file matches *.ts or *.tsx",
  "action": "run",
  "command": "npx prettier --write {file}"
}
```

### Session Lifecycle Hook
```json
{
  "event": "Stop",
  "action": "evaluate",
  "tasks": [
    "Extract learnable patterns",
    "Save session context",
    "Log performance metrics"
  ]
}
```

---

## 4. SLASH COMMAND PATTERNS

### Simple Command
```markdown
# /verify

Run verification loop:
1. Build check
2. Type check
3. Lint check
4. Test suite
5. Security scan
6. Diff review

Output: VERIFICATION REPORT (READY/NOT READY)
```

### Multi-Phase Command
```markdown
# /plan

## Phase 1: Requirements
- Clarify scope with user
- Identify constraints

## Phase 2: Analysis
- Review existing code
- Map dependencies

## Phase 3: Design
- Architecture decisions
- Risk assessment

## Phase 4: Output
- Step-by-step implementation plan
- File change list
- Test strategy
```

### Agent-Spawning Command
```markdown
# /orchestrate

1. Parse user intent
2. Decompose into subtasks
3. Spawn specialized agents:
   - planner for design
   - architect for structure
   - tdd-guide for testing
4. Coordinate results
5. Present unified output
```

---

## 5. CONTEXT MANAGEMENT PATTERNS

### Strategic Compaction
```markdown
## When to Compact
- After completing major milestone
- Before starting new feature
- Context usage > 80%

## What to Preserve
- Current task state
- Recent decisions
- Active file references
- Todo list

## What to Archive
- Completed tasks
- Historical context
- Verbose outputs
```

### Memory Persistence
```markdown
## Cross-Session Storage
1. Learnings → ~/.claude/learnings/
2. Patterns → Serena memories
3. Decisions → CLAUDE.local.md
4. Context → Letta memory blocks

## Retrieval Priority
1. Local CLAUDE.md (project-specific)
2. Serena memories (semantic search)
3. Episodic memory (conversation history)
4. Global CLAUDE.md (user preferences)
```

---

## 6. TDD WORKFLOW PATTERN

### Red-Green-Refactor Cycle
```markdown
## RED Phase
1. Write failing test first
2. Run test - verify it fails
3. Commit: "test: add failing test for X"

## GREEN Phase
1. Write MINIMAL code to pass
2. Run test - verify it passes
3. Commit: "feat: implement X"

## REFACTOR Phase
1. Improve code quality
2. Run tests - verify still passing
3. Commit: "refactor: clean up X"
```

### 80% Coverage Requirement
```markdown
## Coverage Gates
- Unit tests: 80% line coverage
- Integration tests: Critical paths
- E2E tests: User journeys

## Before Completion
1. Run coverage report
2. Identify gaps
3. Add missing tests
4. Verify threshold met
```

---

## 7. SELF-IMPROVEMENT LOOP

### Pattern Extraction (On Session Stop)
```python
# Pseudo-code for continuous learning
patterns = {
    "error_resolution": [],  # How errors were fixed
    "user_corrections": [],  # What user corrected
    "successful_approaches": [],  # What worked well
    "anti_patterns": []  # What to avoid
}

# Save to ~/.claude/skills/learned/
```

### Capability Enhancement
```markdown
## Daily Improvement Cycle
1. Review session learnings
2. Update skill instructions
3. Refine prompt templates
4. Test improvements
5. Persist successful changes

## Weekly Optimization
1. Analyze pattern trends
2. Identify skill gaps
3. Research new techniques
4. Integrate best practices
```

---

## 8. ORCHESTRATION PATTERNS

### Parallel Agent Dispatch
```markdown
## When to Parallelize
- Independent subtasks
- No data dependencies
- Time-critical operations

## Dispatch Template
Use Task tool with multiple parallel calls:
- Agent 1: Research task
- Agent 2: Code analysis
- Agent 3: Test generation

## Coordination
- Wait for all to complete
- Merge results
- Resolve conflicts
- Present unified output
```

### Sequential Pipeline
```markdown
## When to Sequence
- Data dependencies exist
- Order matters
- State must propagate

## Pipeline Template
1. Agent A: Input → Processed
2. Agent B: Processed → Analyzed
3. Agent C: Analyzed → Output

## Checkpointing
- Save state after each stage
- Enable resume on failure
- Track progress
```

---

*Document Version: 1.0 | Last Updated: 2026-01-23*
