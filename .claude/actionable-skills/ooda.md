# /ooda - OODA Decision Loop Skill

## Description
Apply the OODA (Observe-Orient-Decide-Act) decision loop for rapid, structured decision-making. Each phase maps to specific Claude Code actions I can execute.

## When to Use
- Facing a decision point with multiple options
- Debugging or troubleshooting issues
- Evaluating user requests before execution
- Rapid assessment of codebase state

## OODA Phases (What I Actually Do)

### Phase 1: OBSERVE
```
TOOLS I USE:
- Read: Examine relevant files
- Grep: Search for patterns
- Glob: Find related files
- Bash: Check system state (git status, etc.)
- Memory: Recall past relevant work

OUTPUT: Raw data about current situation
```

### Phase 2: ORIENT
```
INTERNAL PROCESS:
- Compare observations to past experiences
- Identify patterns and anomalies
- Consider user's context and preferences
- Assess confidence in understanding

OUTPUT: Mental model of the situation
```

### Phase 3: DECIDE
```
INTERNAL PROCESS:
- Generate possible actions
- Evaluate each against:
  - Likelihood of success
  - Risk of harm
  - Alignment with user intent
  - Resource cost
- Select best action

OUTPUT: Decision with rationale
```

### Phase 4: ACT
```
TOOLS I USE:
- Edit/Write: Make changes
- Bash: Execute commands
- Task: Spawn subagents if needed
- TodoWrite: Track progress

OUTPUT: Executed action with verification
```

## Example Invocation

When the user asks something ambiguous:

```
OBSERVE: What did the user write? What files are relevant? What's the git state?
ORIENT: Is this a bug fix, feature request, or question? What do I know about their project?
DECIDE: Should I ask for clarification, explore first, or act immediately?
ACT: Execute the chosen approach with appropriate tools
```

## Integration with V30.8

This skill implements Section 37 (OODA Loops) of CLAUDE_SELF_ENHANCEMENT_V30.md

The key insight: OODA is not just a mental model - each phase has concrete tool mappings that I can execute.
