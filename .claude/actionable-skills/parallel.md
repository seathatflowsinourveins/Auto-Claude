# /parallel - Parallel Task Agent Skill

## Description
Spawn multiple specialized Task agents concurrently to explore, research, or execute independent work streams in parallel. Maximizes throughput for complex multi-part tasks.

## When to Use
- Multiple independent subtasks identified
- Research across different domains needed
- Codebase exploration with multiple queries
- Concurrent validation/testing

## Parallel Pattern (What I Actually Do)

### Step 1: Decompose Task
```
INTERNAL PROCESS:
- Break task into independent subtasks
- Identify dependencies (must be ZERO for parallelism)
- Assign appropriate subagent types

DEPENDENCY CHECK:
- If subtasks share state → Sequential
- If subtasks are independent → Parallel
```

### Step 2: Spawn Agents
```
TOOL: Task (multiple in single message)

Available subagent_types:
- Explore: Codebase understanding
- Plan: Architecture design
- Bash: Git/command operations
- general-purpose: Multi-step research
- feature-dev:code-architect: Feature design
- feature-dev:code-reviewer: Code review
```

### Step 3: Gather Results
```
Each agent returns findings
I synthesize into coherent response
```

## Example: Research Task

```
User: "Find how errors are handled in both the API and UI layers"

PARALLEL SPAWN:
[
  Task(subagent_type="Explore", prompt="Find error handling in API layer"),
  Task(subagent_type="Explore", prompt="Find error handling in UI layer")
]

Results:
- Agent 1: API uses middleware at src/api/error-handler.ts:45
- Agent 2: UI uses ErrorBoundary at src/components/ErrorBoundary.tsx:12

SYNTHESIS:
"Errors are handled in two places:
1. API: Middleware in src/api/error-handler.ts:45
2. UI: React ErrorBoundary in src/components/ErrorBoundary.tsx:12"
```

## Example: Multi-File Changes

```
User: "Rename getUserById to fetchUserById everywhere"

Step 1: Find all occurrences (single Explore agent)
Step 2: Edit each file (can be parallel if no dependencies)

PARALLEL SPAWN:
[
  Task(subagent_type="Bash", prompt="Edit file1.ts to rename function"),
  Task(subagent_type="Bash", prompt="Edit file2.ts to rename function"),
  Task(subagent_type="Bash", prompt="Edit file3.ts to rename function")
]
```

## When NOT to Use Parallel

```
SEQUENTIAL REQUIRED:
- Step 2 depends on Step 1's output
- File A must be created before File B references it
- Database migration must complete before seeding
- Build must complete before tests

PARALLEL SAFE:
- Independent research queries
- Validation of separate components
- Reading multiple files for context
- Running independent tests
```

## Agent Selection Guide

| Task Type | Subagent | Why |
|-----------|----------|-----|
| Find files | Explore | Optimized for codebase search |
| Git operations | Bash | Command execution |
| Design system | Plan | Architecture thinking |
| Review code | code-reviewer | Security/quality focus |
| Build feature | code-architect | Implementation blueprint |

## Integration with V30.8

This skill implements the Multi-Agent patterns from Section 56 (CrewAI) and my actual Task tool capabilities.

Key insight: The Task tool allows TRUE parallelism - multiple agents running concurrently - but only for independent work.
