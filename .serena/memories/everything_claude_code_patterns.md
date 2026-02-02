# Everything-Claude-Code Pattern Extraction

**Created**: 2026-01-23
**Source**: `Z:\insider\AUTO CLAUDE\unleash\everything-claude-code-full\`

## Continuous Learning Skill

**Location**: `skills/continuous-learning/SKILL.md`

### Stop Hook Pattern Extraction
```python
# On session stop, extract learnings
patterns = {
    "error_resolution": extract_error_fixes(session),
    "user_corrections": extract_corrections(session),
    "workarounds": extract_workarounds(session),
    "debugging_techniques": extract_debug_patterns(session)
}
save_to_learnings(patterns)
```

### Pattern Types
- **error_resolution**: How errors were diagnosed and fixed
- **user_corrections**: Where user corrected Claude's approach
- **workarounds**: Non-obvious solutions discovered
- **debugging_techniques**: Effective debugging strategies used

## Verification Loop Skill

**Location**: `skills/verification-loop/SKILL.md`

### 6-Phase Verification
1. **Build**: Compile/bundle project
2. **Types**: Run type checker (pyright/tsc)
3. **Lint**: Run linter (ruff/eslint)
4. **Tests**: Run test suite (pytest/vitest)
5. **Security**: Run security scan (semgrep/snyk)
6. **Diff**: Review changes before commit

### Implementation
```bash
# Run all phases
npm run build && \
npx tsc --noEmit && \
npm run lint && \
npm test && \
npm run security && \
git diff --stat
```

## Strategic Compact Skill

**Location**: `skills/strategic-compact/SKILL.md`

### Key Insight
Compact at **logical boundaries**, not arbitrary token thresholds.

### Logical Boundaries
- After completing a feature
- Before starting new task
- After major refactor
- When switching contexts

### Anti-Pattern
❌ Auto-compact at 70% context → Loses mid-task context
✅ Manual compact at task boundary → Preserves logical flow

## Orchestrate Command

**Location**: `commands/orchestrate.md`

### Workflow Types
- **feature**: planner → architect → coder → reviewer
- **bugfix**: debugger → coder → tester
- **refactor**: architect → coder → reviewer
- **security**: security-reviewer → coder → reviewer

### Handoff Document Format
```markdown
# Agent Handoff: [source] → [target]

## Completed Work
- [list of completed items]

## Context for Next Agent
- [relevant context]

## Pending Tasks
- [tasks for next agent]

## Files Modified
- [list of files]
```

## Hooks System

**Location**: `rules/hooks.md`

### Hook Types
- **pre-tool**: Before any tool execution
- **post-tool**: After tool execution
- **session-start**: On session initialization
- **session-stop**: On session end

### Configuration
```json
{
  "hooks": {
    "pre-tool": ["validate_tool_call"],
    "post-tool": ["log_tool_result"],
    "session-start": ["load_context"],
    "session-stop": ["extract_learnings"]
  }
}
```

## Architect Agent

**Location**: `agents/architect.md`

### Responsibilities
- System design decisions
- Component boundaries
- Data flow architecture
- Technology selection

### Trigger Patterns
- "design", "architect", "structure"
- "how should I organize"
- "what's the best approach for"

## Security Rules

**Location**: `rules/security.md`

### OWASP Top 10 Checks
- Injection (SQL, command, XSS)
- Broken authentication
- Sensitive data exposure
- XML external entities
- Broken access control
- Security misconfiguration
- Cross-site scripting
- Insecure deserialization
- Components with vulnerabilities
- Insufficient logging

### Code Review Checklist
- [ ] No hardcoded secrets
- [ ] Input validation on all user input
- [ ] Output encoding for display
- [ ] Parameterized queries
- [ ] Proper error handling (no stack traces)
- [ ] Rate limiting on APIs
- [ ] Authentication on sensitive endpoints
