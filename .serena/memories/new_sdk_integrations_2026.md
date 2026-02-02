# New SDK Integrations Memory (2026-01-22)

## Quick Reference for Cross-Session Access

### 1. Opik - AI Observability Platform
**Path**: `Z:\insider\AUTO CLAUDE\unleash\sdks\opik-full\`

**Key files**:
- Python SDK: `sdks/python/` - `pip install opik`
- TypeScript SDK: `sdks/typescript/`
- Backend: `apps/opik-backend/`
- Frontend: `apps/opik-frontend/`
- Self-host: `./opik.sh` or `./opik.ps1`

**Capabilities**:
- LLM tracing (40M+ traces/day capacity)
- LLM-as-a-judge evaluation (hallucination, relevance, precision)
- Agent optimizer (automatic prompt improvement)
- Guardrails (safety features)
- Production dashboards

**Quick Start**:
```python
import opik
opik.configure(api_key="KEY", workspace="WORKSPACE")

@opik.track
def my_llm_call(prompt):
    return claude.messages.create(...)
```

### 2. Everything Claude Code - Production Configs
**Path**: `Z:\insider\AUTO CLAUDE\unleash\everything-claude-code-full\`

**9 Agents** (subagents for delegation):
- planner, architect, tdd-guide, code-reviewer
- security-reviewer, build-error-resolver
- e2e-runner, refactor-cleaner, doc-updater

**11 Skills** (domain knowledge):
- coding-standards, backend-patterns, frontend-patterns
- continuous-learning, strategic-compact, tdd-workflow
- security-review, eval-harness, verification-loop
- clickhouse-io, project-guidelines-example

**14 Commands** (slash commands):
- /plan, /tdd, /e2e, /code-review, /build-fix
- /refactor-clean, /learn, /checkpoint, /verify
- /eval, /orchestrate, /test-coverage
- /update-codemaps, /update-docs

**6 Rules**: security, coding-style, testing, git-workflow, agents, performance

**Install as Plugin**:
```
/plugin marketplace add affaan-m/everything-claude-code
/plugin install everything-claude-code@everything-claude-code
```

### Project-Specific Usage

| Project | Opik Use | Everything CC Use |
|---------|----------|-------------------|
| AlphaForge | Trace trading decisions | Security reviewer, verification loop |
| Witness | Monitor MediaPipe | Backend patterns, continuous learning |
| Unleash | Evaluate Ralph Loop | All agents, eval harness |

### Full Documentation
See: `Z:\insider\AUTO CLAUDE\unleash\sdks\NEW_SDK_INTEGRATIONS.md`
