# Everything Claude Code - Complete V33 Integration Reference

**Repository**: `Z:\insider\AUTO CLAUDE\unleash\everything-claude-code-full\`
**Author**: Affaan Mustafa (@affaanmustafa)
**Origin**: Anthropic hackathon winner configs, 10+ months of daily use
**Integration Date**: 2026-01-23

---

## COMPONENT INVENTORY

### Plugin Structure
```
everything-claude-code/
├── .claude-plugin/
│   ├── plugin.json          # Plugin metadata
│   └── marketplace.json     # Self-hosted marketplace config
├── agents/                   # 9 specialized subagents
├── commands/                 # 15 slash commands
├── skills/                   # 11 workflow skills
├── rules/                    # 8 always-follow guidelines
├── hooks/                    # hooks.json + implementations
├── scripts/                  # 8 cross-platform Node.js scripts
├── tests/                    # Test suite
├── contexts/                 # Dynamic context injection
└── mcp-configs/              # MCP server configurations
```

---

## 1. AGENTS (9 Total)

### Core Development Agents
| Agent | Model | Tools | Purpose |
|-------|-------|-------|---------|
| **planner** | sonnet | Read, Grep, Glob, Task | Implementation planning, risk identification, phase breakdown |
| **architect** | opus | Read, Grep, Glob, Task, WebSearch | System design, ADR creation, tech stack selection |
| **tdd-guide** | sonnet | Read, Write, Edit, Bash, Glob, Grep | TDD enforcement: RED → GREEN → REFACTOR, 80% coverage |
| **code-reviewer** | sonnet | Read, Grep, Glob, Bash | Code quality: CRITICAL/HIGH/MEDIUM/LOW severity |
| **security-reviewer** | opus | Read, Grep, Glob, Bash | OWASP Top 10, secrets detection, vulnerability scanning |

### Specialized Agents
| Agent | Model | Tools | Purpose |
|-------|-------|-------|---------|
| **build-error-resolver** | sonnet | Read, Write, Edit, Bash, Glob, Grep | TypeScript/build error fixing with analysis format |
| **e2e-runner** | sonnet | Read, Write, Edit, Bash, Glob, Grep, Task | Playwright E2E: Page Object Model, flaky test handling |
| **refactor-cleaner** | sonnet | Read, Write, Edit, Bash, Glob, Grep, Task | Dead code removal: knip, depcheck, ts-prune |
| **doc-updater** | haiku | Read, Write, Edit, Glob, Grep, Bash, Task | Documentation sync, codemap generation |

### Agent Invocation Rules
- **planner**: Complex features, refactoring projects
- **architect**: System design decisions, API boundaries
- **tdd-guide**: New features, bug fixes (PROACTIVE - no prompt needed)
- **code-reviewer**: After writing code (PROACTIVE)
- **security-reviewer**: Before commits
- **build-error-resolver**: When build fails
- **e2e-runner**: Critical user flows
- **refactor-cleaner**: Code maintenance
- **doc-updater**: Documentation updates

---

## 2. COMMANDS (15 Total)

### Core Workflow Commands
| Command | Purpose | Key Behavior |
|---------|---------|--------------|
| `/plan` | Implementation planning | Creates phase-based plan with risks |
| `/tdd` | Test-driven development | Enforces RED→GREEN→REFACTOR cycle |
| `/code-review` | Quality review | Generates CRITICAL/HIGH/MEDIUM/LOW findings |
| `/build-fix` | Fix build errors | Analyzes and fixes type/build errors |
| `/e2e` | E2E test generation | Playwright tests with POM pattern |
| `/refactor-clean` | Dead code removal | Uses knip, depcheck, ts-prune |
| `/verify` | Verification loop | Build→Types→Lint→Tests→Security→Diff |

### Advanced Commands
| Command | Purpose | Key Behavior |
|---------|---------|--------------|
| `/checkpoint` | Save verification state | Phase transition marker |
| `/learn` | Extract patterns | Mid-session pattern extraction |
| `/eval` | Eval-driven development | Capability + Regression evals, pass@k |
| `/orchestrate` | Agent workflows | feature/bugfix/refactor/security pipelines |
| `/update-codemaps` | Architecture docs | INDEX.md, frontend.md, backend.md |
| `/update-docs` | Documentation sync | README, API docs, inline docs |
| `/test-coverage` | Coverage analysis | 80% minimum requirement |
| `/setup-pm` | Package manager | Configure npm/pnpm/yarn/bun |

---

## 3. SKILLS (11 Total)

### Core Development Skills
| Skill | Lines | Key Patterns |
|-------|-------|--------------|
| **backend-patterns** | 583 | Repository, Service Layer, Middleware, Cache-Aside, Rate Limiting |
| **frontend-patterns** | 632 | Component Composition, Compound Components, Render Props, Custom Hooks |
| **coding-standards** | 521 | KISS, DRY, YAGNI, Immutability (CRITICAL - spread operator) |
| **security-review** | 495 | OWASP, Secrets, SQL injection, XSS/CSRF, Blockchain (Solana) |
| **tdd-workflow** | 410 | RED→GREEN→REFACTOR, Unit/Integration/E2E, Mocking patterns |

### Advanced Skills
| Skill | Lines | Key Patterns |
|-------|-------|--------------|
| **continuous-learning** | 81 | Stop hook pattern extraction, Session evaluation |
| **eval-harness** | 222 | Capability/Regression evals, pass@k metrics, Grader types |
| **strategic-compact** | 64 | Manual /compact at logical intervals |
| **verification-loop** | 121 | 6-phase verification: Build→Types→Lint→Tests→Security→Diff |
| **clickhouse-io** | 430 | MergeTree, AggregatingMergeTree, Materialized Views, ETL |
| **project-guidelines-example** | 346 | Project-specific skill template (Zenith example) |

---

## 4. RULES (8 Total)

| Rule | Key Requirements |
|------|------------------|
| **agents.md** | Parallel execution, multi-perspective analysis, immediate agent usage |
| **coding-style.md** | Immutability (CRITICAL), 200-400 lines/file (800 max), Zod validation |
| **git-workflow.md** | Conventional commits, Plan→TDD→Review→Commit flow |
| **hooks.md** | PreToolUse, PostToolUse, Stop hook patterns, TodoWrite best practices |
| **patterns.md** | ApiResponse interface, useDebounce, Repository pattern, Skeleton projects |
| **performance.md** | Haiku (90% Sonnet @ 3x savings), Sonnet (main), Opus (reasoning) |
| **security.md** | Mandatory checks before ANY commit, Secret rotation protocol |
| **testing.md** | 80% minimum, TDD mandatory, tdd-guide + e2e-runner agents |

---

## 5. HOOKS SYSTEM

### hooks.json Structure (6 Hook Types)

#### PreToolUse (5 hooks)
1. **Dev server blocker**: Forces tmux for dev servers
2. **Long-running reminder**: Suggests tmux for npm/pnpm/yarn/cargo/docker
3. **Git push review**: Reminder before push
4. **Doc blocker**: Blocks unnecessary .md/.txt creation
5. **Strategic compact**: Suggests /compact at intervals

#### PostToolUse (4 hooks)
1. **PR logger**: Logs PR URL after `gh pr create`
2. **Prettier**: Auto-formats JS/TS after Edit
3. **TypeScript check**: Runs tsc after .ts/.tsx edits
4. **console.log warning**: Warns about console.log statements

#### PreCompact (1 hook)
- **State saver**: Logs compaction event, updates session file

#### SessionStart (1 hook)
- **Context loader**: Finds recent sessions, learned skills, detects PM

#### SessionEnd (2 hooks)
1. **State persister**: Creates/updates session file
2. **Pattern extractor**: Triggers continuous learning evaluation

#### Stop (1 hook)
- **console.log audit**: Checks all modified files for console.log

---

## 6. CROSS-PLATFORM SCRIPTS (8 Total)

### Library Scripts (`scripts/lib/`)
| Script | Lines | Purpose |
|--------|-------|---------|
| **utils.js** | 369 | Cross-platform utilities: findFiles, grepFile, readStdinJson |
| **package-manager.js** | 391 | PM detection: npm/pnpm/yarn/bun with 6-level priority |

### Hook Scripts (`scripts/hooks/`)
| Script | Lines | Purpose |
|--------|-------|---------|
| **session-start.js** | 62 | Load sessions, learned skills, detect PM |
| **session-end.js** | 83 | Create/update session file |
| **pre-compact.js** | 49 | Log compaction, update session |
| **suggest-compact.js** | 61 | Track tool calls, suggest at threshold |
| **evaluate-session.js** | 79 | Continuous learning: pattern extraction |

### Setup Scripts
| Script | Lines | Purpose |
|--------|-------|---------|
| **setup-package-manager.js** | 207 | CLI for PM configuration |

---

## 7. PACKAGE MANAGER DETECTION

### Detection Priority (6 levels)
1. Environment variable: `CLAUDE_PACKAGE_MANAGER`
2. Project config: `.claude/package-manager.json`
3. package.json: `packageManager` field
4. Lock file: package-lock.json / yarn.lock / pnpm-lock.yaml / bun.lockb
5. Global config: `~/.claude/package-manager.json`
6. Fallback: First available (pnpm > bun > yarn > npm)

### PM Configuration Object
```javascript
PACKAGE_MANAGERS = {
  npm: { lockFile: 'package-lock.json', installCmd: 'npm install', runCmd: 'npm run', execCmd: 'npx' },
  pnpm: { lockFile: 'pnpm-lock.yaml', installCmd: 'pnpm install', runCmd: 'pnpm', execCmd: 'pnpm dlx' },
  yarn: { lockFile: 'yarn.lock', installCmd: 'yarn', runCmd: 'yarn', execCmd: 'yarn dlx' },
  bun: { lockFile: 'bun.lockb', installCmd: 'bun install', runCmd: 'bun run', execCmd: 'bunx' }
}
```

---

## 8. KEY PATTERNS

### Immutability Pattern (CRITICAL)
```javascript
// WRONG: Mutation
function updateUser(user, name) {
  user.name = name  // MUTATION!
  return user
}

// CORRECT: Immutability
function updateUser(user, name) {
  return { ...user, name }
}
```

### ApiResponse Pattern
```typescript
interface ApiResponse<T> {
  success: boolean
  data?: T
  error?: string
  meta?: { total: number; page: number; limit: number }
}
```

### Repository Pattern
```typescript
interface Repository<T> {
  findAll(filters?: Filters): Promise<T[]>
  findById(id: string): Promise<T | null>
  create(data: CreateDto): Promise<T>
  update(id: string, data: UpdateDto): Promise<T>
  delete(id: string): Promise<void>
}
```

### Eval Report Format
```
EVAL REPORT: feature-xyz
========================

Capability Evals:
  create-user:     PASS (pass@1)
  validate-email:  PASS (pass@2)
  Overall:         3/3 passed

Regression Evals:
  login-flow:      PASS
  Overall:         3/3 passed

Metrics:
  pass@1: 67%
  pass@3: 100%

Status: READY FOR REVIEW
```

### Verification Report Format
```
VERIFICATION REPORT
==================

Build:     [PASS/FAIL]
Types:     [PASS/FAIL] (X errors)
Lint:      [PASS/FAIL] (X warnings)
Tests:     [PASS/FAIL] (X/Y passed, Z% coverage)
Security:  [PASS/FAIL] (X issues)
Diff:      [X files changed]

Overall:   [READY/NOT READY] for PR
```

---

## 9. INTEGRATION WITH EXISTING SYSTEMS

### AlphaForge Trading System
- **security-reviewer** agent for code audits
- **verification-loop** skill for pre-deploy checks
- **eval-harness** for trading strategy validation
- Opik tracing integration

### State of Witness Creative
- **backend-patterns** for API design
- **continuous-learning** for aesthetic discoveries
- Opik monitoring integration

### UNLEASH Meta-Project
- All 9 agents active
- Full eval-harness integration
- Cross-session memory via continuous-learning

---

## 10. USAGE QUICK REFERENCE

### Installation
```bash
# Plugin install
/plugin marketplace add affaan-m/everything-claude-code
/plugin install everything-claude-code@everything-claude-code
```

### Daily Workflow
```
1. /plan                 # Plan feature
2. /tdd                  # Write tests first
3. [implement]           # Write code
4. /code-review          # Review quality
5. /verify               # Full verification
6. /commit               # Commit with conventional format
```

### Agent Invocation
```
# Automatic (no prompt needed)
- Complex feature → planner agent
- Code written → code-reviewer agent
- Bug fix → tdd-guide agent

# Parallel execution
Launch 3 agents in parallel:
1. Security analysis
2. Performance review
3. Type checking
```

---

**Document Version**: 1.0.0
**Last Updated**: 2026-01-23
**Status**: V33 Integration Complete
