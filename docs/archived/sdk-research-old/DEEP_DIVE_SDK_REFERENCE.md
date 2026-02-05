# Deep Dive SDK Reference - Opik & Everything Claude Code

**Created**: 2026-01-22
**Purpose**: Comprehensive cross-session reference for seamless access

---

## 1. OPIK - AI Observability Platform

### Location
`Z:\insider\AUTO CLAUDE\unleash\sdks\opik-full\`

### Installation
```bash
pip install opik
opik configure  # Interactive setup
```

### Core API Exports
```python
import opik

# Tracing
@opik.track  # Decorator for automatic tracing
opik.flush_tracker()  # Flush pending traces
opik.start_as_current_span()  # Context manager
opik.start_as_current_trace()  # Context manager

# Runtime control
opik.set_tracing_active(True/False)  # Toggle tracing
opik.is_tracing_active()  # Check status
opik.reset_tracing_to_config_default()  # Reset

# Client
client = opik.Opik()  # Main client

# Objects
opik.Trace, opik.Span  # Tracing objects
opik.Dataset  # Dataset management
opik.Prompt, opik.ChatPrompt  # Prompt templates
opik.Attachment  # File attachments

# Evaluation
opik.evaluate()  # Run evaluations
opik.evaluate_prompt()  # Evaluate prompts
opik.evaluate_experiment()  # Experiment evaluation
opik.evaluate_on_dict_items()  # Dict-based eval

# Simulation
opik.SimulatedUser  # User simulation
opik.run_simulation()  # Run simulation

# Local recording
opik.record_traces_locally()  # Local trace storage
```

### 50+ Evaluation Metrics

#### LLM-as-Judge Metrics
```python
from opik.evaluation.metrics import (
    # Core RAG metrics
    Hallucination,
    AnswerRelevance,
    ContextPrecision,
    ContextRecall,
    Moderation,
    Usefulness,

    # Agent metrics
    AgentTaskCompletionJudge,
    AgentToolCorrectnessJudge,
    TrajectoryAccuracy,

    # Bias detection
    GenderBiasJudge,
    PoliticalBiasJudge,
    ReligiousBiasJudge,
    RegionalBiasJudge,
    DemographicBiasJudge,

    # Quality metrics
    DialogueHelpfulnessJudge,
    QARelevanceJudge,
    SummarizationCoherenceJudge,
    SummarizationConsistencyJudge,
    ComplianceRiskJudge,
    PromptUncertaintyJudge,

    # G-Eval framework
    GEval,
    GEvalPreset,

    # Structured output
    StructuredOutputCompliance,
    SycEval,
)
```

#### Heuristic Metrics (No LLM Required)
```python
from opik.evaluation.metrics import (
    # Text comparison
    Contains,
    Equals,
    RegexMatch,
    LevenshteinRatio,

    # NLP metrics
    SentenceBLEU, CorpusBLEU,
    ROUGE,
    GLEU,
    METEOR,
    ChrF,
    BERTScore,

    # Distribution
    JSDivergence, JSDistance, KLDivergence,
    SpearmanRanking,

    # Quality
    Readability,
    Tone,
    Sentiment, VADERSentiment,

    # Safety
    PromptInjection,
    LanguageAdherenceMetric,

    # Format
    IsJson,
)
```

#### Conversation Metrics
```python
from opik.evaluation.metrics import (
    ConversationThreadMetric,
    ConversationDegenerationMetric,
    KnowledgeRetentionMetric,
    ConversationalCoherenceMetric,
    SessionCompletenessQuality,
    UserFrustrationMetric,
)
```

### 16 Framework Integrations
```python
# Available integrations
from opik.integrations.anthropic import track_anthropic
from opik.integrations.openai import track_openai
from opik.integrations.langchain import track_langchain
from opik.integrations.llama_index import track_llama_index
from opik.integrations.litellm import track_litellm
from opik.integrations.dspy import track_dspy
from opik.integrations.crewai import track_crewai
from opik.integrations.haystack import track_haystack
from opik.integrations.bedrock import track_bedrock
from opik.integrations.genai import track_genai  # Google
from opik.integrations.aisuite import track_aisuite
from opik.integrations.adk import track_adk  # Google ADK
from opik.integrations.guardrails import track_guardrails
from opik.integrations.harbor import track_harbor
from opik.integrations.sagemaker import auth as sagemaker_auth
```

### Usage Example - Claude with Opik
```python
import anthropic
import opik
from opik.integrations.anthropic import track_anthropic

# Configure
opik.configure(api_key="YOUR_KEY", workspace="YOUR_WORKSPACE")

# Wrap client
client = track_anthropic(anthropic.Anthropic())

# Use normally - all calls traced
response = client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

# Manual tracing
@opik.track(name="my_function", tags=["production"])
def my_llm_pipeline(prompt):
    return client.messages.create(...)
```

### Self-Hosting
```bash
cd opik-full
./opik.sh              # Full suite (Windows: .\opik.ps1)
./opik.sh --backend    # Backend + infra only
./opik.sh --guardrails # With guardrails

# Access: http://localhost:5173
```

---

## 2. EVERYTHING CLAUDE CODE - Production Configs

### Location
`Z:\insider\AUTO CLAUDE\unleash\everything-claude-code-full\`

### Installation
```bash
# As Claude Code Plugin (recommended)
/plugin marketplace add affaan-m/everything-claude-code
/plugin install everything-claude-code@everything-claude-code

# Or add to ~/.claude/settings.json:
{
  "extraKnownMarketplaces": {
    "everything-claude-code": {
      "source": {"source": "github", "repo": "affaan-m/everything-claude-code"}
    }
  },
  "enabledPlugins": {
    "everything-claude-code@everything-claude-code": true
  }
}
```

### 9 Specialized Agents

#### 1. Planner (`agents/planner.md`)
- Feature implementation planning
- Requirements analysis, architecture review
- Step breakdown with dependencies
- Risk assessment

#### 2. Architect (`agents/architect.md`)
- System design decisions
- Trade-off analysis with ADRs
- Scalability planning
- Pattern recommendations (CQRS, Event Sourcing)

#### 3. Code Reviewer (`agents/code-reviewer.md`)
- Security checks (CRITICAL): Hardcoded credentials, SQL injection, XSS
- Code quality (HIGH): Large functions, deep nesting, missing error handling
- Performance (MEDIUM): O(n²) algorithms, N+1 queries, missing memoization
- Output: CRITICAL/WARNING/SUGGESTION priorities

#### 4. Security Reviewer (`agents/security-reviewer.md`)
- Vulnerability analysis
- OWASP Top 10 checks
- Dependency auditing

#### 5. TDD Guide (`agents/tdd-guide.md`)
- Test-driven development methodology
- Red-green-refactor cycle

#### 6. E2E Runner (`agents/e2e-runner.md`)
- Playwright E2E testing
- User journey validation

#### 7. Build Error Resolver (`agents/build-error-resolver.md`)
- Fix compilation errors
- Dependency resolution

#### 8. Refactor Cleaner (`agents/refactor-cleaner.md`)
- Dead code removal
- Code smell detection

#### 9. Doc Updater (`agents/doc-updater.md`)
- Documentation synchronization

### 11 Skills (Domain Knowledge)

#### 1. Continuous Learning (`skills/continuous-learning/`)
- Stop hook for pattern extraction
- Auto-saves learned patterns to `~/.claude/skills/learned/`
- Patterns: error_resolution, user_corrections, workarounds

#### 2. Eval Harness (`skills/eval-harness/`)
- Eval-Driven Development (EDD)
- Capability evals, Regression evals
- Grader types: Code-based, Model-based, Human
- Metrics: pass@k, pass^k

#### 3. Verification Loop (`skills/verification-loop/`)
- 6 phases: Build, Type, Lint, Test, Security, Diff
- Continuous verification every 15 minutes
- Output: VERIFICATION REPORT (READY/NOT READY)

#### 4. Strategic Compact (`skills/strategic-compact/`)
- Manual compaction suggestions
- Context management

#### 5. TDD Workflow (`skills/tdd-workflow/`)
- Red-green-refactor
- 80% coverage requirement

#### 6. Security Review (`skills/security-review/`)
- Security checklist
- OWASP patterns

#### 7-11. Domain Patterns
- `coding-standards/` - Language best practices
- `backend-patterns/` - API, database, caching
- `frontend-patterns/` - React, Next.js
- `clickhouse-io/` - ClickHouse analytics
- `project-guidelines-example/` - Example configs

### 14 Slash Commands

| Command | Purpose |
|---------|---------|
| `/plan` | Create implementation plan |
| `/tdd` | Test-driven development |
| `/e2e` | Generate E2E tests |
| `/code-review` | Quality review |
| `/build-fix` | Fix build errors |
| `/refactor-clean` | Dead code removal |
| `/learn` | Extract patterns mid-session |
| `/checkpoint` | Save verification state |
| `/verify` | Run verification loop |
| `/eval` | Run evaluation harness |
| `/orchestrate` | Multi-agent orchestration |
| `/test-coverage` | Coverage analysis |
| `/update-codemaps` | Update code maps |
| `/update-docs` | Update documentation |

### Comprehensive Hook System

#### PreToolUse Hooks
- Block dev servers outside tmux
- Remind about tmux for long commands
- Pause before git push
- Block unnecessary .md file creation
- Strategic compaction suggestions

#### PostToolUse Hooks
- Auto-format with Prettier after JS/TS edits
- TypeScript check after .ts/.tsx edits
- Warn about console.log statements
- Log PR URL after creation

#### Session Lifecycle Hooks
- `SessionStart`: Load previous context
- `PreCompact`: Save state before compaction
- `Stop`: Final console.log audit, persist state, evaluate for patterns

### 6 Rules (Always-Follow)

1. **security.md** - Mandatory security checks
2. **coding-style.md** - Immutability, file organization
3. **testing.md** - TDD, 80% coverage
4. **git-workflow.md** - Commit format, PR process
5. **agents.md** - When to delegate to subagents
6. **performance.md** - Model selection, context management

---

## 3. INTEGRATION PATTERNS

### Opik + Everything Claude Code Synergy

```python
# Use Opik for tracing during /verify
import opik
from opik.evaluation.metrics import Hallucination, AnswerRelevance

@opik.track(name="verification_loop", tags=["verify"])
def run_verification(code_changes):
    # Build check
    build_ok = run_build()

    # Type check with tracing
    type_errors = run_type_check()

    # Security scan
    security_issues = run_security_scan()

    return {
        "build": build_ok,
        "type_errors": type_errors,
        "security": security_issues
    }
```

### Project-Specific Recommendations

#### For AlphaForge Trading
- **Opik**: Trace risk analysis pipelines, evaluate trading decisions
- **Everything CC**: Use `security-reviewer`, `verification-loop`
- **Metrics**: `ComplianceRiskJudge`, `StructuredOutputCompliance`

#### For State of Witness
- **Opik**: Monitor MediaPipe accuracy, trace pose classification
- **Everything CC**: Use `backend-patterns`, `continuous-learning`
- **Metrics**: `ContextPrecision`, `TrajectoryAccuracy`

#### For Unleash Meta-Development
- **Opik**: Evaluate Ralph Loop iterations, track self-improvement
- **Everything CC**: All agents, `eval-harness` skill
- **Metrics**: All bias judges, `AgentTaskCompletionJudge`

---

## 4. QUICK ACCESS PATTERNS

### Start a New Session
```bash
# Opik tracing
pip install opik && opik configure

# Everything CC commands
/plan "feature description"
/verify
```

### Evaluate LLM Output
```python
from opik.evaluation.metrics import Hallucination, AnswerRelevance

hallucination = Hallucination()
relevance = AnswerRelevance()

result = hallucination.score(
    input="What is the capital of France?",
    output="The capital of France is Paris.",
    context=["France is a country in Europe. Paris is its capital."]
)
```

### Run Verification
```
/verify full    # All checks
/verify quick   # Build + types only
/verify pre-pr  # Full + security scan
```

---

*Document Version: 1.0 | Last Updated: 2026-01-22*
