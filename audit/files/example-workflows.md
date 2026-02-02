# Claude CLI Toolkit - Example Workflows

> Production-ready workflows for AlphaForge Trading and State of Witness Creative/ML development

---

## Table of Contents

1. [Trading Workflows](#1-trading-workflows)
2. [Creative/ML Workflows](#2-creativeml-workflows)
3. [Development Lifecycle Workflows](#3-development-lifecycle-workflows)
4. [Automation Workflows](#4-automation-workflows)

---

## 1. Trading Workflows

### Workflow 1.1: Strategy Development Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                  Strategy Development Workflow                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Research          2. Implement         3. Backtest         │
│   ┌─────────┐         ┌─────────┐         ┌─────────┐          │
│   │ Polygon │────────►│  Code   │────────►│VectorBT │          │
│   │  MCP    │         │Strategy │         │Backtest │          │
│   └─────────┘         └─────────┘         └─────────┘          │
│        │                   │                   │                 │
│        │                   │                   │                 │
│        ▼                   ▼                   ▼                 │
│   4. Optimize         5. Validate         6. Deploy             │
│   ┌─────────┐         ┌─────────┐         ┌─────────┐          │
│   │Riskfolio│────────►│  Snyk   │────────►│ Alpaca  │          │
│   │   Lib   │         │  Audit  │         │ (Paper) │          │
│   └─────────┘         └─────────┘         └─────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Step-by-step Claude CLI session:**

```bash
# Start Claude in AlphaForge project
cd ~/projects/alphaforge && claude

# In Claude session:
```

```
You: Research momentum strategies for tech stocks using Polygon data

Claude: [Uses Polygon MCP to fetch historical data]
        [Analyzes momentum patterns]
        [Provides research summary]

You: Implement a momentum crossover strategy based on your findings

Claude: [Creates src/strategies/momentum_crossover.py]
        [Implements entry/exit logic]
        [Adds risk parameters]

You: /backtest momentum_crossover --symbol AAPL --period 2y

Claude: [Uses VectorBT for backtesting]
        [Runs walk-forward optimization]
        [Generates performance metrics]

You: Optimize the portfolio allocation using Riskfolio-Lib

Claude: [Calculates optimal weights]
        [Applies risk constraints]
        [Generates efficient frontier]

You: /security-audit

Claude: [Uses Snyk MCP to scan]
        [Reviews code for vulnerabilities]
        [Provides security report]

You: Paper trade this strategy with $10,000

Claude: [Uses Alpaca MCP - Paper Mode]
        [Places orders]
        [Sets up monitoring]
```

### Workflow 1.2: Daily Market Analysis

```bash
# Morning market analysis workflow
cd ~/projects/alphaforge && claude
```

```
You: Get pre-market data for my watchlist: AAPL, MSFT, GOOGL, NVDA

Claude: [Uses Polygon MCP]
        - AAPL: Pre-market $185.50 (+0.3%)
        - MSFT: Pre-market $420.25 (+0.5%)
        ...

You: What's the risk exposure on my current positions?

Claude: [Uses Alpaca MCP to get positions]
        [Calculates VaR, sector exposure]
        Current Exposure:
        - Tech sector: 75%
        - VaR (95%): $2,340
        - Max drawdown risk: 8.5%

You: Rebalance to reduce tech exposure to 60%

Claude: [Calculates required trades]
        [Uses Alpaca MCP - Paper Mode]
        Proposed trades:
        - Sell 15 NVDA @ market
        - Buy 20 XLF @ limit $42.50
        
        Confirm? (Paper trading mode)
```

### Workflow 1.3: Strategy Audit

```
You: /strategy-audit src/strategies/mean_reversion.py
```

**Claude performs:**
1. Logic review - entry/exit conditions
2. Risk controls check - stop-loss, position sizing
3. Data handling audit - look-ahead bias, survivorship
4. Performance analysis - vectorization, memory
5. Test coverage check
6. Documentation review

**Output:**
```
Strategy Audit: mean_reversion.py

✓ Entry/exit logic clear
✓ Stop-loss implemented
⚠️ Position sizing uses fixed amount (consider % of portfolio)
✓ No look-ahead bias detected
⚠️ Test coverage: 65% (target: 80%)
✗ Missing docstrings on 3 functions

Priority Fixes:
1. [HIGH] Add dynamic position sizing
2. [MEDIUM] Increase test coverage
3. [LOW] Add docstrings
```

---

## 2. Creative/ML Workflows

### Workflow 2.1: Generative Visual Creation

```
┌─────────────────────────────────────────────────────────────────┐
│                  Generative Visual Workflow                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Concept          2. Generate          3. Integrate         │
│   ┌─────────┐         ┌─────────┐         ┌─────────┐          │
│   │ Claude  │────────►│ComfyUI  │────────►│  Touch  │          │
│   │ Design  │         │  MCP    │         │Designer │          │
│   └─────────┘         └─────────┘         └─────────┘          │
│        │                   │                   │                 │
│        │                   │                   │                 │
│        ▼                   ▼                   ▼                 │
│   4. Control          5. Track            6. Iterate            │
│   ┌─────────┐         ┌─────────┐         ┌─────────┐          │
│   │   TD    │────────►│ MLflow  │────────►│  Loop   │          │
│   │  MCP    │         │   W&B   │         │ Refine  │          │
│   └─────────┘         └─────────┘         └─────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Step-by-step Claude CLI session:**

```bash
# Start Claude in State of Witness project
cd ~/projects/state-of-witness && claude

# In Claude session:
```

```
You: Create an abstract particle system that responds to audio

Claude: [Designs visual concept]
        [Creates TD network architecture]
        
        I'll create a particle system with:
        - Audio-reactive forces
        - Color cycling based on frequency bands
        - Turbulence driven by amplitude

You: /td-control create CHOP audiodevicein audio_in

Claude: [Uses TouchDesigner MCP]
        ✓ Created audiodevicein CHOP "audio_in"

You: Create the full particle network with noise and feedback

Claude: [Uses TouchDesigner MCP - multiple operations]
        ✓ Created particle SOP
        ✓ Created noise TOP
        ✓ Created feedback TOP
        ✓ Connected audio_in → particle forces
        ✓ Connected noise → particle color
        
        Network ready. Preview in TD.

You: Generate some organic textures for the background using ComfyUI

Claude: [Uses ComfyUI MCP]
        Generating with workflow: organic_textures
        Parameters:
        - Style: abstract organic
        - Resolution: 1920x1080
        - Variations: 4
        
        [Generates and saves to assets/]

You: Log this experiment to MLflow

Claude: [Uses MLflow MCP]
        Experiment: particle_audio_reactive_v1
        Parameters logged:
        - particle_count: 10000
        - audio_bands: 8
        - feedback_level: 0.85
        
        Artifacts saved:
        - network_screenshot.png
        - parameter_config.json
```

### Workflow 2.2: ML Experiment Loop

```bash
cd ~/projects/state-of-witness && claude
```

```
You: Set up a quality-diversity experiment for generative patterns

Claude: I'll create a MAP-Elites experiment using pyribs:
        
        [Creates experiments/qd_patterns.py]
        
        Configuration:
        - Archive: 100x100 grid
        - Behavior dimensions: complexity, symmetry
        - Objective: aesthetic score
        - Generations: 500

You: Run the experiment with MLflow tracking

Claude: [Starts experiment with tracking]
        
        MLflow Experiment: qd_patterns_001
        
        Generation 100: 
        - Archive coverage: 23%
        - Best fitness: 0.72
        
        Generation 250:
        - Archive coverage: 45%
        - Best fitness: 0.84
        
        [Continues with progress updates...]
        
        Experiment complete.
        - Final coverage: 67%
        - Best fitness: 0.91
        - Diverse patterns: 6,723

You: Show me the best patterns and integrate them into TouchDesigner

Claude: [Retrieves top patterns from MLflow]
        [Uses TD MCP to create texture network]
        
        Created 10 texture nodes with top patterns:
        - pattern_001 (fitness: 0.91, high complexity)
        - pattern_002 (fitness: 0.89, high symmetry)
        ...
```

### Workflow 2.3: Real-time Performance Setup

```
You: Set up a real-time performance system with audio reactivity

Claude: I'll create a complete performance system:

        1. Audio Input Chain:
        [Uses TD MCP]
        ✓ audiodevicein → audiospectrum → math(normalize)
        
        2. Visual Generators:
        ✓ noise TOP (audio-driven seed)
        ✓ feedback loop
        ✓ particle system
        
        3. Output Pipeline:
        ✓ comp (layering)
        ✓ null (output)
        
        4. Control Panel:
        ✓ Created COMP with sliders for:
           - particle_intensity
           - feedback_amount
           - color_shift
           - noise_scale

You: Test the system with parameter sweeps

Claude: [Adjusts parameters in real-time via TD MCP]
        
        Testing parameter combinations:
        - Low energy: particles=1000, feedback=0.3
        - Medium: particles=5000, feedback=0.6
        - High: particles=20000, feedback=0.9
        
        Performance metrics:
        - FPS at low: 60
        - FPS at medium: 58
        - FPS at high: 42 (consider optimization)
```

---

## 3. Development Lifecycle Workflows

### Workflow 3.1: Feature Development (Full Cycle)

```
You: I need to add a new RSI divergence indicator to AlphaForge

Claude: I'll guide you through the full development cycle:

        **DESIGN PHASE**
        [Researches RSI divergence patterns]
        [Creates technical specification]
        
        Spec: src/indicators/rsi_divergence.py
        - Detect bullish/bearish divergences
        - Configurable lookback period
        - Signal strength scoring

        **IMPLEMENT PHASE**
        [Creates indicator class]
        [Adds unit tests]
        [Documents with docstrings]
        
        Files created:
        - src/indicators/rsi_divergence.py
        - tests/test_rsi_divergence.py
        
        **AUDIT PHASE**
        [Runs ruff formatting]
        [Checks type hints with mypy]
        [Runs Snyk security scan]
        
        Audit results: ✓ All checks passed
        
        **INTEGRATE PHASE**
        [Integrates with strategy framework]
        [Updates imports and registry]
        
        **DEBUG PHASE**
        [Runs full test suite]
        pytest: 42 passed, 0 failed
        
        **VALIDATE PHASE**
        [Backtests with historical data]
        RSI Divergence backtest results:
        - Sharpe: 1.24
        - Win rate: 58%
        - Max drawdown: 12%

You: Great, create a PR for this

Claude: [Uses GitHub MCP]
        
        Created PR #47: Add RSI Divergence Indicator
        
        Description includes:
        - Feature overview
        - Technical implementation
        - Backtest results
        - Test coverage report
```

### Workflow 3.2: Code Review Workflow

```
You: Review the latest changes in the strategies folder

Claude: [Scans changed files]
        [Performs comprehensive review]

        **Code Review: strategies/**
        
        ✓ momentum_v2.py
          - Clean implementation
          - Good error handling
          - Tests comprehensive
        
        ⚠️ mean_reversion.py
          - Line 45: Magic number, use constant
          - Line 78: Consider edge case for empty data
          - Line 102: Redundant calculation in loop
        
        ✗ scalping.py
          - No stop-loss implementation
          - Missing position size limits
          - CRITICAL: Uses hardcoded API endpoint
        
        **Recommendations:**
        1. [CRITICAL] Fix scalping.py security issue
        2. [HIGH] Add stop-loss to scalping strategy
        3. [MEDIUM] Refactor magic numbers
        
        Would you like me to fix these issues?
```

---

## 4. Automation Workflows

### Workflow 4.1: Scheduled Health Check

Create `.claude/commands/health-check.md`:

```markdown
Perform system health check:

1. **MCP Server Status**
   - Check all configured MCP connections
   - Report any failures

2. **API Connectivity**
   - Test Alpaca API (paper)
   - Test Polygon API
   - Verify rate limits

3. **Code Quality**
   - Run linters
   - Check test coverage
   - Identify technical debt

4. **Trading System**
   - Check open positions
   - Verify risk limits
   - Review pending orders

Output a health report with status indicators.
```

**Usage:**
```
You: /health-check

Claude: **System Health Check**
        
        MCP Servers:
        ✓ github: Connected
        ✓ alpaca: Connected (Paper)
        ✓ polygon: Connected
        ✓ snyk: Connected
        ✗ sentry: Auth expired
        
        APIs:
        ✓ Alpaca: 847/1000 requests remaining
        ✓ Polygon: 4,521/unlimited
        
        Code Quality:
        ✓ Linting: 0 errors
        ✓ Coverage: 82%
        ⚠️ Tech debt: 3 TODOs found
        
        Trading:
        ✓ Positions: 5 open
        ✓ Risk: Within limits
        ⚠️ 1 pending order > 24h old
        
        Action Items:
        1. Refresh Sentry auth token
        2. Review stale pending order
```

### Workflow 4.2: CI/CD Integration

**GitHub Actions with Claude Code:**

```yaml
# .github/workflows/claude-code-review.yml
name: Claude Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Claude Code
        run: npm install -g @anthropic-ai/claude-code
        
      - name: Run Code Review
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          claude -p "Review the changes in this PR for:
          1. Code quality and best practices
          2. Security vulnerabilities
          3. Test coverage
          4. Documentation
          Output a structured review." \
          --output-format json > review.json
          
      - name: Post Review
        uses: actions/github-script@v6
        with:
          script: |
            const review = require('./review.json');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: review.content
            });
```

### Workflow 4.3: Automated Documentation

```
You: Generate documentation for the strategies module

Claude: [Analyzes strategies/ directory]
        [Extracts docstrings and type hints]
        [Generates comprehensive docs]

        Created: docs/strategies/
        ├── README.md (module overview)
        ├── momentum.md (strategy docs)
        ├── mean_reversion.md
        ├── api_reference.md
        └── examples.md
        
        Documentation includes:
        - Strategy descriptions
        - Parameter references
        - Usage examples
        - Backtest results
        - Risk warnings
```

---

## Quick Workflow Reference

| Workflow | Command | MCP Servers Used |
|----------|---------|------------------|
| Market Research | `/research SYMBOL` | Polygon |
| Backtest | `/backtest STRATEGY` | (VectorBT SDK) |
| Risk Analysis | `/risk-analysis` | Alpaca, (Riskfolio SDK) |
| Security Audit | `/security-audit` | Snyk |
| Generate Visual | `/generate-visual CONCEPT` | ComfyUI, TouchDesigner |
| TD Control | `/td-control OPERATION` | TouchDesigner |
| Track Experiment | `/experiment-track NAME` | MLflow, W&B |
| Health Check | `/health-check` | All |
| Code Review | Direct request | GitHub, Snyk |

---

*These workflows demonstrate Claude CLI's capabilities as a builder tool across the full development lifecycle. Customize them for your specific needs.*
