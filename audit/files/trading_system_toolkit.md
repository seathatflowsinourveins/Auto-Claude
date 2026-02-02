# Complete Trading System Development Toolkit for Claude Code CLI

**Claude Code CLI becomes a comprehensive trading system builder** with the right combination of official MCPs for market data and execution, specialized analysis servers for backtesting and technical indicators, and Python SDKs for portfolio optimization and risk management. This toolkit covers the full development lifecycle: design, audit, research, integrate, debug, and UI design for trading systems.

The key distinction: **Claude CLI is the BUILDER tool, not part of the production trading system**. These MCPs and SDKs maximize Claude's power to develop, test, and deploy trading systems like AlphaForge.

---

## Tier 1: Official Trading MCPs (Production-Ready)

### 1. Alpaca MCP Server (Official)
**Status**: Production-ready, 43 Trading API endpoints  
**Purpose**: Multi-asset trading execution (stocks, ETFs, options, crypto)

```bash
# Installation via UV (recommended)
claude mcp add alpaca -e ALPACA_API_KEY=your_key -e ALPACA_SECRET_KEY=your_secret -- uvx alpaca-mcp

# Alternative: npm installation
npm install alpaca-mcp-server
```

**Configuration**:
```json
{
  "mcpServers": {
    "alpaca": {
      "command": "uvx",
      "args": ["alpaca-mcp"],
      "env": {
        "ALPACA_API_KEY": "<your_api_key>",
        "ALPACA_SECRET_KEY": "<your_secret_key>",
        "ALPACA_PAPER": "true"
      }
    }
  }
}
```

**Key Capabilities**:
- Place orders (market, limit, stop, bracket orders)
- Options trading (calls, puts, spreads)
- Crypto trading (BTC, ETH, etc.)
- Account management and position tracking
- Real-time and historical market data
- Corporate actions (dividends, splits)

**Example Prompts**:
```
"Buy 10 shares of SPY at market"
"Get SPY call options expiring next week within 10% of market price"
"Show me my portfolio performance"
"Place a bull call spread using AAPL options"
```

---

### 2. QuantConnect MCP Server (Official)
**Status**: Production-ready, Docker-based, 60+ API endpoints  
**Purpose**: Algorithmic trading platform with backtesting and live deployment

```bash
# Pull Docker image
docker pull quantconnect/mcp-server

# Add to Claude Code
claude mcp add quantconnect docker run -i --rm \
  -e QUANTCONNECT_USER_ID \
  -e QUANTCONNECT_API_TOKEN \
  quantconnect/mcp-server
```

**Configuration**:
```json
{
  "mcpServers": {
    "quantconnect": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "QUANTCONNECT_USER_ID",
        "-e", "QUANTCONNECT_API_TOKEN",
        "-e", "AGENT_NAME",
        "--platform", "linux/arm64",
        "quantconnect/mcp-server"
      ],
      "env": {
        "QUANTCONNECT_USER_ID": "<your_user_id>",
        "QUANTCONNECT_API_TOKEN": "<your_api_token>",
        "AGENT_NAME": "Claude-AlphaForge"
      }
    }
  }
}
```

**Key Capabilities**:
- Project management (create, edit, compile)
- Backtest execution and analysis
- Parameter optimization
- Live trading deployment
- Research notebook support (coming soon)
- Multi-asset support (equities, futures, forex, crypto)

**Example Prompts**:
```
"Create a new momentum strategy project"
"Run a backtest on my RSI strategy for the last year"
"Deploy my algorithm to paper trading"
"Optimize the moving average periods for my strategy"
```

---

### 3. Polygon.io / Massive.com MCP Server (Official)
**Status**: Production-ready, 35+ financial data tools  
**Purpose**: Institutional-grade market data (stocks, options, forex, crypto)

> **Note**: Polygon.io rebranded to Massive.com in November 2025. Both repositories work identically.

```bash
# Installation
claude mcp add polygon -e POLYGON_API_KEY=your_api_key -- \
  uvx --from git+https://github.com/polygon-io/mcp_polygon@v0.4.0 mcp_polygon

# Or use Massive.com version
claude mcp add massive -e MASSIVE_API_KEY=your_api_key -- \
  uvx --from git+https://github.com/massive-com/mcp_massive@v0.6.0 mcp_massive
```

**Configuration**:
```json
{
  "mcpServers": {
    "polygon": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/polygon-io/mcp_polygon@v0.4.0",
        "mcp_polygon"
      ],
      "env": {
        "POLYGON_API_KEY": "<your_api_key>"
      }
    }
  }
}
```

**Key Capabilities**:
- Real-time and historical trades/quotes
- OHLCV aggregates with adjustable timeframes
- Market snapshots and movers
- Ticker details and reference data
- Dividends, splits, and financials
- Options chains and forex/crypto data

**Example Prompts**:
```
"Get the latest price for AAPL stock"
"Show me yesterday's trading volume for MSFT"
"What were the biggest stock market gainers today?"
"Get me the latest crypto market data for BTC-USD"
```

---

## Tier 2: Specialized Analysis MCPs

### 4. MaverickMCP (VectorBT-Powered Backtesting)
**Status**: Production-ready, 29+ financial tools  
**Purpose**: Professional-grade technical analysis and backtesting

```bash
# Clone and setup
git clone https://github.com/wshobson/maverick-mcp
cd maverick-mcp
docker-compose up -d
```

**Configuration**:
```json
{
  "mcpServers": {
    "maverick-mcp": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8003/sse/"]
    }
  }
}
```

**Key Capabilities**:
- **VectorBT Integration**: High-performance vectorized backtesting
- **15+ Built-in Strategies**: ML-powered adaptive, ensemble, regime-aware algorithms
- **Technical Analysis**: RSI, MACD, Bollinger Bands, support/resistance
- **Portfolio Optimization**: Position sizing, risk metrics
- **Walk-Forward Optimization**: Out-of-sample testing
- **Monte Carlo Simulations**: Robustness testing

**Example Prompts**:
```
"Run a backtest on AAPL using the momentum strategy for the last 6 months"
"Compare mean reversion vs trend following strategies on SPY"
"Optimize the RSI strategy parameters for TSLA with walk-forward analysis"
"Show me the Sharpe ratio and maximum drawdown for a portfolio of tech stocks"
```

---

### 5. Finance MCP Server (Technical Analysis)
**Status**: Production-ready, 10+ technical indicators  
**Purpose**: Stock data and technical analysis for Claude Desktop

```bash
pip install finance-mcp-server
# Or development install
git clone https://github.com/akshatbindal/finance-mcp-server.git
cd finance-mcp-server
pip install -e .
```

**Key Capabilities**:
- Complete company profiles and market metrics
- Historical price data with multiple timeframes
- Real-time quotes with change calculations
- 10+ professional technical indicators
- Options chain data
- Institutional holders and earnings calendar

---

## Tier 3: Essential Python SDKs (No MCP, Use via CLI)

These SDKs don't have dedicated MCP servers but are critical for trading system development. Claude CLI accesses them through Python execution.

### 6. VectorBT (Backtesting Engine)
**Status**: Production-ready, fastest Python backtester  
**Purpose**: Vectorized backtesting and portfolio simulation

```bash
pip install vectorbt
# Or full installation
pip install -U "vectorbt[full]"
```

**Usage in Claude CLI**:
```python
import vectorbt as vbt

# Download data and backtest
price = vbt.YFData.download('BTC-USD').get('Close')
fast_ma = vbt.MA.run(price, 10)
slow_ma = vbt.MA.run(price, 50)
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)
pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=10000)
print(pf.stats())
```

**Key Features**:
- 2-4x faster than SWIG-based alternatives
- Parameter grid optimization at C speed
- Numba-accelerated computations
- Pandas/Polars integration

---

### 7. Riskfolio-Lib (Portfolio Optimization)
**Status**: Production-ready, most comprehensive Python portfolio lib  
**Purpose**: Quantitative strategic asset allocation

```bash
pip install riskfolio-lib
# For MOSEK solver (recommended for complex models)
pip install mosek
```

**Key Features**:
- 24 convex risk measures (VaR, CVaR, EVaR, drawdown-based)
- 4 objective functions (min risk, max return, max Sharpe, max utility)
- Hierarchical Risk Parity (HRP) and HERC
- Black-Litterman model
- Factor models and risk budgeting
- Graph theory constraints

**Usage in Claude CLI**:
```python
import riskfolio as rp

# Create portfolio object
port = rp.Portfolio(returns=returns_df)
port.assets_stats(method_mu='hist', method_cov='hist')

# Optimize for maximum Sharpe ratio
w = port.optimization(model='Classic', rm='MV', obj='Sharpe')

# Visualize efficient frontier
ax = rp.plot_frontier(w_frontier, port, rm='MV')
```

---

### 8. TA-Lib (Technical Indicators)
**Status**: Production-ready, 200+ indicators  
**Purpose**: High-performance technical analysis

```bash
# macOS
brew install ta-lib
pip install ta-lib

# Linux
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/ && ./configure && make && sudo make install
pip install ta-lib
```

**Key Indicators**:
- Momentum: RSI, MACD, Stochastic, ADX, CCI
- Overlap: SMA, EMA, Bollinger Bands, SAR
- Volatility: ATR, NATR, TRANGE
- Volume: OBV, AD, ADOSC
- Pattern Recognition: Candlestick patterns

**Usage in Claude CLI**:
```python
import talib
import numpy as np

close = np.random.random(100)
sma = talib.SMA(close, timeperiod=20)
rsi = talib.RSI(close, timeperiod=14)
upper, middle, lower = talib.BBANDS(close, timeperiod=20)
```

---

### 9. skfolio (scikit-learn Compatible Portfolio)
**Status**: Production-ready, scikit-learn integration  
**Purpose**: Portfolio optimization with ML pipelines

```bash
pip install skfolio
```

**Key Features**:
- scikit-learn compatible API (fit/predict/transform)
- Cross-validation for portfolio models
- Hyperparameter tuning with GridSearchCV
- Stress testing and robustness analysis

---

## Complete Trading Toolkit Configuration

```json
{
  "mcpServers": {
    "alpaca": {
      "command": "uvx",
      "args": ["alpaca-mcp"],
      "env": {
        "ALPACA_API_KEY": "${ALPACA_API_KEY}",
        "ALPACA_SECRET_KEY": "${ALPACA_SECRET_KEY}",
        "ALPACA_PAPER": "true"
      }
    },
    "quantconnect": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "QUANTCONNECT_USER_ID",
        "-e", "QUANTCONNECT_API_TOKEN",
        "quantconnect/mcp-server"
      ],
      "env": {
        "QUANTCONNECT_USER_ID": "${QC_USER_ID}",
        "QUANTCONNECT_API_TOKEN": "${QC_API_TOKEN}"
      }
    },
    "polygon": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/polygon-io/mcp_polygon@v0.4.0",
        "mcp_polygon"
      ],
      "env": {
        "POLYGON_API_KEY": "${POLYGON_API_KEY}"
      }
    },
    "maverick": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8003/sse/"]
    }
  }
}
```

---

## Development Lifecycle Coverage

| Phase | Tools | Purpose |
|-------|-------|---------|
| **Design** | QuantConnect, Riskfolio-Lib | Strategy architecture, portfolio allocation |
| **Research** | Polygon, VectorBT, TA-Lib | Market data, backtesting, indicators |
| **Audit** | Snyk MCP, SonarQube MCP | Security scanning, code quality |
| **Integrate** | Alpaca MCP, GitHub MCP | Execution, version control |
| **Debug** | Sentry MCP, Grafana MCP | Error tracking, metrics |
| **UI Design** | React artifacts, TradingView widgets | Dashboards, visualizations |

---

## Token Overhead Analysis

| MCP Server | Approximate Token Cost | Priority |
|------------|----------------------|----------|
| Alpaca | ~3,000 tokens | **Critical** |
| QuantConnect | ~4,000 tokens | High |
| Polygon | ~3,500 tokens | High |
| MaverickMCP | ~2,500 tokens | Medium |

**Recommended Configuration**: Enable 3-4 trading MCPs maximum per session to preserve context window for actual development work.

---

## Installation Script

```bash
#!/bin/bash
# trading_toolkit_install.sh

echo "Installing Claude Code Trading System Toolkit..."

# Tier 1: Official MCPs
claude mcp add alpaca -e ALPACA_API_KEY=$ALPACA_API_KEY -e ALPACA_SECRET_KEY=$ALPACA_SECRET_KEY -- uvx alpaca-mcp

claude mcp add polygon -e POLYGON_API_KEY=$POLYGON_API_KEY -- uvx --from git+https://github.com/polygon-io/mcp_polygon@v0.4.0 mcp_polygon

# Docker-based MCPs
docker pull quantconnect/mcp-server

# Python SDKs
pip install vectorbt riskfolio-lib ta-lib skfolio

echo "Trading toolkit installation complete!"
```

---

## Key Takeaways

1. **Official MCPs First**: Alpaca, QuantConnect, and Polygon are production-ready and officially maintained
2. **MaverickMCP for Backtesting**: Combines VectorBT's speed with MCP accessibility
3. **Python SDKs via CLI**: Riskfolio-Lib and TA-Lib have no MCPs but are accessible through Claude's Python execution
4. **Token Budget**: Keep 3-4 trading MCPs active to preserve context for development
5. **Paper Trading**: Always use paper trading mode (`ALPACA_PAPER=true`) during development

This toolkit transforms Claude CLI into a comprehensive trading system builder capable of the full development lifecycle—from strategy design through backtesting, security auditing, and deployment.
