# SDK Code Patterns V31 - Production-Ready Implementations

**Created**: 2026-01-23
**Purpose**: Detailed code patterns for seamless cross-session access

---

## 1. A2A PROTOCOL (Agent-to-Agent Communication)

### Google A2A Protocol - JSON-RPC 2.0 over HTTP
```python
# Agent Card for discovery (Google A2A spec)
agent_card = {
    "name": "trading-risk-analyst",
    "description": "Analyzes portfolio risk and suggests actions",
    "url": "https://agent.example.com/a2a",
    "capabilities": ["risk_analysis", "portfolio_review"],
    "authentication": {"type": "bearer", "scheme": "oauth2"}
}

# LiteLLM A2A Gateway
from litellm import completion
from litellm.a2a_protocol import A2AClient, AgentCard

# Register agent
client = A2AClient()
client.register_agent(AgentCard(**agent_card))

# Send task to another agent
response = await client.send_task(
    agent_id="trading-risk-analyst",
    task="Evaluate portfolio risk for AAPL 1000 shares @ $185.50",
    context={"urgency": "high", "market_hours": True}
)

# Streaming response
async for chunk in client.stream_task(agent_id="analyst", task=prompt):
    print(chunk, end="")
```

---

## 2. FASTMCP V2 - Server Composition & Enterprise Auth

### Server Composition Pattern
```python
from fastmcp import FastMCP
from fastmcp.server import Server

# Create specialized servers
api_server = FastMCP(name="API Server")
auth_server = FastMCP(name="Auth Server")
data_server = FastMCP(name="Data Server")

@api_server.tool()
async def get_data(query: str) -> dict:
    return {"result": query}

# Parent server with composition
main_server = FastMCP(name="Main Gateway")
main_server.mount("/api", api_server)
main_server.mount("/auth", auth_server)
main_server.mount("/data", data_server)

# Multi-server proxy pattern
config = {
    "mcpServers": {
        "weather": {"url": "https://weather.example.com/mcp"},
        "calendar": {"url": "https://calendar.example.com/mcp"},
        "trading": {"url": "https://trading.example.com/mcp"}
    }
}
composite_proxy = FastMCP.as_proxy(config, name="Unified Proxy")
```

### Enterprise OAuth Patterns
```python
from fastmcp import FastMCP
from fastmcp.server.auth.providers.github import GitHubProvider
from fastmcp.server.auth.providers.google import GoogleProvider
from fastmcp.server.auth import JWTValidator
import os

# GitHub OAuth
github_auth = GitHubProvider(
    client_id=os.environ["GITHUB_CLIENT_ID"],
    client_secret=os.environ["GITHUB_CLIENT_SECRET"],
    base_url="http://localhost:8000",
    required_scopes=["read:user", "read:org"]
)

# Google OAuth
google_auth = GoogleProvider(
    client_id=os.environ["GOOGLE_CLIENT_ID"],
    client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
    base_url="http://localhost:8000"
)

# JWT Validation with JWKS
jwt_validator = JWTValidator(
    jwks_url="https://auth.example.com/.well-known/jwks.json",
    audience="my-app",
    issuer="https://auth.example.com"
)

# Server with auth
mcp = FastMCP(name="Secure Server", auth=github_auth)

@mcp.tool()
async def protected_action(user: str) -> str:
    # user comes from authenticated context
    return f"Action for {user}"
```

---

## 3. DSPY GEPA OPTIMIZER - Reflective Prompt Evolution

### Core GEPA Pattern
```python
import dspy
from dspy import GEPA

# Configure LLM
dspy.configure(lm=dspy.LM("anthropic/claude-opus-4-5-20251101"))

# Define metric with feedback (CRITICAL for GEPA)
def metric_with_feedback(gold, pred, trace=None):
    """GEPA requires feedback in failed cases for reflection."""
    if gold.answer == pred.answer:
        return 1.0
    else:
        # Textual feedback enables reflective improvement
        feedback = (
            f"Incorrect. Expected '{gold.answer}' but got '{pred.answer}'. "
            f"Consider: {gold.reasoning_hint}"
        )
        return {'score': 0.0, 'feedback': feedback}

# Alternative: GEPAFeedbackMetric wrapper
from dspy.evaluate import GEPAFeedbackMetric

base_metric = lambda gold, pred: gold.answer == pred.answer
gepa_metric = GEPAFeedbackMetric(
    base_metric,
    feedback_template="Wrong answer. Expected: {expected}, Got: {actual}. Think about {hint}."
)

# GEPA optimizer configuration
gepa = GEPA(
    metric=metric_with_feedback,
    reflection_lm=dspy.LM("anthropic/claude-sonnet-4-20250514"),  # Cheaper for reflection
    candidate_selection_strategy="pareto",  # Pareto-optimal selection
    auto="light",  # or "medium", "heavy"
    perfect_score=1.0,
    failure_score=0.0,
    track_stats=True,
    num_candidates=10,
    num_iterations=5
)

# Compile optimized program
class QAProgram(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.generate(question=question)

optimized_program = gepa.compile(
    student=QAProgram(),
    trainset=train_data,
    valset=val_data
)

# Save optimized prompts
optimized_program.save("optimized_qa.json")
```

### GEPA Performance Notes
- Outperforms MIPROv2 by 10-14% on benchmarks
- Feedback-driven reflection is key differentiator
- Pareto selection balances quality vs efficiency
- Use "light" for quick iterations, "heavy" for production

---

## 4. TEMPORAL DURABLE EXECUTION

### Basic Workflow Pattern
```python
from temporalio import workflow, activity
from temporalio.client import Client
from temporalio.worker import Worker
from datetime import timedelta
from pydantic import BaseModel

class OrderData(BaseModel):
    symbol: str
    quantity: int
    price: float

@activity.defn
async def validate_order(order: OrderData) -> bool:
    # Validation logic
    return order.quantity > 0 and order.price > 0

@activity.defn
async def execute_trade(order: OrderData) -> str:
    # Trade execution - this is durable
    return f"Executed {order.quantity} {order.symbol} @ {order.price}"

@activity.defn  
async def notify_completion(trade_id: str) -> None:
    # Send notifications
    pass

@workflow.defn
class TradeWorkflow:
    """Crash-proof trading workflow with retry logic."""
    
    @workflow.run
    async def run(self, order: OrderData) -> str:
        # Activity 1: Validate
        is_valid = await workflow.execute_activity(
            validate_order,
            order,
            start_to_close_timeout=timedelta(seconds=30)
        )
        
        if not is_valid:
            raise ValueError("Order validation failed")
        
        # Activity 2: Execute with retries
        trade_id = await workflow.execute_activity(
            execute_trade,
            order,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=workflow.RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=1),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(seconds=30)
            )
        )
        
        # Activity 3: Notify (fire-and-forget style)
        await workflow.execute_activity(
            notify_completion,
            trade_id,
            start_to_close_timeout=timedelta(seconds=10)
        )
        
        return trade_id

# Worker setup
async def main():
    client = await Client.connect("localhost:7233")
    
    async with Worker(
        client,
        task_queue="trading-tasks",
        workflows=[TradeWorkflow],
        activities=[validate_order, execute_trade, notify_completion]
    ):
        # Worker runs until cancelled
        await asyncio.Future()

# Client usage
async def submit_trade():
    client = await Client.connect("localhost:7233")
    result = await client.execute_workflow(
        TradeWorkflow.run,
        OrderData(symbol="AAPL", quantity=100, price=185.50),
        id="trade-001",
        task_queue="trading-tasks"
    )
    print(f"Trade completed: {result}")
```

### Pydantic-AI + Temporal Integration
```python
from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import TemporalAgent, PydanticAIPlugin, AgentPlugin

# Create Pydantic-AI agent
agent = Agent(
    'anthropic/claude-opus-4-5-20251101',
    instructions="You are an expert trading analyst."
)

# Wrap for Temporal durability
temporal_agent = TemporalAgent(agent)

@workflow.defn
class AnalysisWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_agent.run(prompt)
        return result.output

# Worker with Pydantic-AI plugin
async def run_worker():
    client = await Client.connect(
        'localhost:7233',
        plugins=[PydanticAIPlugin()]
    )
    
    async with Worker(
        client,
        task_queue='analysis',
        workflows=[AnalysisWorkflow],
        plugins=[AgentPlugin(temporal_agent)]
    ):
        await asyncio.Future()
```

---

## 5. PYRIBS MAP-ELITES (Quality-Diversity)

### GridArchive + CMA-ME Pattern
```python
import numpy as np
from ribs.archives import GridArchive, CVTArchive
from ribs.emitters import GaussianEmitter, ImprovementEmitter
from ribs.schedulers import Scheduler

# Define archive (behavioral space)
archive = GridArchive(
    solution_dim=10,  # Solution vector dimension
    dims=[50, 50],    # 50x50 grid
    ranges=[          # Behavioral descriptor ranges
        (0.0, 1.0),   # Energy measure
        (0.0, 1.0)    # Coherence measure
    ],
    dtype={
        "solution": np.float32,
        "objective": np.float32,
        "measures": np.float32
    }
)

# Alternative: CVT Archive (Centroidal Voronoi Tessellation)
cvt_archive = CVTArchive(
    solution_dim=10,
    cells=1000,  # Number of cells
    ranges=[(0, 1), (0, 1)]
)

# Emitters (solution generators)
emitters = [
    GaussianEmitter(archive, sigma=0.1, batch_size=32)
    for _ in range(5)
]

# Or improvement-based emitters
improvement_emitters = [
    ImprovementEmitter(archive, sigma=0.1, batch_size=32)
    for _ in range(5)  
]

# Scheduler orchestrates ask-tell
scheduler = Scheduler(archive, emitters)

# Evaluation function (domain-specific)
def evaluate_solutions(solutions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        objectives: Quality scores (higher is better)
        measures: Behavioral descriptors (2D for this archive)
    """
    objectives = np.zeros(len(solutions))
    measures = np.zeros((len(solutions), 2))
    
    for i, sol in enumerate(solutions):
        # Your domain-specific evaluation
        objectives[i] = compute_quality(sol)
        measures[i] = [compute_energy(sol), compute_coherence(sol)]
    
    return objectives, measures

# Main optimization loop
for iteration in range(1000):
    # Ask for new solutions
    solutions = scheduler.ask()
    
    # Evaluate solutions
    objectives, measures = evaluate_solutions(solutions)
    
    # Tell results back
    scheduler.tell(objectives, measures)
    
    # Log progress
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: "
              f"Archive size={len(archive)}, "
              f"Coverage={archive.stats.coverage:.2%}, "
              f"Max fitness={archive.stats.obj_max:.4f}")

# Retrieve best solutions
elite = archive.best_elite
print(f"Best solution: {elite.solution}, fitness: {elite.objective}")

# Sample diverse solutions
samples = archive.sample_elites(10)
for s in samples:
    print(f"Solution: {s.solution[:3]}..., Measures: {s.measures}")
```

### Creative Exploration (Witness Project)
```python
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter

class ShaderExplorer:
    """MAP-Elites for shader parameter exploration."""
    
    def __init__(self):
        # 2D behavioral space: visual_energy, color_coherence
        self.archive = GridArchive(
            solution_dim=8,  # Shader parameters
            dims=[25, 25],
            ranges=[(0, 1), (0, 1)]
        )
        
        self.emitters = [
            EvolutionStrategyEmitter(
                self.archive,
                x0=np.zeros(8),
                sigma0=0.5
            )
        ]
        
        self.scheduler = Scheduler(self.archive, self.emitters)
    
    async def explore(self, iterations: int = 100):
        for _ in range(iterations):
            params = self.scheduler.ask()
            
            # Evaluate aesthetics (async for GPU rendering)
            objectives, measures = await self.evaluate_aesthetics(params)
            
            self.scheduler.tell(objectives, measures)
        
        return self.archive
    
    async def evaluate_aesthetics(self, params):
        """Evaluate shader parameters for visual quality."""
        objectives = []
        measures = []
        
        for p in params:
            # Render with TouchDesigner MCP
            render = await render_shader(p)
            
            # Compute aesthetic metrics
            quality = compute_aesthetic_score(render)
            energy = compute_visual_energy(render)
            coherence = compute_color_coherence(render)
            
            objectives.append(quality)
            measures.append([energy, coherence])
        
        return np.array(objectives), np.array(measures)
```

---

## 6. LLM-GUARD SAFETY PIPELINE

### Input/Output Scanning
```python
from llm_guard import scan_prompt, scan_output
from llm_guard.input_scanners import (
    PromptInjection,
    Toxicity,
    Secrets,
    InvisibleText,
    BanTopics,
    Language
)
from llm_guard.output_scanners import (
    FactualConsistency,
    Bias,
    MaliciousURLs,
    Relevance,
    Sensitive
)

# Configure input scanners
input_scanners = [
    PromptInjection(threshold=0.9),
    Toxicity(threshold=0.7),
    Secrets(),
    InvisibleText(),
    BanTopics(topics=["violence", "illegal activities"]),
    Language(valid_languages=["en"])
]

# Configure output scanners
output_scanners = [
    FactualConsistency(minimum_score=0.7),
    Bias(threshold=0.8),
    MaliciousURLs(),
    Relevance(threshold=0.5),
    Sensitive()
]

class SafetyPipeline:
    def __init__(self):
        self.input_scanners = input_scanners
        self.output_scanners = output_scanners
    
    def scan_input(self, prompt: str) -> tuple[str, bool, dict]:
        """Scan and sanitize input prompt."""
        sanitized, results_valid, results = scan_prompt(
            self.input_scanners,
            prompt
        )
        
        risk_scores = {
            scanner.__class__.__name__: result
            for scanner, result in zip(self.input_scanners, results)
        }
        
        return sanitized, results_valid, risk_scores
    
    def scan_output(self, prompt: str, output: str) -> tuple[str, bool, dict]:
        """Scan LLM output for safety issues."""
        sanitized, results_valid, results = scan_output(
            self.output_scanners,
            prompt,
            output
        )
        
        risk_scores = {
            scanner.__class__.__name__: result
            for scanner, result in zip(self.output_scanners, results)
        }
        
        return sanitized, results_valid, risk_scores

# Usage
pipeline = SafetyPipeline()

# Check input
safe_prompt, input_valid, input_risks = pipeline.scan_input(user_prompt)
if not input_valid:
    raise SecurityError(f"Input rejected: {input_risks}")

# Generate response
response = await llm.generate(safe_prompt)

# Check output
safe_output, output_valid, output_risks = pipeline.scan_output(safe_prompt, response)
if not output_valid:
    raise SecurityError(f"Output rejected: {output_risks}")
```

---

## 7. UNIFIED SAFETY (NeMo + LLM-Guard)

```python
from nemoguardrails import LLMRails, RailsConfig
from llm_guard import scan_prompt, scan_output
from llm_guard.input_scanners import PromptInjection, Toxicity
from llm_guard.output_scanners import FactualConsistency

class UnifiedSafetyPipeline:
    """3-layer safety: LLM-Guard input → NeMo → LLM-Guard output"""
    
    def __init__(self, rails_config_path: str):
        self.rails = LLMRails(RailsConfig.from_path(rails_config_path))
        self.input_scanners = [PromptInjection(), Toxicity()]
        self.output_scanners = [FactualConsistency()]
    
    async def safe_generate(self, prompt: str) -> str:
        # Layer 1: LLM-Guard input scan
        sanitized, valid, risk = scan_prompt(self.input_scanners, prompt)
        if not valid:
            raise SecurityError(f"Input rejected: {risk}")
        
        # Layer 2: NeMo Guardrails (topical, dialog flow)
        response = await self.rails.generate_async(
            messages=[{"role": "user", "content": sanitized}]
        )
        
        # Layer 3: LLM-Guard output scan
        _, valid, risk = scan_output(
            self.output_scanners, 
            sanitized, 
            response["content"]
        )
        if not valid:
            raise SecurityError(f"Output rejected: {risk}")
        
        return response["content"]
```

---

## Quick Access Patterns

### Start Any Session
```python
# Core imports
from dspy import GEPA
from temporalio import workflow
from ribs.archives import GridArchive
from fastmcp import FastMCP
from llm_guard import scan_prompt

# Initialize
dspy.configure(lm=dspy.LM("anthropic/claude-opus-4-5-20251101"))
```

### Project-Specific Starters

**UNLEASH**: `from dspy import GEPA; from temporalio import workflow`
**WITNESS**: `from ribs.archives import GridArchive; from ribs.schedulers import Scheduler`
**TRADING**: `from temporalio import workflow; from nemoguardrails import LLMRails`

---

*Memory Version: 31.0 | Auto-accessible across all sessions*
