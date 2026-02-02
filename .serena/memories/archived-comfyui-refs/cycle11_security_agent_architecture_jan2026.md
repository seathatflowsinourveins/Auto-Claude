# Cycle 11: Security & Agent Architecture
## January 25, 2026 - Perpetual Enhancement Loop

---

## Executive Summary

Cycle 11 reveals two critical themes:
1. **MCP Security Crisis**: 43% of servers vulnerable, multiple high-severity CVEs
2. **Agent Memory Evolution**: Multi-graph architectures replacing monolithic stores

---

## 1. MCP Security - Critical Updates

### Vulnerability Landscape

| Metric | Value |
|--------|-------|
| Servers vulnerable to command injection | 43% |
| Installations affected by CVEs | 437,000+ |
| CVSS scores of identified CVEs | 7.3 - 9.6 |
| Organizations lacking AI access controls | 97% |

### Key Attack Vectors

#### 1. Confused Deputy Problem
The agent acts on behalf of the user but can be manipulated:
```
Attacker → Malicious Prompt → Agent → Privileged Action → System Compromise
```

#### 2. Tool Poisoning
Malicious tool definitions injected via:
- PyPI packages (npm equivalent attacks)
- Modified tool schemas
- MITM on tool discovery

#### 3. Session Hijacking
- Prompt injection to capture session
- Impersonation attacks
- Token theft via filesystem access

### Mandatory Security Requirements

#### OAuth 2.1 + PKCE (Required for Remote MCP)
```python
from mcp.auth import OAuth21Client

client = OAuth21Client(
    client_id="your_client_id",
    redirect_uri="http://localhost:3000/callback",
    pkce=True,  # MANDATORY
    scopes=["read", "execute"]
)

# Authorization flow
auth_url = client.get_authorization_url(
    code_challenge=client.generate_code_challenge(),
    code_challenge_method="S256"
)
```

#### Security Checklist (Production MCP)
```json
{
  "security": {
    "oauth21_pkce": true,
    "sandbox_all_tools": true,
    "network_isolation": true,
    "scope_minimization": true,
    "input_validation": "strict",
    "output_sanitization": true,
    "audit_logging": true,
    "rate_limiting": {
      "enabled": true,
      "max_requests_per_minute": 60
    }
  },
  "blocked_operations": [
    "shell_exec_arbitrary",
    "file_delete_recursive",
    "network_external_unrestricted"
  ]
}
```

### Defense in Depth Strategy
1. **Network Controls**: Block external exfiltration
2. **File Permissions**: ReadOnly by default
3. **Tool Whitelisting**: Explicit allow, implicit deny
4. **Input Sanitization**: All tool inputs validated
5. **Output Monitoring**: Detect sensitive data leakage

---

## 2. Claude Agent SDK Architecture

### The Agent Harness Pattern (Official Anthropic)

From Thariq Shihipar's workshop - the canonical implementation:

```typescript
// The Agent Loop: Context → Thought → Action → Observation
class AgentHarness {
  context: Context;
  tools: Tool[];
  
  async run(task: string): Promise<Result> {
    this.context.add(task);
    
    while (!this.isComplete()) {
      // THINK: Model reasons about current state
      const thought = await this.think(this.context);
      
      // ACT: Execute tool based on thought
      const action = await this.act(thought);
      
      // OBSERVE: Capture result
      const observation = await this.observe(action);
      
      // UPDATE: Append to context
      this.context.add(thought, action, observation);
    }
    
    return this.extractResult();
  }
}
```

### Context Engineering Patterns

#### Filesystem as Memory
```typescript
// Use ls and cat to build dynamic context
async function buildContext(path: string): Promise<string> {
  const structure = await tools.ls(path);
  const relevantFiles = selectRelevant(structure);
  
  const contexts = await Promise.all(
    relevantFiles.map(f => tools.cat(f))
  );
  
  return contexts.join('\n---\n');
}
```

#### Safety Permissions
```typescript
interface ToolPermissions {
  file: 'ReadOnly' | 'ReadWrite';
  network: 'Blocked' | 'AllowList' | 'Open';
  shell: 'Disabled' | 'SafeCommands' | 'Unrestricted';
}

const PRODUCTION_PERMISSIONS: ToolPermissions = {
  file: 'ReadOnly',      // Default safe
  network: 'AllowList',  // Explicit allows only
  shell: 'SafeCommands'  // Whitelisted commands
};
```

### Building a Code Review Agent (Nader Dabit Pattern)
```typescript
import { Agent, Tool } from '@anthropic-ai/claude-agent-sdk';

const codeReviewAgent = new Agent({
  model: 'claude-opus-4-5-20251101',
  tools: [
    Tool.Read,      // File reading
    Tool.Search,    // Code search
    Tool.Analyze    // Pattern analysis
  ],
  prompt: `You are a security-focused code reviewer.
    Analyze the codebase for:
    1. Security vulnerabilities (OWASP Top 10)
    2. Logic bugs
    3. Performance issues
    Return structured feedback as JSON.`
});

const result = await codeReviewAgent.run({
  path: './src',
  focus: ['security', 'bugs']
});
```

---

## 3. Agent Memory Architecture (arxiv Papers)

### arXiv:2601.12560 - Agentic AI Taxonomy

**Key Insight**: Unified taxonomy from single-loop agents to hierarchical multi-agent systems

```
Agent Types:
├── Single-Loop Agents (ReAct, CoT)
├── Multi-Step Agents (Plan-Execute)
├── Hierarchical Agents (Orchestrator + Workers)
└── Multi-Agent Systems (Collaborative)
```

### arXiv:2601.03236 - MAGMA Multi-Graph Memory

**Problem**: Monolithic memory stores entangle temporal, causal, and entity information

**Solution**: Orthogonal graphs with policy-guided traversal

```python
class MAGMAMemory:
    """Multi-Graph Agentic Memory Architecture"""
    
    graphs = {
        "semantic": SemanticGraph(),    # Meaning relationships
        "temporal": TemporalGraph(),    # Time relationships
        "causal": CausalGraph(),        # Cause-effect chains
        "entity": EntityGraph()         # Named entities
    }
    
    def retrieve(self, query: Query) -> Context:
        # Policy-guided traversal over relational views
        results = []
        for graph_name, graph in self.graphs.items():
            if self.policy.should_traverse(query, graph_name):
                results.extend(graph.traverse(query))
        
        return self.construct_context(results)
```

### arXiv:2601.06037 - TeleMem Long-Term Multimodal

**Key Innovation**: Narrative dynamic extraction for coherent user profiles

```python
class TeleMem:
    """Long-term multimodal memory system"""
    
    def process_dialogue(self, dialogue: List[Turn]):
        # Extract only dialogue-grounded information
        narratives = self.extract_narratives(dialogue)
        
        # Structured writing pipeline
        batched = self.batch(narratives)
        clustered = self.cluster(batched)
        
        # Write to memory with conflict resolution
        for cluster in clustered:
            self.write_with_resolution(cluster)
```

### arXiv:2601.15709 - AgentSM Semantic Memory

**Application**: Text-to-SQL with structured execution traces

```python
class AgentSM:
    """Semantic Memory for Agentic SQL"""
    
    def store_trace(self, query: str, sql: str, result: Any):
        # Structure as program, not raw scratchpad
        program = self.compile_to_program(query, sql)
        self.memory.store(program)
    
    def retrieve_similar(self, new_query: str) -> Program:
        # Reuse prior execution traces
        return self.memory.find_similar(new_query)
```

---

## 4. Cross-Project Application

### WITNESS
- **MCP Security**: Sandbox TouchDesigner + ComfyUI servers
- **Agent Harness**: Apply to creative exploration loop
- **MAGMA Memory**: Separate aesthetic/temporal/entity graphs

### TRADING
- **Security**: OAuth 2.1 for all external APIs
- **Defense in Depth**: Network isolation for market data
- **Causal Graph**: Track trade decision causality

### UNLEASH
- **Full Security Audit**: Apply 43% vulnerability check
- **TeleMem Pattern**: Long-term research memory
- **AgentSM**: SQL-based research retrieval

---

## 5. Security Configuration Template

```json
{
  "mcp_security_v2": {
    "version": "2026-01-25",
    "auth": {
      "oauth21": true,
      "pkce": true,
      "token_rotation": "1h"
    },
    "sandbox": {
      "filesystem": {
        "root": "./project",
        "mode": "read_only",
        "exceptions": ["./output"]
      },
      "network": {
        "mode": "allow_list",
        "allowed": ["api.anthropic.com", "localhost"]
      },
      "shell": {
        "mode": "safe_commands",
        "allowed": ["git status", "npm test", "python -m pytest"]
      }
    },
    "monitoring": {
      "audit_log": true,
      "anomaly_detection": true,
      "sensitive_data_scan": true
    }
  }
}
```

---

## Tags
`#mcp-security` `#agent-sdk` `#memory-architecture` `#arxiv` `#cycle11`

## References
- APISec University: MCP Security Best Practices
- Network Intelligence: MCP Security Checklist
- Model Context Protocol: Official Security Docs
- Thariq Shihipar (Anthropic): Agent SDK Workshop
- Nader Dabit: Complete Agent SDK Guide
- arXiv:2601.12560: Agentic AI Taxonomy
- arXiv:2601.03236: MAGMA Multi-Graph Memory
- arXiv:2601.06037: TeleMem Long-Term Memory
- arXiv:2601.15709: AgentSM Semantic Memory
